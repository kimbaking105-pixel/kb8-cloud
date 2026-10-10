#!/usr/bin/env python3
"""Verify KB8 two-head v2 checkpoint, exported weights, NumPy and candidate C++.

This is a static inference check on ten fixed FENs. It is not a playing-strength
or Elo test, and it never reads or evaluates the training shard/test split.
"""
from __future__ import annotations

import argparse
import math
import re
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np

MAGIC = b"KB8F32\0\0"
VERSION = 2
FEATURES = 6144
WIDTH = 128
PERSPECTIVES = 2
HIDDEN = 16
BUCKETS = 4
PAYLOAD_FLOATS = 803144
SCORE_SCALE = np.float32(400.0)

FT_WEIGHT_OFFSET = 0
FT_BIAS_OFFSET = FT_WEIGHT_OFFSET + FEATURES * WIDTH
SHARED_OFFSET = FT_BIAS_OFFSET + WIDTH
SHARED_BIAS_OFFSET = SHARED_OFFSET + BUCKETS * HIDDEN * 2 * WIDTH
SCORE_HEAD_OFFSET = SHARED_BIAS_OFFSET + BUCKETS * HIDDEN
SCORE_BIAS_OFFSET = SCORE_HEAD_OFFSET + BUCKETS * HIDDEN
WDL_HEAD_OFFSET = SCORE_BIAS_OFFSET + BUCKETS
WDL_BIAS_OFFSET = WDL_HEAD_OFFSET + BUCKETS * HIDDEN
assert WDL_BIAS_OFFSET + BUCKETS == PAYLOAD_FLOATS

FENS = [
    ("start-white", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("start-black", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"),
    ("middlegame-white", "r2q1rk1/pp2bppp/2npbn2/2p1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 10"),
    ("middlegame-black", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NPBN2/PP2QPPP/R2R2K1 b - - 0 12"),
    ("rook-endgame", "8/5pk1/6p1/3R4/4P3/6P1/5PK1/8 w - - 0 35"),
    ("queen-endgame-black", "8/2q3k1/8/3P4/4K3/8/8/8 b - - 0 51"),
    ("king-pawn-white", "8/8/8/3k4/8/4K3/4P3/8 w - - 0 1"),
    ("king-pawn-black", "8/8/8/3k4/8/4K3/4P3/8 b - - 0 1"),
    ("low-material", "8/8/8/8/4k3/8/8/4K2R w - - 0 1"),
    ("promotion-race", "8/P7/8/3k4/8/4K3/7p/8 b - - 0 1"),
]


def load_weights(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 44:
        raise ValueError("network file is shorter than the 44-byte header")
    header = struct.unpack("<8s7IfI", data[:44])
    magic, version, features, width, perspectives, hidden, buckets, count, scale, crc = header
    expected = (MAGIC, VERSION, FEATURES, WIDTH, PERSPECTIVES, HIDDEN, BUCKETS, PAYLOAD_FLOATS)
    actual = (magic, version, features, width, perspectives, hidden, buckets, count)
    if actual != expected:
        raise ValueError(f"network header/architecture mismatch: {actual!r}")
    if np.float32(scale) != SCORE_SCALE:
        raise ValueError(f"unexpected score scale: {scale}")
    payload = data[44:]
    if len(payload) != PAYLOAD_FLOATS * 4:
        raise ValueError(f"payload length mismatch: {len(payload)} bytes")
    if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
        raise ValueError("payload CRC32 mismatch")
    weights = np.frombuffer(payload, dtype="<f4")
    if not np.isfinite(weights).all():
        raise ValueError("network has a non-finite weight")
    return weights


def parse_fen_features(fen: str):
    fields = fen.split()
    if len(fields) != 6 or fields[1] not in ("w", "b"):
        raise ValueError(f"invalid FEN: {fen}")
    pieces: list[tuple[int, int, int]] = []  # square a1=0, color 0/1, piece type 0..5
    kings = [-1, -1]
    ranks = fields[0].split("/")
    if len(ranks) != 8:
        raise ValueError(f"invalid FEN placement: {fen}")
    for fen_rank, rank_text in enumerate(ranks):
        file_index = 0
        board_rank = 7 - fen_rank
        for char in rank_text:
            if char in "12345678":
                file_index += int(char)
            elif char.lower() in "pnbrqk":
                if file_index >= 8:
                    raise ValueError(f"FEN rank is wider than 8: {fen}")
                color = 0 if char.isupper() else 1
                piece_type = "pnbrqk".index(char.lower())
                square = board_rank * 8 + file_index
                pieces.append((square, color, piece_type))
                if piece_type == 5:
                    kings[color] = square
                file_index += 1
            else:
                raise ValueError(f"invalid FEN character {char!r}")
        if file_index != 8:
            raise ValueError(f"FEN rank does not cover 8 squares: {fen}")
    if not 2 <= len(pieces) <= 32 or min(kings) < 0:
        raise ValueError(f"invalid kings or piece count in FEN: {fen}")

    stm_color = 0 if fields[1] == "w" else 1
    numpy_views: list[list[int]] = []
    torch_views: list[list[int]] = []
    for perspective in (stm_color, 1 - stm_color):
        oriented_king = kings[perspective] ^ (56 if perspective == 1 else 0)
        king_file = oriented_king & 7
        mirror = 7 if king_file >= 4 else 0
        king_bucket = (4 if (oriented_king >> 3) >= 4 else 0) + (king_file ^ mirror)
        feature_pairs: list[tuple[int, int]] = []
        for square, color, piece_type in pieces:
            plane = 6 * int(color != perspective) + piece_type
            oriented_square = square ^ (56 if perspective == 1 else 0)
            mapped_square = oriented_square ^ mirror
            feature = king_bucket * 768 + plane * 64 + mapped_square
            if not 0 <= feature < FEATURES:
                raise ValueError("feature index out of range")
            feature_pairs.append((square, feature))
        # The training tensor sums features in FEN scan order; C++ scans square 0..63.
        torch_views.append([feature for _, feature in feature_pairs])
        numpy_views.append([feature for _, feature in sorted(feature_pairs)])
    material_bucket = min(3, (len(pieces) - 1) // 8)
    return numpy_views, torch_views, material_bucket


def clip01(value: np.float32) -> np.float32:
    if value < np.float32(0.0):
        return np.float32(0.0)
    if value > np.float32(1.0):
        return np.float32(1.0)
    return value


def sigmoid32(value: np.float32) -> np.float32:
    x = float(value)
    if x >= 40.0:
        return np.float32(1.0)
    if x <= -40.0:
        return np.float32(0.0)
    return np.float32(1.0 / (1.0 + math.exp(-x)))


def round_cp(score_norm: np.float32) -> int:
    scaled = float(np.float32(score_norm * SCORE_SCALE))
    rounded = math.floor(scaled + 0.5) if scaled >= 0.0 else math.ceil(scaled - 0.5)
    return max(-15000, min(15000, rounded))


def numpy_eval(weights: np.ndarray, fen: str) -> tuple[int, float, float, float]:
    views, _, bucket = parse_fen_features(fen)
    accumulators = []
    ft_bias = weights[FT_BIAS_OFFSET:FT_BIAS_OFFSET + WIDTH]
    for active in views:
        accum = ft_bias.copy()
        for feature in active:
            row = weights[FT_WEIGHT_OFFSET + feature * WIDTH:FT_WEIGHT_OFFSET + (feature + 1) * WIDTH]
            np.add(accum, row, out=accum)
        accum = np.where(accum < 0.0, 0.0, np.where(accum > 1.0, 1.0, accum)).astype(np.float32, copy=False)
        accumulators.append(accum)
    joined = np.concatenate(accumulators).astype(np.float32, copy=False)

    hidden = np.empty(HIDDEN, dtype=np.float32)
    shared_base = SHARED_OFFSET + bucket * HIDDEN * 2 * WIDTH
    shared_bias_base = SHARED_BIAS_OFFSET + bucket * HIDDEN
    for h in range(HIDDEN):
        value = np.float32(weights[shared_bias_base + h])
        row = shared_base + h * (2 * WIDTH)
        for j in range(2 * WIDTH):
            value = np.float32(value + np.float32(weights[row + j] * joined[j]))
        hidden[h] = clip01(value)

    score_norm = np.float32(weights[SCORE_BIAS_OFFSET + bucket])
    score_base = SCORE_HEAD_OFFSET + bucket * HIDDEN
    wdl_logit = np.float32(weights[WDL_BIAS_OFFSET + bucket])
    wdl_base = WDL_HEAD_OFFSET + bucket * HIDDEN
    for h in range(HIDDEN):
        score_norm = np.float32(score_norm + np.float32(weights[score_base + h] * hidden[h]))
        wdl_logit = np.float32(wdl_logit + np.float32(weights[wdl_base + h] * hidden[h]))
    return round_cp(score_norm), float(sigmoid32(wdl_logit)), float(score_norm), float(wdl_logit)


def validate_checkpoint_and_build_torch(checkpoint_path: Path, exported_weights: np.ndarray):
    try:
        import torch
    except ImportError as exc:
        raise ValueError("CPU PyTorch is required for checkpoint-vs-export parity") from exc

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only support
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or checkpoint.get("architecture") != "KB8-HalfKA-128-2head-v2":
        raise ValueError("checkpoint architecture is not KB8-HalfKA-128-2head-v2")
    target = checkpoint.get("target", {})
    if not isinstance(target, dict) or target.get("score_scale_cp") != float(SCORE_SCALE):
        raise ValueError("checkpoint score-scale metadata does not match v2 format")

    shapes = {
        "ft.weight": (FEATURES, WIDTH),
        "ft_bias": (WIDTH,),
        "shared1": (BUCKETS, HIDDEN, 2 * WIDTH),
        "shared1_bias": (BUCKETS, HIDDEN),
        "score_head": (BUCKETS, HIDDEN),
        "score_bias": (BUCKETS,),
        "wdl_head": (BUCKETS, HIDDEN),
        "wdl_bias": (BUCKETS,),
    }
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise ValueError("checkpoint state_dict keys do not match the v2 architecture")
    chunks = []
    for name, shape in shapes.items():
        tensor = state[name]
        if tuple(tensor.shape) != shape or tensor.dtype != torch.float32:
            raise ValueError(f"checkpoint tensor {name} has wrong shape or dtype")
        array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        if not np.isfinite(array).all():
            raise ValueError(f"checkpoint tensor {name} contains non-finite values")
        chunks.append(array.tobytes(order="C"))
    checkpoint_weights = np.frombuffer(b"".join(chunks), dtype="<f4")
    if checkpoint_weights.size != PAYLOAD_FLOATS or not np.array_equal(checkpoint_weights, exported_weights):
        raise ValueError("exported payload does not exactly match the local checkpoint")

    class KB8TwoHeadV2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ft = torch.nn.Embedding(FEATURES, WIDTH)
            self.ft_bias = torch.nn.Parameter(torch.empty(WIDTH))
            self.shared1 = torch.nn.Parameter(torch.empty(BUCKETS, HIDDEN, 2 * WIDTH))
            self.shared1_bias = torch.nn.Parameter(torch.empty(BUCKETS, HIDDEN))
            self.score_head = torch.nn.Parameter(torch.empty(BUCKETS, HIDDEN))
            self.score_bias = torch.nn.Parameter(torch.empty(BUCKETS))
            self.wdl_head = torch.nn.Parameter(torch.empty(BUCKETS, HIDDEN))
            self.wdl_bias = torch.nn.Parameter(torch.empty(BUCKETS))

        def forward(self, feature_indices, material_bucket):
            valid = feature_indices.ne(65535).unsqueeze(-1)
            safe_indices = feature_indices.long().clamp(max=FEATURES - 1)
            rows = self.ft(safe_indices)
            accum = (rows * valid.to(rows.dtype)).sum(dim=2) + self.ft_bias
            accum = torch.clamp(accum, 0.0, 1.0)
            joined = accum.reshape(accum.shape[0], 2 * WIDTH)
            shared = torch.einsum("bd,khd->bkh", joined, self.shared1) + self.shared1_bias
            shared = torch.clamp(shared, 0.0, 1.0)
            score_by_bucket = torch.einsum("bkh,kh->bk", shared, self.score_head) + self.score_bias
            wdl_by_bucket = torch.einsum("bkh,kh->bk", shared, self.wdl_head) + self.wdl_bias
            selected = material_bucket.long().view(-1, 1)
            score = score_by_bucket.gather(1, selected).squeeze(1)
            wdl = wdl_by_bucket.gather(1, selected).squeeze(1)
            return score, wdl

    torch.set_num_threads(1)
    model = KB8TwoHeadV2()
    model.load_state_dict(state, strict=True)
    model.eval()
    return torch, model, checkpoint


def torch_eval(torch, model, fen: str) -> tuple[int, float, float, float]:
    _, views, bucket = parse_fen_features(fen)
    features = np.full((1, 2, 32), 65535, dtype=np.int64)
    for view_index, active in enumerate(views):
        features[0, view_index, :len(active)] = active
    feature_tensor = torch.from_numpy(features)
    bucket_tensor = torch.tensor([bucket], dtype=torch.int64)
    with torch.no_grad():
        score_out, wdl_out = model(feature_tensor, bucket_tensor)
        score_norm = float(score_out.item())
        wdl_logit = float(wdl_out.item())
        wdl_probability = float(torch.sigmoid(wdl_out).item())
    return round_cp(np.float32(score_norm)), wdl_probability, score_norm, wdl_logit


BLEND_WEIGHT = 1.0
BLEND_SCALE = 300.0
BLEND_TOLERANCE_CP = 2


def blend_expect(score_cp: int, prob: float) -> int:
    """Mirror of the candidate's kb8_blend_heads (python float64 vs engine float32)."""
    p = min(0.995, max(0.005, float(prob)))
    wdl_cp = BLEND_SCALE * math.log(p / (1.0 - p))
    blended = (1.0 - BLEND_WEIGHT) * score_cp + BLEND_WEIGHT * wdl_cp
    return int(round(max(-30000.0, min(30000.0, blended))))


def run_engine(engine: Path, network: Path) -> tuple[list[int], list[float], list[int]]:
    network_abs = network.resolve()
    quoted_network = network_abs.name.replace('"', '')
    commands = [
        "uci",
        "isready",
        f'setoption name EvalFile value "{quoted_network}"',
        "setoption name EvalMode value NNUE_WDL",
        "isready",
    ]
    for _, fen in FENS:
        commands.extend((f"position fen {fen}", "evalheads"))
    commands.append("quit")
    completed = subprocess.run(
        [str(engine.resolve())],
        cwd=str(network_abs.parent),
        input="\n".join(commands) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=90,
        check=False,
    )
    output = completed.stdout
    if completed.returncode != 0:
        raise RuntimeError(f"candidate exited {completed.returncode}:\n{output}")
    if "KB8 network loaded" not in output:
        raise RuntimeError(f"candidate did not confirm the v2 network load:\n{output}")
    if "EvalMode NNUE_WDL" not in output:
        raise RuntimeError(f"candidate did not enable NNUE_WDL:\n{output}")
    matches = re.findall(
        r"info string evalheads cp (-?\d+) wdl_prob ([0-9.eE+-]+) blend (-?\d+) mode NNUE_WDL", output
    )
    if len(matches) != len(FENS):
        raise RuntimeError(f"got {len(matches)} head outputs for {len(FENS)} positions:\n{output}")
    return ([int(cp) for cp, _, _ in matches],
            [float(prob) for _, prob, _ in matches],
            [int(blend) for _, _, blend in matches])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="two-head v2 candidate executable")
    parser.add_argument("--network", required=True, help="exported KB8 v2 .nnue file")
    parser.add_argument("--checkpoint", help="local original .pt; enables exact PyTorch comparison")
    parser.add_argument("--tolerance-cp", type=int, default=1)
    parser.add_argument("--tolerance-wdl", type=float, default=1e-5)
    parser.add_argument("--tolerance-blend", type=int, default=BLEND_TOLERANCE_CP)
    parser.add_argument("--report", default="nnue_pilot_0003_deep/kb8_wdl_parity_report.txt")
    args = parser.parse_args()

    engine = Path(args.engine).expanduser()
    network = Path(args.network).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser() if args.checkpoint else None
    if not engine.is_file():
        raise SystemExit(f"Candidate not found: {engine}")
    if not network.is_file():
        raise SystemExit(f"Network not found: {network}")
    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    if args.tolerance_cp < 0 or args.tolerance_wdl < 0:
        raise SystemExit("tolerances must be non-negative")

    try:
        weights = load_weights(network)
        numpy_results = [numpy_eval(weights, fen) for _, fen in FENS]
        torch_module = torch_model = checkpoint = None
        if checkpoint_path is not None:
            torch_module, torch_model, checkpoint = validate_checkpoint_and_build_torch(checkpoint_path, weights)
            torch_results = [torch_eval(torch_module, torch_model, fen) for _, fen in FENS]
        cpp_scores, cpp_wdl, cpp_blend = run_engine(engine, network)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(f"Two-head v2 parity check failed: {exc}") from exc

    max_score_delta = 0
    max_wdl_delta = 0.0
    lines = [
        "KB8-HalfKA-128 WDL step-1 — checkpoint / NumPy / C++ parity + blend check",
        "===========================================================================",
        f"engine: {engine.resolve()}",
        f"network: {network.resolve()}",
        f"checkpoint: {checkpoint_path.resolve() if checkpoint_path else '(not supplied; no PyTorch comparison)'}",
        f"selected_seed: {checkpoint.get('selected_seed', 'unknown') if checkpoint else 'unknown'}",
        f"selected_epoch: {checkpoint.get('selected_epoch', 'unknown') if checkpoint else 'unknown'}",
        f"positions: {len(FENS)}",
        f"score_tolerance_cp: {args.tolerance_cp}",
        f"wdl_probability_tolerance: {args.tolerance_wdl}",
        f"blend_weight: {BLEND_WEIGHT}",
        f"blend_logit_scale_cp: {BLEND_SCALE}",
        f"blend_tolerance_cp: {args.tolerance_blend}",
        "heldout_test: not read or evaluated",
        "",
    ]
    if checkpoint_path is not None:
        lines.extend((
            "case                    Torch_cp NumPy_cp C++_cp  dCpp-Torch  Torch_WDL   NumPy_WDL   C++_WDL",
            "----------------------  -------- -------- ------  ----------  ----------  ----------  ----------",
        ))
    else:
        lines.extend((
            "case                    NumPy_cp C++_cp  dCpp-NumPy  NumPy_WDL   C++_WDL",
            "----------------------  -------- ------  ----------  ----------  ----------",
        ))

    for index, (name, _) in enumerate(FENS):
        numpy_score, numpy_prob, _, _ = numpy_results[index]
        cpp_score = cpp_scores[index]
        cpp_prob = cpp_wdl[index]
        if checkpoint_path is not None:
            torch_score, torch_prob, _, _ = torch_results[index]
            max_score_delta = max(max_score_delta, abs(cpp_score - torch_score), abs(cpp_score - numpy_score), abs(torch_score - numpy_score))
            max_wdl_delta = max(max_wdl_delta, abs(cpp_prob - torch_prob), abs(cpp_prob - numpy_prob), abs(torch_prob - numpy_prob))
            lines.append(
                f"{name:<22} {torch_score:>8} {numpy_score:>8} {cpp_score:>6} "
                f"{cpp_score - torch_score:>+10} {torch_prob:>10.7f} {numpy_prob:>10.7f} {cpp_prob:>10.7f}"
            )
        else:
            max_score_delta = max(max_score_delta, abs(cpp_score - numpy_score))
            max_wdl_delta = max(max_wdl_delta, abs(cpp_prob - numpy_prob))
            lines.append(
                f"{name:<22} {numpy_score:>8} {cpp_score:>6} {cpp_score - numpy_score:>+10} "
                f"{numpy_prob:>10.7f} {cpp_prob:>10.7f}"
            )
    max_blend_delta = 0
    lines.append("WDL blend check (expected = python mirror of the candidate formula):")
    lines.append(f"{'case':<22} {'ref':>6} {'expected':>8} {'cpp':>6} {'delta':>5}")
    for index, (name, _) in enumerate(FENS):
        if checkpoint_path is not None:
            ref_score, ref_prob = torch_results[index][0], torch_results[index][1]
        else:
            ref_score, ref_prob = numpy_results[index][0], numpy_results[index][1]
        expected = blend_expect(ref_score, ref_prob)
        got = cpp_blend[index]
        delta = abs(got - expected)
        if delta > max_blend_delta:
            max_blend_delta = delta
        lines.append(f"{name:<22} {ref_score:>6} {expected:>8} {got:>6} {delta:>5}")
    passed = (max_score_delta <= args.tolerance_cp
              and max_wdl_delta <= args.tolerance_wdl
              and max_blend_delta <= args.tolerance_blend)
    lines.extend((
        "",
        f"max_abs_score_delta_cp: {max_score_delta}",
        f"max_abs_wdl_probability_delta: {max_wdl_delta:.9g}",
        f"max_abs_blend_delta_cp: {max_blend_delta}",
        f"result: {'PASS' if passed else 'FAIL'}",
        "This confirms static inference agreement only; it is not an Elo or strength result.",
    ))
    report_path = Path(args.report).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Text report: {report_path.resolve()}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
