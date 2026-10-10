#!/usr/bin/env python3
"""Paired fixed-node smoke match: frozen HCE baseline vs KB8 two-head v2.

This is a short gameplay smoke test, not a rating/Elo test. It uses only fixed
opening move sequences and the compiled engines; it never reads the labeled
shard or any train/validation/test split.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import subprocess
import time
from pathlib import Path

try:
    import chess
except ImportError as exc:
    raise SystemExit("python-chess is required; the Windows launcher installs it if missing.") from exc

OPENINGS = [
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["e2e4", "c7c5", "g1f3", "d7d6"],
    ["e2e4", "e7e6", "d2d4", "d7d5"],
    ["e2e4", "c7c6", "d2d4", "d7d5"],
    ["d2d4", "d7d5", "c2c4", "e7e6"],
    ["d2d4", "g8f6", "c2c4", "e7e6"],
    ["c2c4", "e7e5", "g1f3", "g8f6"],
    ["g1f3", "d7d5", "d2d4", "g8f6"],
    ["b1c3", "d7d5", "e2e4", "e7e6"],
    ["b2b3", "d7d5", "c1b2", "g8f6"],
    ["c2c4", "c7c5", "g1f3", "b8c6"],
    ["e2e4", "g8f6", "e4e5", "d7d5"],
]


class UCIEngine:
    def __init__(self, path: Path, hash_mb: int, working_directory: Path,
                 network_name: str | None = None):
        self.path = path.resolve()
        self.name = self.path.name
        self.proc = subprocess.Popen(
            [str(self.path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, cwd=str(working_directory.resolve()),
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        self.send("uci")
        self.until("uciok")
        self.send(f"setoption name Hash value {hash_mb}")
        if network_name is not None:
            # ASCII basename avoids Windows ANSI-codepage problems with the
            # user's Cyrillic Desktop path; process cwd is the network folder.
            self.send(f'setoption name EvalFile value "{network_name}"')
            line = self.until("KB8 network loaded")
            print(f"NNUE load: {line.strip()}", flush=True)
            self.send("setoption name EvalMode value NNUE_WDL")
            line = self.until("EvalMode NNUE_WDL")
            print(f"NNUE mode: {line.strip()}", flush=True)
        self.ready()

    def send(self, text: str) -> None:
        if self.proc.stdin is None:
            raise RuntimeError(f"{self.name}: stdin is closed")
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def until(self, token: str) -> str:
        if self.proc.stdout is None:
            raise RuntimeError(f"{self.name}: stdout is closed")
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.name} exited while waiting for {token}")
            if token in line:
                return line

    def ready(self) -> None:
        self.send("isready")
        self.until("readyok")

    def reset(self) -> None:
        self.send("ucinewgame")
        self.ready()

    def choose(self, moves: list[str], nodes: int, board: chess.Board) -> chess.Move:
        move_text = " ".join(moves)
        self.send("position startpos moves" + (" " + move_text if move_text else ""))
        self.send(f"go nodes {nodes}")
        if self.proc.stdout is None:
            raise RuntimeError(f"{self.name}: stdout closed during search")
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.name} exited during search")
            if line.startswith("bestmove "):
                fields = line.split()
                if len(fields) < 2 or fields[1] == "0000":
                    raise RuntimeError(f"{self.name} returned no move for {board.fen()}")
                move = chess.Move.from_uci(fields[1])
                if move not in board.legal_moves:
                    raise RuntimeError(f"{self.name} returned illegal move {fields[1]} for {board.fen()}")
                return move

    def close(self) -> None:
        try:
            self.send("quit")
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
            self.proc.wait(timeout=5)


def play_game(baseline: UCIEngine, candidate: UCIEngine, opening: list[str],
              candidate_white: bool, nodes: int, max_plies: int) -> tuple[str, int, bool]:
    board = chess.Board()
    moves: list[str] = []
    for uci in opening:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise RuntimeError(f"invalid fixed opening move {uci}")
        board.push(move)
        moves.append(uci)

    baseline.reset()
    candidate.reset()
    while not board.is_game_over(claim_draw=True) and len(moves) < max_plies:
        candidate_to_move = (board.turn == chess.WHITE) == candidate_white
        player = candidate if candidate_to_move else baseline
        move = player.choose(moves, nodes, board)
        moves.append(move.uci())
        board.push(move)

    capped = not board.is_game_over(claim_draw=True)
    result = "1/2-1/2" if capped else board.result(claim_draw=True)
    return result, len(moves), capped


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="separately compiled frozen-HCE executable")
    parser.add_argument("--baseline-source", required=True, help="unchanged frozen baseline C++ source")
    parser.add_argument("--candidate", required=True, help="separate KB8 two-head v2 executable")
    parser.add_argument("--network", required=True, help="exported two-head v2 .nnue")
    parser.add_argument("--pairs", type=int, default=12,
                        help="opening/color-swap pairs, 1..200 (repeats past 12 get seeded extensions)")
    parser.add_argument("--nodes", type=int, default=8192, help="fixed nodes per move")
    parser.add_argument("--max-plies", type=int, default=120, help="draw at this cap for smoke testing")
    parser.add_argument("--hash-mb", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--report", default="nnue_pilot_0002_twohead/kb8_twohead_v2_match_smoke.txt")
    args = parser.parse_args()

    baseline_path = Path(args.baseline).expanduser().resolve()
    baseline_source = Path(args.baseline_source).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()
    candidate_source = Path(__file__).resolve().parent / "engine_build11_kb8_wdl_candidate.cpp"
    network_path = Path(args.network).expanduser().resolve()
    for file_path in (baseline_path, baseline_source, candidate_path, candidate_source, network_path):
        if not file_path.is_file():
            raise SystemExit(f"Not found: {file_path}")
    if not (1 <= args.pairs <= 200) or args.nodes <= 0 or args.max_plies < 20 or args.hash_mb < 1:
        raise SystemExit("Use 1..200 pairs, positive nodes/hash, and max-plies >= 20")

    rng = random.Random(args.seed)
    base_order = list(range(len(OPENINGS)))
    rng.shuffle(base_order)
    # Beyond 12 pairs, each fixed line repeats with a deterministic 0-6 random
    # legal-ply extension so repetitions are not identical starting positions.
    opening_specs: list[tuple[int, int]] = []
    for i in range(args.pairs):
        opening_specs.append((base_order[i % len(OPENINGS)], i // len(OPENINGS)))

    def build_opening(base_index: int, repeat: int, serial: int) -> list[str]:
        moves = list(OPENINGS[base_index])
        if repeat == 0:
            return moves
        ext_rng = random.Random(args.seed * 1_000_003 + serial)
        board = chess.Board()
        for uci in moves:
            board.push(chess.Move.from_uci(uci))
        for _ in range(ext_rng.randint(0, 6)):
            if board.is_game_over(claim_draw=True):
                break
            choice = ext_rng.choice(list(board.legal_moves))
            moves.append(choice.uci())
            board.push(choice)
        return moves

    results: list[tuple[str, bool, str, int, bool]] = []
    extended_count = 0
    started = time.time()
    baseline = candidate = None
    work_dir = network_path.parent
    try:
        baseline = UCIEngine(baseline_path, args.hash_mb, work_dir)
        candidate = UCIEngine(candidate_path, args.hash_mb, work_dir, network_path.name)
        for serial, (base_index, repeat) in enumerate(opening_specs):
            opening_moves = build_opening(base_index, repeat, serial)
            extended_count += int(repeat > 0)
            opening_label = f"{base_index+1:02d}" + (f"-r{repeat}" if repeat else "")
            for candidate_white in (True, False):
                result, plies, capped = play_game(
                    baseline, candidate, opening_moves, candidate_white,
                    args.nodes, args.max_plies,
                )
                results.append((opening_label, candidate_white, result, plies, capped))
                color = "white" if candidate_white else "black"
                print(f"game={len(results):02d}/{2*args.pairs} opening={opening_label} "
                      f"candidate={color} result={result} plies={plies} cap_draw={capped}", flush=True)
    finally:
        if candidate is not None:
            candidate.close()
        if baseline is not None:
            baseline.close()

    wins = draws = losses = capped_draws = 0
    for _, candidate_white, result, _, capped in results:
        if result == "1/2-1/2":
            draws += 1
            capped_draws += int(capped)
        else:
            candidate_won = (result == "1-0") if candidate_white else (result == "0-1")
            if candidate_won:
                wins += 1
            else:
                losses += 1

    elapsed = time.time() - started
    lines = [
        "KB8 two-head v2 gameplay smoke: candidate vs frozen HCE baseline",
        "================================================================",
        f"baseline_executable: {baseline_path}",
        f"baseline_executable_sha256: {sha256(baseline_path)}",
        f"baseline_source: {baseline_source}",
        f"baseline_source_sha256: {sha256(baseline_source)}",
        f"candidate_executable: {candidate_path}",
        f"candidate_executable_sha256: {sha256(candidate_path)}",
        f"candidate_source: {candidate_source}",
        f"candidate_source_sha256: {sha256(candidate_source)}",
        f"network: {network_path}",
        f"network_sha256: {sha256(network_path)}",
        f"paired_openings: {args.pairs}",
        f"base_openings: {len(OPENINGS)}",
        f"extended_opening_repeats: {extended_count}",
        "opening_policy: fixed lines shuffled by seed; repeats get 0-6 seeded random legal plies",
        f"games: {len(results)}",
        f"fixed_nodes_per_move: {args.nodes}",
        f"hash_mb_per_engine: {args.hash_mb}",
        f"max_plies_draw_cap: {args.max_plies}",
        f"seed: {args.seed}",
        f"result_candidate_W-D-L: {wins}-{draws}-{losses}",
        f"cap_draws: {capped_draws}",
        f"candidate_score_pct: {(wins + 0.5 * draws) / max(1, len(results)):.3f}",
        f"elapsed_seconds: {elapsed:.2f}",
        "",
        "Per-game results (each opening repeated with colors swapped):",
    ]
    for i, (opening_label, candidate_white, result, plies, capped) in enumerate(results, start=1):
        lines.append(f"{i:02d} opening={opening_label} candidate={'W' if candidate_white else 'B'} "
                     f"result={result} plies={plies} capped={capped}")
    lines.extend((
        "",
        "Interpretation: short paired fixed-node smoke only; not an Elo estimate or strength claim.",
        "Frozen source was compiled to a new test executable; no baseline file was replaced.",
    ))
    report = Path(args.report).expanduser()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nText report: {report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
