#!/usr/bin/env python3
"""Export a KB8-HalfKA-128 two-head v2 checkpoint to the candidate reference format.

The 44-byte little-endian KB8F32 header uses version 2. The float32 payload is
ordered as embedding, accumulator bias, shared bucket-specific layer, then
separate score and WDL heads. This is an unquantized parity format, not a
production-optimized NNUE format.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import struct
import zlib
from pathlib import Path

MAGIC = b"KB8F32\0\0"
VERSION = 2
FEATURES = 8 * 12 * 64
WIDTH = 128
PERSPECTIVES = 2
HIDDEN = 16
BUCKETS = 4
SCORE_SCALE = 400.0
TENSOR_SHAPES = {
    "ft.weight": (FEATURES, WIDTH),
    "ft_bias": (WIDTH,),
    "shared1": (BUCKETS, HIDDEN, 2 * WIDTH),
    "shared1_bias": (BUCKETS, HIDDEN),
    "score_head": (BUCKETS, HIDDEN),
    "score_bias": (BUCKETS,),
    "wdl_head": (BUCKETS, HIDDEN),
    "wdl_bias": (BUCKETS,),
}
PAYLOAD_FLOATS = sum(math.prod(shape) for shape in TENSOR_SHAPES.values())


def load_checkpoint(torch, path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only support
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or checkpoint.get("architecture") != "KB8-HalfKA-128-2head-v2":
        architecture = checkpoint.get("architecture") if isinstance(checkpoint, dict) else None
        raise SystemExit(f"Unexpected checkpoint architecture: {architecture!r}")
    target = checkpoint.get("target")
    if not isinstance(target, dict) or target.get("score_scale_cp") != SCORE_SCALE:
        raise SystemExit(f"Unexpected or missing score target metadata: {target!r}")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise SystemExit("Checkpoint has no state_dict")
    missing = sorted(set(TENSOR_SHAPES) - set(state))
    extra = sorted(set(state) - set(TENSOR_SHAPES))
    if missing or extra:
        raise SystemExit(f"state_dict keys differ; missing={missing}, unexpected={extra}")
    return checkpoint


def serialize_state(torch, checkpoint: dict) -> bytes:
    import numpy as np

    chunks = []
    for name, expected_shape in TENSOR_SHAPES.items():
        tensor = checkpoint["state_dict"][name]
        if not hasattr(tensor, "detach") or not tensor.is_floating_point():
            raise SystemExit(f"{name} must be a floating-point tensor")
        if tensor.dtype != torch.float32:
            raise SystemExit(f"{name} must be float32; found {tensor.dtype}")
        tensor = tensor.detach().cpu().contiguous()
        if tuple(tensor.shape) != expected_shape:
            raise SystemExit(f"{name}: expected {expected_shape}, got {tuple(tensor.shape)}")
        array = tensor.numpy().astype("<f4", copy=False)
        if not np.isfinite(array).all():
            raise SystemExit(f"{name} contains NaN or infinity")
        chunks.append(array.tobytes(order="C"))
    payload = b"".join(chunks)
    if len(payload) != PAYLOAD_FLOATS * 4:
        raise SystemExit(f"Internal payload size mismatch: got {len(payload)} bytes")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="best_kb8_twohead_v2.pt")
    parser.add_argument("--out", default="nnue_pilot_0002_twohead/kb8_twohead_v2.nnue")
    parser.add_argument("--force", action="store_true", help="replace a prior exported network/report only")
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise SystemExit("NumPy and CPU PyTorch are required (the v2 training run used them).") from exc

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    destination = Path(args.out).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    report_path = destination.with_suffix(destination.suffix + ".txt")
    if not args.force and (destination.exists() or report_path.exists()):
        raise SystemExit(f"Refusing to overwrite export output; use --force: {destination}")

    checkpoint = load_checkpoint(torch, checkpoint_path)
    payload = serialize_state(torch, checkpoint)
    payload_crc = zlib.crc32(payload) & 0xFFFFFFFF
    header = struct.pack(
        "<8s7IfI", MAGIC, VERSION, FEATURES, WIDTH, PERSPECTIVES, HIDDEN,
        BUCKETS, PAYLOAD_FLOATS, SCORE_SCALE, payload_crc,
    )
    if len(header) != 44:
        raise SystemExit(f"Internal header size mismatch: {len(header)}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(header + payload)
    temporary.replace(destination)

    report = [
        "KB8-HalfKA-128 two-head v2 export report",
        "==========================================",
        "status: ok",
        "architecture: KB8-HalfKA-128-2head-v2",
        f"checkpoint: {checkpoint_path}",
        f"selected_seed: {checkpoint.get('selected_seed', 'unknown')}",
        f"selected_epoch: {checkpoint.get('selected_epoch', 'unknown')}",
        f"output: {destination}",
        f"format_version: {VERSION}",
        f"payload_floats: {PAYLOAD_FLOATS}",
        f"payload_bytes: {len(payload)}",
        f"file_bytes: {len(header) + len(payload)}",
        f"payload_crc32: {payload_crc:08x}",
        f"payload_sha256: {hashlib.sha256(payload).hexdigest()}",
        f"score_scale_cp: {SCORE_SCALE}",
        "payload_heads: direct CP score + auxiliary WDL probability",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"Text report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
