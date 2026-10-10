#!/usr/bin/env python3
"""Convert a deep_v4 hidden=16 checkpoint to the KB8-HalfKA-128-2head-v2 format.

The deep_v4 model with hidden [16] has exactly the v2 architecture (same accumulator,
same clamp, same 16-unit bucket layer, same two heads). Only key names and one
transpose differ, so the exporter and the engine can read it unchanged.

Checks that the converted v2 model gives the same score and WDL outputs as the
deep_v4 model on random positions (max abs difference must be tiny).

Usage:
  py -3 adapt_h16_to_v2.py nnue_pilot_0006_h16\\best_deep_v4.pt --out nnue_pilot_0006_h16\\best_kb8_twohead_v2.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np

import train_first_kb8 as trainer
import train_kb8_deep_v4 as deep
import train_kb8_twohead_v2 as base


def convert_state(sd: dict) -> dict:
    return {
        "ft.weight": sd["ft.weight"],
        "ft_bias": sd["ft_bias"],
        # deep_v4 layer 0: (buckets, 256, 16) -> v2 shared1: (buckets, 16, 256)
        "shared1": sd["hidden_w.0"].permute(0, 2, 1).contiguous(),
        "shared1_bias": sd["hidden_b.0"],
        "score_head": sd["score_head"],
        "score_bias": sd["score_bias"],
        "wdl_head": sd["wdl_head"],
        "wdl_bias": sd["wdl_bias"],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--rows", type=int, default=4096)
    args = p.parse_args()

    import torch

    src_path = Path(args.checkpoint)
    if not src_path.is_file():
        raise SystemExit(f"Checkpoint not found: {src_path}")
    ck = torch.load(src_path, map_location="cpu")
    if ck.get("hidden") != [16]:
        raise SystemExit(f"This adapter is for hidden=[16] only; checkpoint has hidden={ck.get('hidden')}")

    sd = ck["state_dict"]
    v2_state = convert_state(sd)

    # Parity: deep_v4 model vs converted v2 model on random inputs.
    deep_model = deep.make_model(torch, [16], 0.0)()
    deep_model.load_state_dict(sd, strict=True)
    deep_model.eval()
    v2_model = base.make_model(torch)()
    v2_model.load_state_dict(v2_state, strict=True)
    v2_model.eval()

    rng = np.random.default_rng(1)
    n = args.rows
    feats = rng.integers(0, trainer.FEATURES, size=(n, 2, trainer.MAX_PIECES), dtype=np.int64)
    pad_mask = rng.random((n, 2, trainer.MAX_PIECES)) < 0.5
    feats[pad_mask] = trainer.PAD
    buckets = rng.integers(0, 4, size=(n,), dtype=np.int64)
    x = torch.from_numpy(feats)
    b = torch.from_numpy(buckets)
    with torch.no_grad():
        s1, w1 = deep_model(x, b)
        s2, w2 = v2_model(x, b)
    ds = float((s1 - s2).abs().max())
    dw = float((w1 - w2).abs().max())
    print(f"parity: max |score diff| = {ds:.3e}, max |wdl diff| = {dw:.3e}")
    if not (ds < 1e-4 and dw < 1e-4):
        print("PARITY FAILED: converted weights do not reproduce the deep_v4 outputs.")
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": "KB8-HalfKA-128-2head-v2",
        "target": {"score_scale_cp": 400.0,
                   "source": "converted from deep_v4 hidden=[16] checkpoint"},
        "selected_seed": ck.get("selected_seed", "unknown"),
        "selected_epoch": ck.get("selected_epoch", "unknown"),
        "state_dict": v2_state,
    }, out)
    print(f"converted checkpoint written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
