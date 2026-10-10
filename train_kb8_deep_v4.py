#!/usr/bin/env python3
"""KB8 two-head trainer with configurable hidden layers, dropout and L2 (train-only probe).

Reuses the data cache, family split and losses of the first pilot scripts.
Only training and validation are run; the held-out test split is never evaluated.
Outputs: checkpoint (.pt), history CSV, text report. No JSON.

Examples:
  py -3 train_kb8_deep_v4.py selfdist_labels_merged --out nnue_pilot_0006_h16 --hidden 16
  py -3 train_kb8_deep_v4.py selfdist_labels_merged --out nnue_pilot_0006_h256 --hidden 64,32
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np

import train_first_kb8 as trainer
import train_kb8_twohead_v2 as base

ARCH_PREFIX = "KB8-HalfKA-128-deep-v4"
WIDTH = trainer.WIDTH
FEATURES = trainer.FEATURES
PAD = trainer.PAD
BUCKETS = 4


def make_model(torch, hidden_sizes, dropout):
    nn = torch.nn

    class KB8DeepV4(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_sizes = list(hidden_sizes)
            self.ft = nn.Embedding(FEATURES, WIDTH)
            self.ft_bias = nn.Parameter(torch.full((WIDTH,), 0.10))
            nn.init.normal_(self.ft.weight, mean=0.0, std=0.025)
            self.drop = nn.Dropout(dropout)
            # per-bucket layer stack: layer 0 maps 2*WIDTH -> h0, layer i maps h(i-1) -> h(i)
            dims = [2 * WIDTH] + self.hidden_sizes
            self.hidden_w = nn.ParameterList()
            self.hidden_b = nn.ParameterList()
            for din, dout_ in zip(dims[:-1], dims[1:]):
                w = nn.Parameter(torch.empty(BUCKETS, din, dout_))
                nn.init.xavier_uniform_(w.view(BUCKETS * din, dout_), gain=0.7)
                self.hidden_w.append(w)
                self.hidden_b.append(nn.Parameter(torch.zeros(BUCKETS, dout_)))
            last = self.hidden_sizes[-1]
            self.score_head = nn.Parameter(torch.empty(BUCKETS, last))
            self.score_bias = nn.Parameter(torch.zeros(BUCKETS))
            self.wdl_head = nn.Parameter(torch.empty(BUCKETS, last))
            self.wdl_bias = nn.Parameter(torch.zeros(BUCKETS))
            nn.init.normal_(self.score_head, mean=0.0, std=0.02)
            nn.init.normal_(self.wdl_head, mean=0.0, std=0.02)

        def forward(self, feature_indices, material_bucket):
            valid = feature_indices.ne(PAD).unsqueeze(-1)
            safe = feature_indices.long().clamp(max=FEATURES - 1)
            rows = self.ft(safe)
            accum = (rows * valid.to(rows.dtype)).sum(dim=2) + self.ft_bias
            accum = torch.clamp(accum, 0.0, 1.0)
            joined = accum.reshape(accum.shape[0], 2 * WIDTH)  # STM first
            h = None
            for i, (w, b) in enumerate(zip(self.hidden_w, self.hidden_b)):
                if i == 0:
                    h = torch.einsum("bd,kdh->bkh", joined, w) + b
                else:
                    h = torch.einsum("bkh,khg->bkg", h, w) + b
                h = torch.clamp(h, 0.0, 1.0)
                h = self.drop(h)
            score_by = torch.einsum("bkh,kh->bk", h, self.score_head) + self.score_bias
            wdl_by = torch.einsum("bkh,kh->bk", h, self.wdl_head) + self.wdl_bias
            selected = material_bucket.long().view(-1, 1)
            score = score_by.gather(1, selected).squeeze(1)
            wdl = wdl_by.gather(1, selected).squeeze(1)
            return score, wdl

    return KB8DeepV4


def train_one_seed(torch, model_class, arrays, train_ids, val_ids, out_dir: Path,
                   seed: int, epochs: int, batch_size: int, lr: float, wd: float):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model_class()
    dense = [p for n, p in model.named_parameters() if not n.startswith("ft")]
    optimizer = torch.optim.AdamW([
        {"params": [model.ft.weight, model.ft_bias], "weight_decay": 0.0},
        {"params": dense, "weight_decay": wd},
    ], lr=lr)
    batches = math.ceil(len(train_ids) / batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * batches), eta_min=lr * 0.1)
    best_sel, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_ids)
        tot = sco = wd_sum = 0.0
        count = 0
        t0 = time.time()
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            x, bucket, score, result = base.batch_tensors(torch, arrays, ids)
            optimizer.zero_grad(set_to_none=True)
            s_out, w_out = model(x, bucket)
            total, s_loss, w_loss = base.compute_losses(torch, s_out, w_out, score, result)
            total.backward()
            optimizer.step()
            sched.step()
            n = len(ids)
            tot += float(total.item()) * n
            sco += float(s_loss.item()) * n
            wd_sum += float(w_loss.item()) * n
            count += n
        model.eval()
        val = base.evaluate(torch, model, arrays, val_ids, batch_size)
        row = {
            "seed": seed, "epoch": epoch,
            "train_total": tot / count, "train_score_huber": sco / count, "train_wdl_brier": wd_sum / count,
            "val_selection": val["selection_loss"], "val_score_huber": val["score_huber_norm_unclamped"],
            "val_score_mae_unclamped_cp": val["score_mae_unclamped_cp"],
            "val_score_pearson_unclamped": val["score_pearson_unclamped"],
            "val_wdl_brier": val["wdl_brier"], "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(f"seed={seed} epoch={epoch}/{epochs} train={row['train_total']:.6f} "
              f"val_select={row['val_selection']:.6f} val_wdl={row['val_wdl_brier']:.6f} "
              f"val_mae={row['val_score_mae_unclamped_cp']:.1f} sec={row['seconds']}", flush=True)
        if val["selection_loss"] < best_sel:
            best_sel = val["selection_loss"]
            best_epoch = epoch
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    return {"seed": seed, "best_epoch": best_epoch, "best_selection": best_sel,
            "state": best_state}, history


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("--out", required=True)
    p.add_argument("--hidden", default="16", help="comma-separated hidden sizes, e.g. 16 or 64,32")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=20261010)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--keep-cache", action="store_true")
    args = p.parse_args()
    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]
    if not hidden or any(h < 1 for h in hidden):
        p.error("bad --hidden")
    if args.epochs < 1 or args.restarts < 1:
        p.error("epochs and restarts must be positive")

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"Output folder is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required (same Python as the first pilot).") from exc
    torch.set_num_threads(max(1, min(args.threads, os.cpu_count() or 1)))

    started = time.time()
    input_path = Path(args.input)
    cache_dir = out_dir / "_training_cache"
    with trainer.open_text_data(input_path) as (_, manifest, _):
        expected = manifest.get("records") if manifest else None
    n, data_info, _ = trainer.prepare_cache(input_path, cache_dir, expected)
    arrays = trainer.load_memmaps(cache_dir, n)
    split = arrays["split"]
    train_ids = np.flatnonzero(split == 0)
    val_ids = np.flatnonzero(split == 1)
    if len(train_ids) < 1000 or len(val_ids) < 100:
        raise SystemExit("Too few train or validation records")

    model_class = make_model(torch, hidden, args.dropout)
    all_hist, results = [], []
    best = None
    for seed in range(args.seed, args.seed + args.restarts):
        res, hist = train_one_seed(torch, model_class, arrays, train_ids, val_ids, out_dir,
                                   seed, args.epochs, args.batch_size, args.lr, args.wd)
        results.append(res)
        all_hist.extend(hist)
        if best is None or res["best_selection"] < best["best_selection"]:
            best = res

    arch = f"{ARCH_PREFIX}-{'-'.join(str(h) for h in hidden)}"
    torch.save({"architecture": arch, "hidden": hidden, "dropout": args.dropout, "wd": args.wd,
                "selected_seed": best["seed"], "selected_epoch": best["best_epoch"],
                "state_dict": best["state"]}, out_dir / "best_deep_v4.pt")

    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_hist[0].keys()))
        w.writeheader()
        w.writerows(all_hist)

    lines = [
        f"KB8 deep v4 probe report (train/validation only)",
        f"architecture: {arch}",
        f"dropout: {args.dropout}  weight_decay: {args.wd}  epochs: {args.epochs}  restarts: {args.restarts}",
        f"input: {input_path}",
        f"train_records: {len(train_ids)}  validation_records: {len(val_ids)}  test_records: not evaluated",
        f"selected_seed: {best['seed']}  selected_epoch: {best['best_epoch']}  best_selection: {best['best_selection']:.6f}",
        f"elapsed_seconds: {time.time() - started:.1f}",
        "history.csv holds per-epoch train and validation curves.",
        "Note: validation peak epoch is the checkpoint choice; no early stopping.",
    ]
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    del arrays, split
    if not args.keep_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
