#!/usr/bin/env python3
"""Two-head KB8-HalfKA-128 pilot trainer using the existing local labeled shard.

The shared HalfKA accumulator feeds two separate bucket-specific heads:
(1) robust teacher-score regression, exported later for engine search;
(2) auxiliary WDL/result probability. Clamped mate scores are excluded only from
score regression, but remain useful for the WDL head. The held-out test split is
not evaluated: the first pilot already reported it. Outputs are TXT/CSV/PyTorch checkpoints only; no JSON.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np

import train_first_kb8 as trainer

ARCH = "KB8-HalfKA-128-2head-v2"
WIDTH = trainer.WIDTH
FEATURES = trainer.FEATURES
MAX_PIECES = trainer.MAX_PIECES
SCORE_SCALE = trainer.SCORE_SCALE
SCORE_LOSS_WEIGHT = 1.0
WDL_LOSS_WEIGHT = 0.5
HUBER_BETA_SCORE_UNITS = 1.0
BUCKET_LABELS = {0: "2-8", 1: "9-16", 2: "17-24", 3: "25-32"}


def make_model(torch):
    nn = torch.nn

    class KB8TwoHeadV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.ft = nn.Embedding(FEATURES, WIDTH)
            self.ft_bias = nn.Parameter(torch.full((WIDTH,), 0.10))
            self.shared1 = nn.Parameter(torch.empty(4, 16, 2 * WIDTH))
            self.shared1_bias = nn.Parameter(torch.zeros(4, 16))
            self.score_head = nn.Parameter(torch.empty(4, 16))
            self.score_bias = nn.Parameter(torch.zeros(4))
            self.wdl_head = nn.Parameter(torch.empty(4, 16))
            self.wdl_bias = nn.Parameter(torch.zeros(4))
            nn.init.normal_(self.ft.weight, mean=0.0, std=0.025)
            nn.init.xavier_uniform_(self.shared1, gain=0.7)
            nn.init.normal_(self.score_head, mean=0.0, std=0.02)
            nn.init.normal_(self.wdl_head, mean=0.0, std=0.02)

        def forward(self, feature_indices, material_bucket):
            valid = feature_indices.ne(trainer.PAD).unsqueeze(-1)
            safe_indices = feature_indices.long().clamp(max=FEATURES - 1)
            rows = self.ft(safe_indices)
            accum = (rows * valid.to(rows.dtype)).sum(dim=2) + self.ft_bias
            accum = torch.clamp(accum, 0.0, 1.0)
            joined = accum.reshape(accum.shape[0], 2 * WIDTH)  # STM perspective first
            shared = torch.einsum("bd,khd->bkh", joined, self.shared1) + self.shared1_bias
            shared = torch.clamp(shared, 0.0, 1.0)
            score_by_bucket = torch.einsum("bkh,kh->bk", shared, self.score_head) + self.score_bias
            wdl_by_bucket = torch.einsum("bkh,kh->bk", shared, self.wdl_head) + self.wdl_bias
            selected = material_bucket.long().view(-1, 1)
            score = score_by_bucket.gather(1, selected).squeeze(1)  # normalized cp: cp / SCORE_SCALE
            wdl = wdl_by_bucket.gather(1, selected).squeeze(1)      # STM score probability logit
            return score, wdl

    return KB8TwoHeadV2


def batch_tensors(torch, arrays, batch_ids):
    x = torch.from_numpy(np.asarray(arrays["features"][batch_ids], dtype=np.int64))
    bucket = torch.from_numpy(np.asarray(arrays["bucket"][batch_ids], dtype=np.int64))
    score = torch.from_numpy(np.asarray(arrays["score"][batch_ids], dtype=np.float32))
    result = torch.from_numpy(np.asarray(arrays["result"][batch_ids], dtype=np.float32))
    return x, bucket, score, result


def score_mask(torch, score):
    return torch.abs(score) < 14999.5


def compute_losses(torch, score_output, wdl_output, score_cp, result_stm):
    functional = torch.nn.functional
    mask = score_mask(torch, score_cp)
    if bool(mask.any()):
        score_target = score_cp[mask] / SCORE_SCALE
        score_loss = functional.smooth_l1_loss(
            score_output[mask], score_target, reduction="mean", beta=HUBER_BETA_SCORE_UNITS
        )
    else:
        score_loss = score_output.new_zeros(())
    result_target = (result_stm + 1.0) * 0.5
    result_prob = torch.sigmoid(wdl_output)
    wdl_loss = functional.mse_loss(result_prob, result_target, reduction="mean")
    total = SCORE_LOSS_WEIGHT * score_loss + WDL_LOSS_WEIGHT * wdl_loss
    return total, score_loss, wdl_loss


def pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def evaluate(torch, model, arrays, ids: np.ndarray, batch_size: int) -> dict:
    model.eval()
    count = 0
    score_huber_sum = score_huber_count = 0.0
    wdl_brier_sum = 0.0
    all_scores = []
    all_pred_cp = []
    all_results = []
    all_wdl_probs = []
    per_bucket = {b: {"score": [], "pred_cp": [], "result": [], "prob": []} for b in range(4)}

    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            selected = ids[start:start + batch_size]
            x, bucket_t, score_t, result_t = batch_tensors(torch, arrays, selected)
            score_out, wdl_out = model(x, bucket_t)
            mask_t = score_mask(torch, score_t)
            result_prob_t = torch.sigmoid(wdl_out)
            if bool(mask_t.any()):
                score_huber_sum += float(torch.nn.functional.smooth_l1_loss(
                    score_out[mask_t], score_t[mask_t] / SCORE_SCALE,
                    reduction="sum", beta=HUBER_BETA_SCORE_UNITS
                ).item())
                score_huber_count += int(mask_t.sum().item())
            result_target_t = (result_t + 1.0) * 0.5
            wdl_brier_sum += float(torch.square(result_prob_t - result_target_t).sum().item())

            score_np = score_t.cpu().numpy().astype(np.float64)
            pred_np = (score_out * SCORE_SCALE).cpu().numpy().astype(np.float64)
            result_np = result_t.cpu().numpy().astype(np.float64)
            prob_np = result_prob_t.cpu().numpy().astype(np.float64)
            bucket_np = bucket_t.cpu().numpy().astype(np.int64)
            all_scores.extend(score_np.tolist())
            all_pred_cp.extend(pred_np.tolist())
            all_results.extend(result_np.tolist())
            all_wdl_probs.extend(prob_np.tolist())
            for i, b in enumerate(bucket_np):
                per_bucket[int(b)]["score"].append(float(score_np[i]))
                per_bucket[int(b)]["pred_cp"].append(float(pred_np[i]))
                per_bucket[int(b)]["result"].append(float(result_np[i]))
                per_bucket[int(b)]["prob"].append(float(prob_np[i]))
            count += len(selected)

    score = np.asarray(all_scores, dtype=np.float64)
    pred = np.asarray(all_pred_cp, dtype=np.float64)
    result = np.asarray(all_results, dtype=np.float64)
    prob = np.asarray(all_wdl_probs, dtype=np.float64)
    mask = np.abs(score) < 14999.5
    mae_all = float(np.mean(np.abs(pred - score))) if count else float("nan")
    mae_unclamped = float(np.mean(np.abs(pred[mask] - score[mask]))) if mask.any() else float("nan")
    bias_unclamped = float(np.mean(pred[mask] - score[mask])) if mask.any() else float("nan")
    corr_all = pearson(pred, score)
    corr_unclamped = pearson(pred[mask], score[mask]) if mask.any() else None
    result_target = (result + 1.0) * 0.5
    score_huber = score_huber_sum / score_huber_count if score_huber_count else float("nan")
    wdl_loss = wdl_brier_sum / count if count else float("nan")
    selection = SCORE_LOSS_WEIGHT * score_huber + WDL_LOSS_WEIGHT * wdl_loss

    bucket_metrics = {}
    for b in range(4):
        values = per_bucket[b]
        y = np.asarray(values["score"], dtype=np.float64)
        p = np.asarray(values["pred_cp"], dtype=np.float64)
        r = np.asarray(values["result"], dtype=np.float64)
        q = np.asarray(values["prob"], dtype=np.float64)
        if not len(y):
            bucket_metrics[b] = {"records": 0}
            continue
        m = np.abs(y) < 14999.5
        target_wdl = (r + 1.0) * 0.5
        bucket_metrics[b] = {
            "records": int(len(y)),
            "clamped": int((~m).sum()),
            "mae_unclamped_cp": float(np.mean(np.abs(p[m] - y[m]))) if m.any() else None,
            "bias_unclamped_cp": float(np.mean(p[m] - y[m])) if m.any() else None,
            "pearson_all": pearson(p, y),
            "pearson_unclamped": pearson(p[m], y[m]) if m.any() else None,
            "wdl_brier": float(np.mean((q - target_wdl) ** 2)),
            "wdl_pred_mean": float(np.mean(q)),
            "result_mean": float(np.mean(target_wdl)),
        }

    return {
        "records": count,
        "selection_loss": selection,
        "score_huber_norm_unclamped": score_huber,
        "wdl_brier": wdl_loss,
        "score_mae_all_cp": mae_all,
        "score_mae_unclamped_cp": mae_unclamped,
        "score_bias_unclamped_cp": bias_unclamped,
        "score_pearson_all": corr_all,
        "score_pearson_unclamped": corr_unclamped,
        "wdl_pred_mean": float(np.mean(prob)) if count else float("nan"),
        "result_mean": float(np.mean(result_target)) if count else float("nan"),
        "clamped_count": int((~mask).sum()),
        "buckets": bucket_metrics,
    }


def train_one_seed(torch, model_class, arrays, train_ids, val_ids, out_dir: Path,
                   seed: int, epochs: int, batch_size: int,
                   learning_rate: float) -> tuple[dict, dict, list[dict]]:
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    model = model_class()
    optimizer = torch.optim.AdamW([
        {"params": [model.ft.weight, model.ft_bias], "weight_decay": 0.0},
        {"params": [model.shared1, model.shared1_bias, model.score_head, model.score_bias,
                     model.wdl_head, model.wdl_bias], "weight_decay": 1e-5},
    ], lr=learning_rate)
    batches_per_epoch = math.ceil(len(train_ids) / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * batches_per_epoch), eta_min=learning_rate * 0.1
    )
    best_selection = float("inf")
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = np_rng.permutation(train_ids)
        score_sum = score_seen = wdl_sum = seen = 0.0
        for start in range(0, len(order), batch_size):
            batch_ids = order[start:start + batch_size]
            x, bucket, score, result = batch_tensors(torch, arrays, batch_ids)
            optimizer.zero_grad(set_to_none=True)
            score_out, wdl_out = model(x, bucket)
            loss, score_loss, wdl_loss = compute_losses(torch, score_out, wdl_out, score, result)
            mask = score_mask(torch, score)
            loss.backward()
            optimizer.step()
            scheduler.step()
            size = len(batch_ids)
            score_count = int(mask.sum().item())
            score_sum += float(score_loss.detach().item()) * score_count
            score_seen += score_count
            wdl_sum += float(wdl_loss.detach().item()) * size
            seen += size

        train_score = score_sum / max(1, score_seen)
        train_wdl = wdl_sum / max(1, seen)
        val = evaluate(torch, model, arrays, val_ids, batch_size)
        row = {
            "seed": seed,
            "epoch": epoch,
            "train_total_loss": SCORE_LOSS_WEIGHT * train_score + WDL_LOSS_WEIGHT * train_wdl,
            "train_score_huber_norm": train_score,
            "train_wdl_brier": train_wdl,
            "validation": val,
        }
        history.append(row)
        print(
            f"seed={seed} epoch={epoch}/{epochs} train={row['train_total_loss']:.6f} "
            f"val_select={val['selection_loss']:.6f} val_score_mae_unclamped="
            f"{val['score_mae_unclamped_cp']:.1f} val_wdl_brier={val['wdl_brier']:.6f}",
            flush=True,
        )
        if val["selection_loss"] < best_selection:
            best_selection = val["selection_loss"]
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    seed_path = out_dir / f"seed_{seed}_twohead_best.pt"
    torch.save({"architecture": ARCH, "seed": seed, "epoch": best_epoch,
                "target": {"score_scale_cp": SCORE_SCALE,
                           "score_loss": "Huber(score_cp/400), non-clamped records only",
                           "wdl_loss": "Brier(result probability)",
                           "score_loss_weight": SCORE_LOSS_WEIGHT,
                           "wdl_loss_weight": WDL_LOSS_WEIGHT},
                "state_dict": best_state}, seed_path)
    best_model = model_class()
    best_model.load_state_dict(best_state, strict=True)
    best_validation = evaluate(torch, best_model, arrays, val_ids, batch_size)
    return {"seed": seed, "best_epoch": best_epoch,
            "best_validation": best_validation, "checkpoint": seed_path.name}, best_state, history


def format_metric(value, digits=6):
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def write_report(out_dir: Path, input_name: str, data_info: dict,
                 selected_result: dict, selected_state: dict, model_class, torch,
                 arrays, val_ids, histories: list[dict], seed_metrics: list[dict],
                 elapsed_seconds: float, batch_size: int) -> None:
    model = model_class()
    model.load_state_dict(selected_state, strict=True)
    val = evaluate(torch, model, arrays, val_ids, batch_size=batch_size)
    rows = [
        "KB8-HalfKA-128 two-head v2 pilot — validation-only report",
        "=========================================================",
        "status: twohead_v2_training_complete_static_only",
        f"input: {input_name}",
        f"manifest_records: {data_info['records']}",
        f"train_records: {data_info['train_records']}",
        f"validation_records: {data_info['validation_records']}",
        f"heldout_test_records: {data_info['test_records']} (not evaluated)",
        f"train_families: {data_info['train_families']}",
        f"validation_families: {data_info['validation_families']}",
        f"heldout_test_families: {data_info['test_families']} (not evaluated)",
        f"selected_seed: {selected_result['seed']}",
        f"selected_epoch: {selected_result['best_epoch']}",
        f"restarts: {len(seed_metrics)}",
        f"selection: min(validation score_huber_norm + {WDL_LOSS_WEIGHT} * validation WDL Brier)",
        "score head: direct teacher score_cp / 400; clamped +/-15000 labels excluded from score loss only",
        f"WDL head: result_stm mapped to 0/0.5/1; all validation records used",
        f"score_loss_weight: {SCORE_LOSS_WEIGHT}; wdl_loss_weight: {WDL_LOSS_WEIGHT}; Huber beta: {HUBER_BETA_SCORE_UNITS}",
        f"score_huber_norm_unclamped: {format_metric(val['score_huber_norm_unclamped'])}",
        f"score_MAE_all_cp: {format_metric(val['score_mae_all_cp'], 1)}",
        f"score_MAE_unclamped_cp: {format_metric(val['score_mae_unclamped_cp'], 1)}",
        f"score_bias_unclamped_cp: {format_metric(val['score_bias_unclamped_cp'], 1)}",
        f"score_Pearson_all: {format_metric(val['score_pearson_all'], 4)}",
        f"score_Pearson_unclamped: {format_metric(val['score_pearson_unclamped'], 4)}",
        f"WDL_Brier: {format_metric(val['wdl_brier'])}",
        f"WDL_predicted_mean_probability: {format_metric(val['wdl_pred_mean'], 4)}",
        f"WDL_actual_mean_probability: {format_metric(val['result_mean'], 4)}",
        f"score_clamped_records: {val['clamped_count']}/{val['records']}",
        f"checkpoint: best_kb8_twohead_v2.pt",
        f"elapsed_seconds: {elapsed_seconds:.2f}",
        "",
        "Best validation checkpoint per restart:",
        "seed epoch selection_loss score_MAE_unclamped_cp WDL_Brier checkpoint",
    ]
    for seed_result in seed_metrics:
        seed_val = seed_result["best_validation"]
        rows.append(
            f"{seed_result['seed']} {seed_result['best_epoch']} {seed_val['selection_loss']:.6f} "
            f"{seed_val['score_mae_unclamped_cp']:.1f} {seed_val['wdl_brier']:.6f} "
            f"{seed_result['checkpoint']}"
        )
    rows.extend((
        "",
        "Validation by material bucket (CP metrics ignore clamped scores):",
        "bucket pieces       n   clamp   score_MAE_cp   bias_cp   Pearson_uncl   WDL_Brier   pred_p   result_p",
    ))
    for b in range(4):
        m = val["buckets"][b]
        if not m["records"]:
            rows.append(f"{b} {BUCKET_LABELS[b]:<8} 0 n/a")
            continue
        rows.append(
            f"{b:>6} {BUCKET_LABELS[b]:<8} {m['records']:>7} {m['clamped']:>6} "
            f"{format_metric(m['mae_unclamped_cp'],1):>13} {format_metric(m['bias_unclamped_cp'],1):>9} "
            f"{format_metric(m['pearson_unclamped'],4):>13} {format_metric(m['wdl_brier']):>10} "
            f"{format_metric(m['wdl_pred_mean'],4):>7} {format_metric(m['result_mean'],4):>10}"
        )
    rows.extend((
        "",
        "Epoch history: seed epoch train_total train_score_huber train_wdl_brier val_selection val_score_MAE_unclamped val_score_Pearson_unclamped val_WDL_Brier",
    ))
    for h in histories:
        v = h["validation"]
        rows.append(
            f"{h['seed']} {h['epoch']} {h['train_total_loss']:.6f} {h['train_score_huber_norm']:.6f} "
            f"{h['train_wdl_brier']:.6f} {v['selection_loss']:.6f} {v['score_mae_unclamped_cp']:.1f} "
            f"{format_metric(v['score_pearson_unclamped'],4)} {v['wdl_brier']:.6f}"
        )
    rows.extend((
        "",
        "No training/evaluation was performed on the held-out test split.",
        "This is a static fit report, not a game match, Elo result, or strength claim.",
        "The previous pilot checkpoint and frozen engine baseline were not modified.",
    ))
    (out_dir / "training_report.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")

    with (out_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "epoch", "train_total_loss", "train_score_huber_norm", "train_wdl_brier",
            "val_selection_loss", "val_score_mae_unclamped_cp", "val_score_mae_all_cp",
            "val_score_pearson_all", "val_score_pearson_unclamped", "val_wdl_brier",
        ])
        for h in histories:
            v = h["validation"]
            writer.writerow([
                h["seed"], h["epoch"], h["train_total_loss"], h["train_score_huber_norm"],
                h["train_wdl_brier"], v["selection_loss"], v["score_mae_unclamped_cp"],
                v["score_mae_all_cp"], v["score_pearson_all"], v["score_pearson_unclamped"], v["wdl_brier"],
            ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=".", help="existing folder, JSONL, GZIP JSONL, or artifact ZIP")
    parser.add_argument("--out", default="nnue_pilot_0002_twohead", help="new output folder; separate from the first pilot")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.restarts < 1 or args.batch_size < 1:
        parser.error("epochs, restarts and batch-size must be positive")

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    checkpoint_path = out_dir / "best_kb8_twohead_v2.pt"
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise SystemExit(f"Refusing to reuse non-empty output path: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required; use the Python installation from the first pilot.") from exc
    torch.set_num_threads(max(1, min(args.threads, os.cpu_count() or 1)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    started = time.time()
    with trainer.open_text_data(input_path) as (_, manifest, _):
        expected_records = manifest.get("records") if manifest else None
    cache_dir = out_dir / "_training_cache"
    n, data_info, manifest = trainer.prepare_cache(input_path, cache_dir, expected_records)
    arrays = trainer.load_memmaps(cache_dir, n)
    split = arrays["split"]
    train_ids = np.flatnonzero(split == 0)
    val_ids = np.flatnonzero(split == 1)
    test_count = int(np.sum(split == 2))  # metadata only; the test split is never evaluated
    if len(train_ids) < 1000 or len(val_ids) < 100:
        raise SystemExit("Too few train or validation records in the existing family split")

    model_class = make_model(torch)
    histories = []
    seed_results = []
    selected_state = None
    selected_result = None
    for seed in range(args.seed, args.seed + args.restarts):
        result, state, history = train_one_seed(
            torch, model_class, arrays, train_ids, val_ids, out_dir, seed,
            args.epochs, args.batch_size, args.lr,
        )
        seed_results.append(result)
        histories.extend(history)
        if selected_result is None or result["best_validation"]["selection_loss"] < selected_result["best_validation"]["selection_loss"]:
            selected_result = result
            selected_state = state

    if selected_state is None or selected_result is None:
        raise RuntimeError("No v2 checkpoint selected")
    selected_checkpoint = {
        "architecture": ARCH,
        "target": {
            "score_scale_cp": SCORE_SCALE,
            "score_loss": "Huber(score_cp/400), unclamped only",
            "wdl_loss": "Brier(result probability)",
            "score_loss_weight": SCORE_LOSS_WEIGHT,
            "wdl_loss_weight": WDL_LOSS_WEIGHT,
            "huber_beta_score_units": HUBER_BETA_SCORE_UNITS,
            "selection": "validation score_huber_norm + 0.5 * validation WDL Brier",
        },
        "selected_seed": selected_result["seed"],
        "selected_epoch": selected_result["best_epoch"],
        "state_dict": selected_state,
    }
    torch.save(selected_checkpoint, checkpoint_path)
    elapsed_seconds = time.time() - started
    write_report(out_dir, data_info["data_member"], data_info, selected_result,
                 selected_state, model_class, torch, arrays, val_ids, histories,
                 seed_results, elapsed_seconds, args.batch_size)

    del arrays, split
    if not args.keep_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"\nTwo-head v2 pilot complete.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Text report: {out_dir / 'training_report.txt'}")
    print(f"Held-out test records: {test_count} (not evaluated)")
    print("No baseline file was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
