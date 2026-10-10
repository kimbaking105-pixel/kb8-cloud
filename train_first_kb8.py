#!/usr/bin/env python3
"""First local pilot trainer for KB8-HalfKA-128.

This is an offline research pilot, not yet a drop-in evaluator for the C++ engine.
It reads the labeled shard locally, splits by opening family, prepares a disk-backed
feature cache, trains two independent CPU initializations, and writes a compact
training_report.json plus the best PyTorch checkpoint.

Dependencies: numpy and CPU PyTorch. No python-chess is required.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterator, Optional, TextIO

import numpy as np

ARCH = "KB8-HalfKA-128-4b"
FEATURES = 8 * 12 * 64
WIDTH = 128
MAX_PIECES = 32
PAD = 65535
SCORE_SCALE = 400.0
LAMBDA_SCORE = 0.75
PIECE_TYPE = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}


def basename(name: str) -> str:
    return PurePosixPath(name.replace("\\", "/")).name.lower()


def select_data_member(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    files = [x for x in archive.infolist() if not x.is_dir()]
    matches = [x for x in files if basename(x.filename).startswith("labeled_shard_")
               and basename(x.filename).endswith((".jsonl.gz", ".jsonl", ".txt"))]
    if not matches:
        matches = [x for x in files if basename(x.filename).endswith((".jsonl.gz", ".jsonl"))]
    if not matches:
        raise SystemExit("Не найден labeled_shard_*.jsonl[.gz] внутри входного ZIP.")
    return sorted(matches, key=lambda x: (not basename(x.filename).endswith(".jsonl.gz"), x.filename))[0]


def manifest_from_zip(archive: zipfile.ZipFile) -> Optional[dict]:
    matches = [x for x in archive.infolist() if not x.is_dir()
               and basename(x.filename).startswith("manifest_")
               and basename(x.filename).endswith(".json")]
    if not matches:
        return None
    with archive.open(sorted(matches, key=lambda x: x.filename)[0]) as stream:
        return json.load(io.TextIOWrapper(stream, encoding="utf-8"))


def manifest_from_directory(root: Path, data_path: Path) -> Optional[dict]:
    candidates = sorted(root.rglob("manifest_*.json"))
    if not candidates:
        candidates = sorted(data_path.parent.glob("manifest_*.json"))
    if not candidates:
        return None
    wanted = data_path.name.lower().replace("labeled_shard_", "").split(".jsonl")[0]
    match = [p for p in candidates if wanted and wanted in p.name.lower()]
    chosen = match[0] if match else candidates[0]
    try:
        return json.loads(chosen.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Не удалось прочитать {chosen.name}: {exc}")


def find_data_file(root: Path) -> Path:
    candidates = [p for p in root.rglob("*") if p.is_file()
                  and p.name.lower().startswith("labeled_shard_")
                  and p.name.lower().endswith((".jsonl.gz", ".jsonl", ".txt"))]
    if not candidates:
        candidates = [p for p in root.rglob("*") if p.is_file()
                      and p.name.lower().endswith((".jsonl.gz", ".jsonl"))]
    if not candidates:
        raise SystemExit(f"В папке {root} не найден labeled_shard_*.jsonl[.gz].")
    return sorted(candidates, key=lambda p: (not p.name.lower().endswith(".jsonl.gz"), str(p)))[0]


@contextlib.contextmanager
def open_text_data(path: Path) -> Iterator[tuple[TextIO, Optional[dict], str]]:
    """Yield text stream, manifest and data member name; keeps ZIP open as needed."""
    if path.is_dir():
        data_path = find_data_file(path)
        manifest = manifest_from_directory(path, data_path)
        if data_path.name.lower().endswith(".gz"):
            with gzip.open(data_path, "rt", encoding="utf-8") as stream:
                yield stream, manifest, str(data_path)
        else:
            with data_path.open("rt", encoding="utf-8") as stream:
                yield stream, manifest, str(data_path)
        return

    if not path.exists():
        raise SystemExit(f"Вход не найден: {path}")

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as archive:
            info = select_data_member(archive)
            manifest = manifest_from_zip(archive)
            raw = archive.open(info, "r")
            try:
                if info.filename.lower().endswith(".gz"):
                    decoded = gzip.GzipFile(fileobj=raw, mode="rb")
                    text = io.TextIOWrapper(decoded, encoding="utf-8")
                else:
                    text = io.TextIOWrapper(raw, encoding="utf-8")
                try:
                    yield text, manifest, info.filename
                finally:
                    text.close()
            finally:
                raw.close()
        return

    manifest = manifest_from_directory(path.parent, path)
    if path.name.lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            yield stream, manifest, str(path)
    else:
        with path.open("rt", encoding="utf-8") as stream:
            yield stream, manifest, str(path)


def split_for_family(seed: object, family_id: object) -> int:
    token = f"{seed}:{family_id}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little") % 100
    if value < 90:
        return 0  # train
    if value < 95:
        return 1  # validation
    return 2      # held-out test


def features_from_fen(fen: str, stm: str) -> tuple[np.ndarray, int]:
    """Return [STM, other] HalfKA feature indices and material bucket."""
    fields = fen.split()
    if len(fields) != 6 or fields[1] not in ("w", "b"):
        raise ValueError("FEN must have six fields and a valid side-to-move")
    if fields[1] != stm:
        raise ValueError("record stm does not match FEN")

    ranks = fields[0].split("/")
    if len(ranks) != 8:
        raise ValueError("FEN placement must contain eight ranks")

    pieces: list[tuple[str, int]] = []
    king_sq = [-1, -1]  # 0 white, 1 black; squares a1=0 .. h8=63
    for fen_rank, rank_text in enumerate(ranks):
        board_rank = 7 - fen_rank
        file_idx = 0
        for char in rank_text:
            if char in "12345678":
                file_idx += int(char)
            elif char.lower() in PIECE_TYPE:
                if file_idx >= 8:
                    raise ValueError("FEN rank is wider than 8 squares")
                sq = board_rank * 8 + file_idx
                pieces.append((char, sq))
                if char == "K":
                    king_sq[0] = sq
                elif char == "k":
                    king_sq[1] = sq
                file_idx += 1
            else:
                raise ValueError(f"invalid FEN piece symbol {char!r}")
        if file_idx != 8:
            raise ValueError("FEN rank does not cover 8 squares")

    if not (2 <= len(pieces) <= MAX_PIECES):
        raise ValueError(f"unexpected piece count {len(pieces)}")
    if king_sq[0] < 0 or king_sq[1] < 0:
        raise ValueError("FEN is missing a king")

    stm_color = 0 if stm == "w" else 1
    views: list[list[int]] = []
    for perspective in (stm_color, 1 - stm_color):
        own_king = king_sq[perspective]
        oriented_king = own_king ^ (56 if perspective == 1 else 0)
        king_file = oriented_king & 7
        mirror = 7 if king_file >= 4 else 0
        mapped_king_file = king_file ^ mirror
        king_rank = oriented_king >> 3
        king_bucket = (4 if king_rank >= 4 else 0) + mapped_king_file

        active: list[int] = []
        for char, square in pieces:
            piece_color = 0 if char.isupper() else 1
            piece_type = PIECE_TYPE[char.lower()]
            plane = 6 * int(piece_color != perspective) + piece_type
            oriented_square = square ^ (56 if perspective == 1 else 0)
            mapped_square = oriented_square ^ mirror
            index = king_bucket * 768 + plane * 64 + mapped_square
            if not 0 <= index < FEATURES:
                raise ValueError("feature index out of range")
            active.append(index)
        views.append(active)

    packed = np.full((2, MAX_PIECES), PAD, dtype=np.uint16)
    for view_i, active in enumerate(views):
        packed[view_i, :len(active)] = np.asarray(active, dtype=np.uint16)
    material_bucket = max(0, min(3, (len(pieces) - 1) // 8))
    return packed, material_bucket


def sigmoid(value: float) -> float:
    if value >= 40.0:
        return 1.0
    if value <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def prepare_cache(
    input_path: Path, cache_dir: Path, expected_records: Optional[int]
) -> tuple[int, dict, Optional[dict]]:
    if not expected_records or expected_records <= 0:
        raise SystemExit("В manifest нет корректного records; нужен файл manifest_*.json.")
    if expected_records > 20_000_000:
        raise SystemExit(f"Защитная остановка: manifest заявляет {expected_records:,} записей.")

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    n = int(expected_records)
    features = np.memmap(cache_dir / "features.bin", dtype="<u2", mode="w+", shape=(n, 2, MAX_PIECES))
    buckets = np.memmap(cache_dir / "bucket.bin", dtype="u1", mode="w+", shape=(n,))
    targets = np.memmap(cache_dir / "target.bin", dtype="<f4", mode="w+", shape=(n,))
    scores = np.memmap(cache_dir / "score.bin", dtype="<f4", mode="w+", shape=(n,))
    results = np.memmap(cache_dir / "result.bin", dtype="<f4", mode="w+", shape=(n,))
    splits = np.memmap(cache_dir / "split.bin", dtype="u1", mode="w+", shape=(n,))

    split_records = Counter()
    split_families = [set(), set(), set()]
    clamped = 0
    written = 0

    with open_text_data(input_path) as (stream, manifest, data_name):
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if written >= n:
                raise SystemExit("В данных записей больше, чем указано в manifest.")
            try:
                record = json.loads(line)
                fen = record["fen"]
                stm = record["stm"]
                score = int(record["score_cp"])
                result_stm = int(record["result_stm"])
                family = record["family_id"]
                seed = record.get("seed", (manifest or {}).get("seed", 0))
                if result_stm not in (-1, 0, 1):
                    raise ValueError("result_stm must be -1, 0, or 1")
                feat, bucket = features_from_fen(fen, stm)
                q_score = sigmoid(score / SCORE_SCALE)
                q_result = (result_stm + 1.0) * 0.5
                target = LAMBDA_SCORE * q_score + (1.0 - LAMBDA_SCORE) * q_result
                split = split_for_family(seed, family)
            except Exception as exc:
                raise SystemExit(f"Ошибка в записи {line_no}: {exc}") from exc

            features[written] = feat
            buckets[written] = bucket
            targets[written] = target
            scores[written] = score
            results[written] = result_stm
            splits[written] = split
            split_records[split] += 1
            split_families[split].add((str(seed), str(family)))
            clamped += int(abs(score) >= 15000)
            written += 1

            if written % 100000 == 0:
                print(f"Preprocessed {written:,}/{n:,} records", flush=True)

    for arr in (features, buckets, targets, scores, results, splits):
        arr.flush()
    del features, buckets, targets, scores, results, splits

    if written != n:
        raise SystemExit(f"Manifest обещает {n:,} записей, прочитано {written:,}.")
    if not split_records[0] or not split_records[1] or not split_records[2]:
        raise SystemExit(
            "После группового разбиения пустая выборка. Для надёжного train/val/test нужно больше family_id."
        )

    info = {
        "records": written,
        "train_records": split_records[0],
        "validation_records": split_records[1],
        "test_records": split_records[2],
        "train_families": len(split_families[0]),
        "validation_families": len(split_families[1]),
        "test_families": len(split_families[2]),
        "score_at_clamp_count": clamped,
        "data_member": data_name,
    }
    return written, info, manifest


def make_model(torch):
    nn = torch.nn

    class KB8HalfKA128(nn.Module):
        def __init__(self):
            super().__init__()
            self.ft = nn.Embedding(FEATURES, WIDTH)
            self.ft_bias = nn.Parameter(torch.full((WIDTH,), 0.10))
            self.head1 = nn.Parameter(torch.empty(4, 16, 256))
            self.head1_bias = nn.Parameter(torch.zeros(4, 16))
            self.head2 = nn.Parameter(torch.empty(4, 16))
            self.head2_bias = nn.Parameter(torch.zeros(4))
            nn.init.normal_(self.ft.weight, mean=0.0, std=0.025)
            nn.init.xavier_uniform_(self.head1, gain=0.7)
            nn.init.normal_(self.head2, mean=0.0, std=0.02)

        def forward(self, feature_indices, material_bucket):
            valid = feature_indices.ne(PAD).unsqueeze(-1)
            safe_indices = feature_indices.long().clamp(max=FEATURES - 1)
            rows = self.ft(safe_indices)
            accum = (rows * valid.to(rows.dtype)).sum(dim=2) + self.ft_bias
            accum = torch.clamp(accum, 0.0, 1.0)
            joined = accum.reshape(accum.shape[0], 2 * WIDTH)  # STM perspective first
            hidden = torch.einsum("bd,khd->bkh", joined, self.head1) + self.head1_bias
            hidden = torch.clamp(hidden, 0.0, 1.0)
            logits = torch.einsum("bkh,kh->bk", hidden, self.head2) + self.head2_bias
            return logits.gather(1, material_bucket.long().view(-1, 1)).squeeze(1)

    return KB8HalfKA128


def load_memmaps(cache_dir: Path, n: int):
    return {
        "features": np.memmap(cache_dir / "features.bin", dtype="<u2", mode="r", shape=(n, 2, MAX_PIECES)),
        "bucket": np.memmap(cache_dir / "bucket.bin", dtype="u1", mode="r", shape=(n,)),
        "target": np.memmap(cache_dir / "target.bin", dtype="<f4", mode="r", shape=(n,)),
        "score": np.memmap(cache_dir / "score.bin", dtype="<f4", mode="r", shape=(n,)),
        "result": np.memmap(cache_dir / "result.bin", dtype="<f4", mode="r", shape=(n,)),
        "split": np.memmap(cache_dir / "split.bin", dtype="u1", mode="r", shape=(n,)),
    }


def batch_tensors(torch, arrays, batch_ids):
    x = torch.from_numpy(np.asarray(arrays["features"][batch_ids], dtype=np.int64))
    bucket = torch.from_numpy(np.asarray(arrays["bucket"][batch_ids], dtype=np.int64))
    target = torch.from_numpy(np.asarray(arrays["target"][batch_ids], dtype=np.float32))
    score = torch.from_numpy(np.asarray(arrays["score"][batch_ids], dtype=np.float32))
    result = torch.from_numpy(np.asarray(arrays["result"][batch_ids], dtype=np.float32))
    return x, bucket, target, score, result


def evaluate(torch, model, arrays, ids: np.ndarray, batch_size: int) -> dict:
    functional = torch.nn.functional
    model.eval()
    n = loss_sum = mae_sum = mae_unclamped_sum = brier_sum = 0.0
    unclamped_n = 0
    sx = sy = sxx = syy = sxy = 0.0
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            selected = ids[start:start + batch_size]
            x, bucket, target, score, result = batch_tensors(torch, arrays, selected)
            logits = model(x, bucket)
            prob = torch.sigmoid(logits)
            per_example_loss = functional.mse_loss(prob, target, reduction="none")
            pred_cp = logits * SCORE_SCALE
            loss_sum += float(per_example_loss.sum().item())
            mae_sum += float(torch.abs(pred_cp - score).sum().item())
            mask = torch.abs(score) < 14999.5
            if bool(mask.any()):
                mae_unclamped_sum += float(torch.abs(pred_cp[mask] - score[mask]).sum().item())
                unclamped_n += int(mask.sum().item())
            q = (result + 1.0) * 0.5
            brier_sum += float(torch.square(prob - q).sum().item())
            p = pred_cp.detach().double()
            y = score.detach().double()
            sx += float(p.sum().item())
            sy += float(y.sum().item())
            sxx += float(torch.square(p).sum().item())
            syy += float(torch.square(y).sum().item())
            sxy += float((p * y).sum().item())
            n += len(selected)
    corr = None
    den_x = n * sxx - sx * sx
    den_y = n * syy - sy * sy
    if n and den_x > 0 and den_y > 0:
        corr = (n * sxy - sx * sy) / math.sqrt(den_x * den_y)
    return {
        "records": int(n),
        "blend_mse": loss_sum / n if n else None,
        "teacher_score_mae_cp": mae_sum / n if n else None,
        "teacher_score_mae_cp_unclamped": mae_unclamped_sum / unclamped_n if unclamped_n else None,
        "teacher_score_pearson": corr,
        "result_brier": brier_sum / n if n else None,
    }


def train_one_seed(torch, Model, arrays, train_ids, val_ids, out_dir: Path,
                   seed: int, epochs: int, batch_size: int, learning_rate: float,
                   threads: int) -> tuple[dict, dict, list[dict]]:
    import copy
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    model = Model()
    optimizer = torch.optim.AdamW([
        {"params": [model.ft.weight, model.ft_bias], "weight_decay": 0.0},
        {"params": [model.head1, model.head1_bias, model.head2, model.head2_bias], "weight_decay": 1e-5},
    ], lr=learning_rate)
    batches_per_epoch = math.ceil(len(train_ids) / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * batches_per_epoch), eta_min=learning_rate * 0.1
    )

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = np_rng.permutation(train_ids)
        train_loss_sum = 0.0
        seen = 0
        epoch_start = time.time()
        for start in range(0, len(order), batch_size):
            selected = order[start:start + batch_size]
            x, bucket, target, _, _ = batch_tensors(torch, arrays, selected)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, bucket)
            loss = torch.nn.functional.mse_loss(torch.sigmoid(logits), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss_sum += float(loss.item()) * len(selected)
            seen += len(selected)
        val = evaluate(torch, model, arrays, val_ids, batch_size)
        row = {
            "seed": seed,
            "epoch": epoch,
            "train_blend_mse": train_loss_sum / max(1, seen),
            "validation": val,
            "seconds": round(time.time() - epoch_start, 2),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"seed={seed} epoch={epoch}/{epochs} "
            f"train_mse={row['train_blend_mse']:.6f} "
            f"val_mse={val['blend_mse']:.6f} "
            f"val_cp_mae={val['teacher_score_mae_cp']:.1f} "
            f"val_corr={val['teacher_score_pearson']}",
            flush=True,
        )
        if val["blend_mse"] < best_val:
            best_val = val["blend_mse"]
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    seed_checkpoint = out_dir / f"seed_{seed}_best.pt"
    torch.save({"architecture": ARCH, "seed": seed, "epoch": best_epoch, "state_dict": best_state}, seed_checkpoint)
    model.load_state_dict(best_state)
    best_metrics = evaluate(torch, model, arrays, val_ids, batch_size)
    return {"seed": seed, "best_epoch": best_epoch, "best_validation": best_metrics,
            "checkpoint": seed_checkpoint.name}, best_state, history


def run_training(args) -> int:
    try:
        import torch
    except ImportError:
        raise SystemExit(
            "Нужен CPU PyTorch. Запусти START_FIRST_NNUE.cmd: он установит нужные библиотеки и повторит запуск."
        )

    torch.set_num_threads(max(1, min(args.threads, os.cpu_count() or 1)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "_training_cache"
    input_path = Path(args.input).expanduser()

    with open_text_data(input_path) as (_, manifest, _):
        expected_records = manifest.get("records") if manifest else None
    n, data_info, manifest = prepare_cache(input_path, cache_dir, expected_records)
    arrays = load_memmaps(cache_dir, n)
    split = arrays["split"]
    train_ids = np.flatnonzero(split == 0)
    val_ids = np.flatnonzero(split == 1)
    test_ids = np.flatnonzero(split == 2)
    if len(train_ids) < 1000 or len(val_ids) < 100 or len(test_ids) < 100:
        raise SystemExit("Слишком мало записей в одной из выборок; проверим grouping до обучения.")

    Model = make_model(torch)
    seeds = [args.seed + offset for offset in range(args.restarts)]
    results = []
    histories = []
    selected_state = None
    selected_result = None
    started = time.time()
    for seed in seeds:
        result, state, history = train_one_seed(
            torch, Model, arrays, train_ids, val_ids, output, seed,
            args.epochs, args.batch_size, args.lr, args.threads,
        )
        results.append(result)
        histories.extend(history)
        if selected_result is None or result["best_validation"]["blend_mse"] < selected_result["best_validation"]["blend_mse"]:
            selected_result = result
            selected_state = state

    assert selected_state is not None and selected_result is not None
    selected_path = output / "best_kb8_halfka128.pt"
    torch.save({
        "architecture": ARCH,
        "target": {"score_scale_cp": SCORE_SCALE, "lambda_score": LAMBDA_SCORE},
        "selected_seed": selected_result["seed"],
        "selected_epoch": selected_result["best_epoch"],
        "state_dict": selected_state,
    }, selected_path)

    # Test is touched only after choosing the best checkpoint on validation.
    final_model = Model()
    final_model.load_state_dict(selected_state)
    test_metrics = evaluate(torch, final_model, arrays, test_ids, args.batch_size)
    elapsed = time.time() - started
    report = {
        "status": "pilot_complete_static_only",
        "architecture": ARCH,
        "interpretation": "Offline static training pilot; not yet integrated into the C++ engine and not an Elo result.",
        "data": {
            "input": str(input_path),
            "manifest_records": manifest.get("records") if manifest else None,
            "nodes": manifest.get("nodes") if manifest else None,
            "seed": manifest.get("seed") if manifest else None,
            **data_info,
        },
        "training": {
            "device": "CPU",
            "threads": torch.get_num_threads(),
            "epochs_max": args.epochs,
            "restarts": args.restarts,
            "batch_size": args.batch_size,
            "learning_rate_start": args.lr,
            "score_scale_cp": SCORE_SCALE,
            "lambda_score": LAMBDA_SCORE,
            "family_split": "90/5/5 deterministic hash of (seed,family_id)",
            "selected_seed": selected_result["seed"],
            "selected_epoch": selected_result["best_epoch"],
            "restarts_summary": results,
            "test_metrics_selected_checkpoint": test_metrics,
            "wall_seconds": round(elapsed, 2),
            "checkpoint": selected_path.name,
        },
        "history": histories,
        "notes": [
            "WDL target is result_stm mapped to 0/0.5/1 and mixed 25% with 75% sigmoid(score_cp/400).",
            "All targets are side-to-move relative; STM features are concatenated first.",
            "No extra data generation is required for this pilot.",
        ],
    }
    report_path = output / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Chat-friendly plain-text summary: the user cannot attach JSON files here.
    txt_lines = [
        "KB8-HalfKA-128 first pilot training report",
        "===========================================",
        f"Status: {report['status']}",
        f"Records: {data_info['records']:,}; teacher nodes: {manifest.get('nodes') if manifest else 'unknown'}",
        f"Grouped split: train {data_info['train_records']:,} / val {data_info['validation_records']:,} / test {data_info['test_records']:,}",
        f"Family groups: train {data_info['train_families']:,} / val {data_info['validation_families']:,} / test {data_info['test_families']:,}",
        f"Score labels at +/-15000 cp: {data_info['score_at_clamp_count']:,}",
        f"Selected seed/epoch: {selected_result['seed']} / {selected_result['best_epoch']}",
        f"Selected validation metrics: {json.dumps(selected_result['best_validation'], ensure_ascii=False)}",
        f"Held-out test metrics: {json.dumps(test_metrics, ensure_ascii=False)}",
        f"Checkpoint: {selected_path.name}",
        f"Wall time seconds: {elapsed:.2f}",
        "",
        "Epoch history:",
        "seed epoch train_mse val_mse val_cp_mae val_cp_mae_unclamped val_corr result_brier seconds",
    ]
    for row in histories:
        val = row["validation"]
        txt_lines.append(
            f"{row['seed']} {row['epoch']} {row['train_blend_mse']:.6f} "
            f"{val['blend_mse']:.6f} {val['teacher_score_mae_cp']:.1f} "
            f"{val['teacher_score_mae_cp_unclamped']:.1f} "
            f"{val['teacher_score_pearson'] if val['teacher_score_pearson'] is not None else 'n/a'} "
            f"{val['result_brier']:.6f} {row['seconds']:.2f}"
        )
    txt_lines += ["", "This is an offline fit report, not an engine match or Elo result."]
    (output / "training_report.txt").write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

    history_path = output / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["seed", "epoch", "train_blend_mse", "val_blend_mse", "val_teacher_mae_cp",
                         "val_teacher_mae_cp_unclamped", "val_teacher_pearson", "val_result_brier", "seconds"])
        for row in histories:
            val = row["validation"]
            writer.writerow([row["seed"], row["epoch"], row["train_blend_mse"], val["blend_mse"],
                             val["teacher_score_mae_cp"], val["teacher_score_mae_cp_unclamped"],
                             val["teacher_score_pearson"], val["result_brier"], row["seconds"]])

    # Close mapped files before deleting the temporary feature cache (important on Windows).
    del arrays, split
    if not args.keep_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"\nDone. Report: {report_path}")
    print(f"Selected checkpoint: {selected_path}")
    print(f"Test blend MSE: {test_metrics['blend_mse']:.6f}")
    print("Send training_report.json back for the next decision; do not send the raw shard.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Folder, .jsonl/.jsonl.gz, or original GitHub artifact ZIP")
    parser.add_argument("--out", default="nnue_pilot_0001", help="Output folder")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--keep-cache", action="store_true", help="Keep the ~140 MB temporary feature cache")
    parser.add_argument("--prepare-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.epochs < 1 or args.restarts < 1 or args.batch_size < 1:
        parser.error("epochs, restarts and batch-size must be positive")

    if args.prepare_only:
        input_path = Path(args.input).expanduser()
        with open_text_data(input_path) as (_, manifest, _):
            expected_records = manifest.get("records") if manifest else None
        cache_dir = Path(args.out).expanduser() / "_training_cache"
        n, info, _ = prepare_cache(input_path, cache_dir, expected_records)
        print(json.dumps({"prepared_records": n, **info}, indent=2))
        print("Prepare-only requested; no model training was run.")
        return 0
    return run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
