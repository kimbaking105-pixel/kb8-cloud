#!/usr/bin/env python3
"""Run K parallel NNUE self-play shard workers for probe C (self-distillation).

Spawns K copies of selfplay_nnue_positions.py; each worker plays the SAME
candidate network against itself (pure-WDL mode), writes its own shard
(selfdist_shards/labeled_shard_cN.jsonl.gz + manifest), then all shards are
concatenated into one raw shard for the deep-label step.

Stdlib only (python-chess is needed by the workers themselves).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tool", default="selfplay_nnue_positions.py")
    ap.add_argument("--engine", required=True, help="candidate exe (NNUE_WDL)")
    ap.add_argument("--eval-file", required=True, help="kb8_twohead_v2.nnue of the v3 pilot")
    ap.add_argument("--out-dir", default="selfdist_shards")
    ap.add_argument("--raw-dir", default="selfdist_raw")
    ap.add_argument("--positions", type=int, default=1000000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nodes", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--games-cap", type=int, default=20000,
                    help="per-worker game cap (positions cap usually hits first)")
    ap.add_argument("--opening-book", default="",
                    help="optional book file; empty = random heuristic openings")
    ap.add_argument("--opening-plies-min", type=int, default=8)
    ap.add_argument("--opening-plies-max", type=int, default=12)
    ap.add_argument("--early-ply-jitter", type=int, default=15)
    ap.add_argument("--max-plies", type=int, default=200,
                    help="game ply cap (cycle 2 uses 200)")
    ap.add_argument("--discard-before-ply", type=int, default=24,
                    help="drop games ending before this absolute ply (length-only guard)")
    return ap.parse_args()


def stream_output(tag: str, proc: subprocess.Popen, lines: list[str]) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        text = line.rstrip("\r\n")
        lines.append(text)
        print(f"[{tag}] {text}", flush=True)


def main() -> int:
    args = parse_args()
    tool = Path(args.tool)
    engine = Path(args.engine).resolve()
    eval_file = Path(args.eval_file).resolve()
    out_dir = Path(args.out_dir)
    raw_dir = Path(args.raw_dir)

    for required, why in ((tool, "selfplay tool"),
                          (engine, "candidate executable"),
                          (eval_file, "network file")):
        if not required.exists():
            print(f"FATAL: {why} not found: {required}", flush=True)
            return 1

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    book_args = []
    if args.opening_book:
        book = Path(args.opening_book)
        if book.exists():
            book_args = ["--opening-book", str(book)]
            print(f"opening book: {book.name}", flush=True)
        else:
            print(f"NOTE: opening book {book} not found; using random openings.", flush=True)
    else:
        print(f"random heuristic openings: {args.opening_plies_min}-{args.opening_plies_max} plies; "
              f"early-ply node jitter: {args.early_ply_jitter} plies (nodes/2..nodes); "
              f"short games ending before ply {args.discard_before_ply} are discarded",
              flush=True)

    per_worker = int(math.ceil(args.positions / args.workers))
    print(f"=== probe C self-play: {args.workers} workers x {per_worker} positions "
          f"(total target {args.positions}), nodes={args.nodes}, seed={args.seed} ===",
          flush=True)

    procs: list[subprocess.Popen] = []
    logs: list[list[str]] = []
    readers: list[threading.Thread] = []
    started = time.time()
    for i in range(args.workers):
        shard_id = f"c{i}"
        cmd = [
            sys.executable, str(tool),
            "--engine", str(engine),
            "--eval-file", str(eval_file),
            "--out-dir", str(out_dir),
            "--shard-id", shard_id,
            "--positions", str(per_worker),
            "--games", str(args.games_cap),
            "--nodes", str(args.nodes),
            "--threads", "1",
            "--hash", "32",
            "--seed", str(args.seed + i),
            "--book-offset", str(i * args.games_cap),
            "--family-offset", str(i * 100000),
            "--opening-plies-min", str(args.opening_plies_min),
            "--opening-plies-max", str(args.opening_plies_max),
            "--early-ply-jitter", str(args.early_ply_jitter),
            "--discard-before-ply", str(args.discard_before_ply),
            "--sample-every", "1",
            "--min-ply", "10",
            "--max-plies", str(args.max_plies),
        ] + book_args
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        procs.append(proc)
        buf: list[str] = []
        logs.append(buf)
        thread = threading.Thread(target=stream_output, args=(f"w{i}", proc, buf), daemon=True)
        thread.start()
        readers.append(thread)
        print(f"[w{i}] started: {per_worker} positions cap", flush=True)

    failed = []
    for i, proc in enumerate(procs):
        rc = proc.wait()
        readers[i].join(timeout=10)
        if rc != 0:
            failed.append((i, rc))
    elapsed = time.time() - started
    if failed:
        print(f"FATAL: workers failed {failed}; elapsed={elapsed:.1f}s", flush=True)
        return 1

    # Concatenate shards into one raw shard (streamed, no index checks).
    records = 0
    games = 0
    merged_from = []
    raw_path = raw_dir / "raw_shard.jsonl.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8", compresslevel=6) as out:
        for i in range(args.workers):
            shard_id = f"c{i}"
            shard_path = out_dir / f"labeled_shard_{shard_id}.jsonl.gz"
            manifest_path = out_dir / f"manifest_{shard_id}.json"
            if not shard_path.exists() or not manifest_path.exists():
                print(f"FATAL: missing shard/manifest for worker {i}", flush=True)
                return 1
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            count = 0
            with gzip.open(shard_path, "rt", encoding="utf-8") as src:
                for line in src:
                    if not line.strip():
                        continue
                    out.write(line if line.endswith("\n") else "\n")
                    count += 1
            if count != manifest.get("records"):
                print(f"FATAL: worker {i}: file has {count} records, "
                      f"manifest says {manifest.get('records')}", flush=True)
                return 1
            records += count
            games += int(manifest.get("games", 0))
            merged_from.append({"file": shard_path.name, "records": count,
                                "games": manifest.get("games"),
                                "elapsed_sec": manifest.get("elapsed_sec")})
            print(f"[merge] {shard_path.name}: {count} records", flush=True)

    (raw_dir / "records.txt").write_text(str(records) + "\n", encoding="utf-8")
    summary = {
        "schema": "probe-c-raw-selfplay-v1",
        "records": records,
        "games": games,
        "workers": args.workers,
        "nodes": args.nodes,
        "seed": args.seed,
        "engine": engine.name,
        "eval_file": eval_file.name,
        "elapsed_sec": round(elapsed, 1),
        "merged_from": merged_from,
    }
    (raw_dir / "raw_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("=== self-play complete ===", flush=True)
    print(f"records: {records}   games: {games}   elapsed: {elapsed:.1f}s", flush=True)
    print(f"raw shard: {raw_path}", flush=True)
    if records < 50000:
        print("WARNING: fewer than 50k records; training may be weak.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
