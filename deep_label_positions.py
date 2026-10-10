#!/usr/bin/env python3
"""Relabel existing shard positions with a deeper frozen-teacher search.

Reads the original labeled shard (jsonl.gz), takes a deterministic chunk of
records by global 0-based index, runs the frozen HCE teacher engine at a fixed
node budget on every FEN, and writes deep_labels_chunkN.jsonl.gz with the same
schema as the shard: all original fields (fen, stm, ply, game_id, family_id,
seed, result_stm) are preserved; score_cp/score_mate are replaced by the deep
search result; index and label_nodes are added for auditing. WDL labels
(result_stm) are copied untouched for every position.

Mate scores are emitted as score_cp = sign * 30000 and score_mate = sign * m,
so the v2/v3 trainer's clamp mask (|score_cp| < 14999.5) excludes them from
the score loss exactly like the original shard's clamped labels.

Stdlib only; designed for GitHub-Actions Linux runners (2 cores, ~7 GB RAM),
streams the shard instead of loading it into memory.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path

CLAMP_CP = 30000
PROGRESS_EVERY = 5000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Teacher:
    """One frozen-engine process driven over UCI."""

    def __init__(self, engine: Path, hash_mb: int):
        self.proc = subprocess.Popen(
            [str(engine.resolve())], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            bufsize=1,
        )
        self._send("uci")
        self._until("uciok")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("isready")
        self._until("readyok")

    def _send(self, line: str) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("teacher stdin closed")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _until(self, token: str) -> str:
        if self.proc.stdout is None:
            raise RuntimeError("teacher stdout closed")
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("teacher engine exited unexpectedly")
            if token in line:
                return line

    def evaluate(self, fen: str, nodes: int) -> tuple[int, int]:
        """Return (score_cp, score_mate) from the side-to-move perspective."""
        self._send("ucinewgame")
        self._send("isready")
        self._until("readyok")
        self._send(f"position fen {fen}")
        self._send(f"go nodes {nodes}")
        cp: int | None = None
        mate: int | None = None
        if self.proc.stdout is None:
            raise RuntimeError("teacher stdout closed")
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("teacher engine exited during search")
            if line.startswith("info ") and " score " in line:
                tokens = line.split()
                for i in range(len(tokens) - 2):
                    if tokens[i] != "score":
                        continue
                    kind, raw = tokens[i + 1], tokens[i + 2]
                    if kind == "cp":
                        try:
                            cp, mate = int(raw), None
                        except ValueError:
                            pass
                    elif kind == "mate":
                        try:
                            mate, cp = int(raw), None
                        except ValueError:
                            pass
            elif line.startswith("bestmove"):
                if mate is not None:
                    sign = 1 if mate > 0 else -1
                    return sign * CLAMP_CP, mate
                if cp is not None:
                    return max(-CLAMP_CP, min(CLAMP_CP, cp)), 0
                raise RuntimeError(f"teacher returned no score for fen: {fen}")

    def close(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
            self.proc.wait(timeout=5)


def label_worker(engine: str, hash_mb: int, shard: str, start: int, end: int,
                 nodes: int, out_path: str, worker_id: int, stats_path: str) -> None:
    teacher = Teacher(Path(engine), hash_mb)
    labeled = failed = mates = 0
    stm_white = stm_black = 0
    score_sum = 0
    score_min: int | None = None
    score_max: int | None = None
    opened_at = time.time()
    with gzip.open(shard, "rt", encoding="utf-8") as src, \
            gzip.open(out_path, "wt", encoding="utf-8") as dst:
        for index, line in enumerate(src):
            if index < start:
                continue
            if index >= end:
                break
            record = json.loads(line)
            score_cp = score_mate = None
            for attempt in range(2):
                try:
                    score_cp, score_mate = teacher.evaluate(record["fen"], nodes)
                    break
                except Exception as exc:
                    print(f"worker{worker_id}: retry after error on index {index}: {exc}",
                          flush=True)
                    teacher.close()
                    try:
                        teacher = Teacher(Path(engine), hash_mb)
                    except Exception:
                        break
            if score_cp is None:
                failed += 1
                continue
            record["score_cp"] = score_cp
            record["score_mate"] = score_mate
            record["index"] = index
            record["label_nodes"] = nodes
            dst.write(json.dumps(record, separators=(",", ":")) + "\n")
            labeled += 1
            mates += int(score_mate != 0)
            stm = record.get("stm")
            stm_white += int(stm == "w")
            stm_black += int(stm == "b")
            score_sum += score_cp
            score_min = score_cp if score_min is None else min(score_min, score_cp)
            score_max = score_cp if score_max is None else max(score_max, score_cp)
            if labeled % PROGRESS_EVERY == 0:
                elapsed = max(1e-9, time.time() - opened_at)
                rate = labeled / elapsed
                remaining = (end - start) - (index - start + 1)
                print(f"worker{worker_id}: labeled={labeled} failed={failed} "
                      f"rate={rate:.1f} pos/s eta_min={remaining / rate / 60:.1f}",
                      flush=True)
    teacher.close()
    stats = {
        "worker": worker_id, "start": start, "end": end, "labeled": labeled,
        "failed": failed, "mates": mates, "elapsed_sec": round(time.time() - opened_at, 1),
        "stm_white": stm_white, "stm_black": stm_black,
        "score_min": score_min, "score_max": score_max,
        "score_mean": (score_sum / labeled) if labeled else None,
    }
    Path(stats_path).write_text(json.dumps(stats), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True, help="original labeled_shard_0001.jsonl.gz")
    parser.add_argument("--engine", required=True, help="frozen teacher executable")
    parser.add_argument("--engine-sha256", default="", help="optional expected sha256 of engine file")
    parser.add_argument("--chunk-index", type=int, required=True, help="0-based chunk number")
    parser.add_argument("--positions-per-chunk", type=int, default=175000)
    parser.add_argument("--nodes", type=int, default=16384)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--out", required=True, help="output deep_labels_chunkN.jsonl.gz")
    parser.add_argument("--report", required=True, help="compact txt report")
    args = parser.parse_args()

    if args.chunk_index < 0 or args.positions_per_chunk <= 0 or args.nodes <= 0 \
            or not (1 <= args.workers <= 8):
        raise SystemExit("invalid chunk/nodes/workers parameters")
    shard_path = Path(args.shard).expanduser().resolve()
    engine_path = Path(args.engine).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    for path in (shard_path, engine_path):
        if not path.is_file():
            raise SystemExit(f"not found: {path}")
    if args.engine_sha256:
        actual = sha256_file(engine_path)
        if actual.lower() != args.engine_sha256.lower():
            raise SystemExit(
                f"engine sha256 mismatch: expected {args.engine_sha256}, got {actual}")

    start = args.chunk_index * args.positions_per_chunk
    end = start + args.positions_per_chunk
    part_dir = out_path.parent / f"{out_path.stem}_parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    boundaries = []
    span = (end - start) / args.workers
    for worker_id in range(args.workers):
        w_start = start + int(worker_id * span)
        w_end = start + int((worker_id + 1) * span) if worker_id < args.workers - 1 else end
        boundaries.append((worker_id, w_start, w_end))

    opened_at = time.time()
    processes = []
    for worker_id, w_start, w_end in boundaries:
        proc = multiprocessing.Process(
            target=label_worker,
            args=(str(engine_path), args.hash_mb, str(shard_path), w_start, w_end,
                  args.nodes, str(part_dir / f"part_{worker_id}.jsonl.gz"),
                  worker_id, str(part_dir / f"stats_{worker_id}.json")),
        )
        proc.start()
        processes.append(proc)
    for proc in processes:
        proc.join()
    if any(proc.exitcode != 0 for proc in processes):
        raise SystemExit("one or more labeling workers failed")

    labeled = failed = mates = 0
    stm_white = stm_black = 0
    first_index: int | None = None
    last_index: int | None = None
    score_min: int | None = None
    score_max: int | None = None
    score_sum = 0.0
    with gzip.open(out_path, "wt", encoding="utf-8") as merged:
        for worker_id, _, _ in boundaries:
            part = part_dir / f"part_{worker_id}.jsonl.gz"
            with gzip.open(part, "rt", encoding="utf-8") as src:
                for line in src:
                    record = json.loads(line)
                    score = int(record["score_cp"])
                    index = int(record["index"])
                    if last_index is not None and index <= last_index:
                        raise SystemExit(
                            f"index order violation: {index} after {last_index}")
                    if first_index is None:
                        first_index = index
                    last_index = index
                    merged.write(line if line.endswith("\n") else line + "\n")
                    labeled += 1
                    mates += int(int(record["score_mate"]) != 0)
                    stm = record.get("stm")
                    stm_white += int(stm == "w")
                    stm_black += int(stm == "b")
                    score_sum += score
                    score_min = score if score_min is None else min(score_min, score)
                    score_max = score if score_max is None else max(score_max, score)
            stats = json.loads((part_dir / f"stats_{worker_id}.json").read_text(encoding="utf-8"))
            failed += int(stats["failed"])

    output_sha = sha256_file(out_path)
    shard_sha = sha256_file(shard_path)
    engine_sha = sha256_file(engine_path)
    max_failed = max(50, int(labeled * 0.005))
    status = "ok" if failed <= max_failed and labeled > 0 else "failed"
    report = [
        "KB8 deep-label chunk report",
        "===========================",
        f"status: {status}",
        f"shard: {shard_path}",
        f"shard_sha256: {shard_sha}",
        f"engine: {engine_path}",
        f"engine_sha256: {engine_sha}",
        f"nodes_per_position: {args.nodes}",
        f"chunk_index: {args.chunk_index}",
        f"positions_per_chunk: {args.positions_per_chunk}",
        f"index_range: [{start}, {end})",
        f"workers: {args.workers}",
        f"labeled: {labeled}",
        f"failed: {failed}",
        f"mates: {mates}",
        f"score_min_cp: {score_min}",
        f"score_max_cp: {score_max}",
        f"score_mean_cp: {round(score_sum / labeled, 1) if labeled else 'n/a'}",
        f"elapsed_seconds: {round(time.time() - opened_at, 1)}",
        f"output: {out_path}",
        f"output_sha256: {output_sha}",
        f"stm_white: {stm_white}",
        f"stm_black: {stm_black}",
        f"observed_index_range: [{first_index}, {last_index}]",
        "index_order_check: strictly_increasing_ok",
        "wdl_labels: result_stm copied unchanged from the shard for every record",
        "schema: original record fields + index + label_nodes; mate => score_cp=+/-30000",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)
    print(f"Text report: {report_path}", flush=True)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
