#!/usr/bin/env python3
"""Merge deep-label chunk files into the folder layout expected by the trainer.

Creates out-dir (default deep_labels_merged/) containing:
  labeled_shard_deep.jsonl.gz  - all records recompressed into one gzip file
  manifest_deep.json           - exact record count, required by the trainer

Inputs may be gzipped (deep_labels_chunkN.jsonl.gz) or already decompressed
(deep_labels_chunkN.jsonl) - detected by the gzip magic bytes.

Each chunk is validated cheaply: non-empty, first record parses as JSON and
contains the fields the trainer reads (fen, stm, score_cp, result_stm,
family_id, seed). Stdlib only; natural chunk ordering (chunk2 before chunk10).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

REQUIRED_FIELDS = ("fen", "stm", "score_cp", "result_stm", "family_id", "seed")


def chunk_key(path: Path) -> tuple[int, str]:
    match = re.search(r"chunk(\d+)", path.name)
    return (int(match.group(1)) if match else 1 << 30, path.name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_text(path: Path):
    """Open a chunk for reading as text; gzip if the file starts with 1f 8b."""
    with path.open("rb") as probe:
        magic = probe.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=".",
                        help="folder containing deep_labels_chunk* files (.jsonl.gz or .jsonl)")
    parser.add_argument("--out-dir", default="deep_labels_merged")
    parser.add_argument("--pattern", default="",
                        help="optional glob override; default accepts .jsonl.gz and plain .jsonl")
    args = parser.parse_args()

    base = Path(args.dir).expanduser().resolve()
    if args.pattern:
        inputs = sorted(base.glob(args.pattern), key=chunk_key)
    else:
        inputs = sorted(
            {p.resolve() for pat in ("deep_labels_chunk*.jsonl.gz", "deep_labels_chunk*.jsonl")
             for p in base.glob(pat) if p.is_file()},
            key=chunk_key)
    if not inputs:
        raise SystemExit(
            f"No deep_labels_chunk*.jsonl(.gz) files in {base}. "
            "Download the deep-labels artifacts from GitHub first.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = out_dir / "labeled_shard_deep.jsonl.gz"
    manifest_path = out_dir / "manifest_deep.json"

    total = 0
    stm_white = stm_black = 0
    result_counts = {"1": 0, "0": 0, "-1": 0}
    prev_last_index: int | None = None
    coverage_first: int | None = None
    coverage_last: int | None = None
    per_file: list[dict] = []
    with gzip.open(merged, "wt", encoding="utf-8") as out:
        for path in inputs:
            count = 0
            first_index: int | None = None
            last_index: int | None = None
            prev_index: int | None = None
            with open_text(path) as src:
                for line in src:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise SystemExit(f"{path.name}: invalid JSON at record {count + 1}: {exc}")
                    if count == 0:
                        missing = [key for key in REQUIRED_FIELDS if key not in record]
                        if missing:
                            raise SystemExit(f"{path.name}: first record misses fields {missing}")
                    index = int(record.get("index", -1))
                    if index < 0:
                        raise SystemExit(f"{path.name}: record {count + 1} has no index")
                    if prev_index is not None and index <= prev_index:
                        raise SystemExit(
                            f"{path.name}: index not strictly increasing ({index} after {prev_index})")
                    prev_index = index
                    if first_index is None:
                        first_index = index
                    last_index = index
                    stm = record.get("stm")
                    stm_white += int(stm == "w")
                    stm_black += int(stm == "b")
                    result_key = str(record.get("result_stm"))
                    if result_key in result_counts:
                        result_counts[result_key] += 1
                    count += 1
                    out.write(line if line.endswith("\n") else line + "\n")
            if count == 0:
                raise SystemExit(f"{path.name}: contains no records")
            if prev_last_index is not None and first_index is not None and first_index <= prev_last_index:
                raise SystemExit(
                    f"{path.name}: index range overlaps the previous chunk "
                    f"({first_index} <= {prev_last_index})")
            prev_last_index = last_index
            if coverage_first is None:
                coverage_first = first_index
            coverage_last = last_index
            per_file.append({"file": path.name, "records": count,
                             "index_range": [first_index, last_index],
                             "sha256": sha256_file(path)})
            total += count
            print(f"merged {path.name}: {count} records, index [{first_index}, {last_index}]",
                  flush=True)

    merged_sha = sha256_file(merged)
    imbalance_pct = (abs(stm_white - stm_black) / total * 100.0) if total else 0.0
    manifest = {
        "schema": "tiny-nnue-labeled-jsonl-v1",
        "records": total,
        "merged_from": per_file,
        "index_coverage": [coverage_first, coverage_last],
        "index_integrity": "strictly_increasing_within_chunks_and_disjoint_across_chunks",
        "stm_white": stm_white,
        "stm_black": stm_black,
        "stm_imbalance_pct": round(imbalance_pct, 4),
        "result_stm_counts": {"win": result_counts["1"], "draw": result_counts["0"],
                              "loss": result_counts["-1"]},
        "sha256_data": merged_sha,
        "labels": "score_cp/score_mate from frozen-teacher deep search; "
                  "result_stm unchanged from labeled_shard_0001",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(f"total records: {total}")
    print(f"index coverage: [{coverage_first}, {coverage_last}] "
          f"(strictly increasing, no cross-chunk overlap)")
    print(f"stm_white: {stm_white} stm_black: {stm_black} imbalance: {imbalance_pct:.3f}%")
    print(f"result_stm counts: win={result_counts['1']} draw={result_counts['0']} "
          f"loss={result_counts['-1']}")
    if imbalance_pct > 5.0:
        print("WARNING: side-to-move imbalance above 5% - inspect the chunks before training")
    print(f"merged file: {merged}")
    print(f"merged sha256: {merged_sha}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
