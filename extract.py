#!/usr/bin/env python3
"""Turn play_games.py JSONL into a training table (TSV, gzip).

Filters (quiet positions only):
  - drop discarded games (max_plies / errors)
  - drop the book prefix (first --skip_plies plies, default 8)
  - drop positions in check, positions where the move played is a capture or a promotion
  - drop mate scores and |cp| > --max_cp (default 2000)
  - keep every --every-th ply (default 2)
Output columns:
  game_id  ply  fen  result_white  outcome_stm  cp_stm  nodes
  result_white: 1.0 / 0.5 / 0.0  (game result, White's view)
  outcome_stm : same, from the side to move
  cp_stm      : engine score from the side to move (for the alternative label)

Labels are not taken from Stockfish. Both labels come from our own games.
"""
import argparse
import gzip
import json
import sys


def res_to_white(r):
    return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", nargs="+", required=True, help="JSONL shard files")
    ap.add_argument("--out", required=True, help="output .tsv.gz")
    ap.add_argument("--skip_plies", type=int, default=8)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--max_cp", type=int, default=2000)
    args = ap.parse_args()

    stats = {"games": 0, "discarded": 0, "positions_in": 0, "positions_out": 0,
             "rejected": {"check": 0, "capture": 0, "promo": 0, "mate": 0,
                          "cp_none": 0, "cp_big": 0, "every": 0, "skip": 0}}
    with gzip.open(args.out, "wt", encoding="utf-8") as out:
        out.write("game_id\tply\tfen\tresult_white\toutcome_stm\tcp_stm\tnodes\n")
        for path in args.inp:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    g = json.loads(line)
                    stats["games"] += 1
                    if g["discarded"]:
                        stats["discarded"] += 1
                        continue
                    rw = res_to_white(g["result"])
                    for i, p in enumerate(g["plies"]):
                        stats["positions_in"] += 1
                        if i < args.skip_plies:
                            stats["rejected"]["skip"] += 1
                            continue
                        if i % args.every != 0:
                            stats["rejected"]["every"] += 1
                            continue
                        if p["in_check"]:
                            stats["rejected"]["check"] += 1
                            continue
                        if p["capture"]:
                            stats["rejected"]["capture"] += 1
                            continue
                        if p["promo"]:
                            stats["rejected"]["promo"] += 1
                            continue
                        if p["mate"] is not None:
                            stats["rejected"]["mate"] += 1
                            continue
                        if p["cp"] is None:
                            stats["rejected"]["cp_none"] += 1
                            continue
                        if abs(p["cp"]) > args.max_cp:
                            stats["rejected"]["cp_big"] += 1
                            continue
                        stm_white = p["fen"].split(" ")[1] == "w"
                        outcome_stm = rw if stm_white else 1.0 - rw
                        out.write(f"{g['game_id']}\t{i}\t{p['fen']}\t{rw}\t{outcome_stm}\t"
                                  f"{p['cp']}\t{p['nodes'] if p['nodes'] is not None else ''}\n")
                        stats["positions_out"] += 1
    print(json.dumps(stats, indent=1), file=sys.stderr)


if __name__ == "__main__":
    main()
