#!/usr/bin/env python3
"""Calibrate the opening pool before self-play.

For each pool line (first --book_plies plies from the start position):
  1. check that every move is legal (python-chess);
  2. drop duplicate positions (same FEN after the book, ignoring clocks);
  3. search the position with the engine: `go nodes N` (default 10000);
  4. keep lines with |score| <= --max_cp (default 100 cp, from White's view).
Writes:
  opening_pool_calibrated.txt   (kept lines, same format as the pool)
  openings_report.json          (counts, score histogram, drops by reason)

Balanced starts keep the outcome labels informative: a position already
decided by 5 pawns gives almost no signal about the eval terms.
"""
import argparse
import json
import sys
from collections import Counter

import chess

import os  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from play_games import Engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--book_plies", type=int, default=8)
    ap.add_argument("--nodes", type=int, default=10000)
    ap.add_argument("--max_cp", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="0 = all lines")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    eng = Engine(args.engine, 16)
    seen = set()
    drops = Counter()
    hist = Counter()
    kept = []
    n = 0
    with open(args.pool, encoding="utf-8") as f:
        for raw in f:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 3:
                drops["format"] += 1
                continue
            if args.limit and n >= args.limit:
                break
            n += 1
            moves = parts[2].split()[: args.book_plies]
            board = chess.Board()
            ok = True
            for m in moves:
                try:
                    board.push_uci(m)
                except ValueError:
                    ok = False
                    break
            if not ok:
                drops["illegal"] += 1
                continue
            key = " ".join(board.fen().split(" ")[:4])
            if key in seen:
                drops["duplicate"] += 1
                continue
            seen.add(key)
            if board.is_game_over():
                drops["game_over"] += 1
                continue
            eng.new_game()
            _best, cp, mate, _nodes = eng.search(chess.STARTING_FEN, moves, args.nodes)
            if mate is not None:
                drops["mate_score"] += 1
                continue
            if cp is None:
                drops["no_score"] += 1
                continue
            # engine score is from side to move; convert to White's view
            cp_white = cp if board.turn == chess.WHITE else -cp
            bucket = max(-300, min(300, (cp_white // 25) * 25))
            hist[bucket] += 1
            if abs(cp_white) > args.max_cp:
                drops["score_range"] += 1
                continue
            kept.append(raw.rstrip("\n"))
    eng.quit()

    with open(f"{args.out_dir}/opening_pool_calibrated.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")
    report = {"lines_read": n, "kept": len(kept), "drops": dict(drops),
              "score_hist_white_cp_bucket25": {str(k): v for k, v in sorted(hist.items())},
              "max_cp": args.max_cp, "nodes": args.nodes, "book_plies": args.book_plies}
    with open(f"{args.out_dir}/openings_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
