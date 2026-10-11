#!/usr/bin/env python3
"""Fixed-node match between two engine binaries (colour-swapped openings).

- Same node budget for both sides (`go nodes N`).
- Openings: random lines from the pool (book_plies moves from startpos),
  each opening is played twice with colours swapped.
- No adjudication. Games that reach --max_plies are reported as unfinished
  and excluded from the score (never counted as draws).
- Output: JSON with W/D/L, score, Elo estimate and 95% CI (logistic model),
  and per-game records.

Usage:
  python3 match.py --a eng_a --b eng_b --pool opening_pool_calibrated.txt \
      --pairs 500 --nodes 10000 --seed 7 --out match.json
"""
import argparse
import json
import math
import random
import sys
import time

import chess

from play_games import Engine, load_pool, game_result  # noqa: F401


def play(white_eng, black_eng, book, nodes, max_plies):
    board = chess.Board()
    for m in book:
        board.push_uci(m)
    played = []
    engines = {chess.WHITE: white_eng, chess.BLACK: black_eng}
    white_eng.new_game()
    black_eng.new_game()
    while True:
        over = game_result(board)
        if over:
            return over[0], over[1], len(played)
        if len(played) >= max_plies:
            return None, "unfinished", len(played)
        eng = engines[board.turn]
        prefix = list(book) + played
        best, _cp, _mate, _n = eng.search(chess.STARTING_FEN, prefix, nodes)
        try:
            mv = chess.Move.from_uci(best)
        except ValueError:
            return None, "bad_move", len(played)
        if mv not in board.legal_moves:
            return None, "illegal_move", len(played)
        board.push(mv)
        played.append(best)


def elo_ci(w, d, l):
    n = w + d + l
    if n == 0:
        return None
    s = (w + 0.5 * d) / n
    # score variance per game
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * (0 - s) ** 2) / n
    if var <= 0:
        # no losses or no wins at all: Elo is not defined without a bound
        return {"score": s, "elo": None, "elo_lo": None, "elo_hi": None,
                "note": "zero variance, Elo undefined (all results identical)"}
    se = math.sqrt(var / n)

    def to_elo(x):
        x = min(max(x, 1e-6), 1 - 1e-6)
        return -400 * math.log10(1 / x - 1)

    lo, hi = s - 1.96 * se, s + 1.96 * se
    return {"score": s, "elo": to_elo(s), "elo_lo": to_elo(lo), "elo_hi": to_elo(hi)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="engine A (score is reported for A)")
    ap.add_argument("--b", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--pairs", type=int, required=True, help="number of openings (2 games each)")
    ap.add_argument("--nodes", type=int, default=10000)
    ap.add_argument("--book_plies", type=int, default=8)
    ap.add_argument("--max_plies", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--wa", default=None, help="weights file for engine A (HCE v2)")
    ap.add_argument("--wb", default=None, help="weights file for engine B (HCE v2)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = load_pool(args.pool)
    ea = Engine(args.a, args.hash, args.wa)
    eb = Engine(args.b, args.hash, args.wb)
    w = d = l = 0
    unfinished = 0
    games = []
    t0 = time.time()
    for pi in range(args.pairs):
        _fen, line = pool[rng.randrange(len(pool))]
        book = line[: args.book_plies]
        for a_is_white in (True, False):
            if a_is_white:
                res, reason, n = play(ea, eb, book, args.nodes, args.max_plies)
            else:
                res, reason, n = play(eb, ea, book, args.nodes, args.max_plies)
            rec = {"pair": pi, "a_white": a_is_white, "result": res,
                   "reason": reason, "plies": n}
            games.append(rec)
            if res is None:
                unfinished += 1
                continue
            if res == "1/2-1/2":
                a_score = 0.5
            elif (res == "1-0") == a_is_white:
                a_score = 1.0
            else:
                a_score = 0.0
            rec["a_score"] = a_score
            if a_score == 1.0:
                w += 1
            elif a_score == 0.5:
                d += 1
            else:
                l += 1
        if (pi + 1) % 25 == 0:
            print(f"pairs {pi+1}/{args.pairs} W{w} D{d} L{l} unfinished {unfinished} "
                  f"t={time.time()-t0:.0f}s", file=sys.stderr)
    ea.quit()
    eb.quit()
    stat = elo_ci(w, d, l) or {}
    out = {"a": args.a, "b": args.b, "pairs": args.pairs, "nodes": args.nodes,
           "wins": w, "draws": d, "losses": l, "unfinished": unfinished,
           "seconds": round(time.time() - t0, 1), **stat, "games": games}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    summary = {k: v for k, v in out.items() if k != "games"}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
