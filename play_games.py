#!/usr/bin/env python3
"""Self-play data generator (fixed nodes per move, no adjudication).

- Each game starts from a random opening line taken from the opening pool
  (deterministic per seed).
- Both sides use the same engine binary, `go nodes N` (default 10000).
- Partidas are NOT adjudicated by eval and NOT cut at 120 plies.
  Only real game ends count: checkmate, stalemate, insufficient material,
  50-move rule, threefold / fivefold repetition.
- A game that reaches --max-plies (default 600) is DISCARDED and counted
  in the summary (never labelled as a draw).
- Per ply we log: fen before move, move played, engine score (cp, from the
  side to move, last completed iteration), nodes searched.
  Labels (outcome or search score) are built later by extract.py.

Usage:
  python3 play_games.py --engine ./eng --pool opening_pool.txt \
      --games 1000 --nodes 10000 --seed 1 --out shard_01.jsonl
"""
import argparse
import json
import random
import re
import subprocess
import sys
import time

import chess

INFO_SCORE = re.compile(r"score cp (-?\d+)")
INFO_MATE = re.compile(r"score mate (-?\d+)")


class Engine:
    def __init__(self, path, hash_mb, weights=None):
        import os
        env = dict(os.environ)
        if weights:
            env["HCE2_WEIGHTS"] = weights
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        self._send("uci")
        self._read_until("uciok")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def _read_until(self, token):
        lines = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine closed stdout")
            line = line.rstrip("\n")
            lines.append(line)
            if line.startswith(token):
                return lines

    def new_game(self):
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")

    def search(self, fen, moves_uci, nodes):
        """Returns (bestmove, score_cp or None, mate or None, nodes_info)."""
        if moves_uci:
            self._send(f"position fen {fen} moves {' '.join(moves_uci)}")
        else:
            self._send(f"position fen {fen}")
        self._send(f"go nodes {nodes}")
        lines = self._read_until("bestmove")
        bestmove = lines[-1].split()[1]
        score_cp, mate, nodes_done = None, None, None
        for l in lines:
            if l.startswith("info") and " depth " in l:
                m = INFO_SCORE.search(l)
                if m:
                    score_cp, mate = int(m.group(1)), None
                m2 = INFO_MATE.search(l)
                if m2:
                    mate, score_cp = int(m2.group(1)), None
                n = re.search(r" nodes (\d+)", l)
                if n:
                    nodes_done = int(n.group(1))
        return bestmove, score_cp, mate, nodes_done

    def quit(self):
        try:
            self._send("quit")
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


def load_pool(path):
    lines = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            fen = parts[1]
            moves = parts[2].split()
            lines.append((fen, moves))
    return lines


def game_result(board):
    """Returns (result_str, reason) if the game is over by rule, else None."""
    if board.is_checkmate():
        return ("1-0" if board.turn == chess.BLACK else "0-1"), "mate"
    if board.is_stalemate():
        return "1/2-1/2", "stalemate"
    if board.is_insufficient_material():
        return "1/2-1/2", "insufficient"
    if board.is_fivefold_repetition():
        return "1/2-1/2", "fivefold"
    if board.is_seventyfive_moves():
        return "1/2-1/2", "75move"
    if board.is_repetition(3):
        return "1/2-1/2", "threefold"
    if board.halfmove_clock >= 100:
        return "1/2-1/2", "50move"
    return None


def play_one(eng, start_fen, start_moves, nodes, max_plies):
    board = chess.Board(start_fen)
    for m in start_moves:
        board.push_uci(m)
    played = []          # UCI moves after the book prefix
    plies = []           # per-ply records
    eng.new_game()
    while True:
        over = game_result(board)
        if over:
            return {"result": over[0], "reason": over[1], "discarded": False,
                    "plies": plies, "book": start_moves, "moves": played}
        if len(played) >= max_plies:
            return {"result": None, "reason": "max_plies", "discarded": True,
                    "plies": plies, "book": start_moves, "moves": played}
        fen_before = board.fen()
        prefix = list(start_moves) + played
        bestmove, score_cp, mate, nodes_done = eng.search(start_fen, prefix, nodes)
        if bestmove in ("0000", "(none)"):
            return {"result": None, "reason": "no_move", "discarded": True,
                    "plies": plies, "book": start_moves, "moves": played}
        try:
            mv = chess.Move.from_uci(bestmove)
        except ValueError:
            return {"result": None, "reason": "bad_move", "discarded": True,
                    "plies": plies, "book": start_moves, "moves": played}
        if mv not in board.legal_moves:
            return {"result": None, "reason": "illegal_move", "discarded": True,
                    "plies": plies, "book": start_moves, "moves": played}
        plies.append({
            "fen": fen_before,
            "move": bestmove,
            "cp": score_cp,
            "mate": mate,
            "nodes": nodes_done,
            "in_check": board.is_check(),
            "capture": board.is_capture(mv),
            "promo": mv.promotion is not None,
        })
        board.push(mv)
        played.append(bestmove)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--nodes", type=int, default=10000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--book_plies", type=int, default=8,
                    help="number of plies taken from the pool line (0 = startpos only)")
    ap.add_argument("--max_plies", type=int, default=600)
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = load_pool(args.pool)
    eng = Engine(args.engine, args.hash)
    stats = {"games": 0, "kept": 0, "discarded": 0, "reasons": {}, "plies_total": 0,
             "max_len": 0}
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as out:
        for gi in range(args.games):
            _fen_end, line = pool[rng.randrange(len(pool))]
            # Pool lines are move lists from the standard start position
            # (checked: all 20000 lines are legal and reach their FEN).
            book = line[: args.book_plies] if args.book_plies > 0 else []
            start_fen = chess.STARTING_FEN
            g = play_one(eng, start_fen, book, args.nodes, args.max_plies)
            g["game_id"] = f"s{args.seed}_g{gi}"
            g["start_fen"] = start_fen
            g["nodes"] = args.nodes
            out.write(json.dumps(g) + "\n")
            stats["games"] += 1
            stats["plies_total"] += len(g["plies"])
            stats["max_len"] = max(stats["max_len"], len(g["plies"]))
            if g["discarded"]:
                stats["discarded"] += 1
            else:
                stats["kept"] += 1
            stats["reasons"][g["reason"]] = stats["reasons"].get(g["reason"], 0) + 1
    eng.quit()
    stats["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(stats, indent=1), file=sys.stderr)


if __name__ == "__main__":
    main()
