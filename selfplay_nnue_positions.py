#!/usr/bin/env python3
"""Generate labeled self-play shards for the Tiny NNUE experiments.

This tool intentionally uses the UCI engine as both player and fixed-search
teacher. It writes streaming JSONL gzip records, so a shard never needs to be
held in RAM. It does not modify the engine source.

The first implementation is deliberately conservative:
- one engine process per worker invocation;
- fixed-node or fixed-time UCI search;
- score captured from the final UCI info line;
- game result added after the game finishes;
- mate-score positions skipped by default;
- full FEN retained for later feature experiments.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import chess
except ImportError as exc:  # pragma: no cover - useful CI error
    raise SystemExit("python-chess is required: python -m pip install python-chess") from exc

OPENINGS = [
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["e2e4", "c7c5", "g1f3", "d7d6"],
    ["e2e4", "e7e6", "d2d4", "d7d5"],
    ["e2e4", "c7c6", "d2d4", "d7d5"],
    ["d2d4", "d7d5", "c2c4", "e7e6"],
    ["d2d4", "g8f6", "c2c4", "e7e6"],
    ["c2c4", "e7e5", "g1f3", "g8f6"],
    ["g1f3", "d7d5", "d2d4", "g8f6"],
    ["b1c3", "d7d5", "e2e4", "e7e6"],
    ["b2b3", "d7d5", "c1b2", "g8f6"],
    ["c2c4", "c7c5", "g1f3", "b8c6"],
    ["e2e4", "g8f6", "e4e5", "d7d5"],
]


def parse_int(value: str) -> int:
    return int(value, 0)


def build_start_position(rng: random.Random, extra_plies: int) -> chess.Board:
    board = chess.Board()
    for uci in rng.choice(OPENINGS):
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise RuntimeError(f"bad opening move {uci}")
        board.push(move)

    central = {chess.C4, chess.D4, chess.E4, chess.F4,
               chess.C5, chess.D5, chess.E5, chess.F5}
    for _ in range(max(0, extra_plies)):
        legal = list(board.legal_moves)
        quiet = [m for m in legal if not board.is_capture(m) and not board.gives_check(m)]
        candidates = quiet or legal

        def move_score(move: chess.Move) -> float:
            value = rng.random() * 3.0
            if move.to_square in central:
                value += 4.0
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                value += 2.0
            if board.is_castling(move):
                value += 3.0
            return value

        ranked = sorted(candidates, key=move_score, reverse=True)
        board.push(rng.choice(ranked[: min(8, len(ranked))]))
    return board


class Engine:
    def __init__(self, path: str, threads: int, hash_mb: int,
                 eval_file: str | None = None):
        exe = Path(path).resolve()
        net_parent = None
        net_name = None
        if eval_file is not None:
            net = Path(eval_file).resolve()
            # ASCII basename + process cwd: avoids ANSI-codepage problems with
            # the user's Cyrillic Desktop path (same trick as the match script).
            net_parent = str(net.parent)
            net_name = net.name
        self.proc = subprocess.Popen(
            [str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
            errors="replace", bufsize=1, cwd=net_parent,
        )
        self.send("uci")
        self.until("uciok")
        self.send(f"setoption name Threads value {threads}")
        self.send(f"setoption name Hash value {hash_mb}")
        if net_name is not None:
            self.send(f'setoption name EvalFile value "{net_name}"')
            self.until("KB8 network loaded")
            self.send("setoption name EvalMode value NNUE_WDL")
            self.until("EvalMode NNUE_WDL")
        self.ready()

    def send(self, line: str) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("engine stdin is closed")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def until(self, token: str) -> None:
        if self.proc.stdout is None:
            raise RuntimeError("engine stdout is closed")
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"engine exited while waiting for {token}")
            if token in line:
                return

    def ready(self) -> None:
        self.send("isready")
        self.until("readyok")

    @staticmethod
    def parse_score(fields: list[str]) -> tuple[int, int] | None:
        # UCI scores are relative to the side to move at the root position.
        for i in range(len(fields) - 2):
            if fields[i] != "score":
                continue
            kind = fields[i + 1]
            try:
                value = int(fields[i + 2])
            except ValueError:
                continue
            if kind == "cp":
                return max(-15000, min(15000, value)), 0
            if kind == "mate":
                return (15000 if value > 0 else -15000), value
        return None

    def search(self, board: chess.Board, nodes: Optional[int], movetime: Optional[int]) -> tuple[chess.Move, Optional[int], int]:
        self.send("position fen " + board.fen())
        if nodes is not None:
            self.send(f"go nodes {nodes}")
        else:
            self.send(f"go movetime {movetime}")

        if self.proc.stdout is None:
            raise RuntimeError("engine stdout is closed during search")
        best_uci: Optional[str] = None
        last_score: Optional[tuple[int, int]] = None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine exited during search")
            fields = line.strip().split()
            if fields and fields[0] == "info":
                parsed = self.parse_score(fields)
                if parsed is not None:
                    last_score = parsed
            if fields and fields[0] == "bestmove":
                best_uci = fields[1] if len(fields) > 1 else None
                break

        if not best_uci or best_uci == "0000":
            raise RuntimeError(f"engine returned no bestmove for {board.fen()}")
        move = chess.Move.from_uci(best_uci)
        if move not in board.legal_moves:
            raise RuntimeError(f"illegal bestmove {best_uci} for {board.fen()}")
        if last_score is None:
            # Some very small node budgets can stop before a completed
            # iteration. Keep the legal move so self-play can continue, but
            # let the caller skip this position instead of fabricating a label.
            return move, None, 0
        return move, last_score[0], last_score[1]

    def close(self) -> None:
        try:
            self.send("quit")
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def load_opening_book(path: Optional[str]) -> list[tuple[int, list[str]]]:
    if not path:
        return []
    entries: list[tuple[int, list[str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise RuntimeError(f"opening book line {line_no}: expected id, key, moves")
            book_id = int(parts[0])
            entries.append((book_id, parts[2].split()))
    if not entries:
        raise RuntimeError(f"opening book is empty: {path}")
    return entries


def build_book_position(moves: list[str]) -> chess.Board:
    board = chess.Board()
    for uci in moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise RuntimeError(f"illegal opening-book move {uci}")
        board.push(move)
    return board


def game_result(board: chess.Board) -> int:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0
    return 1 if outcome.winner == chess.WHITE else -1


def result_from_stm(result_white: int, stm: chess.Color) -> int:
    return result_white if stm == chess.WHITE else -result_white


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-id", default="local")
    ap.add_argument("--positions", type=int, required=True)
    ap.add_argument("--games", type=int, default=100000)
    budget = ap.add_mutually_exclusive_group(required=True)
    budget.add_argument("--nodes", type=int)
    budget.add_argument("--movetime", type=int)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", dest="hash_mb", type=int, default=32)
    ap.add_argument("--seed", type=parse_int, default=0x20261008)
    ap.add_argument("--opening-random-plies", type=int, default=6)
    ap.add_argument("--sample-every", type=int, default=1)
    ap.add_argument("--min-ply", type=int, default=10)
    ap.add_argument("--max-plies", type=int, default=180)
    ap.add_argument("--keep-mates", action="store_true")
    ap.add_argument("--opening-book", default=None)
    ap.add_argument("--eval-file", default=None,
                    help="path to kb8_twohead_v2.nnue; enables NNUE_WDL self-play")
    ap.add_argument("--book-offset", type=int, default=0,
                    help="disjoint book slice start per parallel worker")
    ap.add_argument("--family-offset", type=int, default=0,
                    help="global family_id offset per parallel worker (split safety)")
    ap.add_argument("--opening-plies-min", type=int, default=8,
                    help="random heuristic opening: min plies (no book mode)")
    ap.add_argument("--opening-plies-max", type=int, default=12,
                    help="random heuristic opening: max plies (no book mode)")
    ap.add_argument("--early-ply-jitter", type=int, default=0,
                    help="first N plies after the opening use random node "
                         "budgets in [nodes/2, nodes] (diversity, 0=off)")
    ap.add_argument("--discard-before-ply", type=int, default=0,
                    help="drop games that end before this absolute ply "
                         "(pathology guard; 0=off, by length only — not by result)")
    args = ap.parse_args()

    if args.positions <= 0:
        ap.error("--positions must be positive")
    if args.sample_every <= 0:
        ap.error("--sample-every must be positive")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"labeled_shard_{args.shard_id}.jsonl.gz"
    manifest_path = out_dir / f"manifest_{args.shard_id}.json"
    rng = random.Random(args.seed)
    opening_book = load_opening_book(args.opening_book)
    engine = Engine(args.engine, args.threads, args.hash_mb,
                    eval_file=args.eval_file)
    started = time.time()
    written = 0
    games_done = 0
    white_stm = 0
    black_stm = 0
    mate_skipped = 0
    no_score = 0
    short_games_discarded = 0
    duplicates_skipped = 0
    seen_position_keys: set[str] = set()

    try:
        with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=6) as out:
            for game_index in range(args.games):
                if written >= args.positions:
                    break
                games_done += 1
                engine.send("ucinewgame")
                engine.ready()
                if opening_book:
                    book_id, book_moves = opening_book[(game_index + args.book_offset) % len(opening_book)]
                    board = build_book_position(book_moves)
                    family_id = int(book_id) + int(args.family_offset)
                else:
                    opening_plies = rng.randint(args.opening_plies_min, args.opening_plies_max)
                    board = build_start_position(rng, opening_plies)
                    family_id = int(game_index) + int(args.family_offset)
                game_records: list[dict] = []
                sample_offset = rng.randrange(args.sample_every)
                opening_end = board.ply()

                for ply in range(board.ply(), args.max_plies):
                    if board.is_game_over(claim_draw=True):
                        break
                    stm = board.turn
                    fen = board.fen()
                    use_nodes = args.nodes
                    if args.early_ply_jitter > 0 and (ply - opening_end) < args.early_ply_jitter:
                        use_nodes = rng.randint(max(1, args.nodes // 2), args.nodes)
                    move, score_cp, score_mate = engine.search(board, use_nodes, args.movetime)
                    if score_cp is None:
                        no_score += 1
                    if score_cp is not None and ply >= args.min_ply and (ply - args.min_ply - sample_offset) % args.sample_every == 0:
                        if score_mate and not args.keep_mates:
                            mate_skipped += 1
                        else:
                            game_records.append({
                                "fen": fen,
                                "score_cp": int(score_cp),
                                "score_mate": int(score_mate),
                                "stm": "w" if stm == chess.WHITE else "b",
                                "ply": int(ply),
                                "game_id": int(game_index),
                                "family_id": family_id,
                                "seed": int(args.seed),
                            })
                    board.push(move)

                result_white = game_result(board)
                if args.discard_before_ply > 0 and board.ply() < args.discard_before_ply:
                    short_games_discarded += 1
                    game_records.clear()
                for record in game_records:
                    if written >= args.positions:
                        break
                    position_key = " ".join(record["fen"].split()[:4])
                    digest = hashlib.blake2b(position_key.encode("utf-8"), digest_size=8).hexdigest()
                    if digest in seen_position_keys:
                        duplicates_skipped += 1
                        continue
                    seen_position_keys.add(digest)
                    record["position_key"] = digest
                    record["result_stm"] = result_from_stm(result_white, record["stm"] == "w")
                    out.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
                    written += 1
                    if record["stm"] == "w":
                        white_stm += 1
                    else:
                        black_stm += 1

                if games_done % 10 == 0:
                    elapsed = max(0.001, time.time() - started)
                    print(
                        f"games={games_done} records={written} "
                        f"white_stm={white_stm} black_stm={black_stm} "
                        f"records_per_sec={written / elapsed:.1f}",
                        flush=True,
                    )
            out.flush()
    finally:
        engine.close()

    elapsed = max(0.001, time.time() - started)
    opening_book_sha = sha256_file(Path(args.opening_book)) if args.opening_book else None
    manifest = {
        "schema": "tiny-nnue-labeled-jsonl-v1",
        "shard_id": args.shard_id,
        "engine": str(Path(args.engine).name),
        "positions_target": args.positions,
        "records": written,
        "games": games_done,
        "nodes": args.nodes,
        "movetime_ms": args.movetime,
        "threads": args.threads,
        "hash_mb": args.hash_mb,
        "seed": args.seed,
        "opening_random_plies": args.opening_random_plies,
        "opening_plies_min": args.opening_plies_min,
        "opening_plies_max": args.opening_plies_max,
        "early_ply_jitter": args.early_ply_jitter,
        "opening_book": Path(args.opening_book).name if args.opening_book else None,
        "opening_book_entries": len(opening_book),
        "opening_book_sha256": opening_book_sha,
        "sample_every": args.sample_every,
        "min_ply": args.min_ply,
        "max_plies": args.max_plies,
        "white_stm": white_stm,
        "black_stm": black_stm,
        "mate_skipped": mate_skipped,
        "no_score": no_score,
        "discard_before_ply": args.discard_before_ply,
        "short_games_discarded": short_games_discarded,
        "duplicates_skipped": duplicates_skipped,
        "unique_position_keys": len(seen_position_keys),
        "elapsed_sec": round(elapsed, 3),
        "records_per_sec": round(written / elapsed, 3),
        "sha256_data": sha256_file(out_path),
        "data_file": out_path.name,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
