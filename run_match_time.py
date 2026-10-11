#!/usr/bin/env python3
"""Paired TIME-control match: candidate vs frozen base, same runner.
Usage: run_match_time.py BASE CAND PAIRS BASE_MS INC_MS MAX_PLIES SEED OUT
Clocks: each side starts with BASE_MS, gets INC_MS after each move.
Flag fall (time used > remaining clock) = loss for that side.
"""
import importlib.util, random, sys, time
from pathlib import Path
import chess

HERE = Path(__file__).resolve().parent / "play_kb8_wdl_smoke.py"
spec = importlib.util.spec_from_file_location("pm", str(HERE))
pm = importlib.util.module_from_spec(spec); spec.loader.exec_module(pm)

base_path, cand_path = Path(sys.argv[1]), Path(sys.argv[2])
pairs, base_ms, inc_ms = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
max_plies, seed = int(sys.argv[6]), int(sys.argv[7])
out_path = Path(sys.argv[8])

base = pm.UCIEngine(base_path, 32, Path("/tmp"))
cand = pm.UCIEngine(cand_path, 32, Path("/tmp"))
rng = random.Random(seed)
W = D = L = 0
lines = []
t0 = time.time()
total_plies = 0


def choose_timed(eng, moves, board, wt, bt):
    move_text = " ".join(moves)
    eng.send("position startpos moves" + (" " + move_text if move_text else ""))
    eng.send(f"go wtime {wt} btime {bt} winc {inc_ms} binc {inc_ms}")
    start = time.perf_counter()
    while True:
        line = eng.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{eng.name} exited during search")
        if line.startswith("bestmove "):
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            fields = line.split()
            if len(fields) < 2 or fields[1] == "0000":
                raise RuntimeError(f"{eng.name} returned no move for {board.fen()}")
            move = chess.Move.from_uci(fields[1])
            if move not in board.legal_moves:
                raise RuntimeError(f"{eng.name} illegal move {fields[1]} for {board.fen()}")
            return move, elapsed_ms


def play_timed(opening, cand_white):
    board = chess.Board()
    moves = []
    for uci in opening:
        board.push(chess.Move.from_uci(uci)); moves.append(uci)
    base.reset(); cand.reset()
    clock = {chess.WHITE: base_ms, chess.BLACK: base_ms}
    flag = None
    while not board.is_game_over(claim_draw=True) and len(moves) < max_plies:
        side = board.turn
        candidate_to_move = (side == chess.WHITE) == cand_white
        player = cand if candidate_to_move else base
        move, elapsed = choose_timed(player, moves, board, clock[chess.WHITE], clock[chess.BLACK])
        if elapsed >= clock[side]:
            flag = side
            break
        clock[side] = clock[side] - elapsed + inc_ms
        moves.append(move.uci()); board.push(move)
    if flag is not None:
        result = "0-1" if flag == chess.WHITE else "1-0"
        capped = False
    else:
        capped = not board.is_game_over(claim_draw=True)
        result = "1/2-1/2" if capped else board.result(claim_draw=True)
    return result, len(moves), capped, flag


def write_out(final):
    n_games = W + D + L
    tag = "FINAL " if final else ""
    with open(out_path, "w") as f:
        f.write(f"{tag}games={n_games} cand_W={W} D={D} cand_L={L} "
                f"score={(W + 0.5 * D) / max(1, n_games):.4f} elapsed={time.time() - t0:.0f}s "
                f"plies={total_plies}\n")
        f.write("\n".join(lines) + "\n")


for i in range(pairs):
    b = chess.Board(); ops = []
    n = rng.randint(8, 12)
    while len(ops) < n:
        mvs = list(b.legal_moves)
        if not mvs or b.is_game_over(claim_draw=True):
            b = chess.Board(); ops = []; continue
        mv = rng.choice(mvs); ops.append(mv.uci()); b.push(mv)
    for cand_white in (True, False):
        res, plies, capped, flag = play_timed(ops, cand_white)
        total_plies += plies
        if res == "1/2-1/2":
            D += 1; cr = "D"
        else:
            won = (res == "1-0") == cand_white
            if won:
                W += 1; cr = "W"
            else:
                L += 1; cr = "L"
        lines.append(f"pair={i:03d} cand={'W' if cand_white else 'B'} result={res} "
                     f"plies={plies} capped={capped} flag={'none' if flag is None else ('w' if flag == chess.WHITE else 'b')} cand_res={cr}")
    if (i + 1) % 10 == 0:
        write_out(False)
write_out(True)
base.close(); cand.close()
print("done", W, D, L, f"elapsed={time.time() - t0:.0f}s", flush=True)
