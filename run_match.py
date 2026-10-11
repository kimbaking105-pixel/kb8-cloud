#!/usr/bin/env python3
"""Paired fixed-node HCE match: candidate vs frozen baseline (no net).
Usage: run_match.py BASELINE CANDIDATE PAIRS NODES MAX_PLIES SEED OUT
"""
import importlib.util, random, sys, time
from pathlib import Path
import chess

HERE = Path(__file__).resolve().parent / "play_kb8_wdl_smoke.py"
spec = importlib.util.spec_from_file_location("pm", str(HERE))
pm = importlib.util.module_from_spec(spec); spec.loader.exec_module(pm)

base_path, cand_path = Path(sys.argv[1]), Path(sys.argv[2])
pairs, nodes, max_plies, seed = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
out_path = Path(sys.argv[7])

base = pm.UCIEngine(base_path, 32, Path("/tmp"))
cand = pm.UCIEngine(cand_path, 32, Path("/tmp"))
rng = random.Random(seed)
W = D = L = 0
lines = []
t0 = time.time()


def write_out(final):
    n_games = W + D + L
    tag = "FINAL " if final else ""
    with open(out_path, "w") as f:
        f.write(f"{tag}games={n_games} cand_W={W} D={D} cand_L={L} "
                f"score={(W + 0.5 * D) / max(1, n_games):.4f} elapsed={time.time() - t0:.0f}s\n")
        f.write("\n".join(lines) + "\n")


for i in range(pairs):
    b = chess.Board()
    ops = []
    n = rng.randint(8, 12)
    while len(ops) < n:
        mvs = list(b.legal_moves)
        if not mvs or b.is_game_over(claim_draw=True):
            b = chess.Board()
            ops = []
            continue
        mv = rng.choice(mvs)
        ops.append(mv.uci())
        b.push(mv)
    for cand_white in (True, False):
        res, plies, capped = pm.play_game(base, cand, ops, cand_white, nodes, max_plies)
        if res == "1/2-1/2":
            D += 1
            cr = "D"
        else:
            won = (res == "1-0") == cand_white
            if won:
                W += 1
                cr = "W"
            else:
                L += 1
                cr = "L"
        lines.append(f"pair={i:03d} cand={'W' if cand_white else 'B'} result={res} "
                     f"plies={plies} capped={capped} cand_res={cr}")
    if (i + 1) % 10 == 0:
        write_out(False)
write_out(True)
base.close()
cand.close()
print("done", W, D, L, flush=True)
