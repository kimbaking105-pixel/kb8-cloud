#!/usr/bin/env python3
"""Texel-style tuning for HCE v2 (MSE on game outcomes, logistic link).

Pipeline
  1. read positions TSV(.gz) from extract.py (white-relative result, stm cp)
  2. ask the engine for features: HCE2_DUMP=1 ./eng2  (one FEN per line in,
     one 'D phase scale matMG matEG fMG[167] fEG[167]' line out, white-relative)
  3. split by game_id (5% validation)
  4. fit K on the search-score labels (cp from the engine, not Stockfish)
  5. Adam on outcome labels: s = ((F_mg.w_mg + M_mg)*ph + (F_eg.w_eg + M_eg)*(24-ph))/24 * scale/128
                             p = sigmoid(K*s/400),  loss = MSE(p, result_white) + l2*|w|^2
  6. write integer weights (idx mg eg) to hce2_weights.txt, report validation MSE

Notes
  - Material is fixed (EV_MG/EV_EG) and enters as M_mg/M_eg, not as a weight.
  - Weights with few supporting positions are reported; they are noisy.
"""
import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import zlib

import numpy as np

N_FEAT = 167


def read_positions(path, max_pos=0):
    rows = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        header = f.readline()  # noqa: F841
        for line in f:
            p = line.rstrip("\n").split("\t")
            rows.append((p[0], p[2], float(p[3]), p[5]))
            if max_pos and len(rows) >= max_pos:
                break
    return rows


def dump_features(engine, fens, chunk=50000):
    env = dict(os.environ, HCE2_DUMP="1")
    out_lines = []
    for i in range(0, len(fens), chunk):
        part = fens[i:i + chunk]
        res = subprocess.run([engine], input="\n".join(part) + "\n", capture_output=True,
                             text=True, env=env, check=True)
        out_lines.extend(res.stdout.strip().split("\n"))
    if len(out_lines) != len(fens):
        raise RuntimeError(f"engine returned {len(out_lines)} lines for {len(fens)} fens")
    phase = np.zeros(len(fens), np.float32)
    scale = np.zeros(len(fens), np.float32)
    mm = np.zeros(len(fens), np.float32)
    me = np.zeros(len(fens), np.float32)
    fm = np.zeros((len(fens), N_FEAT), np.float32)
    fe = np.zeros((len(fens), N_FEAT), np.float32)
    bad = []
    for i, line in enumerate(out_lines):
        if line.startswith("ERR"):
            bad.append(i)
            continue
        p = line.split()
        phase[i], scale[i], mm[i], me[i] = int(p[1]), int(p[2]), int(p[3]), int(p[4])
        vals = np.array(p[5:5 + 2 * N_FEAT], dtype=np.float32)
        fm[i] = vals[:N_FEAT]
        fe[i] = vals[N_FEAT:]
    keep = np.ones(len(fens), bool)
    keep[bad] = False
    return keep, phase, scale, mm, me, fm, fe


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_K(cp, y):
    """K for sigmoid(K*cp/400) on search scores: grid search minimising MSE."""
    best = (1e9, 1.0)
    for K in np.arange(0.3, 3.0001, 0.01):
        p = sigmoid(K * cp / 400.0)
        mse = float(np.mean((p - y) ** 2))
        if mse < best[0]:
            best = (mse, float(K))
    return best


def predict(w_mg, w_eg, phase, scale, mm, me, fm, fe):
    a = fm @ w_mg + mm
    b = fe @ w_eg + me
    return (a * phase + b * (24.0 - phase)) / 24.0 * (scale / 128.0)


def train(tr, va, K, epochs, lr, l2, log):
    phase, scale, mm, me, fm, fe, y = tr
    n_w = np.zeros(N_FEAT, np.float32)
    w_mg = n_w.copy()
    w_eg = n_w.copy()
    m_mg = np.zeros(N_FEAT, np.float32)
    v_mg = np.zeros(N_FEAT, np.float32)
    m_eg = np.zeros(N_FEAT, np.float32)
    v_eg = np.zeros(N_FEAT, np.float32)
    b1, b2, eps = 0.9, 0.999, 1e-8
    sc = scale / 128.0
    ph_scale = phase / 24.0 * sc
    eg_scale = (24.0 - phase) / 24.0 * sc
    nlen = len(y)
    for ep in range(1, epochs + 1):
        a = fm @ w_mg + mm
        b = fe @ w_eg + me
        s = (a * phase + b * (24.0 - phase)) / 24.0 * sc
        p = sigmoid(K * s / 400.0)
        err = (p - y)
        loss = float(np.mean(err ** 2))
        # dL/ds
        dLds = 2.0 * err * p * (1 - p) * K / 400.0 / nlen
        g_mg = fm.T @ (dLds * ph_scale) + 2 * l2 * w_mg
        g_eg = fe.T @ (dLds * eg_scale) + 2 * l2 * w_eg
        m_mg = b1 * m_mg + (1 - b1) * g_mg
        v_mg = b2 * v_mg + (1 - b2) * g_mg ** 2
        m_eg = b1 * m_eg + (1 - b1) * g_eg
        v_eg = b2 * v_eg + (1 - b2) * g_eg ** 2
        step = lr * math.sqrt(1 - b2 ** ep) / (1 - b1 ** ep)
        w_mg -= step * m_mg / (np.sqrt(v_mg) + eps)
        w_eg -= step * m_eg / (np.sqrt(v_eg) + eps)
        if ep % 50 == 0 or ep == 1 or ep == epochs:
            vloss = validation(va, w_mg, w_eg, K)
            log(f"epoch {ep:4d} train_mse {loss:.6f} val_mse {vloss:.6f}")
    return w_mg, w_eg


def validation(va, w_mg, w_eg, K, round_int=False):
    phase, scale, mm, me, fm, fe, y = va
    wm, we = (np.round(w_mg), np.round(w_eg)) if round_int else (w_mg, w_eg)
    s = predict(wm, we, phase, scale, mm, me, fm, fe)
    p = sigmoid(K * s / 400.0)
    return float(np.mean((p - y) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="eng2 built from build_2_dev.cpp")
    ap.add_argument("--positions", required=True, help="comma-separated positions .tsv.gz files")
    ap.add_argument("--out", default="hce2_weights.txt")
    ap.add_argument("--report", default="tune_report.json")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--max_pos", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.05)
    args = ap.parse_args()

    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    rows = []
    for path in args.positions.split(","):
        rows.extend(read_positions(path, 0))
    if args.max_pos:
        rows = rows[: args.max_pos]
    log(f"positions read: {len(rows)}")
    fens = [r[1] for r in rows]
    keep, phase, scale, mm, me, fm, fe = dump_features(args.engine, fens)
    y = np.array([r[2] for r in rows], np.float32)
    cp_white = np.array([float(r[3]) * (1 if r[1].split()[1] == "w" else -1) for r in rows], np.float32)
    gid = [r[0] for r in rows]

    # split by game (stable hash), validation = val_frac of games
    is_val = np.array([(zlib.crc32(g.encode()) % 10000) < int(args.val_frac * 10000) for g in gid])
    idx_tr = np.where(keep & ~is_val)[0]
    idx_va = np.where(keep & is_val)[0]
    log(f"kept {int(keep.sum())}  train {len(idx_tr)}  val {len(idx_va)}")

    mse_k, K = fit_K(cp_white[idx_tr], y[idx_tr])
    log(f"K fitted on search scores: {K:.2f} (mse {mse_k:.4f})")

    def pack(ix):
        return (phase[ix], scale[ix], mm[ix], me[ix], fm[ix], fe[ix], y[ix])

    tr = pack(idx_tr)
    va = pack(idx_va)
    base_va = float(np.mean((sigmoid(0.0) - va[6]) ** 2))
    log(f"baseline val mse (predict 0.5): {base_va:.6f}")

    w_mg, w_eg = train(tr, va, K, args.epochs, args.lr, args.l2, log)
    val_float = validation(va, w_mg, w_eg, K)
    val_int = validation(va, w_mg, w_eg, K, round_int=True)
    log(f"val mse float {val_float:.6f}  rounded {val_int:.6f}")

    support = (fm[idx_tr] != 0).sum(0) + (fe[idx_tr] != 0).sum(0)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(N_FEAT):
            f.write(f"{i} {int(round(w_mg[i]))} {int(round(w_eg[i]))}\n")
    report = {"positions": len(rows), "kept": int(keep.sum()), "train": int(len(idx_tr)),
              "val": int(len(idx_va)), "K": K, "K_mse_search": mse_k,
              "val_mse_baseline_half": base_va, "val_mse_float": val_float,
              "val_mse_rounded": val_int, "epochs": args.epochs, "lr": args.lr, "l2": args.l2,
              "weak_features": [i for i in range(N_FEAT) if support[i] < 50],
              "weights": {i: [int(round(w_mg[i])), int(round(w_eg[i]))] for i in range(N_FEAT)}}
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    log("wrote " + args.out)


if __name__ == "__main__":
    main()
