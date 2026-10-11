import glob, math, re
W = D = L = 0
parts = []
for path in sorted(glob.glob("results/match_*.txt")):
    with open(path, encoding="utf-8") as f:
        first = f.readline()
    m = re.search(r"cand_W=(\d+) D=(\d+) cand_L=(\d+)", first)
    if not m:
        parts.append(f"{path}: no FINAL line")
        continue
    w, d, l = map(int, m.groups())
    W += w; D += d; L += l
    parts.append(f"{path}: W={w} D={d} L={l}")
n = W + D + L
score = (W + 0.5 * D) / max(1, n)
# approximate z for score difference vs 0.5 (draw-aware variance)
var = (W * (1 - score) ** 2 + L * score ** 2 + D * (0.5 - score) ** 2) / max(1, n)
z = (score - 0.5) / math.sqrt(var / max(1, n)) if var > 0 else 0.0
elo = -400 * math.log10(1 / score - 1) if 0 < score < 1 else float("nan")
lines = [
    "PATCH MATCH SUMMARY (candidate vs frozen base, fixed nodes, cap 200)",
    f"games={n} candidate_W={W} D={D} candidate_L={L}",
    f"candidate_score={score:.4f}",
    f"approx_elo_diff={elo:.1f}",
    f"approx_z={z:.2f} (accept needs clearly positive z, e.g. >= 2)",
    "",
] + parts
open("summary.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
