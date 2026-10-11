# Speed check: both engines search the same 20 positions to a fixed depth.
# Rounds alternate engine order. Reports total time and nodes per engine.
import subprocess, sys, time, statistics, re

DEPTH = 9
ROUNDS = 3
HASH = 32

def run_engine(path, fens):
    p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    def send(s): p.stdin.write(s + "\n"); p.stdin.flush()
    def wait_for(prefix):
        lines = []
        while True:
            line = p.stdout.readline()
            if not line: raise RuntimeError("engine closed")
            lines.append(line.strip())
            if line.startswith(prefix): return lines
    send("uci"); wait_for("uciok")
    send(f"setoption name Hash value {HASH}")
    send("isready"); wait_for("readyok")
    total_t, total_n = 0.0, 0
    for fen in fens:
        send("ucinewgame"); send("isready"); wait_for("readyok")
        send(f"position fen {fen}")
        t0 = time.perf_counter()
        send(f"go depth {DEPTH}")
        lines = wait_for("bestmove")
        total_t += time.perf_counter() - t0
        nodes = [int(m.group(1)) for l in lines for m in [re.search(r"nodes (\d+)", l)] if m]
        total_n += nodes[-1] if nodes else 0
    send("quit"); p.wait(timeout=10)
    return total_t, total_n

def main():
    base, cand, out = sys.argv[1], sys.argv[2], sys.argv[3]
    fens = [l.strip() for l in open("positions.txt") if l.strip()]
    bt, bn, ct, cn = [], [], [], []
    for r in range(ROUNDS):
        order = [(base, "base"), (cand, "cand")] if r % 2 == 0 else [(cand, "cand"), (base, "base")]
        for path, name in order:
            t, n = run_engine(path, fens)
            if name == "base": bt.append(t); bn.append(n)
            else: ct.append(t); cn.append(n)
    bmed, cmed = statistics.median(bt), statistics.median(ct)
    ratio = cmed / bmed
    lines = [
        f"SPEED CHECK depth={DEPTH} positions={len(fens)} rounds={ROUNDS} hash={HASH}",
        f"base_time_s={[round(x,2) for x in bt]} median={bmed:.2f}",
        f"cand_time_s={[round(x,2) for x in ct]} median={cmed:.2f}",
        f"base_nodes={bn[0]} cand_nodes={cn[0]} (nodes for the same depth)",
        f"time_ratio_cand_over_base={ratio:.3f}",
        f"speed_ok={'YES' if ratio <= 1.05 else 'NO'} (rule: cand not slower than +5%)",
    ]
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
