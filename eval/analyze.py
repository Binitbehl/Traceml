import json, statistics as st

cache = json.load(open("results/retrain_cache.json"))
EVAL_IDX, SEEDS = [0, 1, 2, 3, 4], [0, 1, 2]
methods = ["random", "knn", "tracin"]

print(f"{'test':>6} " + " ".join(f"{m:>16}" for m in methods))
for pos, idx in enumerate(EVAL_IDX):
    row = []
    for m in methods:
        d = [cache[f"{m}|{pos}|{s}"] - cache[f"base|{s}"][pos] for s in SEEDS]
        row.append(f"{st.mean(d):+.2f} (sd {st.pstdev(d):.2f})")
    print(f"{idx:>6} " + " ".join(f"{r:>16}" for r in row))