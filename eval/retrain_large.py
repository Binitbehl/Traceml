import os, sys, json, math, statistics as st
import torch
import matplotlib.pyplot as plt
from common import get_datasets, load_model
from attribution.knn import KNNAttribution
from eval.retrain import K, train_model, test_losses
from eval.retrain_extra import cosine_tracin

# ---------- settings ----------
EVAL_IDX = list(range(20))          # test images to analyze (must be < 100)
SEEDS = [0, 1]                      # raise to [0, 1, 2] for less noise (slower)
METHODS = ["random", "knn_label", "tracin", "tracin_cos"]
CACHE = "results/retrain_large_cache.json"
OLD_CACHE = "results/retrain_cache.json"
COS = "results/tracin_cos_large.pt"


def se(v):
    return st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")


def summarize(cache):
    done = [p for p, idx in enumerate(EVAL_IDX)
            if all(f"{m}|{idx}|{s}" in cache for m in METHODS for s in SEEDS)
            and all(f"base|{s}" in cache for s in SEEDS)]
    if len(done) < 2:
        print("Not enough finished images yet.")
        return
    delta = {m: [] for m in METHODS}
    for p in done:
        idx = EVAL_IDX[p]
        for m in METHODS:
            vals = [cache[f"{m}|{idx}|{s}"] - cache[f"base|{s}"][p] for s in SEEDS]
            delta[m].append(st.mean(vals))

    n = len(done)
    print(f"\nImages finished: {n} of {len(EVAL_IDX)}   seeds: {SEEDS}   top-{K} removed")
    print("\nIncrease in test loss (mean ± standard error across images):")
    for m in METHODS:
        print(f"  {m:11s} {st.mean(delta[m]):+.2f} ± {se(delta[m]):.2f}")

    print("\nPaired differences (A minus B), with the number of images where A > B:")
    pairs = [("tracin_cos", "knn_label"), ("tracin_cos", "tracin"),
             ("tracin", "knn_label"), ("knn_label", "random"),
             ("tracin_cos", "random")]
    for a, b in pairs:
        d = [x - y for x, y in zip(delta[a], delta[b])]
        wins = sum(1 for x in d if x > 0)
        print(f"  {a:11s} - {b:10s} {st.mean(d):+.2f} ± {se(d):.2f}   ({wins}/{n})")

    print("\nPer image (mean over seeds):")
    print(f"{'test':>6} " + " ".join(f"{m:>11}" for m in METHODS))
    for i, p in enumerate(done):
        print(f"{EVAL_IDX[p]:>6} " + " ".join(f"{delta[m][i]:+11.2f}" for m in METHODS))

    plt.bar(METHODS, [st.mean(delta[m]) for m in METHODS],
            yerr=[se(delta[m]) for m in METHODS], capsize=4)
    plt.ylabel("increase in test loss")
    plt.title(f"Removing top-{K} training examples ({n} test images)")
    plt.savefig("results/retrain_large_summary.png", dpi=120)
    json.dump({"images": [EVAL_IDX[p] for p in done], "delta": delta},
              open("results/retrain_large_results.json", "w"), indent=2)
    print("\nsaved results/retrain_large_summary.png")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    if os.path.exists(OLD_CACHE):                 # reuse runs for test images 0-4
        for key, val in json.load(open(OLD_CACHE)).items():
            if key.split("|")[0] in METHODS and key not in cache:
                cache[key] = val

    if "--summary" in sys.argv:                   # just print results so far
        summarize(cache)
        raise SystemExit

    def save():
        json.dump(cache, open(CACHE, "w"))

    train_set, test_set = get_datasets()
    N = len(train_set)
    train_x = torch.stack([train_set[i][0] for i in range(N)])
    train_y = torch.tensor([train_set[i][1] for i in range(N)])
    xs = torch.stack([test_set[i][0] for i in EVAL_IDX])
    ys = torch.tensor([test_set[i][1] for i in EVAL_IDX])
    all_idx = torch.arange(N)

    # attribution scores
    raw = torch.load("results/tracin_scores.pt")[torch.tensor(EVAL_IDX)]   # [T, N]
    if os.path.exists(COS) and torch.load(COS).shape[0] == len(EVAL_IDX):
        cos = torch.load(COS)
    else:
        cos = cosine_tracin(xs, ys, train_x, train_y)
        torch.save(cos, COS)
    knn = KNNAttribution(load_model(), train_set)

    def removal_set(method, pos, seed):
        if method == "tracin":
            return raw[pos].topk(K).indices
        if method == "tracin_cos":
            return cos[pos].topk(K).indices
        if method == "knn_label":
            s = knn.score(xs[pos]).clone()
            s[knn.train_labels != ys[pos]] = -2
            return s.topk(K).indices
        gen = torch.Generator().manual_seed(seed * 1000 + EVAL_IDX[pos])
        return torch.randperm(N, generator=gen)[:K]

    # baselines: full training set, one model per seed, losses for all test images
    for seed in SEEDS:
        if f"base|{seed}" not in cache:
            cache[f"base|{seed}"] = test_losses(
                train_model(all_idx, seed, train_x, train_y), xs, ys)
            save()
            print(f"baseline seed {seed} done", flush=True)

    # removal runs, one test image at a time (so partial results are usable)
    for pos, idx in enumerate(EVAL_IDX):
        for method in METHODS:
            removed = removal_set(method, pos, 0)
            for seed in SEEDS:
                key = f"{method}|{idx}|{seed}"
                if key in cache:
                    continue
                if method == "random":
                    removed = removal_set(method, pos, seed)
                keep = torch.ones(N, dtype=torch.bool)
                keep[removed] = False
                model = train_model(all_idx[keep], seed, train_x, train_y)
                cache[key] = test_losses(model, xs[pos:pos + 1], ys[pos:pos + 1])[0]
                save()
                print(f"{method} test#{idx} seed {seed} done", flush=True)

    summarize(cache)