import os, json, time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from common import get_datasets, load_model, CNN
from attribution.knn import KNNAttribution

# ----- settings (lower these if it is too slow) -----
EVAL_IDX = [0, 1, 2, 3, 4]   # test images to analyze
K = 100                      # training examples removed
SEEDS = [0, 1, 2]            # retraining repeats (averaging reduces noise)
EPOCHS = 12
CACHE = "results/retrain_cache.json"


def lr_at(epoch):
    return 0.05 if epoch < 8 else 0.02 if epoch < 10 else 0.01


def train_model(keep_idx, seed, train_x, train_y):
    torch.manual_seed(seed)
    model = CNN()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.0)
    keep_idx = torch.as_tensor(keep_idx)
    for epoch in range(EPOCHS):
        for g in opt.param_groups:
            g["lr"] = lr_at(epoch)
        model.train()
        perm = keep_idx[torch.randperm(len(keep_idx))]
        for s in range(0, len(perm), 64):
            b = perm[s:s + 64]
            opt.zero_grad()
            F.cross_entropy(model(train_x[b]), train_y[b]).backward()
            opt.step()
    model.eval()
    return model


def test_losses(model, xs, ys):
    with torch.no_grad():
        return F.cross_entropy(model(xs), ys, reduction="none").tolist()


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

    def save_cache():
        json.dump(cache, open(CACHE, "w"))

    train_set, test_set = get_datasets()
    N = len(train_set)
    train_x = torch.stack([train_set[i][0] for i in range(N)])
    train_y = torch.tensor([train_set[i][1] for i in range(N)])
    xs = torch.stack([test_set[i][0] for i in EVAL_IDX])
    ys = torch.tensor([test_set[i][1] for i in EVAL_IDX])

    # attribution scores
    tracin_scores = torch.load("results/tracin_scores.pt")      # [100, N]
    knn = KNNAttribution(load_model(), train_set)

    def removal_set(method, pos, seed):
        i = EVAL_IDX[pos]
        if method == "tracin":
            return tracin_scores[i].topk(K).indices
        if method == "knn":
            return knn.score(xs[pos]).topk(K).indices
        g = torch.Generator().manual_seed(seed * 1000 + i)
        return torch.randperm(N, generator=g)[:K]

    all_idx = torch.arange(N)

    # 1) baselines: full training set, one model per seed
    for seed in SEEDS:
        key = f"base|{seed}"
        if key not in cache:
            t = time.time()
            cache[key] = test_losses(train_model(all_idx, seed, train_x, train_y), xs, ys)
            save_cache()
            print(f"baseline seed {seed} done ({time.time() - t:.0f}s)", flush=True)

    # 2) removal runs
    methods = ["random", "knn", "tracin"]
    for method in methods:
        for pos in range(len(EVAL_IDX)):
            for seed in SEEDS:
                key = f"{method}|{pos}|{seed}"
                if key in cache:
                    continue
                removed = removal_set(method, pos, seed)
                keep = torch.ones(N, dtype=torch.bool)
                keep[removed] = False
                model = train_model(all_idx[keep], seed, train_x, train_y)
                cache[key] = test_losses(model, xs[pos:pos + 1], ys[pos:pos + 1])[0]
                save_cache()
                print(f"{method} test#{EVAL_IDX[pos]} seed {seed} done", flush=True)

    # 3) summary: increase in test loss after removal (higher = more influential)
    print(f"\nMean increase in test loss after removing top-{K} examples:")
    means = {}
    for method in methods:
        deltas = []
        for pos in range(len(EVAL_IDX)):
            for seed in SEEDS:
                deltas.append(cache[f"{method}|{pos}|{seed}"] - cache[f"base|{seed}"][pos])
        means[method] = sum(deltas) / len(deltas)
        print(f"  {method:7s} {means[method]:+.3f}")

    plt.bar(list(means), list(means.values()))
    plt.ylabel("increase in test loss")
    plt.title(f"Effect of removing top-{K} training examples")
    plt.savefig("results/retrain_summary.png", dpi=120)
    print("saved results/retrain_summary.png")