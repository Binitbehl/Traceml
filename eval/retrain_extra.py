import os, json, statistics as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from common import get_datasets, load_model
from attribution.tracin import per_sample_grads
from attribution.knn import KNNAttribution
from eval.retrain import EVAL_IDX, K, SEEDS, CACHE, train_model, test_losses

SCORES = "results/tracin_cos_scores.pt"
EPOCHS_USED = (3, 7, 11, 15, 19, 24)     # same checkpoints as attribution/tracin.py


def cosine_tracin(xs, ys, train_x, train_y):
    scores = torch.zeros(len(xs), len(train_x))
    for ep in EPOCHS_USED:
        m = load_model(f"ckpts/ep{ep}.pt")
        gt = F.normalize(per_sample_grads(m, xs, ys), dim=1)
        for s in range(0, len(train_x), 100):
            g = F.normalize(
                per_sample_grads(m, train_x[s:s + 100], train_y[s:s + 100]), dim=1)
            scores[:, s:s + 100] += gt @ g.T
        print(f"cosine tracin: checkpoint ep{ep} done", flush=True)
    return scores


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    cache = json.load(open(CACHE))          # keeps the earlier runs
    train_set, test_set = get_datasets()
    N = len(train_set)
    train_x = torch.stack([train_set[i][0] for i in range(N)])
    train_y = torch.tensor([train_set[i][1] for i in range(N)])
    xs = torch.stack([test_set[i][0] for i in EVAL_IDX])
    ys = torch.tensor([test_set[i][1] for i in EVAL_IDX])

    if os.path.exists(SCORES):
        cos_scores = torch.load(SCORES)
    else:
        cos_scores = cosine_tracin(xs, ys, train_x, train_y)
        torch.save(cos_scores, SCORES)

    knn = KNNAttribution(load_model(), train_set)
    all_idx = torch.arange(N)

    def removal_set(method, pos):
        if method == "tracin_cos":
            return cos_scores[pos].topk(K).indices
        s = knn.score(xs[pos]).clone()               # label-aware kNN:
        s[knn.train_labels != ys[pos]] = -2          # only the true class is eligible
        return s.topk(K).indices

    for method in ["knn_label", "tracin_cos"]:
        for pos in range(len(EVAL_IDX)):
            removed = removal_set(method, pos)
            for seed in SEEDS:
                key = f"{method}|{pos}|{seed}"
                if key in cache:
                    continue
                keep = torch.ones(N, dtype=torch.bool)
                keep[removed] = False
                model = train_model(all_idx[keep], seed, train_x, train_y)
                cache[key] = test_losses(model, xs[pos:pos + 1], ys[pos:pos + 1])[0]
                json.dump(cache, open(CACHE, "w"))
                print(f"{method} test#{EVAL_IDX[pos]} seed {seed} done", flush=True)

    methods = ["random", "knn", "knn_label", "tracin", "tracin_cos"]
    print(f"\nIncrease in test loss after removing the top-{K} examples (mean over seeds):")
    print(f"{'test':>6} " + " ".join(f"{m:>11}" for m in methods))
    per_method = {m: [] for m in methods}
    for pos, idx in enumerate(EVAL_IDX):
        row = []
        for m in methods:
            d = [cache[f"{m}|{pos}|{s}"] - cache[f"base|{s}"][pos] for s in SEEDS]
            per_method[m].append(st.mean(d))
            row.append(f"{st.mean(d):+11.2f}")
        print(f"{idx:>6} " + " ".join(row))
    means = {m: st.mean(v) for m, v in per_method.items()}
    print(f"{'mean':>6} " + " ".join(f"{means[m]:+11.2f}" for m in methods))

    plt.bar(methods, [means[m] for m in methods])
    plt.ylabel("increase in test loss")
    plt.title(f"Effect of removing top-{K} training examples")
    plt.savefig("results/retrain_summary_extended.png", dpi=120)
    print("saved results/retrain_summary_extended.png")