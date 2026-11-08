import os, json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.func import functional_call, vmap, grad
from common import get_datasets, load_model, CLASSES


def per_sample_grads(model, x, y):
    """One flattened gradient vector per example -> [B, num_params]."""
    params = {k: v.detach() for k, v in model.named_parameters()}

    def loss_fn(p, xi, yi):
        out = functional_call(model, p, (xi.unsqueeze(0),))
        return F.cross_entropy(out, yi.unsqueeze(0))

    g = vmap(grad(loss_fn), in_dims=(None, 0, 0))(params, x, y)
    return torch.cat([v.flatten(1) for v in g.values()], dim=1)


class TracInAttribution:
    """score[t, n] = sum over checkpoints of lr * <grad(test t), grad(train n)>"""

    def __init__(self, train_set, epochs=(3, 7, 11, 15, 19, 24), batch_size=100):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lrs = json.load(open("ckpts/lrs.json"))
        self.train_x = torch.stack([train_set[i][0] for i in range(len(train_set))])
        self.train_y = torch.tensor([train_set[i][1] for i in range(len(train_set))])

    def score_batch(self, xs, ys):
        T, N = len(xs), len(self.train_x)
        scores = torch.zeros(T, N)
        for ep in self.epochs:
            model = load_model(f"ckpts/ep{ep}.pt")
            lr = self.lrs[ep]
            g_test = per_sample_grads(model, xs, ys)                # [T, P]
            for s in range(0, N, self.batch_size):
                xb = self.train_x[s:s + self.batch_size]
                yb = self.train_y[s:s + self.batch_size]
                g_tr = per_sample_grads(model, xb, yb)              # [B, P]
                scores[:, s:s + len(xb)] += lr * (g_test @ g_tr.T)
            print(f"checkpoint ep{ep} done", flush=True)
        return scores


def to_img(x):
    return (x * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()


def show(i, scores, tracin, xs, ys, final_model, k=5):
    s = scores[i]
    helpful = s.topk(k).indices
    harmful = (-s).topk(k).indices
    with torch.no_grad():
        pred = final_model(xs[i].unsqueeze(0)).argmax(1).item()

    fig, axes = plt.subplots(2, k + 1, figsize=(2.3 * (k + 1), 6))
    axes[0][0].imshow(to_img(xs[i]))
    axes[0][0].set_title(f"TEST #{i}\ntrue: {CLASSES[ys[i]]}\npred: {CLASSES[pred]}", fontsize=9)
    axes[1][0].text(0.5, 0.5, "harmful\n(bottom row)", ha="center", va="center")
    for row, idxs, tag in [(0, helpful, "helpful"), (1, harmful, "harmful")]:
        for col, j in enumerate(idxs, start=1):
            j = j.item()
            axes[row][col].imshow(to_img(tracin.train_x[j]))
            axes[row][col].set_title(
                f"{tag} #{j}\n{CLASSES[tracin.train_y[j]]}\nscore {s[j]:.3f}", fontsize=9)
    for ax in axes.flatten():
        ax.axis("off")
    plt.tight_layout()
    path = f"results/tracin_test{i}.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print("saved", path)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    train_set, test_set = get_datasets()
    N_TEST = 100
    xs = torch.stack([test_set[i][0] for i in range(N_TEST)])
    ys = torch.tensor([test_set[i][1] for i in range(N_TEST)])

    tracin = TracInAttribution(train_set)
    scores = tracin.score_batch(xs, ys)
    torch.save(scores, "results/tracin_scores.pt")

    # Same sanity check as kNN: top-10 training examples sharing the true class
    top_frac = bottom_frac = 0.0
    for i in range(N_TEST):
        top = scores[i].topk(10).indices
        bottom = (-scores[i]).topk(10).indices
        top_frac += (tracin.train_y[top] == ys[i]).float().mean().item()
        bottom_frac += (tracin.train_y[bottom] == ys[i]).float().mean().item()
    print(f"TracIn top-10 (helpful):  same class = {top_frac / N_TEST:.2f}")
    print(f"TracIn bottom-10 (harmful): same class = {bottom_frac / N_TEST:.2f}")

    final_model = load_model()
    for i in (0, 1, 2):
        show(i, scores, tracin, xs, ys, final_model)