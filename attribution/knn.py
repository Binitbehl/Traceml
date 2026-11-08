import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from common import get_datasets, load_model, CLASSES


class RandomAttribution:
    """Baseline: random scores. Any real method must beat this."""
    def __init__(self, n_train, seed=0):
        self.n_train = n_train
        self.gen = torch.Generator().manual_seed(seed)

    def score(self, x, y=None):
        return torch.rand(self.n_train, generator=self.gen)


class KNNAttribution:
    """Score = cosine similarity between test and training embeddings."""
    def __init__(self, model, train_set):
        self.model = model
        embs, labels = [], []
        loader = DataLoader(train_set, batch_size=256, shuffle=False)
        with torch.no_grad():
            for x, y in loader:
                embs.append(model.embed(x))
                labels.append(y)
        self.train_emb = F.normalize(torch.cat(embs), dim=1)
        self.train_labels = torch.cat(labels)

    def score(self, x, y=None):
        with torch.no_grad():
            e = F.normalize(self.model.embed(x.unsqueeze(0)), dim=1)
        return (self.train_emb @ e.T).squeeze(1)


def to_img(x):
    return (x * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()


def show(idx, train_set, test_set, model, knn, k=5):
    x, y = test_set[idx]
    with torch.no_grad():
        pred = model(x.unsqueeze(0)).argmax(1).item()
    top = knn.score(x).topk(k)

    fig, axes = plt.subplots(1, k + 1, figsize=(2.3 * (k + 1), 3.1))
    axes[0].imshow(to_img(x))
    axes[0].set_title(f"TEST #{idx}\ntrue: {CLASSES[y]}\npred: {CLASSES[pred]}", fontsize=9)
    for ax, s, j in zip(axes[1:], top.values, top.indices):
        tx, ty = train_set[j.item()]
        ax.imshow(to_img(tx))
        ax.set_title(f"train #{j.item()}\n{CLASSES[ty]}\nsim {s:.2f}", fontsize=9)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    path = f"results/knn_test{idx}.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print("saved", path)


if __name__ == "__main__":
    train_set, test_set = get_datasets()
    model = load_model()
    knn = KNNAttribution(model, train_set)

    # Visual check: one test image, then the first misclassified one
    show(0, train_set, test_set, model, knn)
    for i in range(200):
        x, y = test_set[i]
        with torch.no_grad():
            if model(x.unsqueeze(0)).argmax(1).item() != y:
                show(i, train_set, test_set, model, knn)
                break

    # Sanity check: do the top-10 neighbors share the test image's true class?
    rand = RandomAttribution(len(train_set))
    for name, method in [("kNN", knn), ("random", rand)]:
        frac = 0.0
        for i in range(100):
            x, y = test_set[i]
            top = method.score(x).topk(10).indices
            frac += (knn.train_labels[top] == y).float().mean().item()
        print(f"{name}: avg fraction of top-10 with same class = {frac / 100:.2f}")