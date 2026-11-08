import os, json
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from common import get_datasets, CNN
from attribution.tracin import per_sample_grads

NOISE = 0.10
EPOCHS = 12
CKPT_EPOCHS = (3, 6, 9, 11)
SEEDS = [0, 1]
SAVE_DIR = "ckpts/noisy"          # inside ckpts/, so git ignores it


def lr_at(e):
    return 0.05 if e < 8 else 0.02 if e < 10 else 0.01


def train(keep_idx, labels, seed, train_x, save_dir=None):
    torch.manual_seed(seed)
    model = CNN()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.0)
    keep_idx = torch.as_tensor(keep_idx)
    lrs = []
    for epoch in range(EPOCHS):
        for g in opt.param_groups:
            g["lr"] = lr_at(epoch)
        lrs.append(lr_at(epoch))
        model.train()
        perm = keep_idx[torch.randperm(len(keep_idx))]
        for s in range(0, len(perm), 64):
            b = perm[s:s + 64]
            opt.zero_grad()
            F.cross_entropy(model(train_x[b]), labels[b]).backward()
            opt.step()
        if save_dir:
            torch.save(model.state_dict(), f"{save_dir}/ep{epoch}.pt")
    if save_dir:
        json.dump(lrs, open(f"{save_dir}/lrs.json", "w"))
    model.eval()
    return model


def accuracy(model, tx, ty):
    with torch.no_grad():
        return (model(tx).argmax(1) == ty).float().mean().item()


def self_influence(x, y):
    lrs = json.load(open(f"{SAVE_DIR}/lrs.json"))
    total = torch.zeros(len(x))
    for ep in CKPT_EPOCHS:
        m = CNN()
        m.load_state_dict(torch.load(f"{SAVE_DIR}/ep{ep}.pt"))
        m.eval()
        for s in range(0, len(x), 100):
            g = per_sample_grads(m, x[s:s + 100], y[s:s + 100])
            total[s:s + 100] += lrs[ep] * (g ** 2).sum(1)
        print(f"self-influence: checkpoint ep{ep} done", flush=True)
    return total


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    train_set, test_set = get_datasets()
    N = len(train_set)
    train_x = torch.stack([train_set[i][0] for i in range(N)])
    true_y = torch.tensor([train_set[i][1] for i in range(N)])
    test_x = torch.stack([test_set[i][0] for i in range(2000)])
    test_y = torch.tensor([test_set[i][1] for i in range(2000)])

    # 1) corrupt 10% of labels (always to a different class)
    g = torch.Generator().manual_seed(123)
    n_bad = int(NOISE * N)
    flip = torch.randperm(N, generator=g)[:n_bad]
    shift = torch.randint(1, 10, (n_bad,), generator=g)
    noisy_y = true_y.clone()
    noisy_y[flip] = (true_y[flip] + shift) % 10
    is_noisy = torch.zeros(N, dtype=torch.bool)
    is_noisy[flip] = True
    print(f"corrupted {n_bad} of {N} labels")

    # 2) train on the noisy labels, saving checkpoints
    all_idx = torch.arange(N)
    model = train(all_idx, noisy_y, 0, train_x, save_dir=SAVE_DIR)
    print(f"test accuracy with noisy labels: {accuracy(model, test_x, test_y):.3f}")

    # 3) detection scores (higher = more suspicious)
    si = self_influence(train_x, noisy_y)
    with torch.no_grad():
        loss = F.cross_entropy(model(train_x), noisy_y, reduction="none")
    rnd = torch.rand(N, generator=torch.Generator().manual_seed(7))
    scores = {"self-influence": si, "train loss": loss, "random": rnd}

    print(f"\nDetection of the {n_bad} corrupted labels:")
    print(f"{'method':16s} {'AUROC':>7s} {'precision@' + str(n_bad):>15s}")
    flagged = {}
    for name, s in scores.items():
        top = s.topk(n_bad).indices
        flagged[name] = top
        auroc = roc_auc_score(is_noisy.numpy(), s.numpy())
        prec = is_noisy[top].float().mean().item()
        print(f"{name:16s} {auroc:7.3f} {prec:15.3f}")

    # 4) repair test: remove flagged examples, retrain, compare test accuracy
    flagged["oracle"] = flip
    print("\nTest accuracy after removing flagged examples and retraining:")
    results = {"no cleaning": []}
    for name in flagged:
        results[name] = []
    for seed in SEEDS:
        results["no cleaning"].append(
            accuracy(train(all_idx, noisy_y, seed, train_x), test_x, test_y))
        for name, idxs in flagged.items():
            keep = torch.ones(N, dtype=torch.bool)
            keep[idxs] = False
            m = train(all_idx[keep], noisy_y, seed, train_x)
            results[name].append(accuracy(m, test_x, test_y))
        print(f"seed {seed} done", flush=True)
    for name, accs in results.items():
        print(f"  {name:16s} {sum(accs) / len(accs):.3f}")