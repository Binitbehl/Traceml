import json, statistics as st
import torch
from sklearn.metrics import roc_auc_score
from common import get_datasets, CNN
from attribution.tracin import per_sample_grads
from eval.label_noise import (NOISE, EPOCHS, SEEDS, N_TEST, SAVE_DIR,
                              CKPT_EPOCHS, train, accuracy, per_example_loss)

N_VAL = 500
VAL_START = 5000   # CIFAR-10 train images 5000-5499: clean labels, unused elsewhere


def load_ckpt(ep):
    m = CNN()
    m.load_state_dict(torch.load(f"{SAVE_DIR}/ep{ep}.pt"))
    m.eval()
    return m


def val_influence(train_x, noisy_y, val_x, val_y):
    """score[n] = sum over checkpoints of lr * <sum of validation grads, grad of train n>.
    Positive = helps validation loss. Negative = hurts (suspicious)."""
    lrs = json.load(open(f"{SAVE_DIR}/lrs.json"))
    total = torch.zeros(len(train_x))
    for ep in CKPT_EPOCHS:
        m = load_ckpt(ep)
        g_val = torch.zeros(sum(p.numel() for p in m.parameters()))
        for s in range(0, len(val_x), 100):
            g_val += per_sample_grads(m, val_x[s:s + 100], val_y[s:s + 100]).sum(0)
        for s in range(0, len(train_x), 100):
            g = per_sample_grads(m, train_x[s:s + 100], noisy_y[s:s + 100])
            total[s:s + 100] += lrs[ep] * (g @ g_val)
        print(f"val-influence: checkpoint ep{ep} done", flush=True)
    return total


def rank(x):
    return x.argsort().argsort().float()


if __name__ == "__main__":
    train_set, test_set = get_datasets()
    N = len(train_set)
    train_x = torch.stack([train_set[i][0] for i in range(N)])
    true_y = torch.tensor([train_set[i][1] for i in range(N)])
    test_x = torch.stack([test_set[i][0] for i in range(N_TEST)])
    test_y = torch.tensor([test_set[i][1] for i in range(N_TEST)])

    full = train_set.dataset
    val_x = torch.stack([full[i][0] for i in range(VAL_START, VAL_START + N_VAL)])
    val_y = torch.tensor([full[i][1] for i in range(VAL_START, VAL_START + N_VAL)])

    # recreate exactly the same corrupted labels as label_noise.py
    g = torch.Generator().manual_seed(123)
    n_bad = int(NOISE * N)
    flip = torch.randperm(N, generator=g)[:n_bad]
    shift = torch.randint(1, 10, (n_bad,), generator=g)
    noisy_y = true_y.clone()
    noisy_y[flip] = (true_y[flip] + shift) % 10
    is_noisy = torch.zeros(N, dtype=torch.bool)
    is_noisy[flip] = True

    # detection scores (higher = more suspicious)
    vi = -val_influence(train_x, noisy_y, val_x, val_y)
    loss = per_example_loss(load_ckpt(EPOCHS - 1), train_x, noisy_y)
    combined = rank(loss) + rank(vi)
    rnd = torch.rand(N, generator=torch.Generator().manual_seed(7))
    scores = {"val-influence": vi, "train loss": loss,
              "loss + val-influence": combined, "random": rnd}

    print(f"\nDetection of the {n_bad} corrupted labels:")
    print(f"{'method':22s} {'AUROC':>7s} {'precision@' + str(n_bad):>16s}")
    flagged, detection = {}, {}
    for name, s in scores.items():
        top = s.topk(n_bad).indices
        flagged[name] = top
        auroc = roc_auc_score(is_noisy.numpy(), s.numpy())
        prec = is_noisy[top].float().mean().item()
        detection[name] = {"auroc": auroc, "precision": prec}
        print(f"{name:22s} {auroc:7.3f} {prec:16.3f}")

    # repair test for the two new detectors; earlier rows come from the saved results
    prev = json.load(open("results/label_noise_results.json"))["repair"]
    all_idx = torch.arange(N)
    new = {"val-influence": [], "loss + val-influence": []}
    print("\nRepair test (retraining takes a few minutes)...", flush=True)
    for seed in SEEDS:
        for name in new:
            keep = torch.ones(N, dtype=torch.bool)
            keep[flagged[name]] = False
            m = train(all_idx[keep], noisy_y, seed, train_x)
            new[name].append(accuracy(m, test_x, test_y))
        print(f"seed {seed} done", flush=True)

    rows = {**prev, **new}
    print("\nTest accuracy after removing flagged examples and retraining:")
    for name, accs in sorted(rows.items(), key=lambda kv: -st.mean(kv[1])):
        print(f"  {name:24s} {st.mean(accs):.3f} (sd {st.pstdev(accs):.3f})")

    json.dump({"detection": detection, "repair": rows},
              open("results/label_noise_val_results.json", "w"), indent=2)
    print("saved results/label_noise_val_results.json")