import json, statistics as st
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from common import get_datasets, CNN
from attribution.tracin import per_sample_grads
from eval.label_noise import (NOISE, SEEDS, N_TEST, SAVE_DIR, CKPT_EPOCHS,
                              EPOCHS, train, accuracy, per_example_loss)
from eval.label_noise_val import load_ckpt, rank

N_VAL = 1000
VAL_START = 5000   # CIFAR-10 train images 5000-5999: clean labels, unused elsewhere


def detector_scores(train_x, noisy_y, val_x, val_y):
    """Higher = more suspicious. Returns three score vectors."""
    lrs = json.load(open(f"{SAVE_DIR}/lrs.json"))
    N = len(train_x)
    cos_cc = torch.zeros(N)    # cosine, class-conditional
    cos_all = torch.zeros(N)   # cosine, all validation images
    raw_cc = torch.zeros(N)    # raw, class-conditional

    for ep in CKPT_EPOCHS:
        m = load_ckpt(ep)
        P = sum(p.numel() for p in m.parameters())
        cls_n = torch.zeros(10, P)     # sum of normalized val gradients per class
        cls_r = torch.zeros(10, P)     # sum of raw val gradients per class
        cnt = torch.zeros(10)
        for s in range(0, len(val_x), 100):
            vy = val_y[s:s + 100]
            g = per_sample_grads(m, val_x[s:s + 100], vy)
            cls_n.index_add_(0, vy, F.normalize(g, dim=1))
            cls_r.index_add_(0, vy, g)
            cnt += torch.bincount(vy, minlength=10).float()
        all_mean = cls_n.sum(0) / cnt.sum()
        cls_n /= cnt[:, None]
        cls_r /= cnt[:, None]

        for s in range(0, N, 100):
            y = noisy_y[s:s + 100]
            g = per_sample_grads(m, train_x[s:s + 100], y)
            gn = F.normalize(g, dim=1)
            sl = slice(s, s + 100)
            cos_cc[sl] += (gn * cls_n[y]).sum(1)
            cos_all[sl] += gn @ all_mean
            raw_cc[sl] += lrs[ep] * (g * cls_r[y]).sum(1)
        print(f"checkpoint ep{ep} done", flush=True)

    # low agreement with clean validation data = suspicious
    return -cos_cc, -cos_all, -raw_cc


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

    s_cc, s_all, s_raw = detector_scores(train_x, noisy_y, val_x, val_y)
    loss = per_example_loss(load_ckpt(EPOCHS - 1), train_x, noisy_y)
    rnd = torch.rand(N, generator=torch.Generator().manual_seed(7))
    scores = {
        "random": rnd,
        "train loss": loss,
        "raw, class-conditional": s_raw,
        "cosine, all-val": s_all,
        "cosine, class-conditional": s_cc,
        "loss + cosine class-cond": rank(loss) + rank(s_cc),
    }

    print(f"\nDetection of the {n_bad} corrupted labels:")
    print(f"{'method':28s} {'AUROC':>7s} {'precision@' + str(n_bad):>16s}")
    flagged, detection = {}, {}
    for name, s in scores.items():
        top = s.topk(n_bad).indices
        flagged[name] = top
        auroc = roc_auc_score(is_noisy.numpy(), s.numpy())
        prec = is_noisy[top].float().mean().item()
        detection[name] = {"auroc": auroc, "precision": prec}
        print(f"{name:28s} {auroc:7.3f} {prec:16.3f}")

    # repair test for the two new cosine-based detectors
    all_idx = torch.arange(N)
    new = {"cosine class-cond": [], "loss + cosine class-cond": []}
    print("\nRepair test (retraining takes a few minutes)...", flush=True)
    for seed in SEEDS:
        for name, key in [("cosine class-cond", "cosine, class-conditional"),
                          ("loss + cosine class-cond", "loss + cosine class-cond")]:
            keep = torch.ones(N, dtype=torch.bool)
            keep[flagged[key]] = False
            m = train(all_idx[keep], noisy_y, seed, train_x)
            new[name].append(accuracy(m, test_x, test_y))
        print(f"seed {seed} done", flush=True)

    prev = json.load(open("results/label_noise_val_results.json"))["repair"]
    rows = {**prev, **new}
    print("\nTest accuracy after removing flagged examples and retraining:")
    for name, accs in sorted(rows.items(), key=lambda kv: -st.mean(kv[1])):
        print(f"  {name:26s} {st.mean(accs):.3f} (sd {st.pstdev(accs):.3f})")

    json.dump({"detection": detection, "repair": rows},
              open("results/label_noise_cos_results.json", "w"), indent=2)
    print("saved results/label_noise_cos_results.json")