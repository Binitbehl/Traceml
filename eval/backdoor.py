import os, json, statistics as st
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from common import get_datasets, CNN, CLASSES
from attribution.tracin import per_sample_grads
from eval.label_noise import train, accuracy

# ---------- settings ----------
TARGET = 0                 # backdoor target class (airplane)
POISON_FRAC = 0.04         # 200 of 5000 training images
N_EXPLAIN = 20             # triggered test images to explain
K = 50                     # top-k used for per-image precision
CKPTS = (3, 6, 9, 11)
SEEDS = [0, 1]             # repeats for the repair test
N_TEST = 10000
SAVE_DIR = "ckpts/backdoor"   # inside ckpts/, so git ignores it

PATTERN = torch.tensor([[1, -1, 1, -1, 1],
                        [-1, 1, -1, 1, -1],
                        [1, -1, 1, -1, 1],
                        [-1, 1, -1, 1, -1],
                        [1, -1, 1, -1, 1]]).float()


def add_trigger(x):
    """Stamp a 5x5 checkerboard near the bottom-right corner (works on [3,32,32] or [N,3,32,32])."""
    x = x.clone()
    x[..., -6:-1, -6:-1] = PATTERN
    return x


def predict(model, x, bs=500):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(x), bs):
            out.append(model(x[s:s + bs]).argmax(1))
    return torch.cat(out)


def embed_all(model, x, bs=500):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(x), bs):
            out.append(model.embed(x[s:s + bs]))
    return F.normalize(torch.cat(out), dim=1)


def tracin_scores(xs, ys, train_x, train_y):
    """score[t, n] = sum over checkpoints of lr * <grad(test t), grad(train n)>"""
    lrs = json.load(open(f"{SAVE_DIR}/lrs.json"))
    scores = torch.zeros(len(xs), len(train_x))
    for ep in CKPTS:
        m = CNN()
        m.load_state_dict(torch.load(f"{SAVE_DIR}/ep{ep}.pt"))
        m.eval()
        g_test = per_sample_grads(m, xs, ys)
        for s in range(0, len(train_x), 100):
            g = per_sample_grads(m, train_x[s:s + 100], train_y[s:s + 100])
            scores[:, s:s + 100] += lrs[ep] * (g_test @ g.T)
        print(f"tracin: checkpoint ep{ep} done", flush=True)
    return scores


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_set, test_set = get_datasets()
    N = len(train_set)
    train_x = torch.stack([train_set[i][0] for i in range(N)])
    true_y = torch.tensor([train_set[i][1] for i in range(N)])
    test_x = torch.stack([test_set[i][0] for i in range(N_TEST)])
    test_y = torch.tensor([test_set[i][1] for i in range(N_TEST)])

    # 1) poison the training set
    g = torch.Generator().manual_seed(123)
    cand = (true_y != TARGET).nonzero().squeeze(1)
    n_poison = int(POISON_FRAC * N)
    pidx = cand[torch.randperm(len(cand), generator=g)[:n_poison]]
    poison_x = train_x.clone()
    poison_x[pidx] = add_trigger(train_x[pidx])
    poison_y = true_y.clone()
    poison_y[pidx] = TARGET
    is_poison = torch.zeros(N, dtype=torch.bool)
    is_poison[pidx] = True
    print(f"poisoned {n_poison} of {N} training images -> class {CLASSES[TARGET]}")

    # test set with the trigger (non-target classes only) for attack success rate
    nt_idx = (test_y != TARGET).nonzero().squeeze(1)
    trig_test_x = add_trigger(test_x[nt_idx])

    def evaluate(model):
        clean = accuracy(model, test_x, test_y)
        asr = (predict(model, trig_test_x) == TARGET).float().mean().item()
        return clean, asr

    # 2) train on the poisoned data, saving checkpoints
    all_idx = torch.arange(N)
    model = train(all_idx, poison_y, 0, poison_x, save_dir=SAVE_DIR)
    clean_acc, asr = evaluate(model)
    print(f"clean test accuracy: {clean_acc:.3f}")
    print(f"attack success rate (triggered non-airplane -> airplane): {asr:.3f}")

    # 3) pick triggered test images the model now calls "airplane"
    orig = nt_idx[:80]
    trig = add_trigger(test_x[orig])
    ok = (predict(model, trig) == TARGET).nonzero().squeeze(1)[:N_EXPLAIN]
    if len(ok) < 5:
        print("The backdoor barely worked (too few successful attacks). "
              "Raise POISON_FRAC and rerun.")
        raise SystemExit
    xs_trig = trig[ok]
    ys_trig = torch.full((len(ok),), TARGET)
    xs_clean = test_x[orig][ok]
    ys_clean = test_y[orig][ok]
    T = len(ok)
    print(f"explaining {T} triggered test images")

    # 4) attribution scores: [T, N]
    s_tracin = tracin_scores(xs_trig, ys_trig, poison_x, poison_y)
    s_ctrl = tracin_scores(xs_clean, ys_clean, poison_x, poison_y)   # control: no trigger
    emb = embed_all(model, poison_x)
    s_knn = embed_all(model, xs_trig) @ emb.T
    s_rand = torch.rand(T, N, generator=torch.Generator().manual_seed(7))

    def prec_at_k(s):
        top = s.topk(K, dim=1).indices
        return is_poison[top].float().mean().item()

    print(f"\nFraction of the top-{K} training examples that are poisoned "
          f"(base rate {POISON_FRAC:.2f}):")
    print(f"  random                       {prec_at_k(s_rand):.3f}")
    print(f"  kNN                          {prec_at_k(s_knn):.3f}")
    print(f"  TracIn (triggered images)    {prec_at_k(s_tracin):.3f}")
    print(f"  TracIn (control, no trigger) {prec_at_k(s_ctrl):.3f}")

    # 5) detection by aggregating over all explained images
    agg = {"TracIn": s_tracin.sum(0), "kNN": s_knn.sum(0), "random": s_rand.sum(0)}
    print(f"\nDetecting the {n_poison} poisoned training images (scores summed over {T} images):")
    print(f"{'method':10s} {'AUROC':>7s} {'precision@' + str(n_poison):>16s}")
    flagged, detection = {}, {}
    for name, s in agg.items():
        top = s.topk(n_poison).indices
        flagged[name] = top
        auroc = roc_auc_score(is_poison.numpy(), s.numpy())
        prec = is_poison[top].float().mean().item()
        detection[name] = {"auroc": auroc, "precision": prec}
        print(f"{name:10s} {auroc:7.3f} {prec:16.3f}")

    # 6) repair: remove flagged examples, retrain, re-measure the backdoor
    flagged["oracle"] = pidx
    rows = {"no cleaning": {"clean": [], "asr": []}}
    for name in flagged:
        rows[name] = {"clean": [], "asr": []}
    print("\nRepair test (retraining takes a while)...", flush=True)
    for seed in SEEDS:
        c, a = evaluate(train(all_idx, poison_y, seed, poison_x))
        rows["no cleaning"]["clean"].append(c)
        rows["no cleaning"]["asr"].append(a)
        for name, idxs in flagged.items():
            keep = torch.ones(N, dtype=torch.bool)
            keep[idxs] = False
            c, a = evaluate(train(all_idx[keep], poison_y, seed, poison_x))
            rows[name]["clean"].append(c)
            rows[name]["asr"].append(a)
        print(f"seed {seed} done", flush=True)

    print("\nAfter removing flagged examples and retraining:")
    print(f"{'setting':14s} {'clean acc':>10s} {'attack success':>16s}")
    for name, r in rows.items():
        print(f"{name:14s} {st.mean(r['clean']):10.3f} {st.mean(r['asr']):16.3f}")

    json.dump({"asr_before": asr, "detection": detection, "repair": rows},
              open("results/backdoor_results.json", "w"), indent=2)
    print("saved results/backdoor_results.json")