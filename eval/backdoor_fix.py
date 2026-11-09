import json
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from common import get_datasets, CNN
from attribution.tracin import per_sample_grads
from eval.backdoor import (TARGET, POISON_FRAC, K, CKPTS, SAVE_DIR, N_EXPLAIN,
                           add_trigger, predict)

if __name__ == "__main__":
    train_set, test_set = get_datasets()
    N = len(train_set)
    train_x = torch.stack([train_set[i][0] for i in range(N)])
    true_y = torch.tensor([train_set[i][1] for i in range(N)])
    test_x = torch.stack([test_set[i][0] for i in range(200)])
    test_y = torch.tensor([test_set[i][1] for i in range(200)])

    # rebuild exactly the same poisoned training set as backdoor.py
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

    # same triggered test images as before (final model = ep11)
    final = CNN()
    final.load_state_dict(torch.load(f"{SAVE_DIR}/ep11.pt"))
    final.eval()
    nt_idx = (test_y != TARGET).nonzero().squeeze(1)
    orig = nt_idx[:80]
    trig = add_trigger(test_x[orig])
    ok = (predict(final, trig) == TARGET).nonzero().squeeze(1)[:N_EXPLAIN]
    xs = trig[ok]
    y_target = torch.full((len(ok),), TARGET)
    y_true = test_y[orig][ok]
    T = len(ok)
    print(f"explaining {T} triggered test images")

    lrs = json.load(open(f"{SAVE_DIR}/lrs.json"))
    names = ["raw, target label (original)", "raw, true label (flag negative)",
             "cosine, target label", "cosine, true label (flag negative)",
             "raw, target label, ep3 only"]
    S = {n: torch.zeros(T, N) for n in names}

    for ep in CKPTS:
        m = CNN()
        m.load_state_dict(torch.load(f"{SAVE_DIR}/ep{ep}.pt"))
        m.eval()
        gt_t = per_sample_grads(m, xs, y_target)
        gt_y = per_sample_grads(m, xs, y_true)
        gt_t_n, gt_y_n = F.normalize(gt_t, dim=1), F.normalize(gt_y, dim=1)
        for s in range(0, N, 100):
            gtr = per_sample_grads(m, poison_x[s:s + 100], poison_y[s:s + 100])
            gtr_n = F.normalize(gtr, dim=1)
            sl = slice(s, s + 100)
            S[names[0]][:, sl] += lrs[ep] * (gt_t @ gtr.T)
            S[names[1]][:, sl] += lrs[ep] * (gt_y @ gtr.T)
            S[names[2]][:, sl] += gt_t_n @ gtr_n.T
            S[names[3]][:, sl] += gt_y_n @ gtr_n.T
            if ep == CKPTS[0]:
                S[names[4]][:, sl] += lrs[ep] * (gt_t @ gtr.T)
        print(f"checkpoint ep{ep} done", flush=True)

    # make "higher = more suspicious" for every variant
    sus = {n: (-S[n] if "negative" in n else S[n]) for n in names}

    print(f"\nFraction of top-{K} per image that is poisoned (base rate {POISON_FRAC:.2f}):")
    print(f"\n{'variant':36s} {'top-K frac':>10s} {'AUROC':>7s} {'prec@200':>9s}")
    for n in names:
        top = sus[n].topk(K, dim=1).indices
        frac = is_poison[top].float().mean().item()
        agg = sus[n].sum(0)
        auroc = roc_auc_score(is_poison.numpy(), agg.numpy())
        prec = is_poison[agg.topk(n_poison).indices].float().mean().item()
        print(f"{n:36s} {frac:10.3f} {auroc:7.3f} {prec:9.3f}")