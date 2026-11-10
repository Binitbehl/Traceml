import json
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from common import get_datasets, CNN
from attribution.tracin import per_sample_grads
from eval.backdoor import (TARGET, POISON_FRAC, SAVE_DIR, CKPTS, N_EXPLAIN, K,
                           add_trigger, predict)

if __name__ == "__main__":
    train_set, test_set = get_datasets()
    N = len(train_set)
    train_x = torch.stack([train_set[i][0] for i in range(N)])
    true_y = torch.tensor([train_set[i][1] for i in range(N)])
    test_x = torch.stack([test_set[i][0] for i in range(200)])
    test_y = torch.tensor([test_set[i][1] for i in range(200)])

    # rebuild exactly the same poisoned training set as eval/backdoor.py
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

    # the same kind of triggered test images as before (model = ep11)
    final = CNN()
    final.load_state_dict(torch.load(f"{SAVE_DIR}/ep11.pt"))
    final.eval()
    nt_idx = (test_y != TARGET).nonzero().squeeze(1)
    trig = add_trigger(test_x[nt_idx[:80]])
    ok = (predict(final, trig) == TARGET).nonzero().squeeze(1)[:N_EXPLAIN]
    xs = trig[ok]
    ys = torch.full((len(ok),), TARGET)
    T = len(ok)
    print(f"{T} triggered test images")

    lrs = json.load(open(f"{SAVE_DIR}/lrs.json"))
    raw = torch.zeros(T, N)          # raw TracIn score, summed over checkpoints
    cos = torch.zeros(T, N)          # cosine TracIn score, summed over checkpoints
    norms = {}

    for ep in CKPTS:
        m = CNN()
        m.load_state_dict(torch.load(f"{SAVE_DIR}/ep{ep}.pt"))
        m.eval()
        gt = per_sample_grads(m, xs, ys)
        gt_n = F.normalize(gt, dim=1)
        nrm = torch.zeros(N)
        for s in range(0, N, 100):
            gtr = per_sample_grads(m, poison_x[s:s + 100], poison_y[s:s + 100])
            n = gtr.norm(dim=1)
            nrm[s:s + 100] = n
            raw[:, s:s + 100] += lrs[ep] * (gt @ gtr.T)
            cos[:, s:s + 100] += gt_n @ (gtr / n.clamp(min=1e-12)[:, None]).T
        norms[ep] = nrm
        print(f"checkpoint ep{ep} done", flush=True)

    # ---- 1) are poisoned examples' gradients smaller? ----
    print("\nGradient norm of training examples (poisoned vs clean):")
    print(f"{'ckpt':>5} {'median poisoned':>16} {'median clean':>13} {'ratio':>7} {'AUROC(low norm = poison)':>26}")
    for ep in CKPTS:
        n = norms[ep]
        mp, mc = n[is_poison].median().item(), n[~is_poison].median().item()
        auroc = roc_auc_score(is_poison.numpy(), (-n).numpy())
        print(f"{ep:>5} {mp:16.3f} {mc:13.3f} {mp / mc:7.2f} {auroc:26.3f}")

    # ---- 2) does the raw score follow the gradient norm? ----
    mean_norm = torch.stack([norms[e] for e in CKPTS]).mean(0)
    rho_raw = [spearmanr(raw[i].numpy(), mean_norm.numpy())[0] for i in range(T)]
    rho_cos = [spearmanr(cos[i].numpy(), mean_norm.numpy())[0] for i in range(T)]
    print("\nSpearman correlation between a score and the example's mean gradient norm")
    print(f"(averaged over the {T} triggered test images):")
    print(f"  raw TracIn score     {sum(rho_raw) / T:+.3f}")
    print(f"  cosine TracIn score  {sum(rho_cos) / T:+.3f}")

    # ---- 3) what do the top-K training examples look like? ----
    med = mean_norm.median().item()
    print(f"\nTop-{K} training examples per image (median gradient norm of all examples = {med:.3f}):")
    for name, s in [("raw", raw), ("cosine", cos)]:
        top = s.topk(K, dim=1).indices
        frac = is_poison[top].float().mean().item()
        avg_norm = mean_norm[top].mean().item()
        print(f"  {name:7s} poisoned fraction {frac:.3f}   mean gradient norm {avg_norm:.3f} "
              f"({avg_norm / med:.1f}x the median)")