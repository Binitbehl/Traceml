import json, statistics as st
import torch
import torch.nn.functional as F
from common import get_datasets, CNN
from attribution.tracin import per_sample_grads
from eval.label_noise import train, accuracy
from eval.backdoor import (TARGET, POISON_FRAC, CKPTS, SAVE_DIR, SEEDS,
                           N_TEST, N_EXPLAIN, add_trigger, predict)

if __name__ == "__main__":
    train_set, test_set = get_datasets()
    N = len(train_set)
    train_x = torch.stack([train_set[i][0] for i in range(N)])
    true_y = torch.tensor([train_set[i][1] for i in range(N)])
    test_x = torch.stack([test_set[i][0] for i in range(N_TEST)])
    test_y = torch.tensor([test_set[i][1] for i in range(N_TEST)])

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

    nt_idx = (test_y != TARGET).nonzero().squeeze(1)
    trig_test_x = add_trigger(test_x[nt_idx])

    def evaluate(model):
        clean = accuracy(model, test_x, test_y)
        asr = (predict(model, trig_test_x) == TARGET).float().mean().item()
        return clean, asr

    # the same 20 triggered test images the model calls "airplane"
    final = CNN()
    final.load_state_dict(torch.load(f"{SAVE_DIR}/ep11.pt"))
    final.eval()
    orig = nt_idx[:80]
    trig = add_trigger(test_x[orig])
    ok = (predict(final, trig) == TARGET).nonzero().squeeze(1)[:N_EXPLAIN]
    xs = trig[ok]
    ys = torch.full((len(ok),), TARGET)
    T = len(ok)
    print(f"explaining {T} triggered test images")

    # cosine TracIn scores, summed over checkpoints and over the test images
    total = torch.zeros(N)
    for ep in CKPTS:
        m = CNN()
        m.load_state_dict(torch.load(f"{SAVE_DIR}/ep{ep}.pt"))
        m.eval()
        gt = F.normalize(per_sample_grads(m, xs, ys), dim=1)          # [T, P]
        for s in range(0, N, 100):
            gtr = F.normalize(
                per_sample_grads(m, poison_x[s:s + 100], poison_y[s:s + 100]), dim=1)
            total[s:s + 100] += (gt @ gtr.T).sum(0)
        print(f"checkpoint ep{ep} done", flush=True)

    flagged = total.topk(n_poison).indices
    prec = is_poison[flagged].float().mean().item()
    print(f"cosine TracIn: {prec:.3f} of the {n_poison} flagged examples are poisoned")

    # repair: remove the flagged examples, retrain, re-measure the backdoor
    all_idx = torch.arange(N)
    row = {"clean": [], "asr": []}
    for seed in SEEDS:
        keep = torch.ones(N, dtype=torch.bool)
        keep[flagged] = False
        c, a = evaluate(train(all_idx[keep], poison_y, seed, poison_x))
        row["clean"].append(c)
        row["asr"].append(a)
        print(f"seed {seed} done", flush=True)

    prev = json.load(open("results/backdoor_results.json"))["repair"]
    rows = {**prev, "TracIn (cosine)": row}
    print("\nAfter removing flagged examples and retraining:")
    print(f"{'setting':18s} {'clean acc':>10s} {'attack success':>16s}")
    for name, r in rows.items():
        print(f"{name:18s} {st.mean(r['clean']):10.3f} {st.mean(r['asr']):16.3f}")

    json.dump(rows, open("results/backdoor_cosine_results.json", "w"), indent=2)
    print("saved results/backdoor_cosine_results.json")