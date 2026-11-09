import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
import streamlit as st
from common import get_datasets, CNN, CLASSES
from attribution.tracin import per_sample_grads
from eval.backdoor import TARGET, POISON_FRAC, SAVE_DIR, add_trigger, predict

st.set_page_config(page_title="TraceML: backdoor demo", layout="wide")
st.title("Backdoor forensics: tracing a hijacked prediction")
st.markdown("*Made by Binit Behl*")
st.write(
    "This model was trained on data where 4% of the images carry a small "
    "checkerboard trigger and were relabeled as airplane. Pick a clean image, "
    "add the trigger, and trace the model's wrong answer back to the training "
    "data using cosine-normalized TracIn.")


@st.cache_resource
def setup():
    train_set, test_set = get_datasets()
    N = len(train_set)
    train_x = torch.stack([train_set[i][0] for i in range(N)])
    true_y = torch.tensor([train_set[i][1] for i in range(N)])
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
    model = CNN()
    model.load_state_dict(torch.load(f"{SAVE_DIR}/ep11.pt"))
    model.eval()
    return test_set, poison_x, poison_y, is_poison, model


@st.cache_data(show_spinner="Tracing the prediction (1-2 minutes)...")
def trace(idx):
    test_set, poison_x, poison_y, _, _ = setup()
    x, _ = test_set[idx]
    xt = add_trigger(x).unsqueeze(0)
    yt = torch.tensor([TARGET])
    scores = torch.zeros(len(poison_x))
    for ep in (3, 6, 9, 11):
        m = CNN()
        m.load_state_dict(torch.load(f"{SAVE_DIR}/ep{ep}.pt"))
        m.eval()
        gt = F.normalize(per_sample_grads(m, xt, yt), dim=1)
        for s in range(0, len(poison_x), 100):
            g = F.normalize(per_sample_grads(m, poison_x[s:s + 100],
                                             poison_y[s:s + 100]), dim=1)
            scores[s:s + 100] += (gt @ g.T).squeeze(0)
    return scores


def to_img(x):
    return (x * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()


test_set, poison_x, poison_y, is_poison, model = setup()

with st.sidebar:
    st.header("Settings")
    idx = int(st.number_input("Test image index", 0, len(test_set) - 1, 0))
    k = st.slider("Training examples to show", 5, 10, 10)

x, y = test_set[idx]
if y == TARGET:
    st.warning(f"Test image #{idx} is already an airplane. Pick a different index.")
    st.stop()

xt = add_trigger(x)
p_clean = predict(model, x.unsqueeze(0)).item()
p_trig = predict(model, xt.unsqueeze(0)).item()

c1, c2, _ = st.columns([1, 1, 3])
c1.image(to_img(x), width=180, caption=f"Clean #{idx}")
c1.write(f"True: **{CLASSES[y]}**  \nModel says: **{CLASSES[p_clean]}**")
c2.image(to_img(xt), width=180, caption="Same image + trigger")
c2.write(f"Model says: **{CLASSES[p_trig]}**")

if p_trig != TARGET:
    st.info("The trigger did not fool the model on this image. Try another index.")
    st.stop()

if st.checkbox("Trace the triggered prediction back to the training data"):
    s = trace(idx)
    top = s.topk(k).indices
    st.subheader(f"Training examples most aligned with the hijacked prediction (top {k})")
    cols = st.columns(k)
    for c, j in zip(cols, top):
        j = j.item()
        tag = "POISONED" if is_poison[j] else "clean"
        c.image(to_img(poison_x[j]), width=110,
                caption=f"#{j}: labeled {CLASSES[poison_y[j]]} ({s[j]:.2f}) [{tag}]")
    top50 = s.topk(50).indices
    n_bad = int(is_poison[top50].sum())
    st.write(f"**{n_bad} of the top 50 are poisoned** "
             f"(only {POISON_FRAC:.0%} of the training set is).")
    st.caption("The poisoned labels are used only to check the result. The "
               "tracing method never sees them.")

st.caption("Cosine TracIn over 4 checkpoints. Small CPU-trained CNN on 5,000 "
           "CIFAR-10 images. TraceML, made by Binit Behl.")