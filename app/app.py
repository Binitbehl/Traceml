import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import streamlit as st
from common import get_datasets, load_model, CLASSES
from attribution.knn import KNNAttribution
from attribution.tracin import per_sample_grads

st.set_page_config(page_title="TraceML", layout="wide")
st.title("TraceML: which training examples explain this prediction?")
st.markdown("*Made by Binit Behl*")

@st.cache_resource
def setup():
    train_set, test_set = get_datasets()
    N = len(train_set)
    train_x = torch.stack([train_set[i][0] for i in range(N)])
    train_y = torch.tensor([train_set[i][1] for i in range(N)])
    model = load_model()
    knn = KNNAttribution(model, train_set)
    return test_set, train_x, train_y, model, knn


@st.cache_data(show_spinner="Computing cosine TracIn (about a minute)...")
def cosine_tracin(idx, ckpts=(11, 19, 24)):
    test_set, train_x, train_y, _, _ = setup()
    x, y = test_set[idx]
    xs, ys = x.unsqueeze(0), torch.tensor([y])
    scores = torch.zeros(len(train_x))
    for ep in ckpts:
        m = load_model(f"ckpts/ep{ep}.pt")
        gt = F.normalize(per_sample_grads(m, xs, ys), dim=1)
        for s in range(0, len(train_x), 100):
            g = F.normalize(per_sample_grads(m, train_x[s:s + 100],
                                             train_y[s:s + 100]), dim=1)
            scores[s:s + 100] += (gt @ g.T).squeeze(0)
    return scores


def to_img(x):
    return (x * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()


def show_grid(indices, scores, train_x, train_y, ncols=5):
    cols = st.columns(ncols)
    for c, j in zip(cols, indices):
        j = j.item()
        c.image(to_img(train_x[j]), width=120,
                caption=f"#{j}: {CLASSES[train_y[j]]} ({scores[j]:.2f})")


test_set, train_x, train_y, model, knn = setup()

with st.sidebar:
    st.header("Settings")
    idx = st.number_input("Test image index", 0, len(test_set) - 1, 0)
    method = st.radio("Attribution method",
                      ["kNN (label-aware)", "TracIn (cosine, 3 checkpoints)"])
    k = st.slider("Training examples to show", 3, 10, 5)

x, y = test_set[int(idx)]
with torch.no_grad():
    probs = F.softmax(model(x.unsqueeze(0)), dim=1)[0]
pred = probs.argmax().item()

left, right = st.columns([1, 3])
with left:
    st.image(to_img(x), width=180, caption=f"Test image #{int(idx)}")
    st.write(f"**True:** {CLASSES[y]}")
    st.write(f"**Predicted:** {CLASSES[pred]} ({probs[pred]:.0%})")
        if pred == y:
        st.success("Correct")
    else:
        st.error("Misclassified")

with right:
    if method.startswith("kNN"):
        s = knn.score(x).clone()
        s[knn.train_labels != y] = -2        # only the true class is eligible
        st.subheader(f"Most similar training images of class {CLASSES[y]}")
        show_grid(s.topk(k).indices, s, train_x, train_y, ncols=k)
    else:
        s = cosine_tracin(int(idx))
        st.subheader("Most helpful training examples (support the true label)")
        show_grid(s.topk(k).indices, s, train_x, train_y, ncols=k)
        st.subheader("Most harmful training examples (oppose the true label)")
        show_grid((-s).topk(k).indices, s, train_x, train_y, ncols=k)
        top = train_y[s.topk(50).indices]
        counts = {CLASSES[c]: int((top == c).sum()) for c in range(10)
                  if int((top == c).sum()) > 0}
        st.write("**Classes of the top-50 helpful examples:**", counts)

st.caption("Scores shown in brackets. kNN: cosine similarity of embeddings. "
           "TracIn: summed cosine similarity of loss gradients at 3 checkpoints. "
           "Small CPU-trained CNN on 5,000 CIFAR-10 images. "
           "TraceML, made by Binit Behl.")