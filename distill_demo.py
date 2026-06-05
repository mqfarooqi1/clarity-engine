"""
=============================================================================
 distill_demo.py  —  Knowledge Distillation: big teacher -> tiny student
=============================================================================

Runnable proof of ADVANCED.md section 4.2 — "the best way to use an LLM to
train for accuracy."

THE IDEA
--------
You can't serve a frontier LLM on every production request (too slow, too
expensive). So you use it ONCE, offline, as a TEACHER: it labels a large pool
of data with SOFT probabilities (not just a single answer). Then you train a
small, fast STUDENT to imitate those soft labels. The student ends up nearly as
accurate as the teacher, but tiny enough to serve cheaply.

The key insight this demo proves: a soft label like
    {billing: 0.80, cancellation: 0.15, technical: 0.05}
teaches the student the *geometry* between classes — far more signal than a
hard label that just says "billing". A student trained on soft labels beats an
identical student trained on hard labels.

In the REAL pipeline the teacher is the Claude Opus 4.8 adjudicator in
llm_denoise.py. Here, so it runs offline with no API key, the teacher is a
strong classifier on the full 384-d embeddings — it plays the same role:
an accurate, expensive model whose knowledge we compress into a small one.

SETUP
-----
    pip install -r requirements.txt
RUN
---
    python distill_demo.py
=============================================================================
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

# Reuse the same messy-dataset generator and intents as the main pipeline.
from clarity_engine import build_dataset, INTENTS

rng = np.random.default_rng(11)
BG, FG, GRID = "#0d1117", "#e6edf3", "#21262d"
PALETTE = ["#f5c542", "#4cc9f0", "#52b788"]
N_CLASSES = len(INTENTS)


# --------------------------------------------------------------------------
# A tiny from-scratch softmax classifier that can train on SOFT targets
# (a full probability distribution per row), not just hard one-hot labels.
# --------------------------------------------------------------------------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TinyStudent:
    """Linear softmax head — trainable on soft OR hard targets via cross-entropy."""

    def __init__(self, n_in, n_out):
        self.W = rng.standard_normal((n_in, n_out)) * 0.01
        self.b = np.zeros((1, n_out))

    @property
    def n_params(self):
        return self.W.size + self.b.size

    def fit(self, X, target_dist, epochs=300, lr=0.5, batch=64):
        """target_dist: (N, C) probability rows (soft) or one-hot rows (hard)."""
        n = X.shape[0]
        for _ in range(epochs):
            order = rng.permutation(n)
            for s in range(0, n, batch):
                idx = order[s:s + batch]
                xb, qb = X[idx], target_dist[idx]
                p = softmax(xb @ self.W + self.b)
                # gradient of cross-entropy CE(q, p) wrt logits is (p - q)
                dz = (p - qb) / len(idx)
                self.W -= lr * (xb.T @ dz)
                self.b -= lr * dz.sum(axis=0, keepdims=True)
        return self

    def predict(self, X):
        return softmax(X @ self.W + self.b).argmax(axis=1)


def one_hot(labels, c):
    m = np.zeros((labels.size, c))
    m[np.arange(labels.size), labels] = 1
    return m


# --------------------------------------------------------------------------
def main():
    import os
    os.makedirs("visuals", exist_ok=True)

    print("Building dataset and embedding messages...")
    texts, true_y, _obs_y, _flip = build_dataset()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                     show_progress_bar=True)

    # A clean held-out test set nobody trains on.
    tr, te = train_test_split(np.arange(len(texts)), test_size=0.25,
                              random_state=0, stratify=true_y)

    # ---- TEACHER: big, accurate, expensive (full 384-d embeddings) ----------
    # Stand-in for the frontier LLM. It labels the training pool with SOFT
    # probabilities — that distribution is the knowledge we distil.
    print("Training the TEACHER (big model on full 384-d embeddings)...")
    teacher = LogisticRegression(max_iter=3000, C=6.0)
    teacher.fit(X[tr], true_y[tr])
    teacher_acc = accuracy_score(true_y[te], teacher.predict(X[te]))

    # Temperature-soften the teacher's logits (Hinton distillation). A higher T
    # spreads probability mass onto the runner-up classes, exposing the "dark
    # knowledge" — which wrong answers the teacher considers plausible. That
    # inter-class structure is exactly what a hard argmax label throws away.
    DISTILL_T = 4.0
    teacher_logits = teacher.decision_function(X[tr])        # (N, C) raw scores
    soft_targets = softmax(teacher_logits / DISTILL_T)        # softened soft labels
    hard_targets = one_hot(teacher.predict(X[tr]), N_CLASSES)  # argmax, info-poor

    # ---- STUDENT: tiny + fast (compressed 32-d features) --------------------
    # Compress embeddings to a few dims so the student is genuinely tiny and
    # capacity-limited — the regime where the teacher's soft "dark knowledge"
    # actually helps the student generalise beyond what hard labels convey.
    STUDENT_DIM = 8
    pca = PCA(n_components=STUDENT_DIM, random_state=0).fit(X[tr])
    Xs_tr, Xs_te = pca.transform(X[tr]), pca.transform(X[te])

    print(f"Distilling into a TINY STUDENT ({STUDENT_DIM}-d features)...")
    student_soft = TinyStudent(STUDENT_DIM, N_CLASSES).fit(Xs_tr, soft_targets)
    student_hard = TinyStudent(STUDENT_DIM, N_CLASSES).fit(Xs_tr, hard_targets)

    acc_soft = accuracy_score(true_y[te], student_soft.predict(Xs_te))
    acc_hard = accuracy_score(true_y[te], student_hard.predict(Xs_te))

    # ---- Report -------------------------------------------------------------
    teacher_params = teacher.coef_.size + teacher.intercept_.size
    print("\n==================  DISTILLATION RESULTS  ==================")
    print(f"  TEACHER  (384-d, {teacher_params:>5} params): {teacher_acc:.1%}")
    print(f"  STUDENT  on HARD labels ({STUDENT_DIM}-d, {student_hard.n_params:>3} params): {acc_hard:.1%}")
    print(f"  STUDENT  on SOFT labels ({STUDENT_DIM}-d, {student_soft.n_params:>3} params): {acc_soft:.1%}")
    print(f"  --> soft-label student keeps {acc_soft / teacher_acc:.0%} of teacher "
          f"accuracy at {teacher_params // student_soft.n_params}x fewer params")
    print("============================================================\n")

    # ---- Visual -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    names = ["Teacher\n(big, 384-d)", f"Student HARD\n(tiny, {STUDENT_DIM}-d)",
             f"Student SOFT\n(tiny, {STUDENT_DIM}-d)"]
    vals = [teacher_acc, acc_hard, acc_soft]
    bars = ax.bar(names, vals, color=PALETTE)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}",
                ha="center", color=FG, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_title("Knowledge distillation: a tiny student trained on SOFT labels\n"
                 "nearly matches a big teacher at a fraction of the size",
                 color=FG, fontsize=13, fontweight="bold", loc="left")
    ax.tick_params(colors=FG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    fig.savefig("visuals/4_distillation.png", dpi=130, facecolor=BG,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved visuals/4_distillation.png. Done.")


if __name__ == "__main__":
    main()
