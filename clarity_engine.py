"""
Label cleaning by similarity-weighted k-NN voting, then training on the result.

The idea: a single label is noisy, but meaning is more stable. Embed each
message, find its nearest neighbours, and let them vote on the label (weighted
by cosine similarity). Random label noise scatters across classes while the true
class stays the plurality, so the vote tends to recover the right label. Rows
where the neighbourhood disagrees get flagged as low-confidence.

This is basically label propagation / a soft k-NN smoother — nothing novel, but
easy to inspect. To check whether it helps, we train a classifier on the raw
noisy labels vs. the corrected labels and score both on a clean held-out set.

Everything runs on a synthetic dataset with uniform random label noise (the
friendly case). For real data, swap the generator for a loader and the exact
k-NN for an ANN index; see README.md and ADVANCED.md.

    pip install -r requirements.txt
    python clarity_engine.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

rng = np.random.default_rng(7)

# ---- look & feel ----------------------------------------------------------
BG, FG, GRID = "#0d1117", "#e6edf3", "#21262d"
PALETTE = ["#f5c542", "#4cc9f0", "#f72585", "#52b788", "#b07cf7"]  # one per intent
OUT = "visuals"

# ---- the five intents we want to recover ----------------------------------
INTENTS = ["billing", "technical", "shipping", "cancellation", "praise"]


# --------------------------------------------------------------------------
# 1. Build a realistic, deliberately MESSY dataset.
# --------------------------------------------------------------------------
TEMPLATES = {
    "billing": [
        "I was charged twice for {x}", "why is my invoice so high this month",
        "can I get a refund for {x}", "my card got billed but I cancelled",
        "there is a strange fee on my statement", "the payment did not go through for {x}",
        "you overcharged me again", "I want my money back for {x}",
    ],
    "technical": [
        "the app keeps crashing when I open {x}", "I get an error loading {x}",
        "the website wont load at all", "login is broken on my phone",
        "everything freezes after the update", "the {x} button does nothing",
        "it says server error every time", "the page is completely blank",
    ],
    "shipping": [
        "where is my package it is late", "my order has not arrived yet",
        "the tracking number does not update", "I got the wrong item in my box",
        "delivery was supposed to be yesterday", "the parcel is stuck in transit",
        "can you ship {x} faster", "my shipment shows delivered but its not here",
    ],
    "cancellation": [
        "I want to cancel my subscription", "please close my account",
        "how do I unsubscribe from {x}", "stop charging me I am leaving",
        "cancel my plan effective today", "I no longer want the service",
        "delete my account permanently", "end my membership please",
    ],
    "praise": [
        "thank you so much this is great", "I love the new {x} feature",
        "your support team is amazing", "best service I have used in years",
        "this app made my day so much easier", "absolutely fantastic experience",
        "you guys are wonderful keep it up", "really happy with {x}",
    ],
}
FILLERS = ["my order", "the premium plan", "this", "your product", "the dashboard",
           "checkout", "the mobile app", "delivery"]
# Short, genuinely VAGUE messages that are hard for anyone to label.
VAGUE = ["help", "this is not working", "?", "please fix", "not happy",
         "what is going on", "see above", "same issue", "still waiting", "ok"]


def messy(text: str) -> str:
    """Inject the kind of noise real user text has: case, typos, padding."""
    if rng.random() < 0.4:
        text = text.lower()
    if rng.random() < 0.25:  # drop a vowel somewhere (typo)
        i = next((k for k, c in enumerate(text) if c in "aeiou"), -1)
        if i > 0:
            text = text[:i] + text[i + 1:]
    if rng.random() < 0.2:
        text = text + rng.choice([" thanks", " asap", " !!", " ..."])
    return text


def build_dataset(n_per_intent=170, noise_rate=0.45):
    texts, true_y = [], []
    for ci, intent in enumerate(INTENTS):
        for _ in range(n_per_intent):
            tmpl = rng.choice(TEMPLATES[intent])
            texts.append(messy(tmpl.replace("{x}", rng.choice(FILLERS))))
            true_y.append(ci)
    # sprinkle in genuinely vague messages (assign a random true intent)
    for v in VAGUE * 8:
        texts.append(messy(v))
        true_y.append(int(rng.integers(len(INTENTS))))

    true_y = np.array(true_y)

    # The OBSERVED labels are what a sloppy human gave us: flip `noise_rate`.
    obs_y = true_y.copy()
    flip = rng.random(len(obs_y)) < noise_rate
    obs_y[flip] = rng.integers(0, len(INTENTS), size=flip.sum())
    return np.array(texts), true_y, obs_y, flip


# --------------------------------------------------------------------------
# 2. THE CORE IDEA: recover truth by similarity-weighted neighbour voting.
# --------------------------------------------------------------------------
def semantic_consensus(embeds, observed_labels, k=15):
    """For each row, let its k nearest neighbours vote (weighted by similarity).

    Returns corrected labels, a 0..1 confidence, and the full vote matrix.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeds)
    dist, idx = nn.kneighbors(embeds)
    dist, idx = dist[:, 1:], idx[:, 1:]          # drop self
    sim = 1.0 - dist                              # cosine similarity as weight

    n, c = len(embeds), len(INTENTS)
    votes = np.zeros((n, c))
    for i in range(n):
        for j, w in zip(idx[i], sim[i]):
            votes[i, observed_labels[j]] += max(w, 0.0)
    votes /= votes.sum(axis=1, keepdims=True) + 1e-9
    corrected = votes.argmax(axis=1)
    confidence = votes.max(axis=1)
    return corrected, confidence, votes


# --------------------------------------------------------------------------
# 3. Train + evaluate. Does denoising actually buy us accuracy?
# --------------------------------------------------------------------------
def train_eval(X_tr, y_tr, X_te, y_te, sample_weight=None):
    clf = LogisticRegression(max_iter=2000, C=4.0)
    clf.fit(X_tr, y_tr, sample_weight=sample_weight)
    pred = clf.predict(X_te)
    return accuracy_score(y_te, pred), pred


# --------------------------------------------------------------------------
# 4. Beautiful visuals.
# --------------------------------------------------------------------------
def _style(ax, title):
    ax.set_title(title, color=FG, fontsize=12, fontweight="bold", loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


def scatter(ax, xy, labels, title, alpha=0.85, sizes=18):
    for ci, intent in enumerate(INTENTS):
        m = labels == ci
        ax.scatter(xy[m, 0], xy[m, 1], s=sizes, c=PALETTE[ci], alpha=alpha,
                   edgecolors="none", label=intent)
    _style(ax, title)


def viz_problem(xy, true_y, obs_y, flip):
    fig = plt.figure(figsize=(14, 6.2), facecolor=BG)
    gs = GridSpec(1, 2, figure=fig, wspace=0.08)
    ax1 = fig.add_subplot(gs[0], facecolor=BG)
    scatter(ax1, xy, true_y, "Hidden TRUTH  (what each message really is)")
    ax2 = fig.add_subplot(gs[1], facecolor=BG)
    scatter(ax2, xy, obs_y, f"OBSERVED labels  ({flip.mean():.0%} are wrong)", alpha=0.85)
    # mark the mislabelled points
    ax2.scatter(xy[flip, 0], xy[flip, 1], s=70, facecolors="none",
                edgecolors="white", linewidths=0.7, alpha=0.5, label="mislabelled")
    leg = ax2.legend(loc="upper right", fontsize=8, framealpha=0.2)
    for t in leg.get_texts():
        t.set_color(FG)
    fig.suptitle("STEP 1  —  A vague dataset: meaning is clean, labels are not",
                 color=PALETTE[0], fontsize=16, fontweight="bold", x=0.012, ha="left")
    fig.savefig(f"{OUT}/1_problem.png", dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def viz_denoising(xy, corrected, confidence, flip, recovered):
    fig = plt.figure(figsize=(14, 6.2), facecolor=BG)
    gs = GridSpec(1, 2, width_ratios=[1.4, 1], figure=fig, wspace=0.15)
    ax1 = fig.add_subplot(gs[0], facecolor=BG)
    # colour by corrected label, brightness by confidence
    for ci in range(len(INTENTS)):
        m = corrected == ci
        ax1.scatter(xy[m, 0], xy[m, 1], s=12 + 60 * confidence[m],
                    c=PALETTE[ci], alpha=0.35 + 0.6 * confidence[m], edgecolors="none")
    _style(ax1, "Consensus-CORRECTED labels  (size/brightness = confidence)")

    ax2 = fig.add_subplot(gs[1], facecolor=BG)
    ax2.hist(confidence, bins=25, color=PALETTE[1], alpha=0.9)
    ax2.axvline(0.6, color=PALETTE[2], ls="--", lw=2)
    ax2.text(0.61, ax2.get_ylim()[1] * 0.9, "confidence gate", color=PALETTE[2],
             fontsize=9)
    ax2.set_title("How sure is the consensus?", color=FG, fontsize=12,
                  fontweight="bold", loc="left")
    ax2.set_xlabel("confidence", color=FG); ax2.tick_params(colors=FG)
    for s in ax2.spines.values():
        s.set_color(GRID)
    fig.suptitle(f"STEP 2  —  Neighbour voting recovered "
                 f"{recovered:.0%} of the corrupted labels",
                 color=PALETTE[0], fontsize=16, fontweight="bold", x=0.012, ha="left")
    fig.savefig(f"{OUT}/2_denoising.png", dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def viz_results(accs, cms):
    fig = plt.figure(figsize=(14, 6.2), facecolor=BG)
    gs = GridSpec(1, 3, width_ratios=[1.1, 1, 1], figure=fig, wspace=0.25)

    ax = fig.add_subplot(gs[0], facecolor=BG)
    names = list(accs.keys())
    vals = [accs[n] for n in names]
    bars = ax.bar(names, vals, color=[GRID, PALETTE[3], PALETTE[0]])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}",
                ha="center", color=FG, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_title("Test accuracy on CLEAN data", color=FG, fontsize=12,
                 fontweight="bold", loc="left")
    ax.tick_params(colors=FG, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)

    for k, (label, cm) in enumerate(cms.items()):
        axc = fig.add_subplot(gs[k + 1], facecolor=BG)
        axc.imshow(cm, cmap="magma")
        axc.set_title(label, color=FG, fontsize=11, fontweight="bold", loc="left")
        axc.set_xticks(range(len(INTENTS))); axc.set_yticks(range(len(INTENTS)))
        axc.set_xticklabels(INTENTS, rotation=45, ha="right", color=FG, fontsize=7)
        axc.set_yticklabels(INTENTS, color=FG, fontsize=7)
        for s in axc.spines.values():
            s.set_color(GRID)
    fig.suptitle("STEP 3  —  Training on corrected data gives more accurate answers",
                 color=PALETTE[0], fontsize=16, fontweight="bold", x=0.012, ha="left")
    fig.savefig(f"{OUT}/3_results.png", dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    import os
    os.makedirs(OUT, exist_ok=True)

    print("Building a deliberately messy customer-message dataset...")
    texts, true_y, obs_y, flip = build_dataset()
    print(f"  {len(texts)} messages | {flip.mean():.0%} of labels are corrupted")

    print("Embedding messages into meaning-space (sentence-transformers)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeds = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                          show_progress_bar=True)

    # Hold out a CLEAN test set (true labels) to judge everyone fairly.
    idx = np.arange(len(texts))
    tr, te = train_test_split(idx, test_size=0.25, random_state=0, stratify=true_y)

    print("Running Semantic Consensus Denoising...")
    corrected, confidence, _ = semantic_consensus(embeds, obs_y, k=15)

    # How many of the originally-corrupted labels did we put back correctly?
    recovered = (corrected[flip] == true_y[flip]).mean()

    print("Training three models and scoring on the clean test set...")
    acc_noisy, pred_noisy = train_eval(embeds[tr], obs_y[tr], embeds[te], true_y[te])
    acc_corr, _ = train_eval(embeds[tr], corrected[tr], embeds[te], true_y[te])
    gate = confidence[tr] >= 0.6
    acc_gate, pred_gate = train_eval(embeds[tr][gate], corrected[tr][gate],
                                     embeds[te], true_y[te])

    accs = {
        "raw noisy\nlabels": acc_noisy,
        "consensus\ncorrected": acc_corr,
        "corrected +\nconfidence gate": acc_gate,
    }
    print("\n================  RESULTS  ================")
    print(f"  recovered corrupted labels : {recovered:.1%}")
    print(f"  model on RAW noisy labels   : {acc_noisy:.1%}")
    print(f"  model on CORRECTED labels   : {acc_corr:.1%}")
    print(f"  corrected + confidence gate : {acc_gate:.1%}")
    print("==========================================\n")

    print("Rendering visuals...")
    xy = PCA(n_components=2, random_state=0).fit_transform(embeds)
    viz_problem(xy, true_y, obs_y, flip)
    viz_denoising(xy, corrected, confidence, flip, recovered)
    cms = {
        "Before (noisy)": confusion_matrix(true_y[te], pred_noisy),
        "After (gated)":  confusion_matrix(true_y[te], pred_gate),
    }
    viz_results(accs, cms)
    print(f"Saved 3 figures to '{OUT}/'. Done.")


if __name__ == "__main__":
    main()
