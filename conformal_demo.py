"""
=============================================================================
 conformal_demo.py  —  Tier 5: trustworthy predictions with a guarantee
=============================================================================

Runnable proof of ADVANCED.md section 5.2 — conformal prediction.

A normal classifier hands you one label and a confidence number you can't
really trust. CONFORMAL PREDICTION instead returns a SET of labels with a
distribution-free guarantee: the true label is inside the set at least
(1 - alpha) of the time -- e.g. 90% -- no matter what model you used or what
the data looks like.

On an easy message the set has ONE element ("definitely billing"). On a
genuinely vague message it returns TWO or THREE ("it's billing or
cancellation") -- an honest, mathematically-backed "I'm not sure which."
That is exactly the signal you want for routing vague cases to a human.

METHOD (split conformal / APS — Adaptive Prediction Sets)
---------------------------------------------------------
  1. Fit any classifier; hold out a CALIBRATION set it didn't train on.
  2. For each calibration row, sort the class probabilities high->low and add
     them up until you reach the TRUE class; that cumulative mass is the score.
  3. q-hat = the (1 - alpha) empirical quantile of those scores.
  4. For a new x, sort classes high->low and keep adding them to the set until
     their cumulative probability reaches q-hat.
  Guarantee: P(true label in set) >= 1 - alpha. APS makes the set GROW on
  ambiguous inputs and shrink to a single label on confident ones.

SETUP
-----
    pip install -r requirements.txt
RUN
---
    python conformal_demo.py
=============================================================================
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

from clarity_engine import build_dataset, INTENTS

BG, FG, GRID = "#0d1117", "#e6edf3", "#21262d"
PALETTE = ["#f5c542", "#4cc9f0", "#f72585", "#52b788"]


_rng = np.random.default_rng(0)


def aps_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Randomised APS calibration score: prob mass ranked above the true class,
    plus a uniform fraction of the true class's own mass. Randomisation makes
    coverage land on the target instead of conservatively over-covering."""
    order = np.argsort(-probs, axis=1)
    scores = np.empty(len(probs))
    for i in range(len(probs)):
        cum_before = 0.0
        for c in order[i]:
            if c == labels[i]:
                scores[i] = cum_before + _rng.random() * probs[i, c]
                break
            cum_before += probs[i, c]
    return scores


def qhat_for_alpha(cal_scores: np.ndarray, alpha: float) -> float:
    """Conformal quantile with the finite-sample (n+1) correction."""
    n = len(cal_scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(cal_scores, level, method="higher"))


def prediction_sets(probs: np.ndarray, qhat: float) -> list[list[int]]:
    """Randomised APS set: add classes high->low until cumulative prob reaches
    q-hat, then drop the boundary class with the right probability so the set is
    as small as the guarantee allows."""
    order = np.argsort(-probs, axis=1)
    sets = []
    for i in range(len(probs)):
        cum, s = 0.0, []
        for c in order[i]:
            s.append(int(c))
            cum += probs[i, c]
            if cum >= qhat:
                # boundary randomisation: maybe drop the class that crossed
                if len(s) > 1 and _rng.random() <= (cum - qhat) / probs[i, c]:
                    s.pop()
                break
        sets.append(s)
    return sets


def main():
    import os
    os.makedirs("visuals", exist_ok=True)

    print("Building dataset and embedding messages...")
    texts, y, _obs, _flip = build_dataset()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                     show_progress_bar=True)

    # Three-way split: train the model / calibrate conformal / test coverage.
    idx = np.arange(len(texts))
    tr, rest = train_test_split(idx, test_size=0.5, random_state=0, stratify=y)
    cal, te = train_test_split(rest, test_size=0.5, random_state=0, stratify=y[rest])

    # Mild regularisation (low C) keeps the model from being overconfident, so
    # genuinely vague messages get spread-out probabilities -> larger APS sets.
    clf = LogisticRegression(max_iter=3000, C=5.0).fit(X[tr], y[tr])

    cal_probs = clf.predict_proba(X[cal])
    cal_scores = aps_scores(cal_probs, y[cal])     # APS calibration scores
    test_probs = clf.predict_proba(X[te])

    # ---- Coverage tracks the target across several confidence levels --------
    print("\n=============  CONFORMAL COVERAGE  =============")
    print("  target     achieved   avg set size")
    alphas = [0.20, 0.10, 0.05]
    achieved, set_sizes_at = [], {}
    for a in alphas:
        qh = qhat_for_alpha(cal_scores, a)
        sets = prediction_sets(test_probs, qh)
        covered = np.mean([y[te][i] in sets[i] for i in range(len(te))])
        avg_size = np.mean([len(s) for s in sets])
        achieved.append(covered)
        set_sizes_at[a] = [len(s) for s in sets]
        print(f"   {1 - a:.0%}        {covered:.1%}        {avg_size:.2f}")
    print("===============================================\n")

    # ---- Show the point: confident -> singleton, vague -> bigger set --------
    qh = qhat_for_alpha(cal_scores, 0.10)
    sets = prediction_sets(test_probs, qh)
    names = list(INTENTS)
    singles = [i for i in range(len(te)) if len(sets[i]) == 1]
    multis = [i for i in range(len(te)) if len(sets[i]) >= 2]

    print("Confident messages get a SINGLE-label set:")
    for i in singles[:3]:
        s = ", ".join(names[c] for c in sets[i])
        print(f'   "{texts[te[i]]}"  ->  {{{s}}}')
    print("\nVague messages get a MULTI-label set (route these to a human):")
    for i in multis[:3]:
        s = ", ".join(names[c] for c in sets[i])
        print(f'   "{texts[te[i]]}"  ->  {{{s}}}')
    print()

    # ---- Visual: coverage guarantee + set-size distribution -----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=BG)

    ax1.set_facecolor(BG)
    targets = [1 - a for a in alphas]
    ax1.plot([0.7, 1.0], [0.7, 1.0], "--", color=GRID, lw=1.5, label="ideal (y = x)")
    ax1.scatter(targets, achieved, s=180, color=PALETTE[0], zorder=5,
                edgecolors="white", linewidths=1.2)
    for t, ac in zip(targets, achieved):
        ax1.annotate(f"{ac:.0%}", (t, ac), color=FG, fontsize=9,
                     xytext=(6, -12), textcoords="offset points")
    ax1.set_title("The guarantee holds: achieved coverage >= target",
                  color=FG, fontsize=12, fontweight="bold", loc="left")
    ax1.set_xlabel("target coverage (1 - alpha)", color=FG)
    ax1.set_ylabel("achieved coverage on test", color=FG)
    ax1.tick_params(colors=FG)
    leg = ax1.legend(framealpha=0.15)
    for t in leg.get_texts():
        t.set_color(FG)
    for s in ax1.spines.values():
        s.set_color(GRID)

    ax2.set_facecolor(BG)
    sizes = set_sizes_at[0.10]
    bins = np.arange(0.5, max(sizes) + 1.5, 1)
    ax2.hist(sizes, bins=bins, color=PALETTE[1], alpha=0.9, rwidth=0.85)
    ax2.set_title("Prediction-set sizes at 90% coverage\n(1 = confident,  >=2 = vague)",
                  color=FG, fontsize=12, fontweight="bold", loc="left")
    ax2.set_xlabel("labels in the prediction set", color=FG)
    ax2.set_ylabel("messages", color=FG)
    ax2.set_xticks(range(1, max(sizes) + 1))
    ax2.tick_params(colors=FG)
    for s in ax2.spines.values():
        s.set_color(GRID)

    fig.suptitle("CONFORMAL PREDICTION  —  honest uncertainty with a coverage guarantee",
                 color=PALETTE[0], fontsize=15, fontweight="bold", x=0.012, ha="left")
    fig.savefig("visuals/5_conformal.png", dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("Saved visuals/5_conformal.png. Done.")


if __name__ == "__main__":
    main()
