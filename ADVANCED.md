# Advanced Guide — Extracting Accurate Information from Vague Data, and Training the Best Model with an LLM

This is the deep-dive companion to [`README.md`](README.md). The README shows a
single technique (Semantic Consensus Denoising) end to end. This document maps
the **whole modern playbook** — from cheap statistical denoising to LLM-in-the-loop
relabelling, knowledge distillation, and calibrated, abstaining models — and ends
with a concrete recommendation for **the best way to use an LLM to train for
maximum accuracy.**

> Honesty up front: the individual building blocks below are established
> techniques (confident learning, label propagation, distillation, conformal
> prediction, LoRA). What's advanced is **how they compose into one pipeline** —
> cheap methods clean the easy 90%, an LLM adjudicates the genuinely ambiguous
> 10%, and the final model trains on the result with calibrated confidence and an
> "I don't know" option.

---

## 0. The core idea: this is a *data* problem, not a *model* problem

The instinct when accuracy is low is to reach for a bigger model. On vague,
mislabelled data that's usually the wrong lever. If 30–50% of your labels are
wrong, a larger model just learns the mistakes more faithfully. **The highest-ROI
move is almost always to fix the labels before training.** Everything below is
about doing that correctly, cheaply, and at scale.

```
        ┌─────────────────────────────────────────────────────────────┐
        │  Raw, vague, mislabelled corpus  (millions of rows)          │
        └───────────────┬─────────────────────────────────────────────┘
                        ▼
   TIER 1  Embed + neighbour consensus   ───►  fixes the easy, dense cases (cheap)
                        ▼
   TIER 2  Confident Learning            ───►  statistically flags likely errors
                        ▼
   TIER 3  LLM adjudication (the hard 10%) ──►  reasons through genuine ambiguity
                        ▼
   TIER 4  Train final model on clean labels ► frozen encoder + head, or distil
                        ▼
   TIER 5  Calibrate + abstain + conformal ──► trustworthy, with an "unsure" path
```

The key economic insight: **route work by difficulty.** Don't send every row to an
expensive LLM. Send the cheap methods first; escalate only what they can't resolve.

---

## 1. Tier 1 — Embedding + neighbour consensus (the cheap workhorse)

Covered in the README. Embed every example, find each one's nearest neighbours in
meaning-space, and let them vote (similarity-weighted) on the label. Random label
noise averages out; the true class stays the plurality.

- **Strength:** near-zero marginal cost, fixes the dense, unambiguous majority.
- **Weakness:** fails exactly where the data is *genuinely* vague — sparse regions,
  short messages, examples that sit on a cluster boundary. Those are the cases the
  consensus flags as *low-confidence*. That flag is the hand-off to Tier 3.

---

## 2. Tier 2 — Confident Learning (find label errors statistically)

[Confident Learning](https://arxiv.org/abs/1911.00068) (the method behind
`cleanlab`) estimates the **joint distribution between noisy observed labels and
latent true labels** using a model's own out-of-sample predicted probabilities. It
then ranks which specific examples are most likely mislabelled — without you ever
seeing ground truth.

The recipe:
1. Train a quick classifier with cross-validation to get out-of-fold predicted
   probabilities for every row.
2. For each class pair *(given = i, likely-true = j)*, find examples confidently
   predicted *j* but labelled *i*.
3. Rank and prune (or relabel) the highest-probability errors.

This is complementary to Tier 1: consensus uses *geometry* (neighbours), confident
learning uses *a model's calibrated probabilities*. Agreement between the two is a
strong signal an example is genuinely mislabelled; disagreement is a candidate for
Tier 3.

---

## 3. Tier 3 — LLM-in-the-loop (the part most teams get wrong)

This is where a frontier LLM earns its cost — **not** by relabelling the whole
dataset (expensive and unnecessary), but by adjudicating the residual cases the
cheap tiers couldn't resolve. The implementation lives in
[`llm_denoise.py`](llm_denoise.py).

### 3.1 LLM-as-judge relabelling

Give the model the ambiguous message, the **label taxonomy with definitions**, and
— critically — **the nearest-neighbour examples and their labels as context**. Ask
it to assign the correct label *with reasoning*. The neighbour context turns a
blind guess into an informed decision and grounds the model in your actual data
distribution.

### 3.2 Make the output reliable: structured outputs + adaptive thinking

Two non-negotiables for production:

- **Structured outputs** (`output_config.format` / `messages.parse()` with a
  schema) guarantee the response is valid JSON matching your label set — no regex
  parsing, no "the model wrote prose instead of a label." Use an `enum`/`Literal`
  so the label is provably one of your classes (or `"uncertain"`).
- **Adaptive thinking** (`thinking: {"type": "adaptive"}` on Opus 4.8) lets the
  model reason through genuinely ambiguous cases and *decide for itself* how much
  reasoning each one needs — short for easy calls, deep for hard ones.

### 3.3 Self-consistency for the hardest cases

For the messages even the LLM finds borderline, sample the judgment a few times and
take the majority (or average the confidences). [Self-consistency](https://arxiv.org/abs/2203.11171)
measurably improves accuracy on ambiguous reasoning and, as a bonus, the spread of
the samples is itself a calibrated uncertainty signal.

### 3.4 Calibrated confidence + abstention

Force the model to emit a confidence and allow an `"uncertain"` verdict. Anything
below a threshold is **not** auto-labelled — it's queued for a human. This is the
difference between a pipeline that quietly fabricates labels and one that knows
what it doesn't know.

### 3.5 Active learning: spend the LLM budget where it moves accuracy

Rank the unresolved pool by *uncertainty* (low consensus confidence, high neighbour
disagreement, Tier-1/Tier-2 conflict) and send the LLM the top of that list first.
A few thousand well-chosen adjudications typically recover most of the accuracy a
full relabel would — at a fraction of the cost. This closes the loop: LLM verdicts
become new "anchors" that re-strengthen the cheap consensus for everything nearby.

---

## 4. Tier 4 — Training the final model: the modern recipe

Once the labels are clean, *what* do you train? Three tiers of ambition.

### 4.1 Baseline (do this first): frozen encoder + lightweight head

A pretrained sentence-transformer (frozen) turns text into vectors; a logistic
regression / small MLP on top does the classification. This is what the README
demo trains, and it already hits 90%+. The representation does the heavy lifting;
the head is cheap, fast, and retrainable in seconds when labels change.

### 4.2 The best accuracy/cost trade-off: **LLM-as-teacher knowledge distillation**

This is the single most important answer to *"what's the best way to use an LLM to
train for accuracy?"*

You can't serve a frontier LLM on every request in production — too slow, too
expensive. But you can use it **once, offline, as a teacher** to create a clean,
richly-labelled training set, then **distil** that knowledge into a small, fast
"student" model you own:

```
   Frontier LLM (teacher)
        │  labels + soft probabilities + rationales, offline, on a big unlabelled pool
        ▼
   Clean, large, high-quality training set
        │  train once
        ▼
   Small student model (MiniLM head, or a fine-tuned small transformer)
        │  serve in production: milliseconds, cents
        ▼
   LLM-level accuracy at small-model cost and latency
```

Why it wins:
- **Soft labels carry more signal than hard labels.** The teacher's probability
  distribution ("80% billing, 15% cancellation, 5% technical") teaches the student
  the class *geometry*, not just the answer — a well-known distillation result.
- **You amortise the LLM cost.** Pay the teacher once; serve the student forever.
- **You can generate data, not just label it.** Have the teacher write realistic
  paraphrases to densify sparse regions and rare intents (LLM augmentation), then
  label those too.

> ▶ **Runnable proof:** [`distill_demo.py`](distill_demo.py) demonstrates this end
> to end — a big teacher (384-d) labels the pool with temperature-softened soft
> probabilities, and a tiny student (8-d, **~40× fewer parameters**) distils it,
> recovering ~97% of the teacher's accuracy. The student trained on **soft** labels
> edges out an identical student trained on hard labels — the soft distribution
> carries inter-class "dark knowledge" that a hard argmax throws away. The gap
> widens as the teacher gets less certain and the student more capacity-limited.

### 4.3 When you need the last few points: fine-tune a small transformer (LoRA/QLoRA)

If a frozen encoder + head plateaus, fine-tune a small transformer end-to-end on
the cleaned/distilled labels. **LoRA / QLoRA** (parameter-efficient fine-tuning)
make this cheap: you train a few million adapter parameters instead of the whole
model, on a single GPU. Reach for this only after 4.1 and 4.2 — it's the smallest
lever for the most effort.

### 4.4 Weak supervision when you have *no* labels

If you're starting from an unlabelled pile, programmatic weak supervision
([Snorkel](https://www.snorkel.org/)) lets you write noisy *labelling functions*
(keyword rules, regexes — and now **LLM labelling functions**) and a label model
that reconciles their agreements/disagreements into probabilistic training labels.
LLMs make excellent labelling functions: cheap to write, broad coverage.

---

## 5. Tier 5 — Make the model *trustworthy*, not just accurate

Accuracy without calibration is dangerous: a model that's 70% accurate but reports
99% confidence everywhere will mislead every downstream decision.

### 5.1 Calibration (temperature scaling)

After training, fit a single temperature parameter on a held-out set so the
predicted probabilities match observed accuracy. Measure it with **Expected
Calibration Error (ECE)**. Cheap, and it makes confidence scores mean something.

### 5.2 Conformal prediction (the rigorous, modern option)

[Conformal prediction](https://arxiv.org/abs/2107.07511) gives a **distribution-free
coverage guarantee**: instead of one label, it returns a *set* of labels guaranteed
to contain the true one with, say, 95% probability. On an easy message the set has
one element; on a vague one it returns two or three — an honest, mathematically
backed "it's one of these." This is one of the most practical recent advances for
high-stakes NLP, and it composes with any underlying model.

> ▶ **Runnable proof:** [`conformal_demo.py`](conformal_demo.py) implements split
> conformal prediction with **randomised APS** (Adaptive Prediction Sets). It
> calibrates on a held-out split, then shows the coverage guarantee holding across
> several confidence levels and the set size *adapting* to difficulty: confident
> messages get a single label, while a genuinely vague `"ok"` expands to four
> candidate labels — exactly the signal to route it to a human.

### 5.3 Abstention + human-in-the-loop

The model should be allowed to say *"route to a human."* Combine the confidence
threshold (5.1), a large conformal set (5.2), or low neighbour consensus into an
abstain signal. The abstained cases are exactly your next active-learning batch —
the loop closes again.

---

## 6. Scaling all of this to large, real-world datasets

The README pipeline runs on a toy set in seconds; the same code runs on millions of
rows with four changes:

| Concern | At toy scale | At production scale |
|---|---|---|
| Nearest neighbours | exact (`O(n²)`) | **ANN index** — FAISS / ScaNN / hnswlib, sub-linear |
| Embedding | one CPU pass | **GPU batch** `encode(..., batch_size=256, device="cuda")`, cached to disk |
| LLM adjudication | one call per row | **Batches API** (50% cheaper, async) + **prompt caching** on the shared taxonomy/instructions |
| Storage | NumPy array | a **vector database**; shard and stream |

Two cost multipliers worth calling out:
- **Prompt caching:** the taxonomy, definitions, and few-shot examples are
  identical across every LLM call — cache that prefix and pay ~0.1× for it on every
  subsequent request. (`llm_denoise.py` places the `cache_control` breakpoint on
  the stable system prefix for exactly this reason.)
- **Batch processing:** label-cleaning is not latency-sensitive — run it through
  the Batches API at half price.

---

## 7. The recommended end-to-end pipeline ("the best way")

Putting it all together, here is the pipeline I'd actually ship:

1. **Embed** the whole corpus once (GPU, cached). Build an ANN index.
2. **Tier 1 consensus** to correct the dense, easy majority and produce a
   per-row confidence.
3. **Tier 2 confident learning** to cross-check and surface statistical label
   errors. Where Tier 1 and Tier 2 agree → trust it.
4. **Tier 3 LLM adjudication** (Opus 4.8, structured outputs, adaptive thinking,
   neighbour context, prompt-cached prefix, Batches API) on the **uncertain
   residual only**, ranked by an active-learning score. Allow abstention.
5. **Tier 4 train**: start with a frozen-encoder head on the cleaned labels; if you
   need production-grade speed *and* accuracy, **distil the LLM teacher into a small
   student** (4.2); fine-tune with LoRA (4.3) only if you must.
6. **Tier 5 calibrate** (temperature scaling), wrap predictions in **conformal
   sets**, and **abstain** below threshold — feeding abstentions back to step 4.

This is the synthesis the Clarity Engine is built around: **cheap methods for the
easy cases, a frontier LLM for the genuinely hard ones, and a small, calibrated,
abstaining model in production.**

---

## 8. How to measure that any of this worked

- **Clean held-out accuracy.** Keep a small, carefully-labelled gold test set the
  cleaning pipeline never touches. This is the only number that matters.
- **Label-recovery rate.** On synthetically corrupted data (as the README demo
  does), measure what fraction of corrupted labels you put back correctly.
- **Calibration (ECE).** Are the confidences honest?
- **Coverage & set size** (if using conformal). Does the 95% set actually contain
  the truth 95% of the time, and how big is it?
- **Cost per corrected label.** The metric that decides whether the LLM tier is
  worth it — it usually is, *because* you only send it the hard cases.

---

## 9. Pitfalls

- **Relabelling everything with the LLM.** Wasteful. Triage first.
- **Trusting LLM labels blindly.** Require structured output + confidence + an
  abstain option; spot-check against gold.
- **Leaking the test set.** The gold set must never pass through the cleaner.
- **Over-fitting to the teacher's quirks.** Distil from a *diverse* teacher set and
  validate the student on gold, not on teacher labels.
- **Uncalibrated confidence.** A confident wrong model is worse than an unsure one.
  Calibrate and allow abstention.

---

Built by [Muhammad Farooqi](https://github.com/mqfarooqi1). See
[`clarity_engine.py`](clarity_engine.py) for Tiers 1 & 4 and
[`llm_denoise.py`](llm_denoise.py) for the Tier 3 LLM adjudicator.
