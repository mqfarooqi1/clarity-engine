# Notes: cleaning noisy labels and what to train on

These are my longer notes behind the scripts in this repo — the stuff that
didn't fit in the README. It's a survey of the approaches I considered for
"the labels are wrong, what now," roughly in the order I'd actually reach for
them, plus where an LLM fits. Nothing here is novel; it's me writing down the
standard playbook so I remember it, with pointers to the small demos that make
each piece concrete.

Standard caveat: everything in this repo runs on synthetic data with uniform
random label noise, which is the friendly case. Real noise is correlated with
class and with hard examples, so treat the numbers as illustrative.

---

## Start by treating it as a data problem

When accuracy is low on noisy labels, the reflex is to reach for a bigger model.
Usually that's the wrong lever — a bigger model just fits the wrong labels more
faithfully. Fixing the labels first tends to pay off more for less effort. The
rest of these notes are about doing that without spending a fortune.

The cheap framing that organises everything below: **route work by difficulty.**
Don't send every row to an expensive LLM. Use cheap methods first and escalate
only what they can't resolve.

```
   raw noisy corpus
        │
        ▼  cheap: embed + neighbour voting        -> fixes the dense, easy cases
        ▼  cheap: confident learning              -> flags likely errors statistically
        ▼  expensive: LLM on the hard residual    -> reasons through real ambiguity
        ▼  train the final model on what's left
        ▼  calibrate + allow "unsure"             -> trustworthy, with a human fallback
```

---

## 1. Embedding + neighbour voting (the cheap workhorse)

Covered in the README and implemented in `clarity_engine.py`. Embed everything,
find each row's nearest neighbours, let them vote on the label weighted by
similarity. Random noise averages out; the true class stays the plurality.

- Cheap, and it handles the dense, unambiguous majority.
- It fails exactly where the data is genuinely vague — sparse regions, very short
  messages, boundary cases. Those show up as low-confidence rows, which is the
  natural hand-off point to the LLM step.

This is essentially label propagation / a soft k-NN smoother. I like it because
it's easy to inspect, not because it's clever.

---

## 2. Confident learning (a second opinion)

[Confident Learning](https://arxiv.org/abs/1911.00068) (the method in `cleanlab`)
estimates the joint distribution of noisy-vs-true labels from a model's
out-of-fold predicted probabilities, then ranks which specific rows are most
likely mislabelled. It's a useful cross-check on the neighbour method: one uses
geometry, the other uses a model's probabilities. Where they agree, I'd trust the
correction; where they disagree, that's a candidate for human review or the LLM.

I didn't wire this in (it's a one-liner with `cleanlab` if you want it), but it's
the obvious next thing to add.

---

## 3. Using an LLM for the part that's actually hard

The cheap tiers leave a residual of genuinely ambiguous rows. That's where an LLM
earns its cost — not relabelling the whole dataset, which is slow and expensive,
but adjudicating the cases the cheap methods couldn't. Implemented in
`llm_denoise.py`.

A few things that make this usable rather than a science project:

- **Give it the taxonomy with definitions and the row's neighbours as context.**
  The neighbours ground the model in your actual data instead of its priors.
- **Structured outputs.** Constrain the response to a schema with the label as an
  enum (plus an `"uncertain"` option). You get a valid label every time, no regex
  parsing.
- **Let it abstain.** Force a confidence and allow "uncertain." Anything below a
  threshold goes to a human rather than being silently guessed.
- **Self-consistency for the worst cases.** Sample a few times and take the
  majority; the spread is a rough uncertainty signal.
- **Spend the budget where it moves the needle.** Rank the unresolved pool by
  uncertainty and send the LLM the top of the list. A few thousand good
  adjudications usually recover most of what a full relabel would, far cheaper.

`llm_denoise.py` uses Claude with adaptive thinking, schema-constrained output,
neighbour context, prompt caching on the fixed instructions, and an abstain path.

---

## 4. What to actually train

Once the labels are reasonable:

### 4.1 Start simple: frozen encoder + light head

A pretrained sentence-transformer (frozen) plus a logistic-regression or small-MLP
head. This is what the README trains and it already gets ~90%. The representation
does the work; the head is cheap and retrains in seconds. Do this first.

### 4.2 If you need small-and-fast in production: distillation

You can't serve a frontier LLM on every request, but you can use it once, offline,
as a teacher to label a large pool with soft probabilities, then train a small
student to imitate those. The student ends up cheap to serve and close to the
teacher in accuracy.

The often-quoted advantage of *soft* labels over hard ones (the probability
distribution carries inter-class information a hard label throws away) is real but,
in my experience here, smaller and more situational than the literature suggests —
see the demo note below.

> `distill_demo.py` runs this: a 384-d teacher labels the pool with
> temperature-softened probabilities, and a tiny 8-d student (~45 params) imitates
> it, recovering ~97% of the teacher's accuracy. The soft-label student only beat
> the hard-label one once the student was small enough to be capacity-limited; on
> the easy, separable version they were a wash. So the soft-label benefit showed
> up, but it's not the headline it's sometimes made out to be.

### 4.3 If a frozen head plateaus: fine-tune small (LoRA/QLoRA)

Parameter-efficient fine-tuning of a small transformer on the cleaned labels.
Cheap (you train a few adapter layers, not the whole model), and worth it only
after 4.1 and 4.2 stop improving. Smallest lever, most effort.

### 4.4 If you have no labels at all: weak supervision

[Snorkel](https://www.snorkel.org/)-style: write noisy labelling functions (rules,
regexes, and now LLM prompts) and let a label model reconcile them into
probabilistic labels. LLMs make good labelling functions — cheap to write, broad
coverage.

---

## 5. Making the model trustworthy, not just accurate

A model that's 70% accurate but reports 99% confidence everywhere is worse than
useless downstream.

### 5.1 Calibration

Fit a single temperature parameter on a held-out set so predicted probabilities
match observed accuracy, and measure it with Expected Calibration Error (ECE).
Cheap, and it's the difference between confidence scores that mean something and
ones that don't. I didn't implement it here; I should have.

### 5.2 Conformal prediction

[Conformal prediction](https://arxiv.org/abs/2107.07511) gives a distribution-free
coverage guarantee: instead of one label it returns a set that contains the true
label with, say, 90% probability. Easy rows get a one-element set; vague ones get
two or three. It wraps any model.

> `conformal_demo.py` implements split conformal with randomised APS. Coverage
> holds across a few confidence levels, and set size adapts — confident messages
> get a single label, a vague "ok" expands to several. Two honest notes: plain
> (non-randomised) APS over-covered and produced bloated sets on this data, which
> is why I used the randomised variant; and coverage still runs a little above
> target because the base model is accurate enough to cover with small sets.

### 5.3 Abstention + human-in-the-loop

Let the model route to a human. Combine a low confidence, a large conformal set,
or low neighbour consensus into an abstain signal. The abstained rows are exactly
your next batch to label by hand — which feeds back into step 3.

---

## 6. Scaling to real datasets

The scripts run on a toy set in seconds. The same code runs on millions of rows
with a few changes:

- **Nearest neighbours:** swap exact k-NN for an ANN index (FAISS / ScaNN /
  hnswlib) past ~100k rows.
- **Embedding:** batch on a GPU and cache the vectors; you only embed once.
- **LLM step:** use the Batches API (about half price, async) and prompt-cache the
  shared taxonomy/instructions — that prefix is identical on every call.
- **Storage:** a vector DB; shard and stream.

---

## 7. How I'd measure whether any of this helped

- **Clean held-out accuracy** on a small, carefully-labelled gold set the cleaning
  pipeline never touches. This is the number that matters; everything else is a
  proxy.
- **Label-recovery rate** on synthetically corrupted data (what the README
  reports).
- **Calibration (ECE)** — are the confidences honest?
- **Coverage and set size** if you use conformal.
- **Cost per corrected label** — the metric that decides whether the LLM tier is
  worth turning on.

---

## 8. Things that bit me / things to watch

- Relabelling everything with the LLM is wasteful. Triage first.
- Don't trust LLM labels blind — require structured output, a confidence, and an
  abstain option, and spot-check against gold.
- Keep the gold test set away from the cleaner, or you'll fool yourself.
- The soft-label distillation benefit is data- and capacity-dependent; don't
  assume it.
- Uncalibrated confidence is the quiet failure mode. Calibrate, and allow "unsure."

---

Muhammad Farooqi · https://github.com/mqfarooqi1 · see `clarity_engine.py`,
`llm_denoise.py`, `distill_demo.py`, and `conformal_demo.py` for the code.
