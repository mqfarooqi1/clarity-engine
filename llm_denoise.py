"""
Relabel the ambiguous residual with an LLM.

The k-NN voting in clarity_engine.py handles the dense, easy cases. What's left
is the genuinely ambiguous rows — short, contradictory, or boundary cases where
the neighbourhood vote is low-confidence. The point of using an LLM here is to
spend it only on that residual, not on the whole dataset.

For each message it sends Claude the row, the label taxonomy with definitions,
and the row's noisy neighbours as context, and asks for a label. A few details
that make it usable:

  * Structured output — the label is constrained to the known classes or
    "uncertain", so you always get a valid value and never parse free text.
  * Adaptive thinking — lets the model spend more reasoning on hard rows.
  * Neighbour context — grounds the decision in your data, not the model's priors.
  * Prompt caching — the taxonomy/instructions are identical every call, so that
    prefix is cached on repeat requests.
  * Abstention — low-confidence rows return "uncertain" and route to a human.

For large jobs you'd run this through the Batches API rather than a loop.

    pip install anthropic pydantic
    set ANTHROPIC_API_KEY=...        (PowerShell:  $env:ANTHROPIC_API_KEY="...")
    python llm_denoise.py
"""

from __future__ import annotations

from typing import Literal, Optional

import anthropic
from pydantic import BaseModel, Field


MODEL = "claude-opus-4-8"

# The label taxonomy WITH definitions. Giving the model crisp definitions is
# what turns a guess into a grounded decision.
INTENTS = {
    "billing": "Charges, refunds, invoices, payments, overcharges, fees.",
    "technical": "Bugs, crashes, errors, things not loading or not working.",
    "shipping": "Deliveries, packages, tracking, late or wrong orders.",
    "cancellation": "Cancelling a subscription, closing/deleting an account, unsubscribing.",
    "praise": "Compliments, thanks, positive feedback.",
}

# A Literal type makes the label provably one of the classes (or "uncertain").
# Pydantic turns this into a JSON-schema enum the API enforces.
IntentLabel = Literal[
    "billing", "technical", "shipping", "cancellation", "praise", "uncertain"
]


class LabelDecision(BaseModel):
    """The structured verdict the model must return for one message."""
    label: IntentLabel = Field(
        description="The single best intent, or 'uncertain' if genuinely unclear."
    )
    confidence: float = Field(
        description="Calibrated confidence from 0.0 to 1.0 in the chosen label."
    )
    reasoning: str = Field(
        description="One or two sentences justifying the label."
    )


# Stable system prefix: taxonomy + instructions. Identical on every call, so we
# mark it with cache_control and the API serves it from cache after the first
# request (huge saving when adjudicating thousands of messages).
def _system_blocks() -> list[dict]:
    taxonomy = "\n".join(f"  - {name}: {desc}" for name, desc in INTENTS.items())
    text = (
        "You are an expert data-labeller cleaning a noisy customer-support "
        "dataset. Classify each message into exactly one intent. The intents "
        "and their definitions are:\n\n"
        f"{taxonomy}\n\n"
        "Rules:\n"
        "  - Judge by MEANING, not keywords. Messages are short, messy, and "
        "may have typos.\n"
        "  - You will be shown the message's nearest semantic neighbours and "
        "their (possibly noisy) labels as context. Weigh them, but trust the "
        "message's own meaning over a single neighbour.\n"
        "  - If the message is genuinely too vague to classify confidently, "
        "return 'uncertain' rather than guessing. An honest 'uncertain' is "
        "better than a confident wrong label.\n"
        "  - Always return calibrated confidence: high only when you are sure."
    )
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _user_text(message: str, neighbours: Optional[list[tuple[str, str]]]) -> str:
    parts = [f'MESSAGE TO LABEL:\n"{message}"']
    if neighbours:
        ctx = "\n".join(f'  - [{lab}] "{txt}"' for txt, lab in neighbours)
        parts.append(f"\nNEAREST SEMANTIC NEIGHBOURS (text + noisy label):\n{ctx}")
    parts.append("\nReturn the intent, your confidence, and brief reasoning.")
    return "\n".join(parts)


def adjudicate(
    message: str,
    neighbours: Optional[list[tuple[str, str]]] = None,
    client: Optional[anthropic.Anthropic] = None,
) -> LabelDecision:
    """Ask Claude to label one ambiguous message, grounded in its neighbours.

    `neighbours` is a list of (text, noisy_label) tuples from the semantic
    consensus step. Returns a validated LabelDecision (label may be 'uncertain').
    """
    client = client or anthropic.Anthropic()

    response = client.messages.parse(
        model=MODEL,
        max_tokens=2000,
        thinking={"type": "adaptive"},        # Opus 4.8: reason as much as needed
        system=_system_blocks(),               # cached stable prefix
        messages=[{"role": "user", "content": _user_text(message, neighbours)}],
        output_format=LabelDecision,           # guaranteed schema-valid result
    )
    return response.parsed_output


def adjudicate_batch(
    items: list[dict],
    confidence_gate: float = 0.6,
    client: Optional[anthropic.Anthropic] = None,
) -> list[dict]:
    """Adjudicate many messages. Each item: {"text": str, "neighbours": [...]}.

    Returns the input enriched with 'label', 'confidence', 'reasoning', and an
    'accept' flag (False => abstained, route to a human). For very large jobs,
    swap this loop for the Batches API (50% cheaper, async) — see ADVANCED.md §6.
    """
    client = client or anthropic.Anthropic()
    out = []
    for it in items:
        d = adjudicate(it["text"], it.get("neighbours"), client=client)
        out.append({
            **it,
            "label": d.label,
            "confidence": d.confidence,
            "reasoning": d.reasoning,
            "accept": d.label != "uncertain" and d.confidence >= confidence_gate,
        })
    return out


# --------------------------------------------------------------------------
def _demo() -> None:
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the live demo. Showing the plan only.\n")
        print("This module would send these vague messages to Claude Opus 4.8,")
        print("each with its semantic neighbours as context, and return a")
        print("schema-validated {label, confidence, reasoning} — abstaining when unsure.")
        return

    # A few genuinely ambiguous messages the cheap tiers would flag as low-confidence,
    # each paired with the noisy neighbours semantic consensus found nearby.
    residual = [
        {"text": "charged again after I left",
         "neighbours": [("stop charging me I am leaving", "cancellation"),
                        ("you overcharged me again", "billing"),
                        ("I want to cancel my subscription", "cancellation")]},
        {"text": "still not working",
         "neighbours": [("the app keeps crashing", "technical"),
                        ("where is my package", "shipping"),
                        ("it says server error every time", "technical")]},
        {"text": "thanks but the box was empty",
         "neighbours": [("thank you so much this is great", "praise"),
                        ("I got the wrong item in my box", "shipping"),
                        ("my order has not arrived yet", "shipping")]},
    ]

    print(f"Adjudicating {len(residual)} ambiguous messages with {MODEL}...\n")
    results = adjudicate_batch(residual)
    for r in results:
        flag = "ACCEPT" if r["accept"] else "ABSTAIN -> human"
        print(f'"{r["text"]}"')
        print(f'   -> {r["label"]}  (conf {r["confidence"]:.2f})  [{flag}]')
        print(f'      {r["reasoning"]}\n')


if __name__ == "__main__":
    _demo()
