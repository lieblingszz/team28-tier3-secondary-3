"""
tier3_pipeline.py
-----------------
Tier 3 Silicon Sample Benchmark Pipeline
Group: Farah Adeeba, Jing Ma, Max Pellert, Marcia Ferreira

Approach: Query-specific RAG on scientific literature + TWO-STEP
relative-scoring prediction.

Step 1 — RELATIVE SCORING: Given all 16 interventions, score each one
         on a -3 (meaningfully harmful) to +3 (strongest positive effect)
         scale for a given outcome, with ties allowed. This still forces
         the model to compare interventions against each other, but no
         longer fabricates a full 1-16 ordering when several interventions
         are genuinely close or have ~zero effect.

Step 2 — EFFECT SIZES: For each intervention, estimate the actual effect
         size. The Step 1 score is passed along only as *supporting*
         comparative evidence — the model is told to estimate the
         absolute ATE independently from the intervention content and
         the retrieved literature, not to mechanically scale the score.

RAG design: retrieval is query-specific rather than one shared context
reused for every outcome/intervention. Step 1 (which compares all 16
interventions at once for a single outcome) uses an outcome-specific
query. Step 2 (per-intervention effect estimate) uses an
intervention x outcome specific query, so e.g. the "Consensus"
intervention pulls trust/consensus literature for the trust_post outcome,
but behavior/donation literature for the donation outcome.

Usage:
    python tier3_pipeline.py --dry_run --quick
    python tier3_pipeline.py --model gpt4o --n_samples 3 --quick
    python tier3_pipeline.py --model gpt4o --n_samples 5
"""

import os
import sys
import json
import csv
import time
import re
import random
import argparse
import threading
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from rag_vector_db import VectorRAG

# ── Raw-response logging (K.2 registration requirement) ───────────────
# call_api() can optionally append one JSON line per API call to a jsonl
# file -- the complete, unprocessed model response, before any parsing --
# for provenance/audit purposes. Off by default (log_path=None) so this
# is fully backward compatible with every existing call site. A single
# lock serializes writes since call sites may fire concurrently (parallel
# thread pool in tier3_pipeline_parallel_fulltext.py).
_LOG_LOCK = threading.Lock()


def log_raw_call(log_path: str, model_key: str, resolved_model: str | None,
                  prompt: str, response: str, meta: dict | None = None) -> None:
    if not log_path:
        return
    record = {
        "ts":             datetime.now().isoformat(timespec="seconds"),
        "model_key":      model_key,
        "resolved_model": resolved_model,  # exact snapshot the API actually served, when available
        "prompt":         prompt,
        "response":       response,
    }
    if meta:
        record.update(meta)
    with _LOG_LOCK:
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

# ── Lab repo path ─────────────────────────────────────────────────────
LAB_REPO = Path("./LLMmegastudy-main/simulation")

# ── API config ────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "gpt4o":  {"provider": "openai",    "model": "gpt-4o",            "env": "OPENAI_API_KEY"},
    "gpt5":   {"provider": "openai",    "model": "gpt-5.4",           "env": "OPENAI_API_KEY",
               "reasoning_model": True},
    "claude": {"provider": "anthropic", "model": "claude-sonnet-4-6",  "env": "ANTHROPIC_API_KEY"},
}
# GPT-5 (and the o1/o3 reasoning family) reject the old Chat Completions
# params: no `max_tokens` (must be `max_completion_tokens`), and no custom
# `temperature` (only the default, 1, is accepted). "reasoning_model": True
# switches both of those off below. Reasoning tokens also count against
# max_completion_tokens, so a small pad isn't reliable -- these models can
# spend anywhere from a few hundred to several thousand tokens "thinking"
# before writing anything visible, and 300 extra was nowhere near enough
# in practice. Two changes fix it: cap how much it's allowed to think via
# reasoning_effort="low" (short scoring/estimation prompts like ours don't
# need deep reasoning), AND pad generously so a hidden reasoning burst
# still leaves room for the actual answer.
REASONING_TOKEN_PADDING = 2000
# "none" is a real off-switch for gpt-5.4's reasoning -- OpenAI's docs
# describe it as "No chain-of-thought. Fastest, cheapest. Behaves like a
# non-thinking model." (an older revision of this comment said "minimal"
# was the closest thing to off because no true off-switch existed --
# that's no longer accurate for this model). With "none", the model
# should skip its internal reasoning pass entirely, same as gpt-4o.
# Still goes through the reasoning_model branch below because gpt-5.4
# requires max_completion_tokens (not max_tokens) and a fixed temperature
# regardless of reasoning_effort -- that's a separate API-compatibility
# quirk of the model family, unrelated to how much it reasons.
REASONING_EFFORT        = "none"   # "none" | "minimal" | "low" | "medium" | "high" | "xhigh"

# ── Calibration ────────────────────────────────────────────────────────
# Was 0.56 (Ashokkumar et al. 2026's PRIMARY-archive slope). Our target is
# a megastudy, not the primary archive, so we tested the paper's own
# megastudy-specific figure (~0.34-0.39) against real Voelkel et al. (2025)
# ground truth (4 outcomes, n=40 predicted/real pairs) instead of assuming.
# Empirically-best-fit b on that real data = 0.321 (pooled RMSE 1.110);
# 0.35 (RMSE 1.123) is nearly identical and matches the paper's own
# megastudy range, so we use that as the working value. Both clearly beat
# 0.56 (RMSE 1.255) on real data. This is a GLOBAL constant, not
# per-intervention -- we have no real ground truth for our actual 16
# interventions (Voelkel's conditions are a different stimulus set), so
# there's no data to fit a finer-grained version of this specific number.
CALIBRATION_FACTOR = 0.35

# ── Reverse-coded official outcomes ───────────────────────────────────
# Per the benchmark's survey_items.json ("outcomes" -> rule), most raw
# items map to their official outcome via "copy" or "mean" (no sign
# change). funding_perceptions is the one exception: rule="reverse100"
# on funding_5 ("...spending too much/too little...", 0=far too little,
# 100=far too much). The preregistration confirms the official scale is
# reverse-coded so higher = supports more funding, and Tier 3 ATEs must
# be submitted "already reverse-coded if applicable". Our prompts show
# the model funding_5's RAW wording/scale (so the model predicts the
# raw-direction effect), so the predicted ATE must be sign-flipped here
# before it reaches the official outcome — reversing a 100-x level
# transform turns a treatment-vs-control DIFFERENCE into its negation,
# not into (100 - effect).
REVERSE_CODED_OUTCOMES = {"funding_perceptions"}

RETRY_LIMIT = 3
RETRY_DELAY = 2

# ── Official Tier 3 outcome names + aggregation rules ─────────────────
# Maps official variable name -> list of raw item keys to average
# Scale range: used to convert to percentage points of scale range
OFFICIAL_OUTCOMES = {
    "trust_multidimensional": {
        "items": ["trust_competence_1","trust_competence_2","trust_competence_3",
                  "trust_integrity_1","trust_integrity_2","trust_integrity_3",
                  "trust_benevolence_1","trust_benevolence_2","trust_benevolence_3",
                  "trust_openness_1","trust_openness_2","trust_openness_3"],
        "scale_range": 100,  # 0-100
    },
    "trust_post": {
        "items": ["trust_post_1"],
        "scale_range": 100,
    },
    "distrust_post": {
        "items": ["distrust_1"],
        "scale_range": 100,
    },
    "funding_perceptions": {
        "items": ["funding_5"],
        "scale_range": 100,
    },
    "policy_role_mean": {
        "items": ["policy_1","policy_2","policy_3","policy_4"],
        "scale_range": 100,
    },
    "inst_trust_mean": {
        "items": ["inst_trust_epa","inst_trust_nasa","inst_trust_noaa",
                  "inst_trust_uni","inst_trust_gov"],
        "scale_range": 100,
    },
    "belief_post": {
        "items": ["belief_post_1"],
        "scale_range": 100,
    },
    "concern_mean": {
        "items": ["concern_1","concern_2","concern_3"],
        "scale_range": 100,
    },
    "policy_general": {
        "items": ["policy_general_1"],
        "scale_range": 100,
    },
    "policy_specific_mean": {
        "items": ["policy_specific_1","policy_specific_2","policy_specific_3",
                  "policy_specific_4","policy_specific_5","policy_specific_6",
                  "policy_specific_7"],
        "scale_range": 100,
    },
    "behavior_mean": {
        "items": ["individual_meat","individual_transport","individual_solar",
                  "individual_fly","individual_talk","individual_donate"],
        "scale_range": 100,
    },
    "donation_ams": {
        "items": ["donation"],
        "scale_range": 10,   # $0-10
    },
    "newsletter_signup": {
        "items": ["newsletter"],
        "scale_range": 1,    # 0-1 binary
    },
}

# Reverse map: raw item key -> official outcome name, used to build a
# human-readable "construct" label for RAG queries (e.g. "donation_ams"
# for the "donation" item, "trust_multidimensional" for the trust items).
ITEM_TO_OUTCOME = {
    item: out_name
    for out_name, spec in OFFICIAL_OUTCOMES.items()
    for item in spec["items"]
}


def get_construct_label(item_key: str) -> str:
    """Human-readable construct name for the given raw item, for use in
    RAG retrieval queries. Falls back to the raw item key if the item
    isn't mapped to an official outcome (e.g. items outside the 13
    preregistered outcomes)."""
    outcome_name = ITEM_TO_OUTCOME.get(item_key, item_key)
    return outcome_name.replace("_", " ")


# ════════════════════════════════════════════════════════════════════
# 1. LOAD OFFICIAL DATA
# ════════════════════════════════════════════════════════════════════

def load_stimuli() -> dict:
    data = json.loads((LAB_REPO / "data/survey_items.json").read_text())
    stimuli = {}
    for name, filename in data["stimuli"].items():
        path = LAB_REPO / "stimuli" / filename
        if path.exists():
            stimuli[name] = path.read_text().strip()
    print(f"[STIMULI] Loaded {len(stimuli)} official stimuli")
    return stimuli


def load_survey_items() -> dict:
    data = json.loads((LAB_REPO / "data/survey_items.json").read_text())
    return data["items"]


# ════════════════════════════════════════════════════════════════════
# 1b. QUERY-SPECIFIC RAG QUERIES
# ════════════════════════════════════════════════════════════════════

def build_outcome_query(item_spec: dict, construct_label: str) -> str:
    """
    Outcome-specific retrieval query, used for the Step 1 scoring prompt.
    Step 1 necessarily compares all 16 interventions side by side for a
    single outcome, so the query is anchored on the outcome/construct
    rather than any one intervention.
    """
    return (
        f"{construct_label}: {item_spec['question']} "
        f"What does the behavioral science / science communication "
        f"literature say about interventions that increase or decrease "
        f"this outcome?"
    )


def build_pair_query(intervention_text: str, item_spec: dict, construct_label: str) -> str:
    """
    Intervention x outcome specific retrieval query, used for the Step 2
    effect-size prompt. This is the core of the per-pair RAG design: the
    same intervention pulls different literature depending on which
    outcome is being predicted (e.g. trust/consensus literature for a
    trust outcome vs. behavior/donation literature for a donation
    outcome).
    """
    return (
        f"{intervention_text[:300]} "
        f"Outcome: {construct_label} — {item_spec['question']}"
    )


# ════════════════════════════════════════════════════════════════════
# 2. STEP 1 PROMPT — RELATIVE SCORING of all interventions for one outcome
# ════════════════════════════════════════════════════════════════════

def build_scoring_prompt(
    conditions:   dict,
    item_spec:    dict,
    rag_context:  str,
) -> str:
    """
    Ask the model to score ALL 16 interventions on a -3..+3 relative
    effectiveness scale for a single outcome, with ties allowed. This
    keeps the benefit of comparing interventions side by side without
    forcing a fabricated 1-16 ordering when several are genuinely close.
    """
    question = item_spec["question"]
    scale    = item_spec["scale_labels"]

    # Build numbered list of interventions
    cond_list = "\n".join(
        f"{i+1}. [{name}]: {text[:200]}..."
        for i, (name, text) in enumerate(conditions.items())
    )

    prompt = f"""You are an expert in behavioral science and science communication research.

RETRIEVED EVIDENCE (literature relevant to this outcome):
{rag_context}

OUTCOME BEING MEASURED:
Question: {question}
Scale: {scale}

TASK — RELATIVE EFFECTIVENESS SCORING:
Below are 16 science communication interventions tested in a large U.S. survey experiment (N~18,000). Each was shown to participants who then answered the outcome question above. A control group saw no message.

{cond_list}

For EACH intervention, estimate its relative effectiveness for this specific outcome on a -3 to +3 scale:

  -3 = likely meaningfully harmful (moves people away from this outcome)
  -2 = moderately negative
  -1 = slightly negative
   0 = approximately no effect
  +1 = small positive effect
  +2 = moderate positive effect
  +3 = strongest expected positive effect among these 16

Ties are allowed and expected — several interventions may genuinely be close in effectiveness, or have no meaningful effect at all. Do not spread scores out artificially just to differentiate them; use 0 freely.

Consider:
- What persuasion strategy each message uses
- What the retrieved evidence and your expertise say about that strategy's effectiveness for THIS outcome specifically
- Some messages may have zero or negative effects for this outcome — that is a realistic result, not an error

Respond ONLY with a JSON object mapping intervention name to score.
Example format: {{"Consensus": 2, "Social justice": 0, "Funding": -1, ...}}
Output JSON only, nothing else."""

    return prompt


# ════════════════════════════════════════════════════════════════════
# 3. STEP 2 PROMPT — Estimate effect size, informed by the relative score
# ════════════════════════════════════════════════════════════════════

def get_effect_range(item_spec: dict) -> tuple[str, str]:
    """
    Shared by build_effect_prompt() and predict_outcome()'s fallback
    retry, so the allowed ATE range/example always match regardless of
    which prompt path is used.
    """
    typ = item_spec.get("type", "slider 0-100")
    if "binary" in typ:
        return "between -0.15 and +0.15", "e.g. +0.05 means 5 percentage points more people said Yes"
    elif "integer 0-10" in typ:
        return "between -3 and +3", "e.g. +1.5 means 1.5 points higher on the 0-10 scale"
    else:
        return "between -10 and +15", "e.g. +6 means 6 points higher on the 0-100 scale vs control"


def build_effect_prompt(
    intervention_name: str,
    intervention_text: str,
    item_spec:         dict,
    rag_context:        str,
    score:              float,
    intro_idx:          int = 0,
    item_key:           str = None,
) -> str:
    """
    Estimate the absolute ATE for one intervention x outcome pair.
    Structured in 4 explicit sections: Task, Retrieved evidence,
    Experimental information, Relative comparison. The Step 1 score is
    passed as supporting comparative evidence only — the prompt
    explicitly tells the model to estimate the ATE independently rather
    than mechanically scaling the score, and there is no longer a
    hard-coded top/mid/bottom calibration range to anchor on.
    """
    question = item_spec["question"]
    scale    = item_spec["scale_labels"]

    intros = [
        "You are an expert in behavioral science and science communication.",
        "You are a behavioral scientist specializing in public attitudes toward science.",
        "You are a social scientist with expertise in climate communication interventions.",
    ]
    intro = intros[intro_idx % len(intros)]

    effect_range, example = get_effect_range(item_spec)

    # funding_5 ("...spending too much/too little...") is the one item
    # whose official outcome (funding_perceptions) is reverse-coded --
    # the pipeline flips the sign in aggregate_to_official() so you do
    # NOT need to reverse it yourself here. This note is only to help
    # you reason correctly about the RAW item's direction, since it is
    # easy to get backwards: an intervention that reveals climate
    # research is actually UNDER-funded (vs. other research areas)
    # should move raw funding_5 DOWN (toward "too little", a negative
    # raw effect) by correcting an overestimate of spending -- not up.
    funding_note = ""
    if item_key == "funding_5":
        funding_note = (
            "\nNote on this item specifically: it asks about spending ADEQUACY directly "
            "(0 = far too little, 100 = far too much). Think through what the intervention "
            "content implies about actual spending levels: content that reveals climate "
            "research funding is smaller than people assume (relative to other research "
            "areas, or in absolute terms) should move this raw item DOWN, toward \"too "
            "little\" (a negative raw effect) -- it corrects an overestimate. Content that "
            "emphasizes waste, corporate/private funding influence, or excess should move it "
            "UP, toward \"too much\" (a positive raw effect). You do not need to reverse-code "
            "your answer yourself -- just predict the effect on the RAW item as worded above."
        )

    prompt = f"""{intro}

1. TASK
Predict the average treatment effect (ATE) of this intervention relative to a no-message control group, for the specified outcome. Base your estimate on the intervention content, the outcome, and the retrieved evidence below. Effects in behavioral experiments are often modest — do not assume the intervention must have a positive effect. The true effect may be positive, approximately zero, or negative.

2. RETRIEVED EVIDENCE
{rag_context}

3. EXPERIMENTAL INFORMATION
INTERVENTION: {intervention_name}
TEXT: \"\"\"{intervention_text[:300]}\"\"\"

OUTCOME: {question}
SCALE: {scale}
Note: read this question's own wording carefully before predicting a direction. Some outcomes are worded as their own distinct construct (e.g. "distrust", "how much do you disagree") rather than as a mirror of general trust or belief — a message that builds trust does not automatically move every other item in the "positive" direction. Predict the effect on exactly what this question asks, not on climate trust in general.{funding_note}

4. RELATIVE COMPARISON
In an independent comparison step across all 16 interventions, this intervention received a relative-effect score of {score:+.1f} on a -3 to +3 scale (0 = no clear effect vs. the other interventions; ties were allowed). This score is on a DIFFERENT scale than your answer and is supporting context only. Do NOT copy this number or rescale it mechanically (e.g. score x constant) — that discards the intervention content and retrieved evidence above. Independently reason about the likely direction and magnitude on the outcome's own scale.

In AT MOST 15 words, reason about the likely direction and rough magnitude given the intervention content and evidence (not the score above). Keep it brief — you MUST leave room for the answer line below. Then, on a new line, write your final answer as:
ANSWER: <number {effect_range}> ({example})"""

    return prompt


# ════════════════════════════════════════════════════════════════════
# 4. API CALL
# ════════════════════════════════════════════════════════════════════

def make_client(model_key: str):
    cfg     = MODEL_CONFIG[model_key]
    api_key = os.environ.get(cfg["env"])
    if not api_key:
        raise ValueError(f"{cfg['env']} not set.\nRun: export {cfg['env']}='your-key'")
    if cfg["provider"] == "openai":
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def call_api_manual(prompt: str, max_tokens: int = None) -> str:
    """
    No-API-budget path: print the prompt so it can be copy-pasted into any
    Claude interface (claude.ai, Claude Code, etc.), then read the pasted
    reply back from the terminal. Returns whatever text was pasted, which
    then flows through the exact same parse_scores()/parse_number() logic
    as a normal API response -- nothing downstream needs to change.
    """
    print("\n" + "=" * 78)
    print("COPY EVERYTHING BETWEEN THE LINES BELOW INTO CLAUDE, THEN COME BACK HERE")
    print("=" * 78)
    print(prompt)
    print("=" * 78)
    print("Paste Claude's reply below. When done, type END on its own line and press Enter.")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def call_api(client, model_key: str, prompt: str,
             max_tokens: int = 500, dry_run: bool = False,
             manual: bool = False, log_path: str | None = None,
             log_meta: dict | None = None) -> str | None:
    if dry_run:
        return None  # handled by callers

    if manual:
        return call_api_manual(prompt, max_tokens)

    cfg = MODEL_CONFIG[model_key]
    for attempt in range(RETRY_LIMIT):
        try:
            if cfg["provider"] == "openai":
                kwargs = {
                    "model":    cfg["model"],
                    "messages": [{"role": "user", "content": prompt}],
                }
                if cfg.get("reasoning_model"):
                    # gpt-5 / o1 / o3 style: max_completion_tokens, not
                    # max_tokens; no custom temperature (default only);
                    # reasoning_effort kept low (these are short scoring /
                    # estimation prompts, not hard reasoning tasks) and the
                    # budget padded generously so reasoning tokens don't
                    # crowd out the actual answer.
                    kwargs["max_completion_tokens"] = max_tokens + REASONING_TOKEN_PADDING
                    kwargs["reasoning_effort"]      = REASONING_EFFORT
                else:
                    kwargs["max_tokens"]  = max_tokens
                    kwargs["temperature"] = 0.5
                resp = client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content
                if not content:
                    raise ValueError(
                        "Empty response content — the reasoning model likely spent its "
                        f"whole token budget thinking (currently max_completion_tokens="
                        f"{max_tokens + REASONING_TOKEN_PADDING}, reasoning_effort="
                        f"'{REASONING_EFFORT}'). Try raising REASONING_TOKEN_PADDING further "
                        "at the top of this file."
                    )
                result = content.strip()
                log_raw_call(log_path, model_key, getattr(resp, "model", None),
                             prompt, result, log_meta)
                return result
            else:
                resp = client.messages.create(
                    model=cfg["model"], max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                # resp.content can start with a ThinkingBlock (no .text
                # attribute) when extended thinking is active -- pick the
                # actual text block instead of assuming content[0] is it.
                text_blocks = [b.text for b in resp.content if b.type == "text"]
                if not text_blocks:
                    raise ValueError("No text block in response (only thinking/tool blocks?)")
                result = text_blocks[0].strip()
                log_raw_call(log_path, model_key, getattr(resp, "model", None),
                             prompt, result, log_meta)
                return result
        except Exception as e:
            print(f"  [ERROR] attempt {attempt+1}: {e}")
            if attempt < RETRY_LIMIT - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def parse_number(text: str) -> float | None:
    """
    Extracts the final numeric ATE answer. Looks for an explicit
    "ANSWER: <number>" marker first (the effect prompt now asks for a
    short reasoning sentence before the number, to stop the model from
    reflexively copying the Step-1 relative score under a tight token
    budget) and takes the number immediately after it. Falls back to
    the LAST number in the text (not the first) for any response that
    doesn't follow the marker format, since a preceding reasoning
    sentence may mention other numbers (e.g. the scale range) before
    the actual answer.
    """
    if text is None:
        return None
    marker_match = re.search(r"ANSWER:\s*(-?\d+\.?\d*)", text, re.IGNORECASE)
    if marker_match:
        return float(marker_match.group(1))
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return float(numbers[-1]) if numbers else None


def parse_scores(text: str, condition_names: list) -> dict | None:
    """Parse JSON relative-score response. Values are clipped to
    [-3, 3]; ties are allowed (no uniqueness requirement, unlike a
    ranking)."""
    if text is None:
        return None
    try:
        # Extract JSON from response
        match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if not match:
            return None
        raw = json.loads(match.group())
        if not all(name in raw for name in condition_names):
            return None
        scores = {}
        for name in condition_names:
            v = float(raw[name])
            scores[name] = max(-3.0, min(3.0, v))
        return scores
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════
# 5. TWO-STEP PREDICTION PER OUTCOME
# ════════════════════════════════════════════════════════════════════

def predict_outcome(
    client,
    model_key:      str,
    conditions:     dict,
    item_key:       str,
    item_spec:      dict,
    rag,
    use_rag:        bool,
    n_samples:      int,
    dry_run:        bool,
    manual:         bool = False,
    step1_attempts: int  = 3,
) -> dict:
    """
    Two-step prediction for all interventions on one outcome.
    Returns {condition_name: effect_size}
    """
    cond_names       = list(conditions.keys())
    n_conds          = len(cond_names)
    construct_label  = get_construct_label(item_key)

    # ── RAG context for Step 1 (outcome-specific, shared across the 16
    #    interventions since Step 1 compares them side by side) ────────
    if dry_run:
        outcome_rag_context = "[dry_run — RAG retrieval skipped]"
    elif use_rag:
        outcome_rag_context = rag.retrieve(
            build_outcome_query(item_spec, construct_label), top_k=5
        )
    else:
        outcome_rag_context = "No literature context provided."

    # ── STEP 1: Get relative scores (average over 3 attempts) ─────────
    print(f"    [Step 1] Scoring {n_conds} interventions (relative effectiveness, -3..+3, ties allowed)...")

    if dry_run:
        # Simulate plausible scores, including ties/zeros
        scores = {name: round(random.uniform(-3, 3) * 0.7, 1) for name in cond_names}
    else:
        score_prompt = build_scoring_prompt(conditions, item_spec, outcome_rag_context)
        attempts      = []
        for attempt in range(step1_attempts):  # take the mean over attempts
            text   = call_api(client, model_key, score_prompt, max_tokens=600, manual=manual)
            parsed = parse_scores(text, cond_names)
            if parsed:
                attempts.append(parsed)

        if not attempts:
            print("    [WARN] Scoring failed, using neutral fallback (all 0)")
            scores = {name: 0.0 for name in cond_names}
        else:
            # Mean score across attempts — ties are preserved, no forced
            # re-sort into a unique ordering.
            scores = {
                name: round(sum(a[name] for a in attempts) / len(attempts), 2)
                for name in cond_names
            }

    top_by_score = sorted(cond_names, key=lambda n: scores[n], reverse=True)[:3]
    print(f"    [Step 1] Done. Top by score: {top_by_score}")

    # ── STEP 2: Estimate effect size per intervention ─────────────────
    print(f"    [Step 2] Estimating effect sizes (intervention x outcome specific retrieval)...")

    effects = {}
    for cond_name, cond_text in conditions.items():
        score = scores[cond_name]

        if dry_run:
            # Simulate realistic effects based on the relative score
            base  = (score / 3.0) * 8  # roughly -8 to +8
            noise = random.uniform(-1.5, 1.5)
            effects[cond_name] = round(base + noise, 2)
            continue

        if use_rag:
            # Two separate retrievals (intervention text, outcome/construct),
            # merged and deduped — instead of one concatenated query, where
            # a topically dense intervention text (e.g. "Model accuracy")
            # was drowning out short outcome signals like "donation ams"
            # and starving that outcome of relevant literature. See
            # VectorRAG.retrieve_merged() docstring for the full rationale.
            pair_rag_context = rag.retrieve_merged(
                [cond_text[:400], build_outcome_query(item_spec, construct_label)],
                top_k_each=4, max_total=6,
            )
        else:
            pair_rag_context = "No literature context provided."

        values = []
        for i in range(n_samples):
            effect_prompt = build_effect_prompt(
                intervention_name = cond_name,
                intervention_text = cond_text,
                item_spec         = item_spec,
                rag_context        = pair_rag_context,
                score               = score,
                intro_idx           = i,
                item_key            = item_key,
            )
            # max_tokens raised 20 -> 110: the prompt asks for a short (<=15
            # word) reasoning sentence before the ANSWER: line, since a
            # tight budget left the model no room to do anything but
            # reflexively echo the Step-1 score shown just above it. 110
            # gives headroom even when the model runs a bit verbose.
            text = call_api(client, model_key, effect_prompt, max_tokens=110, manual=manual)
            val  = parse_number(text)
            if val is None:
                # Rare fallback: the model's reasoning ran long and got cut
                # off before reaching "ANSWER:" (verbosity varies by
                # intervention/outcome and isn't fully preventable via
                # max_tokens alone). Retry once with a compact, no-RAG-
                # context, answer-only prompt instead of dropping the
                # sample -- costs little since it skips the reasoning step
                # and only fires on failure.
                fb_range, _ = get_effect_range(item_spec)
                fallback_prompt = (
                    f"Predict the average treatment effect (ATE) of this intervention "
                    f"on the outcome below, relative to a no-message control.\n"
                    f"INTERVENTION: {cond_name}\n"
                    f"OUTCOME: {item_spec['question']}\n"
                    f"SCALE: {item_spec['scale_labels']}\n"
                    f"Output ONLY a single number {fb_range}. No words, no reasoning."
                )
                text = call_api(client, model_key, fallback_prompt, max_tokens=15, manual=manual)
                val = parse_number(text)
            if val is not None:
                values.append(val)

        effects[cond_name] = round(sum(values)/len(values), 3) if values else None

    return effects


# ════════════════════════════════════════════════════════════════════
# 6. AGGREGATE RAW ITEMS → OFFICIAL OUTCOMES
# ════════════════════════════════════════════════════════════════════

def aggregate_to_official(results: dict) -> dict:
    """
    Convert raw item-level results into official 13 composite outcomes.
    Keeps values in ORIGINAL UNITS per outcome scale.
    Organizers convert to percentage points at scoring time.

    Input:  {item_key: {condition: effect_size}}
    Output: {official_outcome: {condition: effect_in_original_units}}
    """
    official = {}

    for out_name, spec in OFFICIAL_OUTCOMES.items():
        items       = spec["items"]
        scale_range = spec["scale_range"]
        official[out_name] = {}

        # Get all conditions from first available item
        all_conds = set()
        for item in items:
            if item in results:
                all_conds.update(results[item].keys())

        for cond in all_conds:
            vals = []
            for item in items:
                if item in results and cond in results[item]:
                    v = results[item][cond]
                    if v is not None:
                        vals.append(v)

            if vals:
                v = sum(vals) / len(vals)
                if out_name in REVERSE_CODED_OUTCOMES:
                    # Model was prompted with the raw (non-reverse-coded)
                    # item wording/scale, so its predicted ATE is on the
                    # raw direction — flip sign to match the official
                    # reverse-coded outcome (higher = supports more
                    # funding). See REVERSE_CODED_OUTCOMES comment above.
                    v = -v
                # Keep in ORIGINAL units — organizers convert to pp at scoring
                # trust/concern/policy: 0-100 scale points
                # donation_ams: $0-10
                # newsletter_signup: 0-1 probability
                official[out_name][cond] = round(v, 4)
            else:
                official[out_name][cond] = None

    return official


def save_submission_format(
    results_official: dict,
    conditions:       list,
    output_dir:       str,
    label:            str,
    timestamp:        str,
    calibrated:       bool = True,
) -> str:
    """
    Save in official Tier 3 submission format:
    condition, outcome, ate
    (16 conditions x 13 outcomes = 208 rows)
    """
    suffix  = "calibrated" if calibrated else "raw"
    subfile = f"{output_dir}/tier3_submission_{suffix}_{label}_{timestamp}.csv"

    with open(subfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "outcome", "ate"])
        writer.writeheader()

        for cond in conditions:
            for out_name in OFFICIAL_OUTCOMES:
                v = results_official.get(out_name, {}).get(cond)
                if v is not None and calibrated:
                    v = round(v * CALIBRATION_FACTOR, 4)
                writer.writerow({
                    "condition": cond,
                    "outcome":   out_name,
                    "ate":       v,
                })

    return subfile


# ════════════════════════════════════════════════════════════════════
# 8. MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════

def run_tier3(
    model_key:      str  = "gpt4o",
    n_samples:      int  = 3,
    dry_run:        bool = False,
    output_dir:     str  = "./results",
    item_subset:    list = None,
    use_rag:        bool = True,
    manual:         bool = False,
    step1_attempts: int  = 3,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    label       = f"tier3_{model_key}_{'rag' if use_rag else 'norag'}"
    raw_file    = f"{output_dir}/tier3_raw_{label}_{timestamp}.csv"
    cal_file    = f"{output_dir}/tier3_calibrated_{label}_{timestamp}.csv"
    detail_file = f"{output_dir}/tier3_rankings_{label}_{timestamp}.csv"

    stimuli      = load_stimuli()
    survey_items = load_survey_items()
    conditions   = {k: v for k, v in stimuli.items() if k != "control"}

    if item_subset:
        survey_items = {k: v for k, v in survey_items.items() if k in item_subset}
    item_keys = list(survey_items.keys())

    # RAG — instantiate once (the vector DB / TF-IDF index), but retrieval
    # itself now happens per outcome (Step 1) and per intervention x outcome
    # (Step 2) inside predict_outcome, instead of a single shared context
    # reused for every outcome and intervention.
    if use_rag:
        rag = VectorRAG()
        print("[RAG] Vector DB ready — retrieval is query-specific per outcome / intervention x outcome")
    else:
        rag = None
        print("[RAG] DISABLED — vanilla baseline (no literature)")

    n_conds   = len(conditions)
    n_items   = len(item_keys)
    api_est   = n_items * (step1_attempts + n_conds * n_samples)

    print(f"\n{'='*60}")
    print(f"  Tier 3 Pipeline — Relative Scoring + Effect Estimation")
    print(f"  Group  : Farah, Jing, Max, Marcia")
    print(f"  Model  : {'manual copy-paste (no API key needed)' if manual else MODEL_CONFIG[model_key]['model']}")
    print(f"  Items  : {n_items}")
    print(f"  Conds  : {n_conds} interventions")
    print(f"  RAG    : {'ON — query-specific per outcome / pair' if use_rag else 'OFF — vanilla baseline'}")
    print(f"  Method : Step 1=Relative scoring (-3..+3, ties allowed), Step 2=Effect sizes")
    if manual:
        print(f"  Est steps: ~{api_est} manual copy-paste rounds (+ cheap retrieval calls if RAG is ON)")
    else:
        print(f"  Est API: ~{api_est} LLM calls (+ cheap retrieval calls if RAG is ON)")
    print(f"  Dry run: {dry_run}")
    print(f"{'='*60}\n")

    client  = None if (dry_run or manual) else make_client(model_key)
    results = {}   # {item_key: {condition: effect}}

    for i, (item_key, item_spec) in enumerate(survey_items.items()):
        print(f"\n[{i+1}/{n_items}] Outcome: {item_key}")

        item_effects = predict_outcome(
            client         = client,
            model_key      = model_key,
            conditions     = conditions,
            item_key       = item_key,
            item_spec      = item_spec,
            rag            = rag,
            use_rag        = use_rag,
            n_samples      = n_samples,
            dry_run        = dry_run,
            manual         = manual,
            step1_attempts = step1_attempts,
        )
        results[item_key] = item_effects

    # ── Save CSVs ────────────────────────────────────────────────────
    cond_names = list(conditions.keys())

    with open(raw_file, "w", newline="") as rf, \
         open(cal_file, "w", newline="") as cf:

        rw = csv.DictWriter(rf, fieldnames=["condition"] + item_keys)
        cw = csv.DictWriter(cf, fieldnames=["condition"] + item_keys)
        rw.writeheader()
        cw.writeheader()

        for cond in cond_names:
            raw_row = {"condition": cond}
            cal_row = {"condition": cond}
            for item_key in item_keys:
                v = results.get(item_key, {}).get(cond)
                raw_row[item_key] = v
                cal_row[item_key] = round(v * CALIBRATION_FACTOR, 3) if v else None
            rw.writerow(raw_row)
            cw.writerow(cal_row)

    # ── Aggregate to official 13 outcomes ───────────────────────────
    print("\n[AGGREGATING] Computing official 13 composite outcomes...")
    results_official = aggregate_to_official(results)

    # ── Save official submission format ──────────────────────────────
    cond_list = list(conditions.keys())
    sub_raw = save_submission_format(
        results_official, cond_list, output_dir, label, timestamp, calibrated=False
    )
    sub_cal = save_submission_format(
        results_official, cond_list, output_dir, label, timestamp, calibrated=True
    )

    print(f"\n{'='*60}")
    print(f"  ✅ TIER 3 COMPLETE")
    print(f"  Internal raw        : {raw_file}")
    print(f"  Internal calibrated : {cal_file}")
    print(f"")
    print(f"  ── SUBMISSION FILES (official format) ──")
    print(f"  Raw submission      : {sub_raw}")
    print(f"  Calibrated submit   : {sub_cal}  ← USE THIS")
    print(f"  Format: condition, outcome, ate (208 rows)")
    print(f"  Units: percentage points of scale range")
    print(f"{'='*60}")

    return results


# ════════════════════════════════════════════════════════════════════
# 9. CLI
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tier 3 RAG pipeline — relative scoring + effect estimation."
    )
    parser.add_argument("--model",      type=str,  default="gpt4o",
                        choices=["gpt4o", "gpt5", "claude"])
    parser.add_argument("--n_samples",  type=int,  default=3,
                        help="Ensemble size for effect estimation (default: 3)")
    parser.add_argument("--dry_run",    action="store_true")
    parser.add_argument("--output_dir", type=str,  default="./results")
    parser.add_argument("--quick",      action="store_true",
                        help="Run only 3 items for quick testing")
    parser.add_argument("--no_rag",     action="store_true",
                        help="Disable RAG — vanilla baseline for comparison")
    parser.add_argument("--manual",     action="store_true",
                        help="No API budget needed: prints each prompt for you to paste into "
                             "any Claude interface (claude.ai, Claude Code, etc.), then reads "
                             "the pasted reply back from the terminal. No API key required.")
    parser.add_argument("--step1_attempts", type=int, default=3,
                        help="Step 1 scoring attempts to average (default: 3). Lower this "
                             "(e.g. 1) with --manual to cut down the number of copy-paste rounds.")
    parser.add_argument("--item_subset", type=str, default=None,
                        help="Comma-separated outcome keys to run, e.g. "
                             "'trust_competence_1' for a single-outcome smoke test. "
                             "Overrides --quick if both are given.")
    args = parser.parse_args()

    item_subset = None
    if args.quick:
        item_subset = ["trust_competence_1", "concern_1", "policy_general_1"]
    if args.item_subset:
        item_subset = [k.strip() for k in args.item_subset.split(",") if k.strip()]

    run_tier3(
        model_key      = args.model,
        n_samples      = args.n_samples,
        dry_run        = args.dry_run,
        output_dir     = args.output_dir,
        item_subset    = item_subset,
        use_rag        = not args.no_rag,
        manual         = args.manual,
        step1_attempts = args.step1_attempts,
    )
