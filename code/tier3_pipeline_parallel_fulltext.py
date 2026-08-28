"""
tier3_pipeline_parallel_fulltext.py
-------------------------------------
Combines the two separate fixes built earlier into one new file (neither
tier3_pipeline.py nor tier3_pipeline_parallel.py is touched by this):

  1. FULL TEXT (from tier3_pipeline_fulltext.py) — removes all three
     truncation points ([:200] in Step 1's intervention listing, [:300] in
     Step 2's "TEXT:" section, [:400] in Step 2's RAG retrieval query).
     tier3_pipeline_parallel.py still has all three, because it imports
     build_scoring_prompt/build_effect_prompt UNCHANGED from
     tier3_pipeline.py and additionally truncates its own RAG query to
     cond_text[:400] — this file fixes all of that.

  2. PARALLEL (from tier3_pipeline_parallel.py) — the independent LLM
     calls within one outcome (Step 1's step1_attempts scoring calls, and
     Step 2's condition x n_samples effect calls) run concurrently via a
     thread pool instead of one at a time. Same prompts, same parsing,
     same calibration/aggregation — only the call scheduling changes.

Default model is gpt4o (matches tier3_pipeline_fulltext.py's default).

Cost/latency note (from the full-text file, still applies): full
intervention text makes prompts noticeably bigger — Step 1's list of all
16 interventions goes from ~3,400 to ~46,600 characters per attempt. Still
well inside gpt-4o's context window, but more input tokens billed, and
more latency per call. Parallelism here is what keeps that latency
increase from compounding across step1_attempts and n_samples.

--manual mode always stays sequential (there's no way to parallelize a
human copy-pasting into a chat window).

Usage:
    python tier3_pipeline_parallel_fulltext.py --model gpt4o --max_workers 5
    python tier3_pipeline_parallel_fulltext.py --model gpt4o --max_workers 5 --item_subset trust_competence_1
    python tier3_pipeline_parallel_fulltext.py --model gpt4o --dry_run --quick
"""

import argparse
import csv
import hashlib
import random
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from tier3_pipeline import (
    MODEL_CONFIG,
    CALIBRATION_FACTOR,
    load_stimuli,
    load_survey_items,
    get_construct_label,
    build_outcome_query,
    get_effect_range,
    make_client,
    call_api,
    parse_number,
    parse_scores,
    aggregate_to_official,
    save_submission_format,
    VectorRAG,
)


# ════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS — same as tier3_pipeline_fulltext.py, no truncation
# ════════════════════════════════════════════════════════════════════

def build_scoring_prompt_fulltext(conditions: dict, item_spec: dict, rag_context: str) -> str:
    """Same as tier3_pipeline.build_scoring_prompt(), but each
    intervention is listed in full instead of text[:200]."""
    question = item_spec["question"]
    scale    = item_spec["scale_labels"]

    cond_list = "\n".join(
        f"{i+1}. [{name}]: {text}"
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


def build_effect_prompt_fulltext(
    intervention_name: str,
    intervention_text: str,
    item_spec:         dict,
    rag_context:        str,
    score:              float,
    intro_idx:          int = 0,
    item_key:           str = None,
) -> str:
    """Same as tier3_pipeline.build_effect_prompt(), but the "TEXT:"
    section carries the full intervention text instead of text[:300]."""
    question = item_spec["question"]
    scale    = item_spec["scale_labels"]

    intros = [
        "You are an expert in behavioral science and science communication.",
        "You are a behavioral scientist specializing in public attitudes toward science.",
        "You are a social scientist with expertise in climate communication interventions.",
    ]
    intro = intros[intro_idx % len(intros)]

    effect_range, example = get_effect_range(item_spec)

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
TEXT: \"\"\"{intervention_text}\"\"\"

OUTCOME: {question}
SCALE: {scale}
Note: read this question's own wording carefully before predicting a direction. Some outcomes are worded as their own distinct construct (e.g. "distrust", "how much do you disagree") rather than as a mirror of general trust or belief — a message that builds trust does not automatically move every other item in the "positive" direction. Predict the effect on exactly what this question asks, not on climate trust in general.{funding_note}

4. RELATIVE COMPARISON
In an independent comparison step across all 16 interventions, this intervention received a relative-effect score of {score:+.1f} on a -3 to +3 scale (0 = no clear effect vs. the other interventions; ties were allowed). This score is on a DIFFERENT scale than your answer and is supporting context only. Do NOT copy this number or rescale it mechanically (e.g. score x constant) — that discards the intervention content and retrieved evidence above. Independently reason about the likely direction and magnitude on the outcome's own scale.

In AT MOST 15 words, reason about the likely direction and rough magnitude given the intervention content and evidence (not the score above). Keep it brief — you MUST leave room for the answer line below. Then, on a new line, write your final answer as:
ANSWER: <number {effect_range}> ({example})"""

    return prompt


# ════════════════════════════════════════════════════════════════════
# TWO-STEP PREDICTION PER OUTCOME — parallel + full-text
# ════════════════════════════════════════════════════════════════════

def predict_outcome_parallel_fulltext(
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
    max_workers:    int  = 5,
    log_path:       str  = None,
) -> dict:
    """Same two-step logic as predict_outcome_fulltext(), but the
    independent calls within Step 1 and Step 2 run concurrently through a
    thread pool instead of one at a time. Falls back to sequential
    execution when manual=True or max_workers <= 1.

    log_path, if set, is forwarded to every call_api() call so the
    complete raw response (before any parsing) gets archived -- see
    log_raw_call() in tier3_pipeline.py. Off (None) by default."""

    cond_names      = list(conditions.keys())
    n_conds         = len(cond_names)
    construct_label = get_construct_label(item_key)
    sequential      = manual or max_workers <= 1

    if dry_run:
        outcome_rag_context = "[dry_run — RAG retrieval skipped]"
    elif use_rag:
        outcome_rag_context = rag.retrieve(
            build_outcome_query(item_spec, construct_label), top_k=5
        )
    else:
        outcome_rag_context = "No literature context provided."

    # ── STEP 1: relative scores (full-text intervention list) ─────────
    print(f"    [Step 1] Scoring {n_conds} interventions, FULL TEXT "
          f"({'sequential' if sequential else f'{step1_attempts} attempts, up to {max_workers} at once'})...")

    if dry_run:
        scores = {name: round(random.uniform(-3, 3) * 0.7, 1) for name in cond_names}
    else:
        score_prompt = build_scoring_prompt_fulltext(conditions, item_spec, outcome_rag_context)
        attempts = []

        if sequential:
            for attempt_idx in range(step1_attempts):
                text   = call_api(client, model_key, score_prompt, max_tokens=600, manual=manual,
                                   log_path=log_path,
                                   log_meta={"item_key": item_key, "step": "step1_scoring", "attempt": attempt_idx})
                parsed = parse_scores(text, cond_names)
                if parsed:
                    attempts.append(parsed)
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, step1_attempts)) as ex:
                futures = [
                    ex.submit(call_api, client, model_key, score_prompt, max_tokens=600,
                              log_path=log_path,
                              log_meta={"item_key": item_key, "step": "step1_scoring", "attempt": attempt_idx})
                    for attempt_idx in range(step1_attempts)
                ]
                for f in as_completed(futures):
                    parsed = parse_scores(f.result(), cond_names)
                    if parsed:
                        attempts.append(parsed)

        if not attempts:
            print("    [WARN] Scoring failed, using neutral fallback (all 0)")
            scores = {name: 0.0 for name in cond_names}
        else:
            scores = {
                name: round(sum(a[name] for a in attempts) / len(attempts), 2)
                for name in cond_names
            }

    top_by_score = sorted(cond_names, key=lambda n: scores[n], reverse=True)[:3]
    print(f"    [Step 1] Done. Top by score: {top_by_score}")

    # ── STEP 2: effect sizes (full-text RAG query + TEXT section) ─────
    print(f"    [Step 2] Estimating effect sizes, FULL TEXT "
          f"({'sequential' if sequential else f'up to {max_workers} at once'})...")

    if dry_run:
        effects = {}
        for cond_name in cond_names:
            score = scores[cond_name]
            base  = (score / 3.0) * 8
            noise = random.uniform(-1.5, 1.5)
            effects[cond_name] = round(base + noise, 2)
        return effects

    def fallback_answer(cond_name):
        fb_range, _ = get_effect_range(item_spec)
        fallback_prompt = (
            f"Predict the average treatment effect (ATE) of this intervention "
            f"on the outcome below, relative to a no-message control.\n"
            f"INTERVENTION: {cond_name}\n"
            f"OUTCOME: {item_spec['question']}\n"
            f"SCALE: {item_spec['scale_labels']}\n"
            f"Output ONLY a single number {fb_range}. No words, no reasoning."
        )
        text = call_api(client, model_key, fallback_prompt, max_tokens=15, manual=manual,
                         log_path=log_path,
                         log_meta={"item_key": item_key, "step": "step2_fallback", "condition": cond_name})
        return parse_number(text)

    def run_one_sample(cond_name, effect_prompt, sample_idx):
        text = call_api(client, model_key, effect_prompt, max_tokens=110, manual=manual,
                         log_path=log_path,
                         log_meta={"item_key": item_key, "step": "step2_effect",
                                    "condition": cond_name, "sample": sample_idx})
        val  = parse_number(text)
        if val is None:
            val = fallback_answer(cond_name)
        return cond_name, val

    # Build every (condition, sample) prompt up front. RAG retrieval
    # (rag.retrieve_merged) is a local vector-DB lookup, not an LLM call,
    # so it stays sequential — not the bottleneck. Full cond_text now goes
    # into the retrieval query too (was cond_text[:400] in the parallel
    # file) — text-embedding-3-small's ~8191-token limit comfortably
    # covers even the longest intervention (~11,400 chars ≈ ~2,900 tokens).
    jobs = []  # (cond_name, prompt, sample_idx)
    for cond_name, cond_text in conditions.items():
        score = scores[cond_name]
        if use_rag:
            pair_rag_context = rag.retrieve_merged(
                [cond_text, build_outcome_query(item_spec, construct_label)],
                top_k_each=4, max_total=6,
            )
        else:
            pair_rag_context = "No literature context provided."

        for i in range(n_samples):
            effect_prompt = build_effect_prompt_fulltext(
                intervention_name = cond_name,
                intervention_text = cond_text,
                item_spec         = item_spec,
                rag_context        = pair_rag_context,
                score               = score,
                intro_idx           = i,
                item_key            = item_key,
            )
            jobs.append((cond_name, effect_prompt, i))

    values_by_cond = {name: [] for name in cond_names}

    if sequential:
        for cond_name, prompt, sample_idx in jobs:
            _, val = run_one_sample(cond_name, prompt, sample_idx)
            if val is not None:
                values_by_cond[cond_name].append(val)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(run_one_sample, cond_name, prompt, sample_idx)
                for cond_name, prompt, sample_idx in jobs
            ]
            for f in as_completed(futures):
                cond_name, val = f.result()
                if val is not None:
                    values_by_cond[cond_name].append(val)

    effects = {}
    for cond_name in cond_names:
        vals = values_by_cond[cond_name]
        effects[cond_name] = round(sum(vals) / len(vals), 3) if vals else None

    return effects


# ════════════════════════════════════════════════════════════════════
# TOP-LEVEL RUN — parallel + full-text
# ════════════════════════════════════════════════════════════════════

def run_tier3_parallel_fulltext(
    model_key:      str  = "gpt4o",
    n_samples:      int  = 3,
    dry_run:        bool = False,
    output_dir:     str  = "./results",
    item_subset:    list = None,
    use_rag:        bool = True,
    manual:         bool = False,
    step1_attempts: int  = 3,
    max_workers:    int  = 5,
    log_raw:        bool = True,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    label       = f"tier3_{model_key}_{'rag' if use_rag else 'norag'}_parallel_fulltext"
    raw_file    = f"{output_dir}/tier3_raw_{label}_{timestamp}.csv"
    cal_file    = f"{output_dir}/tier3_calibrated_{label}_{timestamp}.csv"
    # K.2 registration requirement: complete, unprocessed model responses,
    # archived + hashed + timestamped. One JSON line per API call (Step 1
    # scoring attempts, Step 2 effect samples, Step 2 fallback retries).
    # dry_run/manual never hit call_api's log_raw_call() path, so no log
    # file is created for those modes regardless of log_raw.
    log_path = f"{output_dir}/raw_log_{label}_{timestamp}.jsonl" if (log_raw and not dry_run and not manual) else None

    stimuli      = load_stimuli()
    survey_items = load_survey_items()
    conditions   = {k: v for k, v in stimuli.items() if k != "control"}

    if item_subset:
        survey_items = {k: v for k, v in survey_items.items() if k in item_subset}
    item_keys = list(survey_items.keys())

    if use_rag:
        rag = VectorRAG()
        print("[RAG] Vector DB ready")
    else:
        rag = None
        print("[RAG] DISABLED — vanilla baseline (no literature)")

    n_conds    = len(conditions)
    n_items    = len(item_keys)
    sequential = manual or max_workers <= 1
    api_est    = n_items * (step1_attempts + n_conds * n_samples)

    print(f"\n{'='*60}")
    print(f"  Tier 3 Pipeline (PARALLEL + FULL TEXT) — Relative Scoring + Effect Estimation")
    print(f"  Group  : Farah, Jing, Max, Marcia")
    print(f"  Model  : {'manual copy-paste (no API key needed)' if manual else MODEL_CONFIG[model_key]['model']}")
    print(f"  Items  : {n_items}")
    print(f"  Conds  : {n_conds} interventions")
    print(f"  RAG    : {'ON' if use_rag else 'OFF'}")
    print(f"  Text   : FULL intervention text everywhere (no [:200]/[:300]/[:400] truncation)")
    print(f"  Concurrency: {'sequential (forced — manual mode)' if manual else (max_workers if max_workers > 1 else 'sequential (max_workers=1)')}")
    if manual:
        print(f"  Est steps: ~{api_est} manual copy-paste rounds (+ cheap retrieval calls if RAG is ON)")
    else:
        print(f"  Est API: ~{api_est} LLM calls (+ cheap retrieval calls if RAG is ON) — bigger prompts than tier3_pipeline.py, costs more per call")
    print(f"  Dry run: {dry_run}")
    print(f"{'='*60}\n")

    client  = None if (dry_run or manual) else make_client(model_key)
    results = {}

    for i, (item_key, item_spec) in enumerate(survey_items.items()):
        print(f"\n[{i+1}/{n_items}] Outcome: {item_key}")
        results[item_key] = predict_outcome_parallel_fulltext(
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
            max_workers    = max_workers,
            log_path       = log_path,
        )

    # ── Save CSVs (identical format to tier3_pipeline.py) ─────────────
    cond_names = list(conditions.keys())
    with open(raw_file, "w", newline="") as rf, open(cal_file, "w", newline="") as cf:
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

    print("\n[AGGREGATING] Computing official 13 composite outcomes...")
    results_official = aggregate_to_official(results)

    cond_list = list(conditions.keys())
    sub_raw = save_submission_format(results_official, cond_list, output_dir, label, timestamp, calibrated=False)
    sub_cal = save_submission_format(results_official, cond_list, output_dir, label, timestamp, calibrated=True)

    print(f"\n{'='*60}")
    print(f"  ✅ TIER 3 COMPLETE (parallel + full-text run)")
    print(f"  Internal raw        : {raw_file}")
    print(f"  Internal calibrated : {cal_file}")
    print(f"")
    print(f"  ── SUBMISSION FILES (official format) ──")
    print(f"  Raw submission      : {sub_raw}")
    print(f"  Calibrated submit   : {sub_cal}  ← USE THIS")
    print(f"  Format: condition, outcome, ate (208 rows)")
    print(f"  Units: percentage points of scale range")
    if log_path and Path(log_path).exists():
        sha256 = hashlib.sha256(Path(log_path).read_bytes()).hexdigest()
        n_lines = sum(1 for _ in open(log_path))
        print(f"")
        print(f"  ── RAW RESPONSE LOG (K.2) ──")
        print(f"  Log file            : {log_path}")
        print(f"  Lines (API calls)   : {n_lines}")
        print(f"  SHA-256             : {sha256}")
        print(f"  ↑ copy this hash + line count into the registration form's K.2 item")
    print(f"{'='*60}")

    return results


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tier 3 RAG pipeline — parallel (thread-pooled) + full-text (no truncation), gpt4o default."
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
                        help="No API budget needed — copy-paste mode. Always runs "
                             "sequentially (parallelism doesn't apply to manual rounds).")
    parser.add_argument("--step1_attempts", type=int, default=3,
                        help="Step 1 scoring attempts to average (default: 3).")
    parser.add_argument("--item_subset", type=str, default=None,
                        help="Comma-separated outcome keys, e.g. 'trust_competence_1'")
    parser.add_argument("--max_workers", type=int, default=5,
                        help="How many LLM calls to fire concurrently (default: 5). "
                             "Set to 1 to force sequential. Ignored under --manual.")
    parser.add_argument("--no_raw_log", action="store_true",
                        help="Skip writing the raw_log_*.jsonl file of complete unprocessed "
                             "model responses (on by default -- see K.2 registration item).")
    args = parser.parse_args()

    item_subset = None
    if args.quick:
        item_subset = ["trust_competence_1", "concern_1", "policy_general_1"]
    if args.item_subset:
        item_subset = [k.strip() for k in args.item_subset.split(",") if k.strip()]

    run_tier3_parallel_fulltext(
        model_key      = args.model,
        n_samples      = args.n_samples,
        dry_run        = args.dry_run,
        output_dir     = args.output_dir,
        item_subset    = item_subset,
        use_rag        = not args.no_rag,
        manual         = args.manual,
        step1_attempts = args.step1_attempts,
        max_workers    = args.max_workers,
        log_raw        = not args.no_raw_log,
    )
