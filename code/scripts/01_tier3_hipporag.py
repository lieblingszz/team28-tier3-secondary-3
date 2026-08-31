"""
01_tier3_hipporag.py
--------------------
Tier 3 Silicon Sample Benchmark Pipeline -- HippoRAG retrieval only.

The VectorRAG (ChromaDB / TF-IDF) backend and its options have been removed,
so the only retrieval path is the OpenIE knowledge graph + Personalized
PageRank one described in hipporag_rag.py. Nothing else about the method
changed.
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

Run it through run_hipporag.sh (which checks the vLLM server, records the
GPU state and tees a timestamped log) rather than directly. Direct use, from
the repo root, needs the HippoRAG venv:

    ./code/.venv-hipporag/bin/python code/scripts/01_tier3_hipporag.py --model qwen38 --quick
    ./code/.venv-hipporag/bin/python code/scripts/01_tier3_hipporag.py --model qwen38 --graph_weight 0.7
"""

import os
import sys
import csv
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

# This file is only ever RUN, never imported (its name starts with a digit),
# so a single flat import of the sibling modules is enough.
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    OFFICIAL_OUTCOMES, REVERSE_CODED_OUTCOMES, MODEL_CONFIG,
    load_stimuli, load_survey_items, get_construct_label,
    build_outcome_query, build_pair_query,
    build_scoring_prompt, build_effect_prompt, get_effect_range,
    make_client, call_api, parse_number, parse_scores,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"


CALIBRATION_FACTOR = 0.35


def _rag_tag(use_rag, graph_weight=1.0):
    """
    Identifies the retrieval configuration in both the output filenames and
    the checkpoint name. The graph/dense fusion weight has to be part of it:
    without it a --graph_weight 0.7 run is indistinguishable from a stock
    HippoRAG one except by timestamp, and -- worse -- would resume from its
    checkpoint, producing a file that is half one configuration and half the
    other.
    """
    if not use_rag:
        return "norag"

    tag = "rag_hipporag"
    if graph_weight is not None and graph_weight < 1.0:
        tag += f"_g{int(round(graph_weight * 100))}"
    return tag


def _checkpoint_path(output_dir, model_key, rag_tag):
    """
    Deliberately has no timestamp in the name: a restart has to find the
    previous attempt's file. Keyed on model and retrieval configuration so a
    run never resumes from a differently weighted one, whose per-outcome
    effects came from different retrieved evidence.
    """
    return f"{output_dir}/tier3_checkpoint_{model_key}_{rag_tag}.json"


def save_checkpoint(path, results):
    """
    Written after every completed outcome, so a run killed partway leaves
    the finished outcomes behind instead of nothing -- the full runs so far
    have been terminated externally (JupyterHub reclaiming the container)
    at 91% and 42/44, each time discarding hours of work.

    os.replace() is atomic on POSIX: a kill during the write leaves either
    the old checkpoint or the new one, never a truncated file.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=1)
    os.replace(tmp, path)


def write_outputs(results, conditions, item_keys, raw_file, cal_file,
                  output_dir, label, timestamp, verbose=False):
    """
    Writes the four tier-3 CSVs -- the same set every other tier-3 run in
    results/ produces, in the same format.

    Called after every completed outcome as well as at the end, so an
    interrupted run leaves usable CSVs rather than only the JSON
    checkpoint. Outcomes not yet predicted are written as empty cells, so
    a partial file is visibly partial.
    """
    with open(raw_file, "w", newline="") as rf, \
         open(cal_file, "w", newline="") as cf:

        rw = csv.DictWriter(rf, fieldnames=["condition"] + item_keys)
        cw = csv.DictWriter(cf, fieldnames=["condition"] + item_keys)
        rw.writeheader()
        cw.writeheader()

        for cond in conditions:
            raw_row = {"condition": cond}
            cal_row = {"condition": cond}
            for item_key in item_keys:
                v = results.get(item_key, {}).get(cond)
                raw_row[item_key] = v
                cal_row[item_key] = round(v * CALIBRATION_FACTOR, 3) if v else None
            rw.writerow(raw_row)
            cw.writerow(cal_row)

    if verbose:
        print("\n[AGGREGATING] Computing official 13 composite outcomes...")

    results_official = aggregate_to_official(results)
    cond_list = list(conditions)

    return (
        save_submission_format(results_official, cond_list, output_dir,
                               label, timestamp, calibrated=False),
        save_submission_format(results_official, cond_list, output_dir,
                               label, timestamp, calibrated=True),
    )


def load_checkpoint(path, survey_items):
    if not os.path.exists(path):
        return {}

    try:
        with open(path) as f:
            saved = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[RESUME] ignoring unreadable checkpoint {path}: {e}")
        return {}

    # Only outcomes this run is actually asked for -- so --quick after a
    # full run does not pull in the other 41.
    return {k: v for k, v in saved.items() if k in survey_items}


# ════════════════════════════════════════════════════════════════════
# 5. TWO-STEP PREDICTION PER OUTCOME
# ════════════════════════════════════════════════════════════════════

def predict_outcome(
    client,
    model_key:    str,
    conditions:   dict,
    item_key:     str,
    item_spec:    dict,
    rag,
    use_rag:      bool,
    n_samples:    int,
    dry_run:      bool,
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
        for attempt in range(3):  # 3 scoring attempts, take the mean
            text   = call_api(client, model_key, score_prompt, max_tokens=600)
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
            # HippoRAGRag.retrieve_merged() for the full rationale.
            pair_rag_context = rag.retrieve_merged(
                [cond_text, build_outcome_query(item_spec, construct_label)],
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
            text = call_api(client, model_key, effect_prompt, max_tokens=110)
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
                text = call_api(client, model_key, fallback_prompt, max_tokens=15)
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
    model_key:              str  = "gpt4o",
    n_samples:              int  = 3,
    dry_run:                bool = False,
    output_dir:             str  = None,
    item_subset:            list = None,
    use_rag:                bool = True,
    hipporag_save_dir:       str  = None,
    hipporag_llm_path:       str  = None,
    hipporag_embedding_model: str = None,
    graph_weight:           float = 1.0,
    fusion:                  str  = None,
    passage_node_weight:    float = None,
    resume:                 bool = True,
):
    output_dir  = str(RESULTS_DIR if output_dir is None else output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    # The fusion weight goes in the label: without it two HippoRAG runs at
    # different weights produce filenames identical except for the timestamp,
    # making them impossible to tell apart in results/.
    rag_tag     = _rag_tag(use_rag, graph_weight)
    label       = f"tier3_{model_key}_{rag_tag}"
    raw_file    = f"{output_dir}/tier3_raw_{label}_{timestamp}.csv"
    cal_file    = f"{output_dir}/tier3_calibrated_{label}_{timestamp}.csv"
    detail_file = f"{output_dir}/tier3_rankings_{label}_{timestamp}.csv"

    stimuli      = load_stimuli()
    survey_items = load_survey_items()
    conditions   = {k: v for k, v in stimuli.items() if k != "control"}

    if item_subset:
        survey_items = {k: v for k, v in survey_items.items() if k in item_subset}
    item_keys = list(survey_items.keys())

    # RAG — the graph and its index are built once here, but retrieval itself
    # happens per outcome (Step 1) and per intervention x outcome (Step 2)
    # inside predict_outcome, instead of a single shared context reused for
    # every outcome and intervention.
    if use_rag:
        # hipporag lives in a SEPARATE venv (.venv-hipporag) from this repo's
        # pinned transformers/torch/vllm versions -- see hipporag_rag.py's
        # module docstring and requirements-hipporag.txt for why. Importing
        # it under .venv will fail; that is the intended signal.
        from hipporag_rag import HippoRAGRag
        rag_kwargs = {}
        if hipporag_save_dir:
            rag_kwargs["save_dir"] = hipporag_save_dir
        if hipporag_llm_path:
            rag_kwargs["llm_model_path"] = hipporag_llm_path
        if hipporag_embedding_model:
            rag_kwargs["embedding_model_name"] = hipporag_embedding_model
        if graph_weight is not None:
            rag_kwargs["graph_weight"] = graph_weight
        if fusion:
            rag_kwargs["fusion"] = fusion
        if passage_node_weight is not None:
            rag_kwargs["passage_node_weight"] = passage_node_weight
        rag = HippoRAGRag(**rag_kwargs)
        print("[RAG] HippoRAG (knowledge graph + PPR) ready — retrieval is query-specific per outcome / intervention x outcome")
    else:
        rag = None
        print("[RAG] DISABLED — vanilla baseline (no literature)")

    n_conds   = len(conditions)
    n_items   = len(item_keys)
    api_est   = n_items * (3 + n_conds * n_samples)

    print(f"\n{'='*60}")
    print(f"  Tier 3 Pipeline — Relative Scoring + Effect Estimation")
    print(f"  Group  : Farah, Jing, Max, Marcia")
    print(f"  Model  : {MODEL_CONFIG[model_key]['model']}")
    print(f"  Items  : {n_items}")
    print(f"  Conds  : {n_conds} interventions")
    print(f"  RAG    : {'ON — query-specific per outcome / pair' if use_rag else 'OFF — vanilla baseline'}")
    print(f"  Method : Step 1=Relative scoring (-3..+3, ties allowed), Step 2=Effect sizes")
    print(f"  Est API: ~{api_est} LLM calls (+ cheap retrieval calls if RAG is ON)")
    print(f"  Dry run: {dry_run}")
    print(f"{'='*60}\n")

    client  = None if dry_run else make_client(model_key)

    # {item_key: {condition: effect}}. dry_run makes up random numbers, so
    # it neither reads nor writes the checkpoint.
    ckpt_file = _checkpoint_path(output_dir, model_key, rag_tag)
    results   = {} if dry_run or not resume else load_checkpoint(ckpt_file, survey_items)

    if results:
        print(f"[RESUME] {len(results)}/{n_items} outcome(s) already complete in "
              f"{ckpt_file} — re-running only the rest")

    for i, (item_key, item_spec) in enumerate(survey_items.items()):
        if item_key in results:
            print(f"\n[{i+1}/{n_items}] Outcome: {item_key} — already done, skipping")
            continue

        print(f"\n[{i+1}/{n_items}] Outcome: {item_key}")

        item_effects = predict_outcome(
            client       = client,
            model_key    = model_key,
            conditions   = conditions,
            item_key     = item_key,
            item_spec    = item_spec,
            rag          = rag,
            use_rag      = use_rag,
            n_samples    = n_samples,
            dry_run      = dry_run,
        )
        results[item_key] = item_effects

        if not dry_run:
            save_checkpoint(ckpt_file, results)
            # Also refresh the CSVs, so an interrupted run leaves the same
            # four files every other tier-3 run produces -- partial, but in
            # the standard format rather than only a JSON checkpoint.
            write_outputs(results, conditions, item_keys, raw_file, cal_file,
                          output_dir, label, timestamp)

    # ── Save CSVs ────────────────────────────────────────────────────
    sub_raw, sub_cal = write_outputs(
        results, conditions, item_keys, raw_file, cal_file,
        output_dir, label, timestamp, verbose=True,
    )

    print(f"\n{'='*60}")
    print(f"  ✅ TIER 3 COMPLETE")
    print(f"  Internal raw        : {raw_file}")
    print(f"  Internal calibrated : {cal_file}")
    print(f"")
    if not dry_run and os.path.exists(ckpt_file):
        os.remove(ckpt_file)   # run finished; nothing left to resume

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
    # Derived from MODEL_CONFIG rather than hardcoded: the old literal list
    # had drifted (it still offered "qwen_local", which no longer exists, and
    # rejected "qwen38", which does).
    parser.add_argument("--model",      type=str,  default="gpt4o",
                        choices=sorted(MODEL_CONFIG),
                        help="'qwen38' (used for both report runs) talks to a self-hosted "
                             "vLLM server — see --base_url / --served_model_name and the "
                             "README. 'qwen_hf' loads a local checkpoint in-process via plain "
                             "transformers instead, no server needed, but cannot share the GPU "
                             "with a vLLM instance.")
    parser.add_argument("--n_samples",  type=int,  default=3,
                        help="Ensemble size for effect estimation (default: 3)")
    parser.add_argument("--dry_run",    action="store_true")
    parser.add_argument("--output_dir", type=str,  default=None,
                        help=f"Default: {RESULTS_DIR}, resolved from this file's location "
                             "so the destination does not depend on where you launch from.")
    parser.add_argument("--quick",      action="store_true",
                        help="Run only 3 items for quick testing")
    parser.add_argument("--no_rag",     action="store_true",
                        help="Disable retrieval entirely — the no-literature ablation baseline")
    parser.add_argument("--base_url",   type=str, default=None,
                        help="Override the qwen_local server URL (default: http://localhost:8000/v1)")
    parser.add_argument("--served_model_name", type=str, default=None,
                        help="Override the model/checkpoint for --model. For qwen_local, must "
                             "match what the server is registered as (e.g. via `vllm serve "
                             "<name>`). For qwen_hf, a local checkpoint directory or HF repo id.")
    parser.add_argument("--hipporag_save_dir", type=str, default=None,
                        help="Index/graph storage dir (default: ../hipporag_index, relative to "
                             "this file)")
    parser.add_argument("--hipporag_llm_path", type=str, default=None,
                        help="Local checkpoint used for OpenIE + fact reranking (default: the "
                             "same Qwen3.5-27B checkpoint as --model qwen_hf)")
    parser.add_argument("--hipporag_embedding_model", type=str, default=None,
                        help="sentence-transformers model for chunk/entity/fact/query encoding "
                             "(default: Qwen/Qwen3-Embedding-4B)")
    parser.add_argument("--graph_weight", type=float, default=1.0,
                        help="Share of the final ranking decided by "
                             "the knowledge graph (PPR); the remainder goes to dense retrieval. "
                             "1.0 (default) is stock HippoRAG with no fusion. 0.7 means 70%% "
                             "graph / 30%% dense. Any value below 1.0 puts the weight in the "
                             "output filenames (rag_hipporag_g70) and in the checkpoint name, so "
                             "runs at different weights never overwrite or resume from each other.")
    parser.add_argument("--fusion", type=str, default=None,
                        choices=["pit", "minmax", "rrf"],
                        help="--graph_weight < 1.0 only: how the two rankings are normalised "
                             "before the weighted sum. 'pit' (default) percentile-rank, which "
                             "keeps the weight meaning the same across queries; 'minmax'; "
                             "'rrf' reciprocal rank fusion.")
    parser.add_argument("--passage_node_weight", type=float, default=None,
                        help="hipporag's PPR seed weight for passage nodes, i.e. how much the "
                             "dense signal seeds the PPR restart vector. Defaults to the "
                             "library's 0.05 whether or not --graph_weight blending is on, so "
                             "the graph arm of the fusion is always stock HippoRAG's ranking.")
    parser.add_argument("--no_resume", action="store_true",
                        help="Ignore any existing per-outcome checkpoint and re-predict every "
                             "outcome from scratch. By default a run resumes where the last one "
                             "stopped, which is what makes an interrupted run recoverable.")
    args = parser.parse_args()

    if args.base_url:
        MODEL_CONFIG[args.model]["base_url"] = args.base_url
    if args.served_model_name:
        MODEL_CONFIG[args.model]["model"] = args.served_model_name

    item_subset = None
    if args.quick:
        item_subset = ["trust_competence_1", "concern_1", "policy_general_1"]

    run_tier3(
        model_key              = args.model,
        n_samples              = args.n_samples,
        dry_run                = args.dry_run,
        output_dir             = args.output_dir,
        item_subset            = item_subset,
        use_rag                = not args.no_rag,
        hipporag_save_dir       = args.hipporag_save_dir,
        hipporag_llm_path       = args.hipporag_llm_path,
        hipporag_embedding_model = args.hipporag_embedding_model,
        graph_weight            = args.graph_weight,
        fusion                  = args.fusion,
        passage_node_weight     = args.passage_node_weight,
        resume                  = not args.no_resume,
    )
