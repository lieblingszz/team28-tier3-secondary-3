#!/usr/bin/env bash
# run_hipporag.sh — the entry point for a full Tier 3 run with HippoRAG
# retrieval: OpenIE knowledge graph + Personalized PageRank over corpus/*.pdf,
# with local Qwen3-Embedding-4B embeddings. No OpenAI/GPT calls anywhere in
# the retrieval path.
#
# Everything is resolved relative to this script, so it runs correctly from
# any working directory:
#
#   ./code/scripts/run_hipporag.sh
#   cd code/scripts && ./run_hipporag.sh         # identical result
#
# The corpus and survey data ship inside code/. The graph index is reused when
# present and rebuilt when absent. The following must be provided:
#
#   code/model/    empty — download the checkpoints from Hugging Face first,
#                  see README.md, "Downloading the models"
#   the venv       create it at code/.venv-hipporag, or point VENV elsewhere
#
#   VENV=/path/to/.venv-hipporag MODELS_DIR=/path/to/models ./run_hipporag.sh
#
# ── The two runs in the report ──────────────────────────────────────────
#
#   MODEL=qwen38 ./run_hipporag.sh                      # stock HippoRAG
#   MODEL=qwen38 ./run_hipporag.sh --graph_weight 0.7   # 70% graph / 30% dense
#
# These are the two Qwen series in ../results/. They differ in one thing
# only: the second fuses HippoRAG's PPR ranking with the dense passage
# ranking at query time. Same corpus, same graph index, same checkpoint, same
# prompts — no re-indexing. The weight goes into the output filename
# (rag_hipporag_g70) AND into the checkpoint name, so the two can never
# overwrite or resume from each other.
#
# ── Two model backends ──────────────────────────────────────────────────
#
#   ./run_hipporag.sh                    # Qwen3.5-27B, loaded in-process (default)
#   MODEL=qwen38 ./run_hipporag.sh       # Qwen3.8-27B, served by vLLM (the report runs)
#
# Why the difference matters. With the default, one 27B checkpoint is loaded
# here and shared between the prediction calls and HippoRAG's fact reranking.
# Qwen3.8-27B cannot work that way: vLLM holds ~64 GiB of the 80 GiB card,
# leaving no room for HippoRAG to load a second 27B (~50 GiB). So MODEL=qwen38
# also points HippoRAG's own LLM calls at the same server via
# HIPPORAG_LLM_BASE_URL, leaving only the 4B embedding model (~8 GiB)
# in-process. One copy of the weights, ~72 GiB total.
#
# MODEL=qwen38 needs the vLLM server up first, from the separate .venv-vllm
# (vllm 0.19.1 — the repo's pinned 0.8.5 cannot serve the qwen3_5
# architecture). See README.md for the serve command.
#
# ── Other usage ─────────────────────────────────────────────────────────
#
#   ./run_hipporag.sh --quick            # 3 outcomes only, fast end-to-end check
#   ./run_hipporag.sh --n_samples 5      # extra args pass through to 01_tier3_hipporag.py
#   ./run_hipporag.sh --no_resume        # ignore the checkpoint, re-predict everything
#
# Note: --dry_run still builds the HippoRAG index first (the pipeline
# constructs the `rag` object before checking dry_run) — it only skips the
# per-outcome LLM calls. Use --quick for a genuinely fast check.
#
# The first run builds the index (OpenIE + graph construction over
# corpus/*.pdf), which is slow because every chunk needs an LLM call for
# entity/triple extraction. Later runs are fast: the index is cached in
# hipporag_index/ and content-hash deduped, so only new or changed chunks are
# reprocessed.
#
# ── Logs ────────────────────────────────────────────────────────────────
#
#   code/scripts/run_hipporag.log               — always the most recent run
#   code/results/run_hipporag_<model>_<ts>.log  — kept per run, never overwritten
#
# The timestamped copy matters because a plain `tee run_hipporag.log`
# truncates: re-running would destroy the evidence from the previous run,
# which is exactly what you need when a long run dies partway.

set -o pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # code/scripts
ROOT="$(cd "$DIR/.." && pwd)"                         # code
# Virtual environments live directly under code/ alongside scripts/, data/,
# corpus/, and results/. Override with VENV when needed.
VENV="${VENV:-$ROOT/.venv-hipporag}"
TS="$(date +%Y%m%d_%H%M%S)"
MODEL="${MODEL:-qwen_hf}"
VLLM_URL="${VLLM_URL:-http://localhost:8000/v1}"
LOG="$DIR/run_hipporag.log"
LOG_KEEP="$ROOT/results/run_hipporag_${MODEL}_${TS}.log"

# The pipeline writes here too — its default output_dir resolves to this same
# directory from the script's own location, not from the caller's cwd.
mkdir -p "$ROOT/results"

if [ ! -x "$VENV/bin/python" ]; then
    echo "error: no python at $VENV/bin/python" >&2
    echo "       build it per code/scripts/README.md, or set VENV=<path>" >&2
    exit 1
fi

case "$MODEL" in
  qwen_hf)
    # One checkpoint loaded in-process, shared by prediction and reranking.
    ;;
  qwen38)
    # Both the prediction calls and HippoRAG's reranking go to vLLM, so the
    # server must be up before we start — otherwise the failure appears
    # ~1 minute in, after the embedding model has already loaded.
    if ! curl -s -m 5 -o /dev/null "$VLLM_URL/models"; then
        echo "error: no vLLM server at $VLLM_URL" >&2
        echo "       start it first — see code/scripts/README.md, 'Start the Qwen server'." >&2
        exit 1
    fi
    export HIPPORAG_LLM_BASE_URL="$VLLM_URL"
    export HIPPORAG_LLM_MODEL="qwen38"
    ;;
  *)
    echo "error: MODEL must be qwen_hf or qwen38 (got '$MODEL')" >&2
    exit 1
    ;;
esac

# Everything below (GPU header + pipeline output) goes through one tee, so the
# log always opens with the GPU state the run actually started from — the
# first full run died with no traceback and no way to tell afterwards whether
# the card had been free at launch.
{
    echo "════════════════════════════════════════════════════════════════"
    echo "run_hipporag.sh   started: $(date -Is)"
    echo "args     : $*"
    echo "model    : $MODEL"
    echo "python   : $VENV/bin/python"
    echo "log      : $LOG_KEEP"
    if [ -n "$HIPPORAG_LLM_BASE_URL" ]; then
        echo "hipporag LLM : $HIPPORAG_LLM_BASE_URL (served '$HIPPORAG_LLM_MODEL')"
        curl -s -m 5 "$VLLM_URL/models"; echo
    else
        echo "hipporag LLM : in-process checkpoint"
    fi
    echo "════════════════════════════════════════════════════════════════"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi
        echo
        echo "-- processes already holding GPU memory --"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory \
                   --format=csv 2>/dev/null || true
    else
        echo "nvidia-smi not found -- cannot report GPU state"
    fi
    echo "════════════════════════════════════════════════════════════════"
    echo

    "$VENV/bin/python" -u "$DIR/01_tier3_hipporag.py" --model "$MODEL" "$@" 2>&1
    status=$?

    echo
    echo "════════════════════════════════════════════════════════════════"
    echo "run_hipporag.sh   finished: $(date -Is)   exit status: $status"
    echo "════════════════════════════════════════════════════════════════"
    exit $status
} 2>&1 | tee "$LOG" "$LOG_KEEP"

# Propagate python's exit status, not tee's.
exit "${PIPESTATUS[0]}"
