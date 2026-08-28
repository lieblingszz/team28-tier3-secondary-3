# Team 28 — Tier 3, secondary entry (Claude + RAG)

Silicon Sample Benchmark, Tier 3. This repo is the **secondary-2** entry: the same RAG pipeline as the primary GPT-4o entry, run with Claude instead.

## Structure

- `code/` — everything needed to run the pipeline: `tier3_pipeline_parallel_fulltext.py` (run this), `tier3_pipeline.py` (imported module — not legacy, required), `rag_vector_db.py`, the prebuilt `chroma_rag_db/` vector index, and `LLMmegastudy-main/simulation/` (the 16 intervention texts + 13 outcome item definitions).
- `results/` — the submitted file: `tier3_submission_calibrated_tier3_claude_rag_parallel_fulltext_20260826_134941.csv` (208 rows, `condition,outcome,ate`), plus its raw (pre-calibration) counterpart.
- `registration.md` — the method registration form for this entry.
- `requirements.txt` — Python dependencies.

## Reproducing this run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key"
cd code
python3 tier3_pipeline_parallel_fulltext.py --model claude --n_samples 3
```

Output lands in `code/results/`. Use the `tier3_submission_calibrated_*` file.

## Calibration

Raw predictions are multiplied by **b = 0.35** (`CALIBRATION_FACTOR` in `tier3_pipeline.py`), fit against real Voelkel et al. (2025/2026) ground truth — see `registration.md` §G.3 for the full derivation. Same constant used across every model in this pipeline family (GPT-4o, GPT-5, Claude), since it corrects a property of LLM-predicted effect sizes generally, not something GPT-4o-specific.
