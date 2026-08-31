# Reproducing the Tier 3 submission

This directory contains Team 28's two-stage prediction pipeline for the
Silicon Sample Benchmark.

1. `01_tier3_hipporag.py` uses Qwen3.8-27B and HippoRAG to predict treatment
   effects for 16 interventions and 13 outcomes.
2. `02_ensemble_tier3.py` combines the two Qwen runs with frozen GPT-4o,
   GPT-5.4, and Claude Sonnet 4.6 predictions.

Run all commands below from the repository root.

## Directory layout

```text
code/
├── .venv/              environment for the ensemble stage
├── .venv-hipporag/     environment for HippoRAG and Qwen prediction
├── .venv-vllm/         environment for the Qwen vLLM server
├── corpus/             43 retrieval PDFs
├── data/               survey items and intervention texts
├── hipporag_index/     graph and embeddings, if available
├── model/              downloaded model weights
├── results/            member predictions, logs, and ensemble output
└── scripts/            pipeline code
```

The three virtual environments live directly under `code/`, alongside the
other pipeline directories. They are separate because HippoRAG, Transformers,
Torch, and vLLM require different versions.

## Included results

`code/results/` contains five calibrated member files and the final ensemble.
Each file has 208 rows in `condition,outcome,ate` format.

| Member | Source |
|---|---|
| Qwen3.8-27B, stock HippoRAG | this pipeline, `graph_weight=1.0` |
| Qwen3.8-27B, graph+dense fusion | this pipeline, `graph_weight=0.7` |
| GPT-4o | frozen member file |
| GPT-5.4 | frozen member file |
| Claude Sonnet 4.6 | frozen member file |

For all details about GPT-4o, GPT-5.4, and Claude Sonnet 4.6, refer to
[team28-tier3-secondary-1](https://github.com/lieblingszz/team28-tier3-secondary-1).

## 1. Create the environments

Python 3.11.6 and one NVIDIA H100 80 GB were used for the Qwen runs. The
ensemble stage needs only pandas and a CPU.

Create the ensemble environment:

```bash
python3.11 -m venv code/.venv
code/.venv/bin/pip install --upgrade pip
code/.venv/bin/pip install "pandas==3.0.5" "numpy==2.2.6" "huggingface_hub[cli]"
```

Create the HippoRAG environment:

```bash
python3.11 -m venv code/.venv-hipporag
code/.venv-hipporag/bin/pip install --upgrade pip
code/.venv-hipporag/bin/pip install hipporag
code/.venv-hipporag/bin/pip install \
  "transformers==5.15.0" "accelerate==1.14.0" "tokenizers==0.22.2"
code/.venv-hipporag/bin/pip install "pymupdf==1.28.2"
```

The explicit Transformers upgrade is required for the Qwen3.8 architecture.
`pip check` may report conflicts with HippoRAG's declared pins; the pipeline
uses its own local LLM and embedding adapters instead of those conflicting
HippoRAG backends. Exact installed versions are listed in
`requirements-hipporag.txt`.

Create the vLLM environment:

```bash
python3.11 -m venv code/.venv-vllm
code/.venv-vllm/bin/pip install --upgrade pip
code/.venv-vllm/bin/pip install "vllm==0.19.1"
```

## 2. Download the models

Stage 1 requires approximately 62 GB of model weights. Stage 2 does not need
either model.

```bash
code/.venv/bin/hf download Qwen/Qwen3.8-27B \
  --local-dir code/model/Qwen3.8-27B

code/.venv/bin/hf download Qwen/Qwen3-Embedding-4B \
  --local-dir code/model/Qwen3-Embedding-4B
```

You may instead set `MODELS_DIR` when the models already exist elsewhere:

```bash
MODELS_DIR=/path/to/models MODEL=qwen38 ./code/scripts/run_hipporag.sh --quick
```

## 3. Start the Qwen server

Start Qwen3.8-27B before running Stage 1:

```bash
nohup code/.venv-vllm/bin/vllm serve code/model/Qwen3.8-27B \
  --served-model-name qwen38 \
  --port 8000 \
  --gpu-memory-utilization 0.78 \
  --max-model-len 32768 \
  --additional-config '{"gdn_prefill_backend":"triton"}' \
  > code/vllm.log 2>&1 &

curl -s http://localhost:8000/v1/models
```

Wait until the final command returns the served model.

## 4. Prepare the HippoRAG index

The default index location is `code/hipporag_index/`. If the directory is
present, Stage 1 reuses it. If it is absent, build it from the 43 PDFs:

```bash
HIPPORAG_LLM_BASE_URL=http://localhost:8000/v1 \
HIPPORAG_LLM_MODEL=qwen38 \
HIPPORAG_EMBED_BATCH=8 \
code/.venv-hipporag/bin/python code/scripts/hipporag_rag.py --build
```

The Qwen server must remain running during the build. A new graph is
stochastic, so it will not reproduce the frozen Qwen member files byte for
byte.

## 5. Run the Qwen predictions

Use the launcher so the vLLM endpoint is checked and each run is logged.

```bash
# Short end-to-end check: three survey items
MODEL=qwen38 ./code/scripts/run_hipporag.sh --quick

# Submitted retrieval configurations
MODEL=qwen38 ./code/scripts/run_hipporag.sh
MODEL=qwen38 ./code/scripts/run_hipporag.sh --graph_weight 0.7
```

The first run uses stock HippoRAG ranking. The second combines PIT-normalized
graph and dense rankings as `0.7 × graph + 0.3 × dense`.

The script retrieves evidence separately for each outcome and each
intervention/outcome pair. It then:

1. scores all interventions relative to one another;
2. estimates absolute effects for each intervention;
3. aggregates 44 survey items into the 13 official outcomes.

A checkpoint is saved after every completed item. Interrupted runs resume
automatically; pass `--no_resume` to start again.

Each run writes raw, calibrated, and submission-format files to
`code/results/`. Logs are written to `code/scripts/run_hipporag.log` and a
timestamped file under `code/results/`.

### Calibration

`01_tier3_hipporag.py` currently writes calibrated files using `b=0.56`. The
frozen member files used by this submission were calibrated with `b=0.35`.
When regenerating a submitted member, use the `submission_raw` output and
apply `calibrated_ate = raw_ate × 0.35`.

## 6. Build the final ensemble

The five calibrated input filenames are fixed in the `MEMBERS` mapping in
`02_ensemble_tier3.py`.

```bash
code/.venv/bin/python code/scripts/02_ensemble_tier3.py
```

The calculation gives every model family equal weight:

```text
Qwen_mean = (Qwen_stock + Qwen_fused) / 2
Final = (Qwen_mean + GPT-4o + GPT-5.4 + Claude) / 4
```

The output is:

```text
code/results/tier3_submission_calibrated_tier3_ensemble_direct.csv
```

It should match:

```text
predictions/team_28_T3_secondary-3_v1.csv
```

Verify the two files directly:

```bash
cmp code/results/tier3_submission_calibrated_tier3_ensemble_direct.csv \
    predictions/team_28_T3_secondary-3_v1.csv
```

## Default paths

Paths are resolved from the scripts, so commands work from any directory.
These environment variables override the defaults when needed:

| Variable | Default |
|---|---|
| `VENV` | `code/.venv-hipporag` |
| `MODELS_DIR` | `code/model` |
| `CORPUS_DIR` | `code/corpus` |
| `DATA_DIR` | `code/data` |
| `HIPPORAG_INDEX` | `code/hipporag_index` |
| `VLLM_URL` | `http://localhost:8000/v1` |
