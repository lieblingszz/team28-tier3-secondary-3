# `code/` — reproducing the Tier 3 submission

Team 28, Silicon Sample Benchmark.

```
code/
├── scripts/          this folder — the pipeline
├── results/          model outputs
├── corpus/           43 PDFs, the retrieval corpus
├── data/             survey_items.json + stimuli/
├── hipporag_index/   prebuilt graph + embeddings (~76 MB)
└── model/            EMPTY — download the checkpoints first (see below)
```

Everything except `model/` and the virtualenvs ships here, so the folder is
self-contained: no path reaches outside `code/`.

Two stages:

| | script | what it does | venv |
|---|---|---|---|
| **01** | `01_tier3_hipporag.py` | HippoRAG retrieval → per-condition × outcome ATEs for one model | `.venv-hipporag` |
| **02** | `02_ensemble_tier3.py` | averages the finished Tier 3 files into one ensemble | `.venv` |

Stage 01 is normally launched through `run_hipporag.sh`, which checks the
vLLM server, records GPU state and tees a timestamped log. Stage 02 needs no
GPU, no models and about a second — it is the part a reviewer can run.

Supporting modules, imported rather than run:

- `hipporag_rag.py` — the retrieval backend: OpenIE knowledge graph +
  Personalized PageRank, with the optional graph/dense fusion. Also runnable
  standalone for `--build`, `--test` and `--sweep`.
- `utils.py` — outcome schema, prompt builders, PDF/text processing, LLM
  client helpers.

The VectorRAG (ChromaDB / TF-IDF) backend that the development tree also
carries has been removed here, since no submitted result used it — see
[What was removed](#what-was-removed).

---

## Paths

Every path resolves from the scripts' own location, never the working
directory, so `./code/scripts/run_hipporag.sh` and
`cd code/scripts && ./run_hipporag.sh` behave identically. Nothing needs
configuring; each default is overridable by environment variable if you want
to point somewhere else:

| variable | default | holds |
|---|---|---|
| `CORPUS_DIR` | `../corpus` | the 43 retrieval PDFs |
| `HIPPORAG_INDEX` | `../hipporag_index` | the graph + embeddings — see [The graph index](#the-graph-index); **if this folder is missing from the bundle, build it with [Building the index from scratch](#building-the-index-from-scratch)** (~16 min) |
| `DATA_DIR` | `../data` | `survey_items.json`, `stimuli/` |
| `MODELS_DIR` | `../model` | the downloaded checkpoints |
| `VENV` | `../../.venv-hipporag` | the HippoRAG virtualenv |

```bash
MODELS_DIR=/path/to/models VENV=/path/to/.venv-hipporag ./code/scripts/run_hipporag.sh
```

`VENV` is the one default that points outside `code/`: a Python virtualenv
bakes absolute paths into its scripts, so it cannot be relocated and has to be
built where it will be used. See [Environment](#environment).

**Stage 02 needs none of this** — no models, no corpus, no GPU. It reads and
writes `code/results/` only, and it is the part a reviewer can run directly.

---

## What is in `results/`

Six files, all in the official submission format (`condition,outcome,ate`,
208 rows) and all calibrated at **b = 0.35**.

Five are the ensemble's members; the sixth is its output.

| file | model | how it was produced |
|---|---|---|
| `…_qwen38_rag_hipporag_20260830_093608.csv` | Qwen3.8-27B | **this code**, stage 01, `--graph_weight 1.0` |
| `…_qwen38_rag_hipporag_g70_20260830_102830.csv` | Qwen3.8-27B | **this code**, stage 01, `--graph_weight 0.7` |
| `…_gpt4o_rag_parallel_fulltext_20260825_132946.csv` | GPT-4o | [team28-tier3-primary](https://github.com/lieblingszz/team28-tier3-primary) |
| `…_gpt5_rag_parallel_fulltext_20260826_135120.csv` | GPT-5 | [team28-tier3-primary](https://github.com/lieblingszz/team28-tier3-primary) |
| `…_claude_rag_parallel_fulltext_20260826_134941.csv` | Claude Sonnet 4.6 | [team28-tier3-primary](https://github.com/lieblingszz/team28-tier3-primary) |
| `…_tier3_ensemble_direct.csv` | — | **this code**, stage 02, from the five above |

### The two Qwen runs

These are the only two files this code produced, and they differ in exactly
one setting:

| | graph weight | retrieval |
|---|---|---|
| `…_hipporag_20260830_093608.csv` | **1.0** (the default) | stock HippoRAG — PPR ranking used as-is, no fusion |
| `…_hipporag_g70_20260830_102830.csv` | **0.7** | `0.7 × graph + 0.3 × dense`, PIT-normalised |

```bash
MODEL=qwen38 ./code/scripts/run_hipporag.sh                      # → stock file
MODEL=qwen38 ./code/scripts/run_hipporag.sh --graph_weight 0.7   # → g70 file
```

Evidence, from the runs' own logs: the 09:36 run records
`args : --hipporag_llm_path code/model/Qwen3.8-27B` and prints no fusion
banner, so it took the stock short-circuit path at `graph_weight >= 1.0`. The
10:28 run records the same path plus `--graph_weight 0.7` and prints
`Fusion on: 0.70 graph / 0.30 dense, pit normalisation, passage_node_weight=0.05`.
Both ran against the same index directory, which was last written at 09:12,
before either started — so the pair isolates the fusion weight and nothing
else.

`1.0` is the default, so it never appears in a filename; only a weight below
1.0 gets a `_g<NN>` tag. Neither file carries a `full_text` annotation: an
earlier pair of runs did, as a manual note that matched no setting, and those
files have been superseded (see *Which model built the graph*).

### The three API-model files

GPT-4o, GPT-5 and Claude were run from a **separate codebase**,
[team28-tier3-primary](https://github.com/lieblingszz/team28-tier3-primary),
with parallel full-text retrieval rather than HippoRAG — hence the different
`rag_parallel_fulltext` tag. Nothing in `code/scripts/` reproduces them; they
are inputs here, included so stage 02 runs standalone. Re-running stage 01
will not regenerate them.

---

## Environment

Python 3.11.6, one H100 80 GB. **Three separate venvs**, and they cannot be
merged — this is the single most common way to break the pipeline.

| venv | holds | why separate |
|---|---|---|
| `.venv` | pandas 3.0.5, numpy 2.2.6, transformers 5.15.0, torch 2.6.0, vllm 0.8.5 | the project's main pins |
| `.venv-hipporag` | hipporag 2.0.0a3 + transformers 5.15.0, torch 2.5.1+cu124 | `pip install hipporag` pins transformers 4.45.2 / torch 2.5.1 / vllm 0.6.6, which silently downgrades `.venv` and breaks the qwen3_5 architecture entirely |
| `.venv-vllm` | vllm 0.19.1 | the main pin, vllm 0.8.5, predates the qwen3_5 architecture and cannot serve Qwen3.8-27B at all |

### Building `.venv-hipporag`

Build it beside `code/`, or anywhere, and point `VENV` at it. A virtualenv
records absolute paths in its own scripts, so it cannot be copied into
`code/` and shipped.

```bash
python3 -m venv .venv-hipporag
./.venv-hipporag/bin/pip install --upgrade pip
./.venv-hipporag/bin/pip install hipporag
# force past hipporag's own transformers pin — required to load a
# qwen3_5-architecture checkpoint
./.venv-hipporag/bin/pip install "transformers==5.15.0" "accelerate==1.14.0" "tokenizers==0.22.2"
./.venv-hipporag/bin/pip install "pymupdf==1.28.2"
```

`pip check` will afterwards report hipporag's declared pins as incompatible.
**This is expected.** `hipporag_rag.py` monkeypatches HippoRAG's LLM and
embedding factories to use `transformers` directly and
`sentence-transformers`, so none of the conflicting code paths (GritLM,
NV-Embed-v2, Contriever, the vllm-offline OpenIE path) is ever reached. See
`requirements-hipporag.txt` for the full pin list and reasoning.

### Downloading the models

`code/model/` ships **empty**. The checkpoints are far too large to include —
Qwen3.8-27B is ~54 GB in bf16 — so they have to be fetched from Hugging Face
before stage 01 can run. Stage 02 does not need them.

Two downloads, ~62 GB total:

```bash
pip install "huggingface_hub[cli]"

# the prediction model — required for both report runs (~54 GB)
hf download Qwen/Qwen3.8-27B \
    --local-dir code/model/Qwen3.8-27B

# the embedding model — chunk / entity / fact / query encoding (~8 GB)
hf download Qwen/Qwen3-Embedding-4B \
    --local-dir code/model/Qwen3-Embedding-4B
```

Older `huggingface_hub` installs use `huggingface-cli download` with the same
arguments. A gated or rate-limited download needs `hf auth login` first.

Expected layout afterwards — **two** checkpoints, nothing else:

```
code/model/
├── Qwen3.8-27B/          served by vLLM  (--model qwen38, both report runs)
└── Qwen3-Embedding-4B/   loaded in-process by HippoRAG
```

The embedding download is optional: if `code/model/Qwen3-Embedding-4B/` is
absent, `sentence-transformers` fetches the hub id
`Qwen/Qwen3-Embedding-4B` once into `~/.cache/huggingface` instead. Putting it
in `model/` keeps everything in one place and works offline. Either way the
*name* hipporag sees stays `Qwen/Qwen3-Embedding-4B` — see the warning below.

**Qwen3.8-27B is the only LLM checkpoint you need.** It built the shipped
index and it answers every query — see
[Which model built the graph](#which-model-built-the-graph) below. On the
`qwen38` route the local checkpoint directory is never actually opened,
because `HIPPORAG_LLM_BASE_URL` sends HippoRAG's fact-reranking calls to the
same vLLM server as the predictions; `LLM_PATH` is read only for its
**basename**, which is what names the index directory. So `LLM_PATH`
resolving to a directory that does not exist is fine, as long as its basename
stays `Qwen3.8-27B` and the graph is not rebuilt.

> **Never substitute a local path for a model *name*.** hipporag derives its
> working directory from `llm_name` + `embedding_model_name`, so passing
> `…/code/model/Qwen3-Embedding-4B` as the *name* creates a second index
> directory, orphans the prebuilt graph, and silently starts a multi-hour
> rebuild. `hipporag_rag.py` keeps the two separate: `EMBEDDING_MODEL` is the
> stable name, `_embedding_weights()` decides where the weights load from.

Point `MODELS_DIR` at an existing copy to skip the download entirely:

```bash
MODELS_DIR=/shared/models ./code/scripts/run_hipporag.sh
```

### Serving Qwen3.8-27B

Both report runs use `MODEL=qwen38`, which needs the server up **first** —
otherwise the run fails about a minute in, after the embedding model has
already loaded.

```bash
nohup ./.venv-vllm/bin/vllm serve "$MODELS_DIR/Qwen3.8-27B" \
    --served-model-name qwen38 --port 8000 \
    --gpu-memory-utilization 0.78 --max-model-len 32768 \
    --additional-config '{"gdn_prefill_backend":"triton"}' \
    > vllm.log 2>&1 &

curl -s http://localhost:8000/v1/models      # wait for this to answer
```

vLLM holds ~64 GiB of the card. With `MODEL=qwen38` the pipeline also points
HippoRAG's own LLM calls at that same server (`HIPPORAG_LLM_BASE_URL`), so
only the 4 B embedding model (~8 GiB) is loaded in-process: one copy of the
27 B weights, ~72 GiB total. This is why the two cannot both load the
checkpoint — `MODEL=qwen_hf` loads it in-process instead and must not be run
alongside a vLLM server.

**No OpenAI or GPT model is used anywhere in the retrieval path.** The graph
was extracted by a local Qwen checkpoint and the embeddings are local
`Qwen3-Embedding-4B`. The `openai` package appears in `.venv-hipporag` only as
a transitive dependency and as the client for our own OpenAI-compatible vLLM
endpoint.

---

## Stage 01 — run the HippoRAG pipeline

```bash
MODEL=qwen38 ./code/scripts/run_hipporag.sh --quick   # 3 outcomes, end-to-end check
MODEL=qwen38 ./code/scripts/run_hipporag.sh           # full run, all 13 outcomes
```

### The two runs in the report

```bash
MODEL=qwen38 ./code/scripts/run_hipporag.sh                      # stock HippoRAG
MODEL=qwen38 ./code/scripts/run_hipporag.sh --graph_weight 0.7   # 70% graph / 30% dense
```

These produced the two Qwen files in `../results/` — see
[What is in `results/`](#what-is-in-results) for which file came from which
command. They differ in exactly one thing: the second fuses HippoRAG's PPR
ranking with the dense passage ranking at query time. Same corpus, same graph
index, same checkpoint, same prompts — **no re-indexing**, only query-time
scoring changes.

`--graph_weight 0.7` is not "graph instead of vectors." Stock HippoRAG
already uses the dense signal, as the PPR restart seeds
(`passage_node_weight=0.05`), and that stays at 0.05 in the fused run too. The
fusion leaves HippoRAG's ranking untouched and adds a second, explicit channel
beside it: both arms are converted to percentile ranks, then combined as
`0.7 × graph + 0.3 × dense`. Percentile-rank (PIT) normalisation is what makes
the weight mean the same thing on every query — PPR scores are power-law
spiked, so under min-max the effective weight drifts query to query.
`--fusion minmax|rrf` switch the normalisation for a robustness check.

Any weight below 1.0 goes into **both** the output filename
(`rag_hipporag_g70`) and the checkpoint name, so runs at different weights can
never overwrite or resume from one another.

### The graph index

`code/hipporag_index/` holds the finished OpenIE graph, the entity/fact/chunk
embeddings and the LLM response cache for the 43-PDF corpus — about 82 MB:

```
hipporag_index/
├── local_Qwen3.8-27B_Qwen_Qwen3-Embedding-4B/
│   ├── graph.pickle          5.2 MB   the knowledge graph itself
│   ├── fact_embeddings/       39 MB
│   ├── entity_embeddings/     25 MB
│   └── chunk_embeddings/     5.5 MB
├── llm_response_cache.jsonl  2.4 MB   replayed on a rebuild, so no call repeats
└── openie_results_ner_…json  4.3 MB
```

If it is present, a run starts querying immediately. If it is **absent**, the
first run rebuilds it automatically from `corpus/` — correct, but it silently
turns a 50-minute run into a 65-minute one, so build it deliberately instead
(next section). A build is also forced by `--build`, and the index is
content-hash deduped, so adding one PDF to `corpus/` reprocesses only that PDF.

Both submitted runs used this same index; no PDF is newer than it, which is
what makes the stock-vs-fused pair a clean single-variable comparison.

### Building the index from scratch

**Needed only if `hipporag_index/` was not shipped** — it is over half the
bundle by size, so it may have been dropped to keep the upload small.
Everything required to rebuild it is here: the 43 PDFs in `corpus/`, the
chunker in `utils.py` and the extraction code in `hipporag_rag.py`.

Prerequisites are the same as for a prediction run — `.venv-hipporag` built,
both models downloaded into `model/`, and **the vLLM server already up**
(see *Serving Qwen3.8-27B* above). Then, from the repository root:

```bash
HIPPORAG_LLM_BASE_URL=http://localhost:8000/v1 \
HIPPORAG_LLM_MODEL=qwen38 \
HIPPORAG_EMBED_BATCH=8 \
./.venv-hipporag/bin/python code/scripts/hipporag_rag.py --build
```

Roughly **16 minutes** on one H100 with the server up. Then run stage 01
normally; it will find the index and not rebuild.

Three things in that command are not optional:

| | why |
|---|---|
| `HIPPORAG_LLM_BASE_URL` + `HIPPORAG_LLM_MODEL` | Without them hipporag loads the 27 B checkpoint **in-process**, on top of the ~64 GiB vLLM is already holding. Guaranteed CUDA OOM. These send the extraction calls to the running server instead. |
| `HIPPORAG_EMBED_BATCH=8` | vLLM *reserves* 78% of the card, leaving ~16.3 GiB. The default batch of 16 needs marginally more than that for `Qwen3-Embedding-4B` over 1k-token chunks and OOMs by a few hundred MB. 8 fits. Irrelevant if no server is running. |
| the default `--llm_model_path` | Its **basename** names the output directory. Leave it alone and you get `local_Qwen3.8-27B_Qwen_Qwen3-Embedding-4B/`, which is what the rest of the pipeline looks for. Pass a path with a different basename and the graph lands somewhere nothing will find. |

A correct build prints, at the end:

```
6,717 phrase nodes, 1,132 passage nodes, 7,849 total
10,402 extracted triples, 34,370 synonymy triples, 59,100 total
```

Expect roughly **34 of 1,132 chunks (3%) to soft-fail NER JSON parsing**
(`'NoneType' object has no attribute 'group'`). This is normal and non-fatal:
the model returned a bare JSON array instead of the `{"named_entities": […]}`
object hipporag's regex wants, so those chunks contribute no entities and the
build continues.

**A rebuilt graph will not reproduce the shipped CSVs exactly.** Extraction is
an LLM pass, so a fresh graph differs slightly from ours and the predictions
shift with it — expect agreement, not equality. Only stage 02 is fully
deterministic. If byte-identical reproduction matters, ask us for the index
directory rather than rebuilding.

### Which model built the graph

The index directory is named `local_Qwen3.8-27B_Qwen_Qwen3-Embedding-4B`, and
that name is **accurate**: the OpenIE pass — entities, triples, edges — was run
by **Qwen3.8-27B** on 2026-08-30, in ~16 minutes against the vLLM server. One
checkpoint therefore does all three jobs:

| stage | model |
|---|---|
| graph construction (OpenIE, one-off) | Qwen3.8-27B, via vLLM |
| fact reranking at query time | Qwen3.8-27B, via vLLM |
| Step 1 / Step 2 predictions | Qwen3.8-27B, via vLLM |

Graph statistics: 6,717 phrase nodes over 1,132 passage nodes (7,849 total),
10,402 extracted triples and 34,370 synonymy triples. 34 of the 1,132 chunks
(3%) soft-failed NER JSON parsing and contribute no entities; the build
continues past them by design.

**This supersedes an earlier index built by Qwen3.5-27B on 2026-08-21**, which
the first pair of submitted runs queried. Corpus and embedding model are
unchanged across the two builds — `chunk_embeddings/vdb_chunk.parquet` is
byte-identical, which is the control that isolates the graph as the only thing
that moved — but the Qwen3.8 pass extracts about 28% more entities. The
results in `../results/` are all from the Qwen3.8 graph; the Qwen3.5-graph
files are not part of this bundle.

> **The directory name is derived, not recorded.** hipporag builds it from
> `llm_name`, which `hipporag_rag.py` sets to `LLM_PATH`'s basename — so a
> rebuild by a *different* checkpoint would still land in a folder named after
> whatever `LLM_PATH` says, with no warning. If you rebuild, set `LLM_PATH`'s
> basename to match whatever actually did the extraction.

### Outputs

Four files per run, in `../results/`, named
`tier3_{raw,calibrated,submission_raw,submission_calibrated}_tier3_<model>_<rag_tag>_<timestamp>.csv`.
The **`tier3_submission_calibrated_…`** one is the official-format file
(`condition,outcome,ate`, 208 rows) — that prefix matters, the
non-`submission` files are a wide internal 16 × 44 layout.

Logs go to `run_hipporag.log` (most recent) and
`../results/run_hipporag_<model>_<timestamp>.log` (kept per run).

### Resuming

A checkpoint is written after every completed outcome, so a run killed partway
resumes rather than starting over — both full runs were in fact terminated
externally by JupyterHub reclaiming the container. `--no_resume` forces a
clean re-run.

### Retrieval-only checks

No LLM generation, minutes rather than hours:

```bash
./.venv-hipporag/bin/python code/scripts/hipporag_rag.py --build --test
./.venv-hipporag/bin/python code/scripts/hipporag_rag.py --sweep
./.venv-hipporag/bin/python code/scripts/hipporag_rag.py --sweep --fusion rrf
```

`--sweep` reports, per weight, the Jaccard overlap of the top-k documents
against `w=1.0` and against `w=0.0`, the mean rank displacement, and the count
of queries where the graph arm came up empty and retrieval fell back to pure
dense. If `w=0.7` overlaps `w=1.0` at ~1.0, the fusion changes nothing the LLM
ever sees and a full run cannot show an effect — worth learning before
spending the hours.

### Calibration

`01_tier3_hipporag.py` applies `CALIBRATION_FACTOR = 0.56` (Ashokkumar et al.
2026) to produce its `calibrated` outputs. **The submitted files use 0.35
instead**, applied separately. The `submission_raw` file from a run is the
unscaled one, so any factor can be applied afterwards:
`calibrated = raw × b`.

---

## Stage 02 — ensemble the Tier 3 results

```bash
./.venv/bin/python code/scripts/02_ensemble_tier3.py
```

Reads five prediction files from `../results/` and writes
`../results/tier3_submission_calibrated_tier3_ensemble_direct.csv` beside
them. Runs under `.venv` — pandas and numpy only.

The five files span four models: two Qwen3.8 runs from this code, plus GPT-4o,
GPT-5 and Claude from
[team28-tier3-primary](https://github.com/lieblingszz/team28-tier3-primary).
See [What is in `results/`](#what-is-in-results). The average is **nested**: the two Qwen3.8
runs are collapsed into one series first, then the four model series are
averaged. That gives every *model* an equal 25%. A flat mean over the five
files would instead hand Qwen3.8 40%, purely because we happened to run it
twice — and the two Qwen runs correlate at r = 0.912, far above any
cross-model pair, so they are close to one vote, not two.

Mean rather than median: with four members the median discards the outer two
and moves discontinuously when they reorder. The failure mode calibration
exists to correct is systematic over-prediction, not outliers.

No performance weighting (the benchmark's held-out data is not available, so
any weights would be fitted to our own guesses) and no inverse-variance
weighting (these files carry point estimates with no standard errors).

`--flat` writes the unnested five-file mean instead, for comparison. It
correlates with the nested mean at r = 0.995, so it is a sensitivity check
rather than a real alternative, and is deliberately not kept alongside the
submission files where a near-duplicate would contaminate any glob.

The member filenames are listed explicitly in the `MEMBERS` dict at the top of
the script — edit that dict to change what goes into the ensemble.

---

## What was removed

Code that only ever served the VectorRAG backend, which no submitted result
used:

- `rag_vector_db.VectorRAG` and its import — the ChromaDB / TF-IDF flat
  similarity retriever.
- `--rag_backend`, `--embedding_backend`, `--local_embedding_model`, and the
  `run_tier3` parameters behind them. Retrieval is always HippoRAG now;
  `--no_rag` still gives the no-literature ablation.
- `utils.build_embedding_function`, `CHROMA_DB_DIR`, `COLLECTION_NAME` — the
  ChromaDB embedding-function factory. HippoRAG encodes through its own
  sentence-transformers backend (`LocalSentenceTransformerEmbedding` in
  `hipporag_rag.py`), so nothing here imports `chromadb`.
- The `try/except ImportError` dual relative/absolute import blocks, which
  existed so the development tree could work both as a package and as loose
  files.

Three things were **changed** rather than removed:

- `--output_dir` defaulted to `"./results"`, relative to the *working
  directory*, so a run launched from the scripts folder wrote into a nested
  `scripts/results/` — which is where one run's outputs actually landed. It
  now resolves from the script's own location.
- The `--passage_node_weight` help text claimed it defaults to 0.0 when
  blending. The code has always kept the library's 0.05 in both arms, on
  purpose, so that the graph arm of the fusion is stock HippoRAG's ranking.
  The help now says what the code does.
- `_embedding_weights()` is new. It lets the embedding weights load from
  `model/Qwen3-Embedding-4B` while the *name* hipporag sees stays
  `Qwen/Qwen3-Embedding-4B`. Without the split, pointing `EMBEDDING_MODEL` at
  a local path changes the derived index directory name, orphans the prebuilt
  graph and silently starts a multi-hour rebuild — found by the smoke test
  below, which OOM'd mid-rebuild rather than reusing the index.

None of these change the method: same prompts, same retrieval, same weights.

---

## Verifying a fresh checkout

No GPU and no downloaded checkpoints needed for any of these:

```bash
# 1. the ensemble reproduces byte-for-byte — the strongest offline check,
#    since stage 02 is fully deterministic
./.venv/bin/python code/scripts/02_ensemble_tier3.py

# 2. every input path resolves inside code/
./.venv-hipporag/bin/python -c "
import sys; sys.path.insert(0, 'code/scripts')
import hipporag_rag as h, utils
for name, path in [('corpus', h.CORPUS_DIR), ('index', h.SAVE_DIR),
                   ('data', utils.LAB_REPO), ('model', utils.MODELS_DIR)]:
    print(f'{name:8s} {path}  exists={path.exists()}')"

# 3. the full write path, without models or retrieval
./.venv-hipporag/bin/python code/scripts/01_tier3_hipporag.py \
    --no_rag --dry_run --quick --output_dir /tmp/smoke
```

Check 2 should report `exists=True` for corpus, index and data. `model` exists
but is empty until you download the checkpoints.

### Full end-to-end check

With the checkpoints downloaded and the vLLM server up:

```bash
MODEL=qwen38 ./code/scripts/run_hipporag.sh --quick --graph_weight 0.7 --no_resume
```

Three outcomes rather than thirteen, so a few minutes. A healthy run prints,
in order:

```
[HippoRAG LLM]   Resuming: 2404 cached responses from code/hipporag_index/
[HippoRAG Embed] Loading Qwen/Qwen3-Embedding-4B on cuda from code/model/...
[HippoRAG]       Fusion on: 0.70 graph / 0.30 dense, pit normalisation
[RAG]            HippoRAG (knowledge graph + PPR) ready
✅ TIER 3 COMPLETE
```

and exits 0 with four files in `results/`. **If you instead see OpenIE
extraction start, stop the run** — it means the index directory it derived
does not match the shipped one, and it is about to rebuild the graph from
scratch. Check that `ls code/hipporag_index/` still shows exactly one
`local_*` directory.
