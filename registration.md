# Silicon Sample Benchmark — method registration form

**Entry covered by this form:** Tier 3, **secondary-k**, Claude + RAG pipeline (`tier3_pipeline_parallel_fulltext.py`, run with `--model claude`), the run saved as `tier3_submission_calibrated_tier3_claude_rag_parallel_fulltext_20260826_134941.csv`.

---

## 0 · Approach identity and output

- **0.1 Team ★**: RAG team silicon sample study. Members/creators: Farah Adeeba (University of Konstanz); Jing Ma (University of Konstanz); Marcia Ferreira Goncalves (Graphwise); Max Pellert (Barcelona Supercomputing Center). Corresponding contact: max.pellert@bsc.es.

- **0.2 Plain-language summary ★**: For each of the 16 climate-communication interventions and 13 outcome measures, we ask Claude to predict the treatment effect directly (no simulated survey respondents): first it ranks all 16 interventions relative to each other for that outcome, then it estimates the actual effect size for each intervention individually, grounding both steps in evidence retrieved from a curated corpus of 30+ published science-communication studies. Raw predictions are then scaled down by a fixed factor to correct for LLMs' known tendency to overestimate effect sizes.

- **0.3 Submission tier & approach family ★**: Tier 3. Direct-forecast family (no persona/respondent simulation): single model (Claude), literature-conditioned (retrieval-augmented), two-step per outcome (relative ranking, then magnitude estimation), zero-shot in the sense that Claude receives no fine-tuning — it is prompted, not trained.

- **0.4 Pipeline diagram** — ordered steps, raw inputs → submitted file:
  1. Load the 16 official intervention texts + control, and the 44 raw survey items (which roll up into the 13 official outcomes).
  2. For each of the 44 raw items:
     a. Retrieve top-5 literature chunks relevant to that outcome (RAG, ChromaDB + `text-embedding-3-small`).
     b. **Step 1 (relative scoring):** Claude scores all 16 interventions at once, −3..+3, for this outcome; repeated 3×, results averaged per intervention.
     c. For each of the 16 interventions, retrieve up to 6 merged literature chunks (intervention text + outcome query).
     d. **Step 2 (effect estimation):** Claude predicts the actual ATE for this intervention/outcome, given the full intervention text, retrieved evidence, and the Step-1 score as soft context; repeated 3×, results averaged.
  3. Average the 44 raw item-level ATEs into the 13 official outcomes (per the benchmark's item→outcome grouping); sign-flip `funding_perceptions` (its raw item, `funding_5`, is worded in the opposite direction from the official outcome — reversing a treatment-vs-control difference means negating it, not subtracting from 100).
  4. Save the raw (uncalibrated) 208-row submission file.
  5. Multiply every value by the calibration factor (0.35) → save the calibrated 208-row submission file. **This is the file submitted.**

- **0.5 Coverage ★** — Full coverage confirmed: all 16 interventions × 13 outcomes = 208 estimates, one file, `condition,outcome,ate` format. Verified row count on the submitted file: 208.

## A · Scope of LLM use

- **A.1 Purpose** — Claude is used for exactly two pipeline stages: (1) Step-1 relative-effectiveness ranking of interventions, (2) Step-2 effect-size magnitude prediction. A separate, smaller embedding model (`text-embedding-3-small`, OpenAI) is used only for RAG retrieval — embedding the corpus and the queries — not for any prediction, and is unchanged regardless of which model (GPT-4o/GPT-5/Claude) answers the prompts. No LLM is used to simulate individual respondents — this tier has no persona step (see D).

- **A.2 Degree of automation ★** — Fully automated at prediction time: one CLI command runs the entire pipeline end-to-end via the Anthropic API with no human editing of intermediate outputs. The code also has a `--manual` copy-paste mode for running without an API key; that mode was not used for this submission — the actual run used the API mode.

## B · Model / system details

- **B.1 Model name(s)** — Provider: Anthropic. Model identifier requested in code: `claude-sonnet-4-6`. I checked Anthropic's current models overview page (`platform.claude.com/docs/en/about-claude/models/overview`, checked 2026-08-28) and this identifier is not listed among Anthropic's currently-documented models (which are Claude Fable 5, Claude Opus 5, Claude Sonnet 5, Claude Haiku 4.5). This doesn't mean the calls failed — a real output file exists, so `claude-sonnet-4-6` clearly resolved to something callable on 2026-08-26 — but it does mean the exact model that served these calls needs confirming from your own Anthropic console/billing history before deposit, since it may be an older, since-deprecated identifier, or the current model under a different name than what's in the code. Unlike OpenAI's `gpt-4o` alias, Anthropic model strings are normally fully-pinned identifiers, not rolling aliases that resolve to a different snapshot per call — so there usually isn't a separate "resolved snapshot" to look up the way GPT-4o's B.1 needs one, but given this identifier isn't on the current docs page, even that assumption needs re-checking here. `[CONFIRM via your Anthropic console]`

- **B.2 Access & context mode** — Anthropic Messages API (`anthropic` Python SDK, `client.messages.create`). Stateless: every call is an independent request (`messages=[{"role": "user", "content": prompt}]`), no thread/session state carried between calls. Exact call date: inferred from the output filename timestamp, **2026-08-26, ~13:49** local system time (`_20260826_134941` suffix on both the raw and calibrated files). `[CONFIRM]` this matches your terminal history / Anthropic usage dashboard, and note the timezone.

- **B.3 Configuration** — The Anthropic branch of `call_api()` does **not** set a `temperature` parameter at all — unlike the GPT-4o branch, which explicitly sets 0.5, this entry ran at Anthropic's API default (1.0, unless Anthropic's SDK default differs — `[CONFIRM]`). No top-p/top-k set either. Max tokens: 600 for Step-1 scoring calls, 110 for Step-2 effect calls, 15 for the Step-2 fallback single-number re-prompt (used only when the main parse fails) — the same values used for every model in this pipeline, passed generically regardless of provider. **Extended thinking**: not explicitly enabled or disabled anywhere in the code — no `thinking` parameter is passed to `client.messages.create()`. The code does defensively handle a response that starts with a thinking block (picks the first actual text block rather than assuming index 0 is text), which implies the possibility was anticipated, but nothing in the code guarantees thinking was off for these calls. `[CONFIRM]` from your Anthropic console/account settings whether extended thinking was active by default for this model at call time. No `seed` parameter — Anthropic's API doesn't expose one — so results are not reproducible bit-for-bit under identical settings. Completions per item: 3 Step-1 scoring attempts (averaged) + 3 Step-2 samples per intervention (averaged), same as every entry.

- **B.4 Customization** — No fine-tuning. RAG: yes — ChromaDB vector store, `text-embedding-3-small` embeddings, over a manually curated corpus (see H.2); this corpus and embedding setup is shared across every model entry in this pipeline family, not Claude-specific. No automated prompt-optimization loop; prompts were manually iterated by the team over the course of the project (see C.1) and held fixed for this run. No tool use, no web search, no agentic/multi-step scaffold beyond the fixed two-step (rank, then estimate) structure described in 0.4 — each call is a single-turn completion.

- **B.5 Persistent memory** — None. No memory persists across calls or across items; each API call in a full run is independent (see K.3 for the count).

- **B.6 Inference stack** — N/A. Claude is accessed via the hosted Anthropic API, not run locally.

- **B.7 Ensembles** — N/A for this entry: a single model (Claude) produces every prediction here. (Repeated same-model sampling that gets averaged is documented under F, not here — that's not a multi-model ensemble.) The GPT-4o+GPT-5+Claude ensemble is submitted as a separate secondary entry with its own registration form.

## C · Prompts

- **C.1 Exact prompts** — Verbatim templates: `build_scoring_prompt_fulltext()` (Step 1) and `build_effect_prompt_fulltext()` (Step 2) in `tier3_pipeline_parallel_fulltext.py`, deposited with the code (see K.1). These are the exact same prompt templates used for every model in this pipeline family — no Claude-specific prompt variant exists; only the model answering them changes via the `--model` flag. Iteratively refined over the project (earlier truncated-text and prompt-wording variants were tested during development); the final versions used for this run were pre-specified before the run, not adjusted in response to seeing this run's own outputs.

- **C.2 System-wide instructions** — None. No `system`-role message is used; the API call sends only a single `user`-role message per request (`call_api()` in `tier3_pipeline.py`, Anthropic branch: `client.messages.create(model=..., max_tokens=..., messages=[{"role": "user", "content": prompt}])` — no `system=` parameter). The "You are an expert in behavioral science and science communication..." framing that opens each prompt is part of the user-role text itself, not a separate system instruction.

- **C.3 Prompt-design rationale** *(optional)* — The two-step structure (relative ranking, then independent magnitude estimation) forces the model to differentiate interventions rather than collapsing them all to similar values, while keeping the magnitude estimate independently reasoned rather than a mechanical rescaling of the rank (the prompt explicitly instructs the model not to just multiply the rank score). Full, untruncated intervention text is used in both steps after an earlier truncated-text version was identified as a limitation. RAG grounding was added so magnitude estimates are anchored to retrieved findings rather than the model's unaided prior. These design choices were made once, for the pipeline as a whole, and apply identically regardless of which model runs it.

## D · Persona / profile construction (Tiers 1–2)

N/A — this is a Tier 3, direct-forecast entry with no simulated respondents or personas.

## E · Stimulus and survey administration

- **E.1 Stimulus presentation** — Verbatim, full intervention text, no paraphrasing and no truncation (this is the specific fix `tier3_pipeline_parallel_fulltext.py` makes over earlier pipeline versions, which truncated intervention text to 200–400 characters at three separate points). No state-contingent stimulus content identified in these 16 interventions.

- **E.2 Survey walk-through** — Not a whole-survey simulation. Each of the 44 raw items is handled independently, one item at a time, with no context carried over between items or between calls. Item processing order = the fixed order of `survey_items.json`; condition order within the Step-1 ranking list = the fixed order interventions are loaded from `survey_items.json`'s stimuli mapping. Neither is randomized per run. This is a genuine limitation worth being upfront about: a fixed ranking-list order is a plausible, if likely small, source of position bias in Step 1 specifically. Scale display: each item's own scale-label text (e.g. "0 = far too little ... 100 = far too much") is inserted into the prompt verbatim from `survey_items.json`. No attention/comprehension checks — not applicable, there are no simulated respondents to check.

- **E.3 Response elicitation** — Step 1: structured output, a JSON object mapping intervention name → score, parsed with `parse_scores()`. Step 2: constrained free text — up to 15 words of reasoning, then a required `ANSWER: <number>` line, parsed with `parse_number()`; if parsing fails, a stricter single-number-only fallback prompt is sent. Not logprob-based.

## F · Stochasticity and aggregation

- **F.1 Runs & seeds** — Step 1: 3 independent scoring attempts per outcome item, run concurrently (thread pool), each attempt scores all 16 interventions at once. Step 2: 3 independent samples per (intervention, item) pair, run concurrently. No `seed` parameter is available via the Anthropic API. Stochasticity for this entry comes from Anthropic's default sampling temperature (the code sets none explicitly — see B.3), not the deliberately-chosen 0.5 used for the GPT-4o entry — results are not reproducible bit-for-bit under identical settings; only the average over repeated draws is what's reported.

- **F.2 Aggregation rule** — Step 1: mean of the 3 parsed scores per intervention (fallback: all-zero/neutral if every attempt fails to parse). Step 2: mean of the 3 parsed ATE values per (intervention, item). Item-level ATEs are then meaned across each official outcome's constituent raw items (e.g. `trust_multidimensional` = mean of 12 raw trust items), sign-flipped for `funding_perceptions`, then the whole 208-row table is multiplied by the calibration constant (0.35) for the submitted, calibrated file.

## G · Validation & post-processing

- **G.1 Human validation** — None; no human reviewed or edited individual model outputs in this pipeline run.

- **G.2 Post-processing** — Numeric extraction (`parse_number`, `parse_scores`) with a stricter single-number fallback re-prompt on parse failure (Step 2); if Step-1 scoring fails to parse on all 3 attempts for an item, that item falls back to a neutral all-zero score vector rather than being dropped. No responses were excluded/dropped outright. `[CONFIRM]` exact parse-failure/fallback count for this entry — not currently logged for the run this submission came from; countable directly from a rerun with logging enabled (see K.2). "Effective N per condition" — N/A, this approach doesn't generate individual synthetic respondents.

- **G.3 Calibration corrections** — Yes. All values are multiplied by a single global constant, `CALIBRATION_FACTOR = 0.35`, applied as the last step before saving the calibrated submission file. This corrects for LLMs' documented tendency to overestimate treatment-effect magnitudes. **What it was fit on:** initially set to 0.56 following Ashokkumar et al. (2026)'s primary-archive correction figure; empirically re-validated by scoring both 0.56 and a range of alternatives against real human ground truth from Voelkel et al. (2025/2026, *Nature Climate Change*) — 4 outcome composites × 10 conditions = 40 predicted/real pairs, through-origin regression + RMSE comparison. 0.35 was chosen as very close to the empirically best-fit value (0.321) on that data and clearly better than 0.56 (pooled RMSE 1.123 vs 1.255). This constant is a single module-level value in `tier3_pipeline.py`, applied identically regardless of which model (`--model gpt4o`/`gpt5`/`claude`) produced the raw predictions — the `--model` flag only changes which model answers the Step-1/Step-2 prompts; the calibration step that runs afterward is unaffected. It is applied globally, not per-intervention or per-outcome, since no real ground truth exists for the actual 16 target interventions to fit a finer-grained version. Cross-ref H.2 (Voelkel's paper is also in the RAG corpus — see I.4 for why that's not circular) and I.2/J.1 below.

## H · Learning and conditioning components

- **H.1 Fine-tuning data** — N/A. No fine-tuning is used anywhere in this pipeline.

- **H.2 Context & retrieval corpora** — A curated corpus of 30+ published papers on climate/science communication, trust, and related behavioral topics, chunked and indexed in a persistent ChromaDB store (`chroma_rag_db/`, 795 chunks at last count), embedded with `text-embedding-3-small`. This corpus and index are shared across every model entry in this pipeline family — the same `chroma_rag_db/` is used whether `--model` is `gpt4o`, `gpt5`, or `claude`. `[CONFIRM]` reconcile "30+ papers" against the `corpus/` folder, which currently has 43 PDFs — some may have been added after the last indexing pass; worth double-checking `chroma_rag_db/` reflects all 43 before depositing. The corpus includes Voelkel et al.'s own paper (`voelkel_2026_megastudy.pdf`) — intentional and appropriate for the actual submission (real literature legitimately informing real predictions), but see I.4 for why this needed special handling in the separate calibration-validation script.

## I · Data inputs, blinding, and competing interests

- **I.1 Competing interests ★** — No team member has any competing interest to disclose. Farah Adeeba's API costs for this Tier 3 entry (Anthropic API usage) were paid personally, out of pocket — no institutional or third-party funding for the API usage specifically.

- **I.2 External human data †** — Two real human datasets informed this approach, beyond the RAG literature corpus, identically to every other entry in this pipeline family:
  1. **Voelkel et al. (2025/2026, Nature Climate Change)** megastudy data — used to empirically validate and refit the calibration constant (G.3, J.1). Not used for training/fine-tuning; used purely to fit one global scalar, applied the same way regardless of which model produced the raw predictions being calibrated.
  2. **Ashokkumar et al. (2026)** — source of the original 0.56 calibration heuristic before it was re-validated and revised to 0.35.
  Additionally, the RAG corpus (H.2) consists of published papers that themselves report human experimental findings, retrieved into context at inference time (in-context, not training) — disclosed here for completeness even though it's the intended mechanism (H.2), not incidental exposure.

- **I.3 Blinding attestation ★** — **mandatory.** "We attest that no team member accessed, solicited, or was shown any human outcome data from this study (the Silicon Sample Benchmark's own target interventions/outcomes), including pilot data, before the prediction lock. Real human data referenced elsewhere in this form (Voelkel et al. 2025/2026; Ashokkumar et al. 2026) are independent, already-published studies unrelated to this benchmark's own target study, used only as disclosed in I.2/G.3/J.1." I, Farah Adeeba, attest to this on behalf of the team for this entry.

- **I.4 Contamination note †** — I could not find an authoritative, current training-cutoff date for `claude-sonnet-4-6` specifically: Anthropic's current models overview page (checked 2026-08-28) does not list this identifier at all (see B.1), so I'm not going to state a cutoff date for it that I can't source properly in a compliance document. Anthropic's own recommended source for this is their Transparency Hub (anthropic.com/transparency), which publishes per-model cutoff dates. `[CONFIRM]` once B.1 is resolved (which model actually served these calls), look up that specific model's training cutoff there and compare against Voelkel et al.'s 2026 publication date and the target study's own materials' first public-release date — the same comparison done for the GPT-4o entry, which showed no contamination risk (GPT-4o's ~October 2023 cutoff predates Voelkel's 2026 publication comfortably). Note: the RAG corpus intentionally includes Voelkel's paper as retrieved-at-inference-time context (H.2) — this is in-context exposure, not training-data contamination, and is a different, non-circular use from the calibration-validation script, which explicitly filters Voelkel citations out of its own retrieval before scoring against Voelkel's ground truth (to avoid the model "predicting" a study using that same study's own summary as evidence).

  Sources: [Models overview - Claude Platform Docs](https://platform.claude.com/docs/en/about-claude/models/overview) (checked 2026-08-28; does not list `claude-sonnet-4-6`) — points to [Anthropic's Transparency Hub](https://www.anthropic.com/transparency) for full per-model cutoff details.

## J · Internal selection procedure

- **J.1 Design-space search †** — Configurations tried during development (visible in the project's own run history): RAG on vs. off; truncated vs. full intervention text; sequential vs. parallel execution (a performance/cost change only, not a modeling choice); at least 3 named prompt variants ("optionA/B/C"). The one external, quantitative validation criterion used was correlation and RMSE against Voelkel et al. (2025/2026) real human data — used specifically to select the calibration constant (0.56 vs. ~0.35 vs. the empirical best-fit 0.321), not to select among the prompt/RAG/truncation variants, which were chosen on other grounds (avoiding known truncation bias; RAG being part of the approach's core design) rather than an internal accuracy metric. Claude itself was not separately tuned or selected via its own design search — it is one of three models (GPT-4o, GPT-5, Claude) run through this same fixed pipeline, for comparison purposes, after the pipeline's own design (prompts, RAG, calibration constant) was settled. No target-study human data was consulted at any point (I.3).

## K · Reproducibility & frozen artifacts

- **K.1 Code & materials** — This entry's repository contains: `code/tier3_pipeline.py`, `code/tier3_pipeline_parallel_fulltext.py`, `code/rag_vector_db.py`, the prebuilt `code/chroma_rag_db/`, `code/LLMmegastudy-main/simulation/` (16 stimulus texts + `survey_items.json`), `requirements.txt`, and this entry's submitted CSV (raw + calibrated) in `results/`. No secrets are in this repository — API keys are read from environment variables only, never from a file. `[CONFIRM]` link/DOI to fill in after Zenodo deposit, then also record it in `metadata.json` → `code_repository`/`code_doi`.

- **K.2 Raw output logs †** — **Gap to address before depositing.** As currently written, `tier3_pipeline.py`/`tier3_pipeline_parallel_fulltext.py` did not save the complete unprocessed model responses (the raw text each API call returned) for the run that produced this entry's submitted file — only the final parsed numbers were kept. This form's own instructions require complete raw logs for Tier 3 "where intermediate generations exist," which they do here. The pipeline has since been updated to log every raw response (timestamp, resolved model, exact prompt, exact response, and which step/item/condition/sample) to a hashed `.jsonl` file automatically. `[CONFIRM]` how to proceed: (a) rerun with `--model claude` before the deadline to produce a fully-logged version of this entry (recommended — also resolves the B.1/B.2 TODOs above), or (b) disclose plainly that raw per-call text wasn't archived for the currently-submitted run.

- **K.3 Computational resources** — Estimated API calls for the full run: `n_items × (step1_attempts + n_conditions × n_samples)` = 44 × (3 + 16×3) = **44 × 51 = 2,244 calls**, assuming this run used the code's defaults (step1_attempts=3, n_samples=3). `[CONFIRM]` no `--step1_attempts`/`--n_samples` override was passed for this specific run. Plus a smaller number of cheap embedding/retrieval calls for RAG (not completions). `[CONFIRM]` total token count and cost from your Anthropic usage dashboard for the exact figures — the pipeline doesn't currently log these itself.

## L · Disclosure class

Given the ★ items above (0.1, 0.2, 0.3, 0.5, A.2, I.1, I.3) must be fully public, and the † items (I.2, I.4, J.1, K.2) must be at minimum escrowed: nothing here currently requires **C · Sealed** — every † item has a clear, disclosable answer once the `[CONFIRM]` items are resolved (mainly B.1's model-identity confirmation, which I.4 depends on). **A · Open** looks achievable. Record the final choice in `metadata.json` → `disclosure_class`.

---

## Next step: the rerun that fills in the remaining `[CONFIRM]` items

```bash
cd ~/Desktop/challenge/challenge
export ANTHROPIC_API_KEY="your-key"
python3 tier3_pipeline_parallel_fulltext.py --model claude --n_samples 3
```

This produces a **new** timestamped submission file plus a hashed `raw_log_..._claude_...jsonl` with the resolved model, SHA-256, and line count printed at the end — resolving B.1, B.2, G.2, and K.2/K.3 together. Before running it, please check your Anthropic console for exactly which model `claude-sonnet-4-6` resolves to — that's the one open question that isn't resolvable from code or public docs alone.
