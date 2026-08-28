# Silicon Sample Benchmark — method registration form

**Entry covered by this form:** Tier 3, **secondary-k**, Claude + RAG pipeline (same `tier3_pipeline_parallel_fulltext.py`, run with `--model claude`), the run saved as `tier3_submission_calibrated_tier3_claude_rag_parallel_fulltext_20260826_134941.csv`.

> This is one of three Tier 3 entries (primary: GPT-4o; secondary: this Claude entry; secondary: the GPT-4o+GPT-5+Claude ensemble) — each has its own form per the instructions at the top of this template. Sections identical to the primary GPT-4o form are noted as such rather than re-derived from scratch, since they come from the exact same pipeline code; only what genuinely differs by model is re-verified here.

---

## 0 · Approach identity and output

- **0.1 Team ★**: Same as the primary Tier 3 entry and the Tier 1 entry — Farah Adeeba (University of Konstanz); Jing Ma (University of Konstanz); Marcia Ferreira Goncalves (Graphwise); Max Pellert (Barcelona Supercomputing Center). Corresponding contact: max.pellert@bsc.es.

- **0.2 Plain-language summary ★**: Identical approach to the primary GPT-4o entry, run with Claude instead: for each of the 16 interventions and 13 outcomes, Claude first ranks all 16 interventions relative to each other, then independently estimates the actual effect size for each one, grounded in evidence retrieved from the same 30+-paper literature corpus. Predictions are scaled by the same fixed calibration factor as every other entry in this family.

- **0.3 Submission tier & approach family ★**: Tier 3. Same family as the primary entry: direct forecast, single model (this entry: Claude, not GPT-4o), literature-conditioned via RAG, no fine-tuning.

- **0.4 Pipeline diagram**: Identical to the primary GPT-4o form — same code path (`tier3_pipeline_parallel_fulltext.py`), only the `--model claude` flag differs. See that form for the full step-by-step diagram; the only change is which model answers the Step-1 scoring and Step-2 effect prompts.

- **0.5 Coverage ★**: All 16 interventions × 13 outcomes = 208 rows, `condition,outcome,ate`. Confirmed against the submitted file.

## A · Scope of LLM use

- **A.1 Purpose**: Identical to the primary entry — Claude is used for Step-1 relative ranking and Step-2 effect-size estimation only. RAG embeddings still use OpenAI's `text-embedding-3-small` regardless of which model does the prediction (the embedding model is not swapped per `--model`).

- **A.2 Degree of automation ★**: Fully automated; no human in the loop at generation time. Same `--manual` fallback mode exists but was not used for this run.

## B · Model / system details

- **B.1 Model name(s)**: Provider: Anthropic. Model identifier requested in code: **`claude-sonnet-4-6`**. **Flag, not just a TODO**: I checked Anthropic's current models overview page (`platform.claude.com/docs/en/about-claude/models/overview`, checked 2026-08-28) and this identifier is **not listed** among Anthropic's currently-documented models (which are Claude Fable 5, Claude Opus 5, Claude Sonnet 5, Claude Haiku 4.5). This doesn't necessarily mean the calls failed — you have a real output file, so `claude-sonnet-4-6` clearly resolved to *something* callable on 2026-08-26 — but it means the exact model that served these calls needs to be confirmed from your own Anthropic console/billing history before deposit, since it may be an older, since-deprecated identifier, or the current one under a different name than what's in the code. Unlike OpenAI, Anthropic model strings are normally fully-pinned identifiers (not rolling aliases), so there usually isn't a separate "resolved snapshot" to look up the way there is for GPT-4o's B.1 — but given this identifier isn't on the current docs page, that assumption itself needs checking here. **TODO**: confirm via your Anthropic console.

- **B.2 Access & context mode**: Anthropic Messages API (`anthropic` Python SDK, `client.messages.create`). Stateless — same as the GPT-4o entry, one message per call, no thread/session state. Exact call date: inferred from the output filename timestamp, **2026-08-26, ~13:49** local system time (`_20260826_134941`). **TODO**: confirm from terminal history / Anthropic usage dashboard.

- **B.3 Configuration**: **Genuinely different from the GPT-4o entry, worth flagging explicitly**: the code's Anthropic branch of `call_api()` does **not** set a `temperature` parameter at all (unlike the GPT-4o branch, which explicitly sets 0.5) — so this entry ran at Anthropic's API default, which is 1.0 unless overridden. No top-p/top-k set either. Max tokens: same values as every entry (600 Step-1, 110 Step-2, 15 fallback), since that parameter is passed generically regardless of provider. **Extended thinking**: not explicitly enabled or disabled in the code — no `thinking` parameter is passed to `client.messages.create()`. The code defensively handles a response that *starts* with a thinking block (picks the first actual text block instead of assuming index 0), which implies the possibility was anticipated, but there's no code-level guarantee thinking was off. **TODO**: confirm from your Anthropic console/account settings whether extended thinking was active by default for this model at call time — this affects both B.3 and, if thinking was active, arguably warrants its own disclosure similar to how the GPT-5 branch's `reasoning_effort` is disclosed. No `seed` parameter set (Anthropic's API doesn't expose one) — not reproducible bit-for-bit. Completions per item: same as every entry, 3 Step-1 attempts + 3 Step-2 samples, averaged.

- **B.4 Customization**: Same as the GPT-4o entry — no fine-tuning; same shared RAG corpus/index; no automated prompt-optimization loop (same prompts, unchanged across models — see C.1); no tool use, web search, or agentic scaffold.

- **B.5 Persistent memory**: None across calls — identical to the GPT-4o entry.

- **B.6 Inference stack**: N/A — hosted API model.

- **B.7 Ensembles**: N/A for *this* entry (Claude alone). The separate ensemble secondary entry (GPT-4o + GPT-5 + Claude averaged) has its own registration form.

## C · Prompts

- **C.1 Exact prompts**: **Identical text to the primary GPT-4o entry** — the same `build_scoring_prompt_fulltext()` / `build_effect_prompt_fulltext()` functions are used regardless of `--model`; no Claude-specific prompt variant exists in the code.

- **C.2 System-wide instructions**: None — same as GPT-4o. The Anthropic branch of `call_api()` also sends only a single `user`-role message (no `system=` parameter passed to `client.messages.create()`).

- **C.3 Prompt-design rationale**: Same rationale as the primary entry — see that form. Not model-specific.

## D · Persona / profile construction (Tiers 1–2)

N/A — Tier 3, direct forecast, no simulated respondents.

## E · Stimulus and survey administration

- **E.1 Stimulus presentation**: Identical to the primary entry — verbatim, full, untruncated intervention text.

- **E.2 Survey walk-through**: Identical to the primary entry, including the same limitation: item and condition order is fixed (not randomized), same as noted there.

- **E.3 Response elicitation**: Identical parsing logic (`parse_scores()`, `parse_number()`, same fallback re-prompt on failure) — model-agnostic code path.

## F · Stochasticity and aggregation

- **F.1 Runs & seeds**: Same repeat counts as every entry (3 Step-1 attempts, 3 Step-2 samples), but **note the B.3 difference**: this entry's stochasticity comes from Anthropic's default temperature (likely 1.0, unconfirmed — see B.3), not the deliberately-chosen 0.5 used for GPT-4o. No seed available via the Anthropic API.

- **F.2 Aggregation rule**: Identical to every entry — mean of repeats, mean across each outcome's constituent raw items, sign-flip for `funding_perceptions`, then × 0.35.

## G · Validation & post-processing

- **G.1 Human validation**: N/A.

- **G.2 Post-processing**: Same parsing/fallback logic as every entry. **TODO**: exact fallback-trigger count for this entry specifically — countable from this entry's own `raw_log_..._claude_..._<timestamp>.jsonl` once the logged rerun (see K.2) is done, same as the primary entry.

- **G.3 Calibration corrections**: **Same value, same derivation, same justification as the primary entry — this is the part of your last question.** `CALIBRATION_FACTOR = 0.35` is a single module-level constant in `tier3_pipeline.py`, applied identically regardless of which model produced the raw predictions (`--model gpt4o` vs `--model claude` only changes which model answers the prompts, not the calibration step that runs afterward). It was fit against real Voelkel et al. (2025/2026) ground truth exactly as described in the primary entry's G.3 — that derivation doesn't change per model, since it corrects a property of LLM-predicted effect sizes generally (documented overestimation), not something specific to GPT-4o. Cross-ref H.2, I.2, J.1 — same content as the primary form.

## H · Learning and conditioning components

- **H.1 Fine-tuning data**: N/A.

- **H.2 Context & retrieval corpora**: Identical corpus to every entry — same `chroma_rag_db/`, same embedding model (`text-embedding-3-small`, OpenAI — used for retrieval regardless of which model does the prediction). Same **TODO** as the primary form: reconcile "30+ papers" against the `corpus/` folder's 43 PDFs.

## I · Data inputs, blinding, and competing interests

- **I.1 Competing interests ★**: Same as the primary entry — no competing interests; API costs paid personally. (This entry used the Anthropic API rather than OpenAI's, but the funding answer is the same.)

- **I.2 External human data †**: Same three sources as the primary entry (Ashokkumar et al. 2026; Voelkel et al. 2025/2026; the RAG corpus) — the calibration and RAG mechanisms are shared across all model entries, not re-derived per model.

- **I.3 Blinding attestation ★** — **mandatory**: Same attestation as the primary entry — I, Farah Adeeba, attest no exposure to any human outcome data from this study before the prediction lock, across every entry, not just this one.

- **I.4 Contamination note †**: **Cannot respond as precisely as the GPT-4o entry did — flagging honestly rather than guessing.** I could not find an authoritative, current cutoff date for `claude-sonnet-4-6` specifically — see the B.1 flag above; this identifier isn't on Anthropic's current models page, and I'm not going to state a cutoff date I can't source properly for a compliance document. Anthropic's own recommended source is their Transparency Hub (anthropic.com/transparency), which should have this per-model. **TODO**: once B.1 is confirmed (which model actually served these calls), look up that specific model's training cutoff there and compare against Voelkel et al.'s 2026 publication date and the target study's own materials release date, same comparison as the GPT-4o entry.

Sources: [Models overview - Claude Platform Docs](https://platform.claude.com/docs/en/about-claude/models/overview) — checked 2026-08-28, does not list `claude-sonnet-4-6`; points to [Anthropic's Transparency Hub](https://www.anthropic.com/transparency) for full cutoff details.

## J · Internal selection procedure

- **J.1 Design-space search †**: Same design history as the primary entry (RAG as core design, full-text fix, prompt variants, calibration-constant validation against Voelkel) — none of that was model-specific tuning; Claude was one of three models run through the same fixed pipeline for comparison purposes (see the correlation/coherence checks discussed during development), not selected via its own separate design search.

## K · Reproducibility & frozen artifacts

- **K.1 Code & materials**: Same repository as the primary entry (same `tier3_pipeline.py`/`tier3_pipeline_parallel_fulltext.py`), plus this entry's own submission CSV. If this is deposited as a genuinely separate Zenodo release (per the "one repo per entry" rule), include the Claude submission file alongside the same code; **TODO**: DOI after deposit.

- **K.2 Raw output logs †**: Same **TODO** as the primary entry — the logging fix (see that form) applies to every model since it lives in the shared `call_api()`. Rerun with `--model claude` to get this entry's own `raw_log_..._claude_..._<timestamp>.jsonl`, hash, and line count.

- **K.3 Computational resources**: Same formula as the primary entry: 44 × (3 + 16×3) = 2,244 completion calls (assuming the same `n_samples`/`step1_attempts` defaults were used for this run — **TODO** confirm). **TODO**: exact tokens/cost from your Anthropic console for this run's date range.

## L · Disclosure class

Same as the primary entry — nothing here requires escrow or sealing once the TODOs (mainly B.1's model-identity confirmation) are resolved. **A · Open** looks achievable, contingent on confirming exactly which Claude model this was.

---

## Next step

Same rerun command pattern as the primary entry, just with `--model claude`:

```bash
cd ~/Desktop/challenge/challenge
export ANTHROPIC_API_KEY="your-key"
python3 tier3_pipeline_parallel_fulltext.py --model claude --n_samples 3
```

This fills in this form's B.1/B.2/K.2/K.3 TODOs the same way it does for the primary GPT-4o entry. **Before running it, please check your Anthropic console for exactly which model `claude-sonnet-4-6` resolves to** — that's the one open question I couldn't resolve from code or public docs alone.
