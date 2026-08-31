# Silicon Sample Benchmark — method registration form

**Entry:** Tier 3, **secondary-3**, literature-conditioned multi-model ensemble. This repository and its Zenodo release constitute one benchmark entry. Items marked ★ are fully public; items marked † are disclosed at least to the benchmark team. This entry is entirely public (Class A).

Reproducibility materials for this entry are available at <https://github.com/lieblingszz/team28-tier3-secondary-3>. Full generation details for the frozen GPT-4o, GPT-5.4, and Claude Sonnet 4.6 members are available at <https://github.com/lieblingszz/team28-tier3-secondary-1>.

---

## 0 · Approach identity and output

- **0.1 Team ★** — RAG team silicon sample study: Farah Adeeba (University of Konstanz), Jing Ma (University of Konstanz), Marcia Ferreira Goncalves (Graphwise), and Max Pellert (Barcelona Supercomputing Center). Corresponding contact: max.pellert@bsc.es.

- **0.2 Plain-language summary ★** — This entry combines literature-conditioned forecasts from GPT-4o, GPT-5.4, Claude Sonnet 4.6, and Qwen3.8-27B. Qwen contributes two forecasts generated from the same model, corpus, knowledge graph, embeddings, intervention texts, survey items, and prompts; only the retrieval ranking differs. The two Qwen forecasts are averaged first, after which the four model-level forecasts receive equal weight. A common calibration factor of `0.35` is applied to all members.

- **0.3 Submission tier and approach family ★** — Tier 3 direct forecasting with retrieval-augmented generation (RAG) and model-level ensembling. The method predicts condition-level treatment effects directly; it does not construct personas or simulate respondent-level survey records.

- **0.4 Pipeline** — The entry followed eight stages.

  1. Extract and clean text from 43 scientific papers, divide it into 1,132 passages, and represent passages, entities, facts, and queries with Qwen3-Embedding-4B.
  2. Use Qwen3.8-27B to construct an OpenIE knowledge graph and support HippoRAG fact reranking.
  3. For each of 44 raw survey items, retrieve five outcome-relevant passages and ask Qwen to score all 16 interventions comparatively from −3 to +3. Average three independent scoring attempts.
  4. For each intervention–item pair, retrieve evidence separately for the intervention and outcome, merge and deduplicate the results, and retain at most six passages. Generate three independent ATE estimates and average successfully parsed values.
  5. Aggregate the 44 item-level forecasts into the 13 official outcomes. Reverse the sign of `funding_perceptions` to match the benchmark direction.
  6. Repeat the Qwen workflow under two retrieval rules: stock HippoRAG PPR and PIT-normalized `0.7 × graph + 0.3 × dense` fusion.
  7. Apply the common factor `b = 0.35` to the two Qwen series and the frozen GPT-4o, GPT-5.4, and Claude series.
  8. Average the Qwen pair, then average the resulting Qwen, GPT-4o, GPT-5.4, and Claude model-level series equally.

- **0.5 Coverage ★** — Complete coverage: 16 interventions × 13 outcomes = 208 ATE estimates. The control condition is represented implicitly as the reference category and therefore does not appear as a row.

## A · Scope of LLM use

- **A.1 Purpose** — LLMs generate the five member forecast series. In the Qwen workflow, Qwen3.8-27B performs comparative intervention scoring, absolute ATE estimation, OpenIE graph construction, and fact reranking. Qwen3-Embedding-4B encodes retrieval objects and queries. Corpus preparation, parsing, item-to-outcome aggregation, calibration, validation, and final ensembling are deterministic non-LLM operations.

- **A.2 Degree of automation ★** — Prediction, calibration, validation, and aggregation are fully automated. No individual forecast cell was manually selected, excluded, or edited.

## B · Model and system details

### B-Qwen · Qwen3.8-27B

- **B.1 Model names** — Forecasting and retrieval LLM: `Qwen/Qwen3.8-27B` (27B parameters). Embedding model: `Qwen/Qwen3-Embedding-4B`. Both are open-weight Hugging Face models; their weights are not included in this repository because of their size.

- **B.2 Access and context mode** — Local inference through an OpenAI-compatible vLLM 0.19.1 endpoint. Calls are stateless and single-turn, contain one user message, and request one completion. No system message or conversation history is retained. The stock run executed from 30 August 2026, 09:36:07–10:26:29 UTC; the fused run executed from 10:28:30–11:16:38 UTC.

- **B.3 Configuration** — Forecasting used temperature `0.5`, disabled thinking, and no specified seed, top-p, top-k, penalties, or stop sequence. Output limits were 600 tokens for comparative scoring, 110 for effect estimation, and 15 for fallback requests. Three comparative-scoring attempts were issued per item, and three ATE samples were issued per intervention–item pair. This yields 2,244 nominal forecasting calls per full Qwen run and 4,488 across the two runs, excluding retrieval calls and any retries. Failed API calls were retried up to three times with increasing delays. HippoRAG LLM operations used temperature `0.3`, a 512-token limit, and disabled thinking. Retrieval settings were `graph_weight = 1.0` for stock ranking and `graph_weight = 0.7` for fusion, with PIT normalization and `passage_node_weight = 0.05`.

- **B.4 Customization** — No fine-tuning, adapters, tool use, external web search, or agentic scaffold was used. HippoRAG 2.0.0a3 operated over an OpenIE knowledge graph with PPR. The fused run added matched dense-passage ranking while holding the model, corpus, graph, embedding model, prompts, and sampling structure fixed.

- **B.5 Persistent memory** — No conversational memory was used. The retrieval index, embeddings, graph, and retrieval-response cache persisted across calls; model conversations did not.

- **B.6 Inference stack** — Python 3.11.6 on one NVIDIA H100 80 GB GPU with tensor parallelism 1. Qwen3.8-27B was served with vLLM 0.19.1 using a maximum model length of 32,768 tokens and GPU-memory utilization of 0.78. HippoRAG used Transformers 5.15.0 and Torch 2.5.1+cu124. Qwen3-Embedding-4B was loaded in process with an embedding batch size of 8 during graph construction.

- **B.7 Ensemble role** — The two calibrated Qwen variants were averaged into one model-level forecast. Each Qwen variant therefore contributes 12.5% of the final ensemble.

### B-Hosted models · GPT-4o, GPT-5.4, and Claude Sonnet 4.6

- **B.1 Model names** — OpenAI GPT-4o (`gpt-4o`), OpenAI GPT-5.4 (`gpt-5.4`), and Anthropic Claude Sonnet 4.6 (`claude-sonnet-4-6`).

- **B.2–B.6 Access, configuration, customization, memory, and inference** — This entry reuses frozen, calibrated forecasts generated by the Team 28 hosted-API RAG workflow. Provider identifiers, access modes, prompts, retrieval configuration, sampling settings, token use, costs, dates, and inference details are reported in <https://github.com/lieblingszz/team28-tier3-secondary-1>. No new hosted-model calls were made for this entry.

- **B.7 Ensemble role** — GPT-4o, GPT-5.4, and Claude Sonnet 4.6 each contribute one calibrated series and receive 25% of the final ensemble weight.

## C · Prompts

- **C.1 Exact prompts** — The exact Qwen prompts are included in the public reproducibility materials and were identical across the stock and fused retrieval runs. The hosted-model prompts are documented in the supporting entry cited above.

- **C.2 System-wide instructions** — Each Qwen call contains a single user-role prompt. The comparative-scoring prompt frames the model as an expert in behavioral science and science communication, provides the outcome wording, scale, retrieved evidence, and all 16 intervention texts, and requests a JSON mapping from intervention names to scores from −3 to +3. Ties and zero effects are explicitly permitted. The effect-estimation prompt provides the complete intervention text, item wording and scale, retrieved evidence, and averaged comparative score. It instructs the model to treat the score only as supporting context, reason independently about direction and magnitude, and return a brief rationale followed by `ANSWER: <number>`.

- **C.3 Prompt-design rationale** — Joint comparative scoring encourages consistent ranking without forcing a unique order. The second stage separates absolute effect-size estimation from the comparative scale and discourages mechanical score rescaling. Pair-specific retrieval prevents long intervention texts from dominating short outcome terms. Holding the prompts fixed across Qwen variants isolates the effect of retrieval ranking.

## D · Persona and profile construction

- **D.1 Profile source** — N/A; no personas or respondent profiles were constructed.

- **D.2 Profile verbalization** — N/A.

- **D.3 Assignment and weighting** — N/A; forecasts are condition-level ATEs rather than respondent-level records.

## E · Stimulus and survey administration

- **E.1 Stimulus presentation** — The complete official intervention texts were supplied directly for forecasting. No text was paraphrased, and no respondent-level exposure or control filler was simulated. Effects were defined relative to the benchmark's no-message control.

- **E.2 Survey walk-through** — The method processed 44 raw survey items independently rather than simulating a respondent completing a survey. For each item, all interventions were compared jointly; intervention-specific ATEs were then estimated in separate calls. Item-level predictions were aggregated according to the benchmark outcome definitions. There was no multi-turn conversation and no context transfer between items or intervention–item pairs.

- **E.3 Response elicitation** — Comparative responses had to contain all 16 intervention names in valid JSON; numeric scores were clipped to [−3, 3]. If all three comparative responses failed to parse, the item used neutral scores of zero. ATE prompts specified ranges of [−10, 15] for 0–100 outcomes, [−3, 3] for donation, and [−0.15, 0.15] for newsletter signup. Parsed ATEs were not mechanically clipped. An unparseable ATE triggered one numeric-only fallback request; successfully parsed samples were averaged, and a cell was missing only if no sample could be parsed. The frozen member files contain no missing cells. Token log-probabilities were not used.

## F · Stochasticity and aggregation

- **F.1 Runs and seeds** — Each Qwen run used three independent comparative-scoring attempts per item and three independent ATE samples per intervention–item pair. No fixed seed was supplied, so regeneration is distributional rather than bit-for-bit. The two final Qwen files are separate stochastic runs using the same graph, corpus, prompts, model, and sample counts; only retrieval ranking differs. Rebuilding the OpenIE graph is also stochastic and may shift regenerated forecasts. The final five-member aggregation is deterministic.

- **F.2 Aggregation rule** — The two Qwen variants were averaged cell by cell: `Qwen_mean = (Qwen_stock + Qwen_g70) / 2`. The final estimate was `Final = (Qwen_mean + GPT-4o + GPT-5.4 + Claude) / 4`. Each model family therefore receives 25% weight. A flat five-file mean was examined as a sensitivity analysis but rejected because it would assign Qwen 40% weight solely because two correlated retrieval variants were available. No performance or inverse-variance weighting was used because target outcomes and member-level standard errors were unavailable.

## G · Validation and post-processing

- **G.1 Human validation** — None. Individual generations and forecast cells were not reviewed or manually corrected.

- **G.2 Post-processing** — Comparative and ATE outputs were parsed automatically. Successfully parsed item-level ATEs were averaged, composite outcomes were formed as arithmetic means of their constituent items, and `funding_perceptions` was sign-reversed because the raw item has the opposite direction. Results remained in original outcome units. Before aggregation, each frozen member was required to contain exactly 208 rows, no missing ATEs, and the identical intervention–outcome grid. The final output was sorted and rounded to four decimal places. The deposited prediction contains no missing values or duplicate cells and passes all 43 repository validation checks.

- **G.3 Calibration corrections** — All five members were multiplied by `b = 0.35` before aggregation. Calibration was developed independently of the target study. An initial factor of `0.56`, motivated by Ashokkumar et al. (2026) was reassessed using 40 prediction–outcome pairs from published Voelkel et al. (2026, Nature Climate Change)'s data. A through-origin regression estimated an optimum near `0.321`; `0.35` was retained and reduced pooled RMSE relative to `0.56` (`1.123` versus `1.255`). The same factor was applied to every model, intervention, and outcome. No target-study outcomes informed calibration. Because the factor is common and linear, calibrating before or after averaging is algebraically equivalent.
- Limitations disclosed plainly: (1) The 43-paper retrieval corpus contains studies on related topics, interventions, and outcomes rather than exact matches to the target study's 16 interventions and 13 outcomes. The forecasts therefore rely partly on analogical transfer from related evidence, whose validity for the target interventions cannot be verified before the benchmark outcomes are revealed. (2) The results depend on the retrieval procedure as well as on the underlying models. Relevant evidence may fail to be retrieved, while semantically similar but less applicable passages may receive high rank, so retrieval and chunk selection introduce an additional source of uncertainty. (3) Although the multi-model ensemble reduces reliance on any single forecast, its members are not fully independent: several share the same literature-conditioning framework, and the two Qwen variants additionally share the same underlying model, corpus, and retrieval index. Common corpus, prompting, retrieval, or calibration biases may therefore persist after averaging.

## H · Learning and conditioning components

- **H.1 Fine-tuning data** — None. No benchmark-specific fine-tuning was performed.

- **H.2 Context and retrieval corpora** — Qwen retrieval used 43 scientific papers divided into 1,132 passages. HippoRAG linked passages, extracted entities or phrases, and facts in an OpenIE graph and used PPR for graph-based ranking. Qwen3-Embedding-4B encoded passages, entities, facts, and queries. Comparative scoring retrieved the top five passages for an outcome-level query. ATE estimation performed separate intervention-text and outcome queries, retained up to four passages per query, merged them by retrieval score, removed duplicates, and supplied at most six passages. Both Qwen variants used the same corpus and graph; only the final passage-ranking rule differed. Corpus and retrieval details for the hosted models are reported in the supporting entry.

## I · Data inputs, blinding, and competing interests

- **I.1 Competing interests ★** — No team member reports a competing interest. Qwen generation used institutionally available local compute. Hosted-provider access and costs are disclosed in the supporting entry; no other financial or in-kind support from LLM-interested entities is reported for this entry.

- **I.2 External human data †** — Published Voelkel et al. (2026, Nature Climate Change)'s data informed calibration-factor validation, and Ashokkumar et al. (2026) informed the initial calibration heuristic. Published experimental findings also entered inference through the retrieval corpora. No target-study outcomes were used.

- **I.3 Blinding attestation ★** — No team member accessed, solicited, or was shown target-study outcomes before the prediction lock, with one disclosed exception: Max Pellert attended an approximately five-minute presentation of preliminary results at the Behavioral Clones workshop, Max Planck Institute for Human Development, Berlin, in May 2026. This exposure was disclosed to the benchmark organizers on 20 August 2026. No specific estimates were retained or used in calibration, model selection, or submission construction, and no other team member was exposed. Farah Adeeba attests on behalf of the team.

- **I.4 Contamination note †** — Provider-specific contamination disclosures for the hosted models are reported in the supporting entry. No authoritative training-data cutoff was identified for Qwen3.8-27B, so no cutoff is inferred from its release date. Published papers were intentionally supplied at inference time through retrieval; this exposure is disclosed and is distinct from possible pretraining exposure.

## J · Internal selection procedure

- **J.1 Design-space search †** — Retrieval-only comparisons examined graph weights and PIT, min–max, and reciprocal-rank normalization without consulting target outcomes. The submitted fused setting (`0.7 × graph + 0.3 × dense`, PIT-normalized) was selected to introduce a meaningful dense-retrieval contribution while retaining the graph signal. Aggregation comparisons considered a flat five-file mean and a nested model-balanced mean. The nested mean was selected to avoid assigning Qwen 40% weight solely because it had two correlated retrieval variants. Mean was preferred to a four-member median because the latter would discard the outer two forecasts. No target-study outcomes informed these choices.

## K · Reproducibility and frozen artifacts

- **K.1 Code and materials** — The public repository contains the Qwen generation and deterministic aggregation workflow, exact prompts, benchmark survey inputs, 43-paper corpus, five calibrated member series, final ensemble, and timestamped Qwen run logs. Large model checkpoints and the HippoRAG index are not included because of size; the index can be rebuilt from the deposited corpus, although stochastic graph construction prevents byte-identical regeneration. Hosted-model materials are documented in the supporting entry. Secrets and API keys are excluded.

- **K.2 Raw output logs †** — Timestamped Qwen run logs, parsed estimates, calibrated member forecasts, and final outputs are retained. The logs record endpoint configuration, hardware state, retrieval settings, progress, and output artifacts. Complete unprocessed text from every Qwen comparative-scoring and ATE call was not archived separately; this is a reproducibility limitation. Hosted-model logging and retention are reported in the supporting entry.

- **K.3 Computational resources** — The two full Qwen runs required approximately 50 minutes and 48 minutes, respectively, on one NVIDIA H100 80 GB GPU, excluding the one-time graph build. Graph construction took approximately 16 minutes. Each full run issued 2,244 nominal forecasting calls, for 4,488 across both Qwen variants, plus retrieval/fact-reranking calls. Qwen3.8-27B occupied approximately 54 GB in bf16, while Qwen3-Embedding-4B required approximately 8 GB. Final aggregation required only pandas, NumPy, and approximately one second of CPU time. This entry made no new hosted-model calls.

## L · Disclosure class

This entry is **A · Open**. All applicable items are public. Entry-specific methods, prompts, inputs, frozen member forecasts, outputs, logs, and reproduction instructions are openly available. Hosted-model documentation is provided through the public supporting entry. A limitation concerning unarchived raw Qwen forecasting responses is disclosed explicitly in K.2.
