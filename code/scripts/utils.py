

import os
import re
import json
import time
from pathlib import Path
import pymupdf  # PyMuPDF


MODELS_DIR = Path(os.environ.get(
    "MODELS_DIR", Path(__file__).parent / "../model")).resolve()
# ════════════════════════════════════════════════════════════════════
# 1. PATHS + OFFICIAL DATA LOADING
# ════════════════════════════════════════════════════════════════════

# Data dir (survey_items.json + stimuli/*.txt). Resolved from this file's
# own location, never the caller's cwd, so it ships with the code. Set
# DATA_DIR to point it anywhere else.
LAB_REPO = Path(os.environ.get(
    "DATA_DIR", Path(__file__).parent / "../data")).resolve()


def load_stimuli() -> dict:
    data = json.loads((LAB_REPO / "survey_items.json").read_text())
    stimuli = {}
    for name, filename in data["stimuli"].items():
        path = LAB_REPO / "stimuli" / filename
        if path.exists():
            stimuli[name] = path.read_text().strip()
    print(f"[STIMULI] Loaded {len(stimuli)} official stimuli")
    return stimuli


def load_survey_items() -> dict:
    data = json.loads((LAB_REPO / "survey_items.json").read_text())
    return data["items"]


# ════════════════════════════════════════════════════════════════════
# 2. OFFICIAL OUTCOME SCHEMA
# ════════════════════════════════════════════════════════════════════

# Maps official variable name -> list of raw item keys to average.
# Scale range: used to convert to percentage points of scale range.
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

# Per the benchmark's survey_items.json ("outcomes" -> rule), most raw
# items map to their official outcome via "copy" or "mean" (no sign
# change). funding_perceptions is the one exception: rule="reverse100"
# on funding_5 ("...spending too much/too little...", 0=far too little,
# 100=far too much). The preregistration confirms the official scale is
# reverse-coded so higher = supports more funding, and Tier 3 ATEs must
# be submitted "already reverse-coded if applicable". Prompts show the
# model funding_5's RAW wording/scale (so the model predicts the
# raw-direction effect), so the predicted ATE must be sign-flipped
# before it reaches the official outcome (see aggregate_to_official() in
# 01_tier3_hipporag.py) — reversing a 100-x level transform turns a
# treatment-vs-control DIFFERENCE into its negation, not into (100 - effect).
REVERSE_CODED_OUTCOMES = {"funding_perceptions"}


def get_construct_label(item_key: str) -> str:
    """Human-readable construct name for the given raw item, for use in
    RAG retrieval queries. Falls back to the raw item key if the item
    isn't mapped to an official outcome (e.g. items outside the 13
    preregistered outcomes)."""
    outcome_name = ITEM_TO_OUTCOME.get(item_key, item_key)
    return outcome_name.replace("_", " ")


# ════════════════════════════════════════════════════════════════════
# 3. QUERY-SPECIFIC RAG QUERIES
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
        f"{intervention_text} "
        f"Outcome: {construct_label} — {item_spec['question']}"
    )


# ════════════════════════════════════════════════════════════════════
# 4. STEP 1 PROMPT — RELATIVE SCORING of all interventions for one outcome
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
        f"{i+1}. [{name}]: {text}..."
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
# 5. STEP 2 PROMPT — Estimate effect size, informed by the relative score
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
# 6. PDF / TEXT PROCESSING
# ════════════════════════════════════════════════════════════════════
#
# (The ChromaDB embedding-function factory that used to sit here served
# VectorRAG only. HippoRAG encodes chunks, entities, facts and queries
# through its own sentence-transformers backend -- see the
# LocalSentenceTransformerEmbedding class in hipporag_rag.py -- so nothing
# in this path imports chromadb.)
# ════════════════════════════════════════════════════════════════════

ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "et al.",
    "Fig.",
    "Figs.",
    "Eq.",
    "Eqs.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Prof.",
    "U.S.",
    "U.K.",
    "No.",
    "Vol.",
    "pp.",
    "p.",
)

KNOWN_HEADERS = {
    "Nature Climate Change",
    "Journal of Environmental Psychology",
    "PLOS ONE",
    "Public Understanding of Science",
    "Communications Earth & Environment",
    "Nature Human Behaviour",
    "Nature Communications",
    "Science Advances",
    "Science Communication",
    "Behavioural Public Policy",
    "Research & Politics",
    "Global Environmental Change",
    "The Lancet Planetary Health",
    "Topics in Cognitive Science",
    "npj Climate Action",
    "Energy Policy",
    "Geophysical Research Letters",
    "PNAS Nexus",
    "Scientific Reports",
    "Journal of Economic Behavior & Organization",
    "Mitigation and Adaptation Strategies for Global Change",
    "Climatic Change",
    "PLOS Climate",
}

KNOWN_HEADERS_NORMALIZED = {header.casefold() for header in KNOWN_HEADERS}


def _citation_from_metadata(meta: dict, pdf_path: str) -> str:
    title = (meta.get("title") or "").strip()
    author = (meta.get("author") or "").strip()

    if author and title:
        return f"{author} — {title}"
    if title:
        return title

    return Path(pdf_path).stem


def pdf_citation(pdf_path: str) -> str:
    """
    The citation alone, without extracting page text. Shares
    _citation_from_metadata with extract_text_from_pdf so the two can
    never disagree -- hipporag_rag.py maps a retrieved chunk's citation
    back to its source filename with this.
    """
    doc = pymupdf.open(pdf_path)
    try:
        return _citation_from_metadata(doc.metadata or {}, pdf_path)
    finally:
        doc.close()


def extract_text_from_pdf(pdf_path: str) -> tuple[str, str]:
    doc = pymupdf.open(pdf_path)
    meta = doc.metadata or {}

    pages = []

    for page in doc:
        text = page.get_text("text")

        if text and text.strip():
            pages.append(text)

    doc.close()

    full_text = "\n".join(pages)

    return full_text, _citation_from_metadata(meta, pdf_path)


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r'(\w)-[ \t]*\n[ \t]*(\w)', r'\1-\2', text)

    lines = text.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()

        if not line:
            cleaned.append("")
            continue

        if len(line) < 30 and re.fullmatch(r'[\d\s|–—-]+', line):
            continue

        if line.casefold() in KNOWN_HEADERS_NORMALIZED:
            continue

        cleaned.append(line)

    text = "\n".join(cleaned)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()


def remove_references(text: str) -> str:
    pattern = re.compile(
        r'(?im)^\s*(references|references\s+and\s+notes|bibliography|literature\s+cited|works\s+cited)\s*$'
    )

    matches = list(pattern.finditer(text))

    for match in reversed(matches):
        position = match.start() / max(len(text), 1)

        if position >= 0.60:
            return text[:match.start()].strip()

    return text.strip()


def split_sentences(text: str) -> list[str]:
    protected = text
    replacements = {}

    for i, abbreviation in enumerate(ABBREVIATIONS):
        token = f"__ABBR_{i}__"
        replacements[token] = abbreviation
        protected = protected.replace(abbreviation, token)

    sentences = re.split(r'(?<=[.!?])\s+', protected)

    restored = []

    for sentence in sentences:
        for token, abbreviation in replacements.items():
            sentence = sentence.replace(token, abbreviation)

        sentence = sentence.strip()

        if sentence:
            restored.append(sentence)

    return restored


def split_long_sentence(sentence: str, chunk_size: int) -> list[str]:
    words = sentence.split()

    if len(words) <= chunk_size:
        return [sentence]

    pieces = []

    for start in range(0, len(words), chunk_size):
        piece = words[start:start + chunk_size]

        if piece:
            pieces.append(" ".join(piece))

    return pieces


def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap_sentences: int = 2,
) -> list[str]:

    sentences = split_sentences(text)
    normalized_sentences = []

    for sentence in sentences:
        pieces = split_long_sentence(sentence, chunk_size)
        normalized_sentences.extend(pieces)

    chunks = []
    current = []
    current_len = 0

    for sentence in normalized_sentences:
        wlen = len(sentence.split())

        if current and current_len + wlen > chunk_size:
            chunk = " ".join(current).strip()

            if chunk:
                chunks.append(chunk)

            if overlap_sentences > 0:
                overlap = current[-overlap_sentences:]
            else:
                overlap = []

            current = overlap + [sentence]
            current_len = sum(len(item.split()) for item in current)
            continue

        current.append(sentence)
        current_len += wlen

    if current:
        chunk = " ".join(current).strip()

        if chunk:
            chunks.append(chunk)

    return chunks

# ════════════════════════════════════════════════════════════════════
# 7. LLM CLIENT + CALL HELPERS
# ════════════════════════════════════════════════════════════════════

# "qwen_local" talks to a self-hosted OpenAI-compatible server (e.g.
# `vllm serve Qwen/Qwen3-32B --port 8000`) instead of a paid API —
# override --served_model_name / --base_url at the CLI if your server
# uses a different model name or port.
#
# Pinned to the established dense Qwen3 line (0.6B-32B), NOT the newer
# Qwen3.8/Qwen3.5 hybrid-architecture releases: vllm==0.8.5 (this repo's
# pin — see requirements.txt for why) predates those architectures and
# cannot serve them at all (confirmed: Qwen3.8-27B fails at config-load
# with "model type qwen3_5 not recognized" on a transformers version old
# enough for vllm's tokenizer code, and fails later in tokenizer setup
# on a transformers version new enough to recognize the architecture —
# no transformers version satisfies both).、

MODEL_CONFIG = {
    "gpt4o":  {"provider": "openai",    "model": "gpt-4o",            "env": "OPENAI_API_KEY"},
    "claude": {"provider": "anthropic", "model": "claude-sonnet-4-6",  "env": "ANTHROPIC_API_KEY"},
    "qwen_hf": {
        "provider":         "transformers",
        "model":            str(MODELS_DIR / "Qwen3.5-27B"),
        "env":              None,
        "disable_thinking": True,  # same reasoning as qwen_local above
    },
    # Qwen3.8-27B served over HTTP by vLLM, NOT loaded in-process. The
    # repo-pinned vllm==0.8.5 cannot serve the qwen3_5 architecture, so this
    # needs the separate .venv-vllm (vllm 0.19.1), started with:
    #
    #   ./code/.venv-vllm/bin/vllm serve $MODELS_DIR/Qwen3.8-27B \
    #       --served-model-name qwen38 --port 8000 \
    #       --gpu-memory-utilization 0.78 --max-model-len 32768 \
    #       --additional-config '{"gdn_prefill_backend":"triton"}'
    #
    # THINKING OFF, exactly as for qwen_hf, so the two models run through
    # an identical method and the only difference between the runs is the
    # checkpoint. Measured on 8 real Step 2 prompts with thinking off:
    # 19-30 completion tokens, all finish_reason "stop", all parse -- the
    # pipeline's native 110-token budget is ample and no wrapper is needed.
    # (Turning thinking ON costs ~10x the tokens and needs the separate
    # tier3_qwen38.py, which raises the budgets, strips the reasoning
    # before parsing, and rejects truncated replies.)
    "qwen38": {
        "provider":         "openai_compatible",
        "model":            "qwen38",              # --served-model-name
        "base_url":         "http://localhost:8000/v1",
        "env":              None,
        "disable_thinking": True,
    },
}

RETRY_LIMIT = 3
RETRY_DELAY = 2

# Keyed by checkpoint path so one process loads the weights once.
_CHECKPOINT_CACHE = {}


def load_checkpoint(path: str):
    """
    Load a local HF checkpoint once per process and reuse it.

    01_tier3_hipporag.py builds the RAG backend and the chat client
    separately, and with --rag_backend hipporag --model qwen_hf both want
    this same 27B checkpoint -- two copies are ~50GB each and do not fit
    on one 80GB card. hipporag_rag.py calls this too, so both paths share
    one set of weights.

    The pad token and left-padding are set here because the batched caller
    (hipporag_rag's batching layer) needs them, and they are inert for
    call_api()'s single-prompt path, which pads nothing and passes
    pad_token_id explicitly.
    """
    if path not in _CHECKPOINT_CACHE:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[MODEL] Loading local checkpoint: {path}")
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, device_map="cuda",
        )
        model.eval()
        _CHECKPOINT_CACHE[path] = (tokenizer, model)
        print("[MODEL] Ready.")

    return _CHECKPOINT_CACHE[path]


def make_client(model_key: str):
    cfg = MODEL_CONFIG[model_key]

    if cfg["provider"] == "openai_compatible":
        # Local server (vLLM/SGLang/etc.) — no API key needed.
        from openai import OpenAI
        return OpenAI(base_url=cfg["base_url"], api_key="not-needed")

    if cfg["provider"] == "transformers":
        return load_checkpoint(cfg["model"])

    api_key = os.environ.get(cfg["env"])
    if not api_key:
        raise ValueError(f"{cfg['env']} not set.\nRun: export {cfg['env']}='your-key'")
    if cfg["provider"] == "openai":
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def call_api(client, model_key: str, prompt: str,
             max_tokens: int = 500, dry_run: bool = False) -> str | None:
    if dry_run:
        return None  # handled by callers

    cfg = MODEL_CONFIG[model_key]
    for attempt in range(RETRY_LIMIT):
        try:
            if cfg["provider"] == "transformers":
                import torch
                tokenizer, model = client
                # return_dict=True: this checkpoint's chat template returns a
                # BatchEncoding (input_ids + attention_mask), not a bare
                # tensor, even for text-only prompts — likely a side effect
                # of the model's vision-language template. Unpacking via
                # **encoded also avoids transformers' "missing attention_mask"
                # warning that a bare input_ids tensor would trigger.
                encoded = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True, return_tensors="pt", return_dict=True,
                    enable_thinking=not cfg.get("disable_thinking", False),
                ).to(model.device)
                input_ids = encoded["input_ids"]
                with torch.no_grad():
                    output_ids = model.generate(
                        **encoded, max_new_tokens=max_tokens,
                        do_sample=True, temperature=0.5,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = output_ids[0][input_ids.shape[-1]:]
                return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            elif cfg["provider"] in ("openai", "openai_compatible"):
                kwargs = dict(
                    model=cfg["model"], max_tokens=max_tokens, temperature=0.5,
                    messages=[{"role": "user", "content": prompt}],
                )
                if cfg.get("disable_thinking"):
                    # vLLM/SGLang extension (ignored by the real OpenAI
                    # API) — Qwen3-style chat template kwarg to skip the
                    # <think>...</think> reasoning trace.
                    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content.strip()
            else:
                resp = client.messages.create(
                    model=cfg["model"], max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp.content[0].text.strip()
        except Exception as e:
            print(f"  [ERROR] attempt {attempt+1}: {e}")
            if attempt < RETRY_LIMIT - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


# ════════════════════════════════════════════════════════════════════
# 8. RESPONSE PARSING
# ════════════════════════════════════════════════════════════════════

def parse_number(text: str) -> float | None:

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
