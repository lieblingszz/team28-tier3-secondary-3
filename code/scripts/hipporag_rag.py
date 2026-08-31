
import os
import re
import sys
import json
import time
import queue
import hashlib
import argparse
import threading
import traceback
from pathlib import Path
from scipy.stats import rankdata
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    MODELS_DIR,
    extract_text_from_pdf, clean_text, remove_references, chunk_text,
    load_checkpoint, pdf_citation,
)

from hipporag import HippoRAG
from hipporag.llm.base import BaseLLM, LLMConfig
from hipporag.embedding_model.base import BaseEmbeddingModel, EmbeddingConfig
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.misc_utils import QuerySolution


# Everything the pipeline needs ships inside code/, so these resolve from
# this file's own location and need no configuration. Override with
# CORPUS_DIR / HIPPORAG_INDEX to point elsewhere.
CORPUS_DIR = Path(os.environ.get(
    "CORPUS_DIR", Path(__file__).parent / "../corpus")).resolve()
SAVE_DIR = Path(os.environ.get(
    "HIPPORAG_INDEX", Path(__file__).parent / "../hipporag_index")).resolve()

# code/model/ ships EMPTY: the checkpoints are tens of GB and have to be
# downloaded from Hugging Face first -- see README.md, "Downloading the
# models". Override the location with MODELS_DIR.
# This basename is also hipporag's IDENTITY for the index: the working
# directory is local_<basename>_<embedding name>/. It must stay
# "Qwen3.8-27B" to match the shipped hipporag_index/ graph, which that
# checkpoint built on 2026-08-30. Changing it orphans the graph.
LLM_PATH = str(MODELS_DIR / "Qwen3.8-27B")

# The embedding model's NAME, which must stay stable: hipporag derives its
# working directory from llm_name + embedding_model_name, so substituting a
# local path here silently orphans the prebuilt index in SAVE_DIR and
# triggers a full rebuild. Where the weights are actually loaded from is a
# separate question -- see _embedding_weights() below.
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"


def _embedding_weights(name):
    """Resolve where to LOAD the embedding weights from, without changing the
    model's name.

    sentence-transformers accepts either a local directory or a hub id. If
    model/<basename> exists we load from there, so a machine with no network
    works and nothing is re-downloaded; otherwise we pass the hub id through
    and it caches into ~/.cache/huggingface. Either way the name hipporag
    sees is unchanged, so the index directory is the same one both times.
    """
    local = MODELS_DIR / Path(name).name
    return str(local) if local.is_dir() else name

# Set HIPPORAG_LLM_BASE_URL to send HippoRAG's own LLM calls (fact reranking
# during retrieval, OpenIE during a rebuild) to an OpenAI-compatible server
# instead of loading a checkpoint in-process.
#
# This exists for the Qwen3.8-27B path: that model is served by vLLM, which
# holds ~64 GiB, leaving no room for HippoRAG to load its own 27B (~50 GiB)
# alongside it. Pointing both at the one server means a single copy of the
# weights on the card, and only the 4B embedding model stays in-process.
#
# Unset (the default), nothing changes: the local transformers backend is
# used exactly as before.
LLM_BASE_URL = os.environ.get("HIPPORAG_LLM_BASE_URL")
LLM_SERVED_MODEL = os.environ.get("HIPPORAG_LLM_MODEL", "qwen38")

# hipporag defaults to 2048. Well-formed OpenIE responses on this corpus
# run 167-460 tokens; only ones that lost the JSON format reached the cap.
TEMPERATURE = 0.3
MAX_NEW_TOKENS = 512

# A batch-size limit is not a memory bound on its own: every sequence in a
# padded batch pays for the longest prompt in it.
MAX_BATCH_TOKENS = 16384

GENERATION_TIMEOUT_S = 900

# ── Graph / dense fusion ────────────────────────────────────────────────
#
# hipporag has no weight controlling how much the graph decides the final
# ranking. passage_node_weight (BaseConfig, default 0.05) looks like one but
# is not: it scales the PPR *restart vector* (HippoRAG.py:1401), so the dense
# signal is diffused through the graph before it reaches the ranking. The
# mapping to "share of influence" is non-linear and topology-dependent, the
# two seed sources are never normalised against each other, and no value of
# it reproduces the dense retriever -- even at 1.0 it is still a diffusion.
# HippoRAG 2's own sweep (arXiv 2502.14802, Table 5) confirms it is a weak
# dial: 2.6 points of Passage Recall@5 across a 50x change.
#
# A real weighted blend has to happen after both rankings exist. hipporag
# computes the dense ranking inside graph_search_with_fact_entities
# (HippoRAG.py:1391) and throws it away -- the function returns only the PPR
# arrays -- so _search_blended() below re-runs the retrieve loop itself and
# keeps both. Neither needs a library patch: both methods are public and
# return ALL passages over the same index space.
#
# GRAPH_WEIGHT 1.0 (the default) short-circuits to stock hipporag, so nothing
# changes unless a weight is asked for.
GRAPH_WEIGHT = float(os.environ.get("HIPPORAG_GRAPH_WEIGHT", "1.0"))
FUSION_MODE = os.environ.get("HIPPORAG_FUSION", "pit")
FUSION_MODES = ("pit", "minmax", "rrf")

# Standard RRF constant (Cormack et al. 2009); large enough that the top few
# ranks do not dominate the sum.
RRF_K = 60

# Chunks are encoded in batches of this size during an index build. hipporag's
# embedding_store calls batch_encode() with no batch_size, so this default is
# what a build actually uses. 16 needs roughly 16 GB of headroom for
# Qwen3-Embedding-4B over 1k+ long chunks -- more than is left when a vLLM
# server already holds 78% of the card, which is exactly how an index build
# alongside a running server OOMs. Lower it with HIPPORAG_EMBED_BATCH.
EMBED_BATCH = int(os.environ.get("HIPPORAG_EMBED_BATCH", "16"))


class LocalTransformersLLM(BaseLLM):
    """
    Serves hipporag's OpenIE and fact-reranking calls from a local
    checkpoint, returning the (text, metadata, cache_hit) triple its call
    sites unpack.

    hipporag fires OpenIE calls from a 32-worker ThreadPoolExecutor, but
    generate() takes one prompt at a time, so unbatched those threads
    serialize on the GPU (~35s/chunk, 24+ hours for this corpus). infer()
    therefore enqueues and blocks; one worker thread drains the queue and
    runs each group through a single padded generate() (~1.6s/item).
    """

    def __init__(self, global_config=None, model_path=LLM_PATH,
                 max_batch_size=8, batch_wait_s=0.3,
                 max_batch_tokens=MAX_BATCH_TOKENS,
                 timeout_s=GENERATION_TIMEOUT_S):
        super().__init__(global_config=global_config)
        self.cache_model_id = Path(model_path).name
        # BaseLLM.__init__ does not call this; only CacheOpenAI's does.
        self._init_llm_config()
        # Shared with utils.call_api's client: tier3_pipeline builds the RAG
        # backend and the chat client separately, and two copies of the 27B
        # do not fit on one card. utils.load_checkpoint also sets the pad
        # token and left-padding that _generate_once's batching needs.
        self.tokenizer, self.model = load_checkpoint(model_path)

        self.max_batch_size = max_batch_size
        self.batch_wait_s = batch_wait_s
        self.max_batch_tokens = max_batch_tokens
        self.timeout_s = timeout_s

        self.cache = {}
        self.cache_lock = threading.Lock()
        self.cache_file = None

        # hipporag saves OpenIE output only after both the NER and triple
        # passes finish for the whole corpus, so an interruption discards
        # every completed call. Persisting responses here lets a restart
        # replay them as cache hits instead.
        save_dir = getattr(self.global_config, "save_dir", None)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, "llm_response_cache.jsonl")
            restored = self._load_cache(path)
            if restored:
                print(f"[HippoRAG LLM] Resuming: {restored} cached responses from {path}")
            self.cache_file = open(path, "a", encoding="utf-8")

        self.requests = queue.Queue()
        self.worker_alive = True
        threading.Thread(target=self._worker, daemon=True).start()

    def _init_llm_config(self):
        self.llm_config = LLMConfig.from_dict({
            "llm_name": self.global_config.llm_name,
            "generate_params": {
                "max_new_tokens": self.global_config.max_new_tokens or MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
            },
        })

    @staticmethod
    def _key(model_id, messages, max_new_tokens, temperature):
        payload = json.dumps(
            [model_id,
             [(m.get("role"), m.get("content")) for m in messages],
             max_new_tokens,
             temperature],
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_cache(self, path):
        if not os.path.exists(path):
            return 0

        restored = 0

        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A torn line is always the last one, from a hard kill.
                    continue
                self.cache[record["k"]] = (record["t"], record["m"])
                restored += 1

        return restored

    def _persist(self, key, text, metadata):
        if self.cache_file is None:
            return

        self.cache_file.write(json.dumps(
            {"k": key, "t": text, "m": metadata}, ensure_ascii=False) + "\n")
        self.cache_file.flush()

    def infer(self, messages, **kwargs):
        params = self.llm_config.generate_params
        max_new_tokens = (kwargs.get("max_completion_tokens")
                          or kwargs.get("max_tokens")
                          or params.get("max_new_tokens", MAX_NEW_TOKENS))
        temperature = TEMPERATURE

        key = self._key(self.cache_model_id, messages, max_new_tokens, temperature)

        with self.cache_lock:
            cached = self.cache.get(key)

        if cached is not None:
            return cached[0], cached[1], True

        if not self.worker_alive:
            raise RuntimeError("HippoRAG LLM batch worker is not running")

        request = {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "done": threading.Event(),
            "result": None,
            "error": None,
            "key": key,
        }
        self.requests.put(request)

        # An unbounded wait would turn any worker-side failure into a
        # silent, permanent hang of every OpenIE thread.
        if not request["done"].wait(timeout=self.timeout_s):
            raise TimeoutError(
                f"HippoRAG LLM did not return within {self.timeout_s}s")

        if request["error"] is not None:
            raise request["error"]

        return request["result"][0], request["result"][1], False

    def _worker(self):
        try:
            while True:
                batch = []
                try:
                    batch = self._collect()
                    self._run(batch)
                except Exception as exc:
                    # This thread must never die: infer() blocks on an event
                    # only this thread sets, so a dead worker parks all 32
                    # OpenIE threads forever with no traceback.
                    traceback.print_exc()
                    self._fail(batch, exc)
        finally:
            self.worker_alive = False

    def _collect(self):
        """Block for one request, then gather more for up to batch_wait_s."""
        batch = [self.requests.get()]
        deadline = time.time() + self.batch_wait_s

        while len(batch) < self.max_batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                batch.append(self.requests.get(timeout=remaining))
            except queue.Empty:
                break

        return batch

    @staticmethod
    def _normalise_openie(prompt, text):
        """
        Qwen returns a bare JSON array for roughly one OpenIE call in four
        -- ["Alvarez", "Americans", ...] instead of the
        {"named_entities": [...]} object hipporag's regex needs. These are
        complete responses (finish_reason 'stop', well under the token
        cap), so hipporag's repair pass cannot help and the chunk silently
        loses its entities. Rewrap them.

        Keyed off the prompt, which names the field it wants, so this can
        only ever touch OpenIE calls -- never the DSPy reranking ones.
        """
        for key in ("named_entities", "triples"):
            if f'"{key}"' in prompt:
                break
        else:
            return text

        if f'"{key}"' in text:
            return text

        body = text.strip()
        if body.startswith("```"):
            body = re.sub(r"^```[A-Za-z]*\s*", "", body)
            body = re.sub(r"\s*```$", "", body).strip()

        if not body.startswith("["):
            return text

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return text

        if not isinstance(parsed, list):
            return text

        return json.dumps({key: parsed}, ensure_ascii=False)

    @staticmethod
    def _fail(batch, exc):
        for request in batch:
            if not request["done"].is_set():
                request["error"] = exc
                request["done"].set()

    def _run(self, batch):
        try:
            for group in self._split(batch):
                try:
                    self._generate(group)
                except Exception as exc:
                    traceback.print_exc()
                    self._fail(group, exc)
        finally:
            self._fail(batch, RuntimeError("batch produced no result"))

    def _split(self, batch):
        """
        Group requests so batch_size * (padded_len + max_new_tokens) stays
        under max_batch_tokens: one 2664-token outlier pads the whole batch
        up to its length, costing several times what typical chunks do.
        """
        for request in batch:
            request["prompt"] = self.tokenizer.apply_chat_template(
                request["messages"], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            request["n_tokens"] = len(self.tokenizer(request["prompt"])["input_ids"])

        groups = []
        current, pad_len, gen_len = [], 0, 0

        for request in batch:
            new_pad = max(pad_len, request["n_tokens"])
            new_gen = max(gen_len, request["max_new_tokens"])

            if current and (len(current) + 1) * (new_pad + new_gen) > self.max_batch_tokens:
                groups.append(current)
                current, new_pad, new_gen = [], request["n_tokens"], request["max_new_tokens"]

            current.append(request)
            pad_len, gen_len = new_pad, new_gen

        if current:
            groups.append(current)

        return groups

    def _generate(self, batch):
        """One padded generate(), halving the batch on CUDA OOM."""
        try:
            self._generate_once(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

            if len(batch) == 1:
                raise

            mid = len(batch) // 2
            print(f"[HippoRAG LLM] CUDA OOM at batch size {len(batch)} -- "
                  f"retrying as {mid} + {len(batch) - mid}", flush=True)
            self._generate(batch[:mid])
            self._generate(batch[mid:])

    def _generate_once(self, batch):
        encoded = self.tokenizer(
            [r["prompt"] for r in batch], return_tensors="pt", padding=True,
        ).to(self.model.device)
        input_len = encoded["input_ids"].shape[1]

        max_new_tokens = max(r["max_new_tokens"] for r in batch)
        temperature = TEMPERATURE

        with torch.no_grad():
            output_ids = self.model.generate(
                **encoded, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                pad_token_id=self.tokenizer.pad_token_id,
            )

        stop_ids = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        stop_ids.discard(None)

        for i, request in enumerate(batch):
            # generate() pads every sequence to the batch-wide length, so
            # one that stopped early carries trailing pad/eos. Counting it
            # would report finish_reason 'length', and hipporag runs those
            # through fix_broken_generated_json() (openie_openai.py:53,99),
            # mangling JSON that was never truncated.
            generated = output_ids[i][input_len:].tolist()
            length = len(generated)

            for position, token in enumerate(generated):
                if token in stop_ids:
                    length = position + 1
                    break

            text = self._normalise_openie(
                request["prompt"],
                self.tokenizer.decode(generated[:length], skip_special_tokens=True).strip())
            metadata = {
                "prompt_tokens": request["n_tokens"],
                "completion_tokens": length,
                "finish_reason": "length" if length >= request["max_new_tokens"] else "stop",
            }

            with self.cache_lock:
                self.cache[request["key"]] = (text, metadata)

            self._persist(request["key"], text, metadata)
            request["result"] = (text, metadata)
            request["done"].set()


class LocalSentenceTransformerEmbedding(BaseEmbeddingModel):
    """Local sentence-transformers model in place of hipporag's OpenAI /
    GritLM / NV-Embed-v2 / Contriever wrappers."""

    def __init__(self, global_config=None, embedding_model_name=None):
        super().__init__(global_config=global_config)

        if embedding_model_name:
            self.embedding_model_name = embedding_model_name

        device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = _embedding_weights(self.embedding_model_name)
        where = "" if weights == self.embedding_model_name else f" from {weights}"
        print(f"[HippoRAG Embed] Loading {self.embedding_model_name} on {device}{where}")
        self.model = SentenceTransformer(weights, device=device)
        self.embedding_dim = self.model.get_embedding_dimension()

        self.embedding_config = EmbeddingConfig.from_dict({
            "embedding_model_name": self.embedding_model_name,
            "norm": getattr(self.global_config, "embedding_return_as_normalized", True),
        })

    def batch_encode(self, texts, **kwargs):
        # Some call sites pass a bare string rather than a list.
        if isinstance(texts, str):
            texts = [texts]

        instruction = kwargs.get("instruction", "")
        prompt = f"Instruct: {instruction}\nQuery: " if instruction else None

        return np.asarray(self.model.encode(
            texts,
            batch_size=kwargs.get("batch_size", EMBED_BATCH),
            prompt=prompt,
            normalize_embeddings=bool(kwargs.get("norm", self.embedding_config.norm)),
            convert_to_numpy=True,
        ))


class RemoteOpenAILLM(BaseLLM):
    """
    HippoRAG's LLM calls sent to an OpenAI-compatible server rather than a
    checkpoint loaded here. Same contract as LocalTransformersLLM:
    infer(messages, **kwargs) -> (text, metadata, cache_hit).

    No batching layer: the server does its own continuous batching, so the
    32 OpenIE threads become 32 concurrent HTTP requests and vLLM schedules
    them. That is why this class is a fraction of the size of the local one.

    Thinking is disabled. HippoRAG's fact-reranking prompt expects a
    structured reply terminated by a marker, and a reasoning trace both
    breaks that parse and costs ~10x the tokens on every one of the ~1,450
    reranking calls a full run makes.
    """

    def __init__(self, global_config=None, base_url=None, served_model=None):
        super().__init__(global_config=global_config)
        self.cache_model_id = served_model
        self._init_llm_config()

        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.served_model = served_model
        print(f"[HippoRAG LLM] Using served model {served_model!r} at {base_url}")

        self.cache = {}
        self.cache_lock = threading.Lock()
        self.cache_file = None

        save_dir = getattr(self.global_config, "save_dir", None)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, "llm_response_cache.jsonl")
            restored = self._load_cache(path)
            if restored:
                print(f"[HippoRAG LLM] Resuming: {restored} cached responses from {path}")
            self.cache_file = open(path, "a", encoding="utf-8")

    def _init_llm_config(self):
        self.llm_config = LLMConfig.from_dict({
            "llm_name": self.global_config.llm_name,
            "generate_params": {
                "max_new_tokens": self.global_config.max_new_tokens or MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
            },
        })

    # The cache format is shared with LocalTransformersLLM, so an index built
    # by one is resumable by the other. _key must be re-wrapped: reading it
    # off the other class yields the plain function, and assigning that into
    # a class body would make it an instance method and pass self as the
    # first argument.
    _key = staticmethod(LocalTransformersLLM._key)
    _load_cache = LocalTransformersLLM._load_cache
    _persist = LocalTransformersLLM._persist

    def infer(self, messages, **kwargs):
        params = self.llm_config.generate_params
        max_new_tokens = (kwargs.get("max_completion_tokens")
                          or kwargs.get("max_tokens")
                          or params.get("max_new_tokens", MAX_NEW_TOKENS))
        temperature = TEMPERATURE

        key = self._key(self.cache_model_id, messages, max_new_tokens, temperature)
        with self.cache_lock:
            cached = self.cache.get(key)
        if cached is not None:
            return cached[0], cached[1], True

        resp = self.client.chat.completions.create(
            model=self.served_model,
            messages=list(messages),
            max_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        
        prompt = "\n".join(
            m.get("content", "")
            for m in messages
            if isinstance(m.get("content"), str)
        )

        text = LocalTransformersLLM._normalise_openie(prompt, text)
        metadata = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            # hipporag routes finish_reason == "length" through
            # fix_broken_generated_json(); the server's own value is already
            # the right signal, so pass it through unchanged.
            "finish_reason": choice.finish_reason,
        }

        with self.cache_lock:
            self.cache[key] = (text, metadata)
        self._persist(key, text, metadata)
        return text, metadata, False


def patch_hipporag_backends(model_path, base_url=None, served_model=None):
    """
    Point hipporag's backend factories at the classes above. Must run
    before the first HippoRAG(...) is constructed.

    base_url selects the remote backend; without it the local one is used.

    hipporag 2.0.0a3's HippoRAG.__init__ takes no extraction_llm / qa_llm /
    embedding_model arguments (the README documents a later version), so
    replacing these factories is the only injection point it exposes. They
    must be patched on the submodule from sys.modules: hipporag/__init__.py
    rebinds the `hipporag.HippoRAG` attribute to the class, and patching
    the class does nothing since its __init__ reads the factories from the
    submodule's globals.
    """
    module = sys.modules["hipporag.HippoRAG"]
    if base_url:
        module._get_llm_class = lambda config: RemoteOpenAILLM(
            config, base_url=base_url, served_model=served_model)
    else:
        module._get_llm_class = lambda config: LocalTransformersLLM(config, model_path)
    module._get_embedding_model_class = lambda embedding_model_name=None: LocalSentenceTransformerEmbedding


class HippoRAGRag:

    def __init__(self, corpus_dir=CORPUS_DIR, save_dir=SAVE_DIR, rebuild=False,
                 llm_model_path=LLM_PATH, embedding_model_name=EMBEDDING_MODEL,
                 max_new_tokens=MAX_NEW_TOKENS, index_on_init=True,
                 llm_base_url=None, llm_served_model=None,
                 graph_weight=None, fusion=None, passage_node_weight=None):
        # Env vars are the default so no CLI plumbing is needed through
        # 01_tier3_hipporag.py; explicit arguments still win.
        llm_base_url = llm_base_url or LLM_BASE_URL
        llm_served_model = llm_served_model or LLM_SERVED_MODEL
        patch_hipporag_backends(llm_model_path, llm_base_url, llm_served_model)

        self.graph_weight = GRAPH_WEIGHT if graph_weight is None else float(graph_weight)
        self.fusion = fusion or FUSION_MODE
        if self.fusion not in FUSION_MODES:
            raise ValueError(f"fusion must be one of {FUSION_MODES} (got {self.fusion!r})")
        self.blending = self.graph_weight < 1.0

        # Counts queries where the graph arm produced nothing and retrieval
        # fell back to pure dense -- see _search_blended().
        self.dense_fallbacks = 0
        self.blended_queries = 0

        # Keep HippoRAG's standard passage_node_weight=0.05 in both stock and
        # blended retrieval. The graph-side arm is therefore the standard
        # HippoRAG ranking, and the external fusion combines that ranking with
        # the matched dense passage ranking.
        if passage_node_weight is None:
            passage_node_weight = 0.05
        self.passage_node_weight = float(passage_node_weight)

        self.corpus_dir = Path(corpus_dir)
        self.citation_to_file = {}

        # force_index_from_scratch and max_new_tokens are BaseConfig fields,
        # not HippoRAG.__init__ arguments.
        #
        # llm_name stays keyed on the CHECKPOINT PATH even when the calls go
        # to a served model, because hipporag derives its working directory
        # from it: changing the name would orphan the existing
        # hipporag_index/local_Qwen3.8-27B_.../ graph and silently trigger a
        # full 2,264-call rebuild.
        self.hippo = HippoRAG(global_config=BaseConfig(
            save_dir=str(save_dir),
            llm_name=f"local/{Path(llm_model_path).name}",
            embedding_model_name=embedding_model_name,
            openie_mode="online",
            force_index_from_scratch=rebuild,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            # A BaseConfig field, so no library patch is needed to change it.
            passage_node_weight=self.passage_node_weight,
        ))

        if self.blending:
            print(f"[HippoRAG] Fusion on: {self.graph_weight:.2f} graph / "
                  f"{1 - self.graph_weight:.2f} dense, {self.fusion} normalisation, "
                  f"passage_node_weight={self.passage_node_weight}")

        # 01_tier3_hipporag.py constructs this and queries immediately.
        # index() is content-hash deduped, so this is cheap once built.
        if index_on_init:
            self.index()

    def load_chunks(self):
        chunks = []
        pdfs = sorted(self.corpus_dir.glob("*.pdf"))

        for pdf_path in pdfs:
            text, citation = extract_text_from_pdf(str(pdf_path))
            self.citation_to_file[citation] = pdf_path.name
            text = remove_references(clean_text(text))
            # index() takes raw strings with no metadata field, so the
            # citation rides in the text; retrieve_with_metadata() splits
            # it back off this prefix.
            chunks.extend(f"{citation}: {chunk}"
                          for chunk in chunk_text(text, chunk_size=400, overlap_sentences=2))

        print(f"[HippoRAG] {len(chunks)} chunks from {len(pdfs)} PDFs")
        return chunks

    def index(self):
        chunks = self.load_chunks()

        if not chunks:
            print("[HippoRAG] No chunks found")
            return

        self.hippo.index(docs=chunks)

    def _source_for(self, citation):
        """
        02_eval_retrieval_quality.py resolves a hit to a paper via
        h["source"], falling back to the citation, and matches it against
        corpus_metadata.json's PDF filenames. Without a source every hit
        counts as unlabelled, scoring this backend at zero recall.
        """
        if not self.citation_to_file:
            self.citation_to_file = {
                pdf_citation(str(p)): p.name
                for p in sorted(self.corpus_dir.glob("*.pdf"))
            }

        return self.citation_to_file.get(citation)

    def _search(self, queries, top_k):
        if not self.blending:
            return self.hippo.retrieve(queries=queries, num_to_retrieve=top_k)
        return self._search_blended(queries, top_k)

    def _search_blended(self, queries, top_k):
        """
        hipporag's own retrieve loop (HippoRAG.py:400-425), reproduced so
        both rankings survive: graph_search_with_fact_entities computes the
        dense ranking internally and returns only the PPR arrays, so the
        blend cannot be assembled from its output.

        Both arms return every passage, sorted, over the same index space
        (HippoRAG.py:1421 asserts the PPR side covers the whole corpus), so
        _fuse() can align them exactly by document id.
        """
        hippo = self.hippo
        # _search_blended() bypasses HippoRAG.retrieve(), whose normal guard
        # prepares query embeddings, passage-node keys, and graph handles.
        if not hasattr(hippo, "query_to_embedding"):
            hippo.prepare_retrieval_objects()

        # Batches both instruction-encodings of every query in one call
        # instead of one-at-a-time inside the per-arm getters below.
        hippo.get_query_embeddings(queries)

        solutions = []

        for query in queries:
            self.blended_queries += 1

            fact_scores = hippo.get_fact_scores(query)
            fact_indices, facts, _ = hippo.rerank_facts(query, fact_scores)

            # The query embedding is already cached by get_query_embeddings,
            # so this second dense call is one matmul, not an encode.
            dense_ids, dense_scores = hippo.dense_passage_retrieval(query)

            graph = None
            if facts:
                try:
                    graph = hippo.graph_search_with_fact_entities(
                        query=query,
                        link_top_k=hippo.global_config.linking_top_k,
                        query_fact_scores=fact_scores,
                        top_k_facts=facts,
                        top_k_fact_indices=fact_indices,
                        passage_node_weight=hippo.global_config.passage_node_weight,
                    )
                except AssertionError:
                    # HippoRAG.py:1412 asserts the PPR seed vector is non-zero.
                    # If the graph arm cannot be formed, fall back to dense,
                    # matching the library's own empty-facts behavior.
                    graph = None

            if graph is None:
                self.dense_fallbacks += 1
                dense_vector = self._scatter(dense_ids, dense_scores, len(dense_ids))
                normalise = {"pit": self._pit, "minmax": self._minmax, "rrf": self._rrf}[self.fusion]
                norm_dense = normalise(dense_vector)
                doc_ids = np.argsort(norm_dense)[::-1]
                doc_scores = norm_dense[doc_ids]
            else:
                doc_ids, doc_scores = self._fuse(*graph, dense_ids, dense_scores)

            docs = [hippo.chunk_embedding_store.get_row(
                        hippo.passage_node_keys[doc_id])["content"]
                    for doc_id in doc_ids[:top_k]]

            solutions.append(QuerySolution(question=query, docs=docs,
                                           doc_scores=doc_scores[:top_k]))

        return solutions

    @staticmethod
    def _scatter(ids, scores, n):
        """Sorted (ids, scores) back to a dense vector indexed by doc id."""
        dense = np.zeros(n)
        dense[ids] = scores
        return dense

    @staticmethod
    def _pit(scores):
        return rankdata(scores, method="max") / len(scores)

    @staticmethod
    def _minmax(scores):
        span = scores.max() - scores.min()
        return (scores - scores.min()) / (span + 1e-12)

    @staticmethod
    def _rrf(scores):
        """1 / (k + rank), rank 1-based and descending by score."""
        order = np.argsort(scores, kind="stable")[::-1]
        ranks = np.empty(len(scores))
        ranks[order] = np.arange(1, len(scores) + 1)
        return 1.0 / (RRF_K + ranks)

    def _fuse(self, graph_ids, graph_scores, dense_ids, dense_scores):
        """graph_weight * norm(graph) + (1 - graph_weight) * norm(dense)."""
        n = len(dense_ids)
        graph = self._scatter(graph_ids, graph_scores, n)
        dense = self._scatter(dense_ids, dense_scores, n)

        normalise = {"pit": self._pit, "minmax": self._minmax, "rrf": self._rrf}[self.fusion]
        fused = (self.graph_weight * normalise(graph)
                 + (1 - self.graph_weight) * normalise(dense))

        doc_ids = np.argsort(fused)[::-1]
        return doc_ids, fused[doc_ids]

    def fallback_summary(self):
        if not self.blending or not self.blended_queries:
            return None
        share = self.dense_fallbacks / self.blended_queries
        return (f"[HippoRAG] Graph arm empty on {self.dense_fallbacks}/"
                f"{self.blended_queries} queries ({share:.1%}) -- those fell back "
                f"to pure dense retrieval")

    @staticmethod
    def _format(docs):
        # 01_tier3_hipporag.py drops this straight into its prompts, so the
        # shape has to match what tier3_pipeline's predict_outcome expects.
        if not docs:
            return "No literature context provided."

        return "Relevant evidence retrieved from prior literature:\n" + "\n".join(
            f"  - {doc}" for doc in docs)

    @staticmethod
    def _report(pairs):
        for score, doc in pairs:
            print(f"{score:.4f} | {doc[:150]}")

    def retrieve(self, query, top_k=5, verbose=False):
        solution = self._search([query], top_k)[0]

        if verbose:
            self._report(zip(solution.doc_scores, solution.docs))

        return self._format(solution.docs)

    def retrieve_merged(self, queries, top_k_each=4, max_total=6, verbose=False):
        pairs = [(float(score), doc)
                 for solution in self._search(queries, top_k_each)
                 for doc, score in zip(solution.docs, solution.doc_scores)]
        pairs.sort(reverse=True)

        if verbose:
            self._report(pairs)

        selected, seen = [], set()

        for _, doc in pairs:
            if doc in seen:
                continue

            selected.append(doc)
            seen.add(doc)

            if len(selected) >= max_total:
                break

        return self._format(selected)

    def retrieve_with_metadata(self, query, top_k=5):
        """
        Note on "similarity": with fusion off it is the raw PPR score, as
        before. With fusion on it is the fused [0,1] quantity, which under
        pit is comparable ACROSS queries -- so retrieve_merged()'s
        cross-query sort is better founded when blending than without it.
        """
        solution = self._search([query], top_k)[0]
        results = []

        for doc, score in zip(solution.docs, solution.doc_scores):
            citation, _, text = doc.partition(": ")
            results.append({"text": text or doc,
                            "citation": citation,
                            "source": self._source_for(citation),
                            "chunk_n": None,
                            "similarity": round(float(score), 4)})

        return results


def sweep(weights, top_k=5, fusion="pit", n_queries=8, **rag_kwargs):
    """
    How much does the blend actually change what the LLM sees?

    Retrieval only -- no generation, so this runs in minutes and answers the
    question that decides whether a full pipeline run is worth the hours. If
    top_k at w=0.7 overlaps w=1.0 at ~1.0, the fusion changes nothing
    downstream and no pipeline run can show an effect.

    Uses the pipeline's own Step 1 queries so the comparison is against
    traffic the system really issues, not invented strings.
    """
    from utils import load_survey_items, get_construct_label, build_outcome_query

    items = list(load_survey_items().items())[:n_queries]
    queries = [build_outcome_query(spec, get_construct_label(key)) for key, spec in items]

    print(f"\n{'='*72}")
    print(f"  Graph/dense fusion sweep -- {len(queries)} queries, top_k={top_k}, "
          f"fusion={fusion}")
    print(f"{'='*72}\n")

    # One instance, one model load. Both arms are weight-independent, so
    # they are computed once per query and only the numpy fusion is repeated
    # -- which also removes a confound: the DSPy fact reranker runs at
    # temperature 0.3, so re-deriving the graph arm per weight could select
    # different facts and attribute reranker noise to the fusion weight.
    rag = HippoRAGRag(index_on_init=False, graph_weight=min(weights + [0.5]),
                      fusion=fusion, **rag_kwargs)
    hippo = rag.hippo
    if not hasattr(hippo, "query_to_embedding"):
        hippo.prepare_retrieval_objects()
    hippo.get_query_embeddings(queries)

    arms, fallbacks = [], 0

    for query in queries:
        fact_scores = hippo.get_fact_scores(query)
        fact_indices, facts, _ = hippo.rerank_facts(query, fact_scores)
        dense = hippo.dense_passage_retrieval(query)

        def graph_at(passage_node_weight):
            if not facts:
                return None
            try:
                return hippo.graph_search_with_fact_entities(
                    query=query, link_top_k=hippo.global_config.linking_top_k,
                    query_fact_scores=fact_scores, top_k_facts=facts,
                    top_k_fact_indices=fact_indices,
                    passage_node_weight=passage_node_weight)
            except AssertionError:
                return None

        # passage_node_weight is read at call time, so the standard 0.05
        # ranking comes from the same instance the fusion runs on. That
        # ranking is both the reference every existing run in results/ was
        # produced with and the graph arm the fusion weight is applied to,
        # which is what makes w=1.00 reproduce stock exactly.
        stock = graph_at(0.05)
        if stock is None:
            fallbacks += 1
        arms.append((stock, dense))

    def docs_for(doc_ids):
        return [hippo.chunk_embedding_store.get_row(
                    hippo.passage_node_keys[i])["content"] for i in doc_ids[:top_k]]

    retrieved = {"stock": [docs_for((s or d)[0]) for s, d in arms]}
    for weight in weights:
        rag.graph_weight = weight
        retrieved[weight] = [
            docs_for(rag._fuse(*stock, *dense)[0] if stock else dense[0])
            for stock, dense in arms
        ]

    def jaccard(a, b):
        sa, sb = set(a), set(b)
        return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0

    def displacement(a, b):
        """Mean |rank change| over docs present in both lists."""
        pos_b = {doc: i for i, doc in enumerate(b)}
        shifts = [abs(i - pos_b[doc]) for i, doc in enumerate(a) if doc in pos_b]
        return sum(shifts) / len(shifts) if shifts else 0.0

    dense_only = min(weights)
    print(f"{'config':>22}  {'J vs stock':>11}  {'J vs dense':>11}  {'mean |drank|':>13}")
    print("-" * 72)

    def row(label, key):
        j_stock = np.mean([jaccard(a, b) for a, b in zip(retrieved[key], retrieved["stock"])])
        j_dense = np.mean([jaccard(a, b) for a, b in zip(retrieved[key], retrieved[dense_only])])
        shift = np.mean([displacement(a, b) for a, b in zip(retrieved[key], retrieved["stock"])])
        print(f"{label:>22}  {j_stock:>11.3f}  {j_dense:>11.3f}  {shift:>13.2f}")

    row("stock (pnw=0.05)", "stock")
    for weight in sorted(weights, reverse=True):
        note = "  standard HippoRAG" if weight == 1.0 else ("  pure dense" if weight == 0.0 else "")
        row(f"w={weight:.2f}{note}", weight)

    print(f"\n  stock is what every run in results/ used -- HippoRAG with its standard "
          f"passage_node_weight=0.05 and no external fusion.\n  Blended rows fuse that "
          f"standard HippoRAG ranking with the matched dense ranking.\n  J vs stock near "
          f"1.00 => the blend retrieves what HippoRAG already did.\n  J vs dense near "
          f"1.00 => it has collapsed to plain dense retrieval.\n  Standard HippoRAG arm "
          f"empty (fell back to dense) on {fallbacks}/{len(queries)} queries.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="Retrieval-only sweep over --sweep_weights: reports how much "
                             "the retrieved set moves as the graph/dense weight changes. "
                             "No generation, no ground truth needed.")
    parser.add_argument("--sweep_weights", type=float, nargs="+",
                        default=[0.0, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--corpus_dir", default=str(CORPUS_DIR))
    parser.add_argument("--save_dir", default=str(SAVE_DIR))
    parser.add_argument("--llm_model_path", default=LLM_PATH)
    parser.add_argument("--embedding_model", default=EMBEDDING_MODEL)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--graph_weight", type=float, default=None,
                        help="Weight on the standard HippoRAG passage ranking; the rest goes "
                             "to matched dense retrieval. 1.0 (default) is stock HippoRAG, unblended.")
    parser.add_argument("--fusion", type=str, default=None, choices=list(FUSION_MODES),
                        help="How the two rankings are normalised before the weighted sum. "
                             "'pit' (default) percentile-rank; 'minmax'; 'rrf' reciprocal rank.")
    parser.add_argument("--passage_node_weight", type=float, default=None,
                        help="HippoRAG's PPR seed weight for passage nodes. Defaults to the "
                             "standard HippoRAG value 0.05 for both stock and blended retrieval.")
    args = parser.parse_args()

    shared = dict(
        corpus_dir=args.corpus_dir,
        save_dir=args.save_dir,
        llm_model_path=args.llm_model_path,
        embedding_model_name=args.embedding_model,
    )

    if args.sweep:
        sweep(args.sweep_weights, top_k=args.top_k,
              fusion=args.fusion or FUSION_MODE, **shared)
        sys.exit(0)

    rag = HippoRAGRag(
        rebuild=args.rebuild,
        index_on_init=False,
        graph_weight=args.graph_weight,
        fusion=args.fusion,
        passage_node_weight=args.passage_node_weight,
        **shared,
    )

    if args.build or args.rebuild:
        rag.index()

    if args.test:
        print(rag.retrieve("scientific consensus and trust in climate scientists",
                           top_k=args.top_k, verbose=True))
        summary = rag.fallback_summary()
        if summary:
            print(summary)
