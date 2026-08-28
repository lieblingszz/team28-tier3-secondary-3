"""
rag_vector_db.py
----------------
Vector-based RAG for the Silicon Sample Benchmark.

Two modes:
  - With OPENAI_API_KEY: uses text-embedding-3-small (best quality)
  - Without key:         uses TF-IDF keyword similarity (no cost, no internet)

Usage:
    python rag_vector_db.py --build          # build the DB
    python rag_vector_db.py --test           # test retrieval on sample personas

    from rag_vector_db import VectorRAG
    rag = VectorRAG()
    ctx = rag.get_context_for_persona(age=65, gender='male', race='White',
              education='high school or less', income='middle', party='Republican')
"""

import os
import json
import math
import argparse
from pathlib import Path
from collections import Counter

# ── corpus ───────────────────────────────────────────────────────────
CORPUS = [
    # MacInnis & Krosnick
    {"id": "macinnis_01", "citation": "MacInnis & Krosnick (Stanford Climate Opinion)",
     "text": "Republicans and conservatives show significantly lower trust in climate scientists compared to Democrats and liberals. Political ideology is the single strongest predictor of trust in scientists' statements about the environment."},
    {"id": "macinnis_02", "citation": "MacInnis & Krosnick (Stanford Climate Opinion)",
     "text": "Fox News viewers show notably lower trust in climate scientists. Mainstream television news viewers show higher trust. Media diet is a significant predictor of trust in environmental scientists."},
    {"id": "macinnis_03", "citation": "MacInnis & Krosnick (Stanford Climate Opinion)",
     "text": "Education level is a strong positive predictor of trust in climate scientists. College graduates and postgraduates trust scientists significantly more than those with high school or less education."},
    {"id": "macinnis_04", "citation": "MacInnis & Krosnick (Stanford Climate Opinion)",
     "text": "Older adults aged 45 and above trust climate scientists less than younger adults. Adults aged 65 and older show the lowest levels of trust. Women trust climate scientists more than men."},
    {"id": "macinnis_05", "citation": "MacInnis & Krosnick (Stanford Climate Opinion)",
     "text": "Low-trust individuals form beliefs about global warming based on personal experience and local temperatures, NOT on scientific consensus messaging. Consensus messages have little effect on people who already distrust scientists. High-trust individuals rely on scientists and hold stable beliefs."},

    # MacInnis & Krosnick — Democrat and young educated specific
    {"id": "macinnis_06", "citation": "MacInnis & Krosnick (Stanford Climate Opinion)",
     "text": "Democrats and liberals show the highest baseline trust in climate scientists. Young educated Democrat women show near-ceiling trust levels so interventions have limited room to increase trust further. For high-trust Democrat audiences, reinforcing existing trust and mobilizing action is more effective than persuasion messaging."},
    {"id": "macinnis_07", "citation": "MacInnis & Krosnick (Stanford Climate Opinion)",
     "text": "Highly educated young Democrats already accept scientific consensus on climate change. For this group, messages emphasizing urgency, collective action, and policy solutions are more effective than basic trust-building. They respond to calls for engagement rather than persuasion."},

    # Vlasceanu — ceiling effects for Democrats
    {"id": "vlasceanu_04", "citation": "Vlasceanu et al. (2024, Science Advances)",
     "text": "Young educated liberal Democrat audiences show ceiling effects in climate trust interventions: they already hold high trust in scientists so additional messaging produces smaller effects. Interventions targeting Democrats should focus on behavioral mobilization rather than attitude change."},

    # van der Linden
    {"id": "linden_01", "citation": "van der Linden et al. (2017, Global Challenges)",
     "text": "Scientific consensus messaging — 97% of climate scientists agree on human-caused warming — increases perceived consensus across the political spectrum including Republicans and conservatives. It does NOT backfire among skeptical audiences."},
    {"id": "linden_02", "citation": "van der Linden et al. (2017, Global Challenges)",
     "text": "Inoculation messages that warn people about misinformation tactics before exposure protect the positive effects of consensus messaging. Pre-bunking is more effective than debunking after the fact."},

    # Rode et al.
    {"id": "rode_01", "citation": "Rode, Clarke & van der Linden (2026)",
     "text": "In a field experiment, scientific consensus messaging depolarized partisans. The trust-increasing effect was stronger for conservatives than liberals, reducing the ideological gap in climate beliefs. Consensus messaging is particularly effective for Republican audiences."},

    # Cologna & Siegrist
    {"id": "cologna_01", "citation": "Cologna & Siegrist (2020)",
     "text": "Trust in climate science directly predicts support for climate policy and willingness to engage in pro-environmental behavior. People who trust scientists are more likely to support government climate action."},
    {"id": "cologna_02", "citation": "Cologna & Siegrist (2020)",
     "text": "Political ideology is the strongest moderator of trust in climate scientists. Conservatives show significantly lower baseline trust. Education level is a positive predictor — more education means more trust."},

    # Cologna PLOS
    {"id": "cologna_plos_01", "citation": "Cologna et al. (PLOS Climate, 2024)",
     "text": "For skeptical and conservative audiences, emphasizing co-benefits of climate action — economic development, job creation, energy independence — is more effective than scientific consensus messaging alone. Co-benefit framing bridges ideological divides."},
    {"id": "cologna_plos_02", "citation": "Cologna et al. (PLOS Climate, 2024)",
     "text": "Personal relevance and local impact framings are effective for audiences with lower incomes and lower education. Messages connecting climate change to immediate local consequences outperform abstract global framing for these groups."},

    # Vlasceanu
    {"id": "vlasceanu_01", "citation": "Vlasceanu et al. (2024, Science Advances)",
     "text": "In a 63-country global climate tournament, young, educated, and liberal audiences showed ceiling effects: they already had high trust in scientists so intervention effects were smaller. Room for improvement is limited among high-trust high-education groups."},
    {"id": "vlasceanu_02", "citation": "Vlasceanu et al. (2024, Science Advances)",
     "text": "Conservative and older audiences showed larger intervention effects from expert-based and consensus messages in the global climate tournament. These groups have more room for trust improvement and respond more to authoritative science communication."},
    {"id": "vlasceanu_03", "citation": "Vlasceanu et al. (2024, Science Advances)",
     "text": "Social norm messages — telling people that most others support climate science — were broadly effective across countries and demographic groups. System justification framings also worked well globally."},

    # Voelkel
    {"id": "voelkel_01", "citation": "Voelkel et al. (2026, Nature Climate Change)",
     "text": "Among the most persuasive climate messages: those emphasizing scientific consensus, expert credibility, and personal qualifications of climate scientists were most effective across a broad U.S. sample."},
    {"id": "voelkel_02", "citation": "Voelkel et al. (2026, Nature Climate Change)",
     "text": "Message effectiveness varied by political partisanship. For Republican audiences, economic and national security framings of climate change performed better than environmental or moral framings."},

    # Goldwert
    {"id": "goldwert_01", "citation": "Goldwert et al. (2026, PNAS Nexus)",
     "text": "Behavioral interventions emphasizing collective action and social norms were effective for catalyzing climate advocacy behaviors. Emotional appeals worked better for high-education audiences."},
    {"id": "goldwert_02", "citation": "Goldwert et al. (2026, PNAS Nexus)",
     "text": "Practical co-benefit framing — jobs, energy costs, local air quality — worked better for lower-education and conservative audiences than abstract environmental or moral appeals."},

    # Roozenbeek
    {"id": "roozenbeek_01", "citation": "Roozenbeek, Traberg & van der Linden (2022)",
     "text": "Psychological inoculation against science misinformation is effective across all demographic groups. Warning people about manipulation tactics before exposure protects their trust in scientists."},
    {"id": "roozenbeek_02", "citation": "Roozenbeek, Traberg & van der Linden (2022)",
     "text": "Pre-bunking is consistently more effective than debunking. Effects are relatively uniform across age groups and education levels, making inoculation a broadly applicable communication strategy."},
]


# ════════════════════════════════════════════════════════════════════
# TF-IDF fallback (no external dependencies)
# ════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    return text.lower().split()

def _build_tfidf(corpus):
    N = len(corpus)
    df = Counter()
    tfs = []
    for doc in corpus:
        tokens = _tokenize(doc["text"])
        tf = Counter(tokens)
        tfs.append(tf)
        df.update(set(tokens))
    idfs = {w: math.log(N / (df[w] + 1)) for w in df}
    return tfs, idfs

def _cosine(vec_a: dict, vec_b: dict) -> float:
    keys = set(vec_a) & set(vec_b)
    dot  = sum(vec_a[k] * vec_b[k] for k in keys)
    na   = math.sqrt(sum(v**2 for v in vec_a.values()))
    nb   = math.sqrt(sum(v**2 for v in vec_b.values()))
    return dot / (na * nb + 1e-9)

def _tfidf_vec(tokens, idfs):
    tf  = Counter(tokens)
    return {w: tf[w] * idfs.get(w, 0) for w in tf}


# ════════════════════════════════════════════════════════════════════
# VectorRAG
# ════════════════════════════════════════════════════════════════════

class VectorRAG:

    def __init__(self, rebuild: bool = False):
        openai_key = os.environ.get("OPENAI_API_KEY")

        if openai_key:
            self._mode = "openai"
            self._init_chroma(openai_key, rebuild)
        else:
            self._mode = "tfidf"
            self._init_tfidf()

    # ── OpenAI / ChromaDB path ──────────────────────────────────────
    def _init_chroma(self, api_key, rebuild):
        import chromadb
        from chromadb.utils import embedding_functions

        self._client = chromadb.PersistentClient(path="./chroma_rag_db")
        self._ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key    = api_key,
            model_name = "text-embedding-3-small",
        )
        print("[RAG] Mode: OpenAI text-embedding-3-small + ChromaDB")

        existing = [c.name for c in self._client.list_collections()]
        if rebuild or "silicon_rag" not in existing:
            try:
                self._client.delete_collection("silicon_rag")
            except Exception:
                pass
            self._col = self._client.create_collection(
                name               = "silicon_rag",
                embedding_function = self._ef,
                metadata           = {"hnsw:space": "cosine"},
            )
            self._col.add(
                ids       = [c["id"]   for c in CORPUS],
                documents = [c["text"] for c in CORPUS],
                metadatas = [{"citation": c["citation"]} for c in CORPUS],
            )
            print(f"[RAG] Indexed {len(CORPUS)} chunks into ChromaDB")
        else:
            self._col = self._client.get_collection(
                name               = "silicon_rag",
                embedding_function = self._ef,
            )
            print(f"[RAG] Loaded ChromaDB ({self._col.count()} chunks)")

    def _chroma_retrieve(self, query: str, top_k: int, verbose: bool) -> str:
        res   = self._col.query(query_texts=[query], n_results=top_k)
        docs  = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        if verbose:
            for i, (d, m, dist) in enumerate(zip(docs, metas, dists)):
                print(f"  [{i+1}] sim={1-dist:.3f} | {m['citation']}")
                print(f"       {d[:80]}...")

        lines, seen = [], set()
        for doc, meta in zip(docs, metas):
            cit = meta["citation"]
            if cit not in seen:
                lines.append(f"  - {cit}: {doc}")
                seen.add(cit)
        return "Relevant evidence retrieved from prior literature:\n" + "\n".join(lines)

    # ── TF-IDF path ─────────────────────────────────────────────────
    def _init_tfidf(self):
        print("[RAG] Mode: TF-IDF keyword similarity (no API key needed)")
        self._tfs, self._idfs = _build_tfidf(CORPUS)

    def _tfidf_retrieve(self, query: str, top_k: int, verbose: bool) -> str:
        q_vec  = _tfidf_vec(_tokenize(query), self._idfs)
        scores = [
            (_cosine(q_vec, _tfidf_vec(_tokenize(c["text"]), self._idfs)), c)
            for c in CORPUS
        ]
        scores.sort(key=lambda x: x[0], reverse=True)
        top    = scores[:top_k]

        if verbose:
            for i, (score, chunk) in enumerate(top):
                print(f"  [{i+1}] sim={score:.3f} | {chunk['citation']}")
                print(f"       {chunk['text'][:80]}...")

        lines, seen = [], set()
        for score, chunk in top:
            cit = chunk["citation"]
            if cit not in seen:
                lines.append(f"  - {cit}: {chunk['text']}")
                seen.add(cit)
        return "Relevant evidence retrieved from prior literature:\n" + "\n".join(lines)

    # ── Public API ──────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = 3, verbose: bool = False) -> str:
        if self._mode == "openai":
            return self._chroma_retrieve(query, top_k, verbose)
        return self._tfidf_retrieve(query, top_k, verbose)

    def _chroma_retrieve_raw(self, query: str, top_k: int) -> list[dict]:
        res   = self._col.query(query_texts=[query], n_results=top_k)
        docs  = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        return [
            {
                "text":       doc,
                "citation":   meta.get("citation"),
                "source":     meta.get("source"),    # PDF path, as stored by add_paper.py
                "chunk_n":    meta.get("chunk_n"),
                "similarity": round(1 - dist, 4),
            }
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    def _tfidf_retrieve_raw(self, query: str, top_k: int) -> list[dict]:
        q_vec  = _tfidf_vec(_tokenize(query), self._idfs)
        scores = [
            (_cosine(q_vec, _tfidf_vec(_tokenize(c["text"]), self._idfs)), c)
            for c in CORPUS
        ]
        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:top_k]
        return [
            {
                "text":       chunk["text"],
                "citation":   chunk.get("citation"),
                "source":     chunk.get("source"),   # not set on the hardcoded seed CORPUS
                "chunk_n":    chunk.get("id"),
                "similarity": round(score, 4),
            }
            for score, chunk in top
        ]

    def retrieve_with_metadata(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Like retrieve(), but returns structured per-chunk results (text,
        citation, source PDF path, similarity) instead of one formatted
        string. Used by eval_retrieval_quality.py to check retrieved
        chunks against corpus_metadata.json's relevance tags — retrieve()
        itself is unchanged, so tier3_pipeline.py is unaffected.
        """
        if self._mode == "openai":
            return self._chroma_retrieve_raw(query, top_k)
        return self._tfidf_retrieve_raw(query, top_k)

    def retrieve_merged(
        self, queries: list[str], top_k_each: int = 4,
        max_total: int = 6, verbose: bool = False,
    ) -> str:
        """
        Retrieve from MULTIPLE queries separately and merge the results
        (deduped by citation, ranked by similarity across all queries),
        instead of concatenating everything into one query string.

        Why this exists: a single query built as
        "intervention_text + outcome question" lets whichever part is
        longer/more topically specific dominate the embedding — e.g. a
        long, topically dense intervention (like "Model accuracy") can
        pull the query entirely toward its own topic and drown out a
        short outcome signal like "donation ams", so donation-specific
        literature never surfaces even when it's in the corpus. Running
        the intervention text and the outcome/construct as two separate
        retrievals guarantees each gets its own dedicated top-k pass,
        then merges the best results from both.
        """
        all_hits = []
        for q in queries:
            if self._mode == "openai":
                all_hits.extend(self._chroma_retrieve_raw(q, top_k_each))
            else:
                all_hits.extend(self._tfidf_retrieve_raw(q, top_k_each))

        all_hits.sort(key=lambda h: h["similarity"], reverse=True)

        if verbose:
            for h in all_hits:
                print(f"  sim={h['similarity']} | {h.get('citation')}")

        lines, seen = [], set()
        for h in all_hits:
            cit = h.get("citation")
            if not cit or cit in seen:
                continue
            lines.append(f"  - {cit}: {h['text']}")
            seen.add(cit)
            if len(lines) >= max_total:
                break

        if not lines:
            return "No literature context provided."
        return "Relevant evidence retrieved from prior literature:\n" + "\n".join(lines)

    def build_query_from_persona(
        self, age, gender, race, education, income, party
    ) -> str:
        age_label = (
            "young adult" if age <= 35 else
            "middle-aged" if age <= 54 else
            "older adult"
        )
        return (
            f"{age_label} {gender} {race} American, "
            f"{party} voter, {education} education, {income} income. "
            f"How do people like this respond to climate science communication? "
            f"What message framings are effective or ineffective for this demographic?"
        )

    def get_context_for_persona(
        self, age, gender, race, education, income, party,
        top_k=3, verbose=False
    ) -> str:
        query = self.build_query_from_persona(
            age, gender, race, education, income, party
        )
        if verbose:
            print(f"\n[RAG] Query: {query}")
        return self.retrieve(query, top_k=top_k, verbose=verbose)


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--test",  action="store_true")
    args = parser.parse_args()

    rag = VectorRAG(rebuild=args.build)

    if args.test:
        personas = [
            dict(age=65, gender="male",   race="White",    education="high school or less", income="middle", party="Republican"),
            dict(age=28, gender="female", race="White",    education="postgraduate",         income="high",   party="Democrat"),
            dict(age=45, gender="male",   race="Hispanic", education="some college",         income="low",    party="Independent"),
        ]
        for p in personas:
            print(f"\n{'='*60}")
            print(f"PERSONA: {p['age']}yo {p['gender']} {p['race']} | {p['party']} | {p['education']}")
            print("="*60)
            ctx = rag.get_context_for_persona(**p, verbose=True)
            print("\nCONTEXT:\n", ctx[:300], "...")