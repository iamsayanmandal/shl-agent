"""
retriever.py - Hybrid retrieval over the SHL catalog

The retrieval strategy combines two approaches:
  1. BM25 (lexical) - good at exact keyword matches like "Java", "OPQ", "verbal"
  2. Dense embeddings via Gemini text-embedding-004 (semantic) - catches synonyms
     and paraphrases like "communication skills" matching "verbal reasoning"

The two rankings are merged using Reciprocal Rank Fusion (RRF), which is a
simple but effective way to combine rankings without needing to tune weights.

Embeddings are cached to disk so the server starts fast on Render.
If the Gemini API is unavailable, the service falls back to BM25 only.
The quality drops a bit but it still works and stays spec-compliant.
"""
from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from .catalog import Catalog, CatalogItem


# Simple regex tokenizer - keeps alphanumerics plus common tech chars like .+#
_TOKEN_RE = re.compile(r"[a-zA-Z0-9.+#-]+")


def tokenize(text: str) -> List[str]:
    """Lowercase tokenization for BM25. Keeps things like 'C++', '.NET', 'C#'."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class HybridRetriever:
    def __init__(self, catalog: Catalog, embeddings: Optional[np.ndarray] = None):
        self.catalog = catalog
        # Shape: (N, D) where N = catalog size, D = embedding dimension
        # None if Gemini embeddings weren't available at startup
        self.embeddings = embeddings
        self._tokenized = [tokenize(it.search_text) for it in catalog.items]
        self.bm25 = BM25Okapi(self._tokenized)

    def bm25_rank(self, query: str, k: int = 30) -> List[Tuple[int, float]]:
        """Run BM25 over the full catalog and return top-k (index, score) pairs."""
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]

    def dense_rank(self, query_emb: np.ndarray, k: int = 30) -> List[Tuple[int, float]]:
        """
        Cosine similarity between the query embedding and all catalog embeddings.
        Assumes embeddings are already L2-normalized (Gemini's API returns them that way),
        so dot product equals cosine similarity.
        """
        if self.embeddings is None:
            return []
        sims = self.embeddings @ query_emb
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idx]

    @staticmethod
    def rrf(rankings: List[List[Tuple[int, float]]], k_rrf: int = 60) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion - merges multiple ranked lists into one.
        Each item gets a score of 1/(k + rank) summed across all lists.
        k=60 is the standard default from the original RRF paper.
        Items appearing in multiple lists naturally get boosted.
        """
        scores: Dict[int, float] = {}
        for ranking in rankings:
            for rank, (idx, _score) in enumerate(ranking):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)
        return sorted(scores.items(), key=lambda x: -x[1])

    def search(
        self,
        query: str,
        top_k: int = 30,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[CatalogItem]:
        """
        Main search method. Runs BM25 and optionally dense retrieval,
        then fuses the rankings with RRF and returns the top-k catalog items.
        """
        rankings = []

        bm = self.bm25_rank(query, k=top_k)
        if bm:
            rankings.append(bm)

        if query_embedding is not None and self.embeddings is not None:
            dn = self.dense_rank(query_embedding, k=top_k)
            if dn:
                rankings.append(dn)

        if not rankings:
            return []

        # If only one ranking source, skip RRF (nothing to fuse)
        if len(rankings) == 1:
            fused = rankings[0]
        else:
            fused = self.rrf(rankings)

        return [self.catalog.items[i] for i, _ in fused[:top_k]]

    def filter_candidates(
        self,
        candidates: List[CatalogItem],
        *,
        max_duration: Optional[int] = None,
        required_test_types: Optional[List[str]] = None,
        required_job_levels: Optional[List[str]] = None,
        required_languages: Optional[List[str]] = None,
        require_remote: bool = False,
        require_adaptive: bool = False,
    ) -> List[CatalogItem]:
        """
        Post-retrieval filter for hard constraints.
        Not currently called by the agent (the LLM handles filtering via
        candidate selection), but available for future use or testing.
        """
        out = []
        for it in candidates:
            if max_duration is not None and it.duration_minutes is not None:
                if it.duration_minutes > max_duration:
                    continue
            if required_test_types:
                if not any(t in it.test_types for t in required_test_types):
                    continue
            if required_job_levels:
                lvls = {l.lower() for l in it.job_levels}
                if not any(rl.lower() in lvls for rl in required_job_levels):
                    continue
            if required_languages:
                langs = {l.lower() for l in it.languages}
                if not any(rl.lower() in langs for rl in required_languages):
                    continue
            if require_remote and not it.remote:
                continue
            if require_adaptive and not it.adaptive:
                continue
            out.append(it)
        return out


# ------------------------------------------------------------------ #
# Embedding cache
# Building embeddings for 153+ items takes ~10 seconds and costs API quota.
# Caching to disk means restarts are instant and don't re-hit the API.
# The cache is invalidated if the catalog size changes (new scrape run).
# ------------------------------------------------------------------ #

EMBED_CACHE_PATH = Path(os.environ.get("SHL_EMBED_CACHE", "data/embeddings.pkl"))


def load_or_build_embeddings(
    catalog: Catalog,
    embed_fn,
    force_rebuild: bool = False,
) -> Optional[np.ndarray]:
    """
    Load embeddings from cache if available and valid, otherwise build them.
    Returns None if embed_fn is None or the API call fails.

    embed_fn should accept List[str] and return a normalized np.ndarray of shape (N, D).
    """
    cache = EMBED_CACHE_PATH
    cache.parent.mkdir(parents=True, exist_ok=True)

    # Try loading from cache first
    if cache.exists() and not force_rebuild:
        try:
            with open(cache, "rb") as f:
                payload = pickle.load(f)
            # Validate cache: must match current catalog size and schema version
            if (
                isinstance(payload, dict)
                and payload.get("count") == len(catalog)
                and payload.get("version") == 1
            ):
                return payload["embeddings"]
        except Exception:
            pass  # Cache corrupted or wrong format - rebuild below

    if embed_fn is None:
        return None

    # Build embeddings from scratch
    try:
        texts = [it.search_text for it in catalog.items]
        embs = embed_fn(texts)
        if embs is None:
            return None
        # Save to cache with metadata for validation on next load
        with open(cache, "wb") as f:
            pickle.dump(
                {"version": 1, "count": len(catalog), "embeddings": embs}, f
            )
        return embs
    except Exception as e:
        print(f"[retriever] embedding build failed: {e}")
        return None
