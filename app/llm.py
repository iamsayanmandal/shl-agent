"""
Gemini wrapper used by the agent.

Uses the new google-genai SDK (google-generativeai is deprecated).
Two responsibilities:
  1) embed_texts / embed_query   -> retrieval embeddings (L2-normalized)
  2) generate_json                -> structured agent decision

If the API call fails or no key is set, callers fall back to heuristics so the
service stays up — matches the assignment's "non-deterministic conversation
should not make the system fall apart" requirement.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import List, Dict, Any, Optional

import numpy as np

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_OK = True
except Exception as _e:
    print(f"[llm] google-genai import failed: {_e}")
    _GENAI_OK = False
    genai = None
    genai_types = None


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEN_MODEL = os.environ.get("GEMINI_GEN_MODEL", "gemini-2.0-flash")
EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "text-embedding-004")

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not _GENAI_OK or not GEMINI_API_KEY:
        return None
    try:
        _client = genai.Client(api_key=GEMINI_API_KEY)
        return _client
    except Exception as e:
        print(f"[llm] client init failed: {e}")
        return None


def is_available() -> bool:
    return _get_client() is not None


def embed_texts(texts: List[str], batch_size: int = 100) -> Optional[np.ndarray]:
    """L2-normalized (N, D) array, or None on failure."""
    client = _get_client()
    if client is None:
        return None
    out_vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        try:
            resp = client.models.embed_content(
                model=EMBED_MODEL,
                contents=chunk,
                config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            for emb in resp.embeddings:
                out_vecs.append(emb.values)
        except Exception as e:
            print(f"[llm] batch embed failed ({e}); retrying per-item")
            for t in chunk:
                try:
                    r = client.models.embed_content(
                        model=EMBED_MODEL,
                        contents=t,
                        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                    )
                    out_vecs.append(r.embeddings[0].values)
                except Exception as e2:
                    print(f"[llm] per-item embed failed: {e2}")
                    return None
        time.sleep(0.05)
    arr = np.array(out_vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return (arr / norms).astype(np.float32)


def embed_query(query: str) -> Optional[np.ndarray]:
    client = _get_client()
    if client is None or not query:
        return None
    try:
        r = client.models.embed_content(
            model=EMBED_MODEL,
            contents=query,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        v = np.array(r.embeddings[0].values, dtype=np.float32)
        n = np.linalg.norm(v) + 1e-9
        return (v / n).astype(np.float32)
    except Exception as e:
        print(f"[llm] query embed failed: {e}")
        return None


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = _JSON_BLOCK_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def generate_json(prompt: str, *, temperature: float = 0.2, max_tokens: int = 1024) -> Optional[Dict[str, Any]]:
    client = _get_client()
    if client is None:
        return None
    try:
        resp = client.models.generate_content(
            model=GEN_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
        )
        text = (resp.text or "").strip()
        return _extract_json(text)
    except Exception as e:
        print(f"[llm] generate_json failed: {e}")
        return None
