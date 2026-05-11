"""
llm.py - LLM wrapper for the assessment recommender

Switched to Groq because it offers a reliable free tier with no quota issues.
The assignment explicitly lists Groq as an acceptable free LLM provider.

Using llama-3.3-70b-versatile via Groq's OpenAI-compatible API endpoint.
This means standard httpx calls work fine without a special SDK.

Two responsibilities:
  1) generate_json  -> structured agent decision (action + reply + selected_ids)
  2) embed_query / embed_texts -> kept as stubs returning None since Groq
     doesn't do embeddings; retrieval falls back to BM25-only which is fine.

If the API call fails or the key is missing, callers fall back to heuristics
so the service stays up regardless.
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Dict, Any, Optional

import httpx
import numpy as np


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Timeout: 25 seconds to stay within the 30s limit the evaluator enforces
REQUEST_TIMEOUT = 25.0


def is_available() -> bool:
    """Check if the Groq API key is present and non-empty."""
    return bool(GROQ_API_KEY)


def embed_texts(texts: List[str], **kwargs) -> Optional[np.ndarray]:
    """
    Groq doesn't support embeddings, so this always returns None.
    The retriever falls back to BM25-only mode which works well enough.
    """
    return None


def embed_query(query: str) -> Optional[np.ndarray]:
    """Same as embed_texts - no embedding support on Groq."""
    return None


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from the model response.
    Handles cases where the model wraps output in markdown code fences.
    """
    if not text:
        return None
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        # Try extracting just the JSON object if there's surrounding text
        m = _JSON_BLOCK_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def generate_json(
    prompt: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Optional[Dict[str, Any]]:
    """
    Call Groq's API and return the parsed JSON decision.
    Returns None on any failure so the agent falls back to heuristics.

    Groq's API is OpenAI-compatible so the request format is standard.
    I'm using a system + user message split because it produces cleaner
    JSON output than dumping everything into a single user message.
    """
    if not GROQ_API_KEY:
        return None

    # Split the prompt at CONVERSATION SO FAR to get system vs user parts
    # This gives the model clearer role separation
    if "CONVERSATION SO FAR:" in prompt:
        parts = prompt.split("CONVERSATION SO FAR:", 1)
        system_part = parts[0].strip()
        user_part = "CONVERSATION SO FAR:" + parts[1]
    else:
        system_part = "You are an SHL Assessment Recommender. Respond only with valid JSON."
        user_part = prompt

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_part},
            {"role": "user", "content": user_part},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            resp = client.post(GROQ_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return _extract_json(text)
    except httpx.TimeoutException:
        print("[llm] groq request timed out")
        return None
    except Exception as e:
        print(f"[llm] groq request failed: {e}")
        return None
