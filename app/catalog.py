"""
Catalog loading and preprocessing.

The catalog should be saved as data/catalog.json (the JSON the user provided).
This module loads it, normalizes test_type from the 'keys' field, and exposes
a Catalog class for retrieval.
"""
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Map SHL "keys" categories to single-letter test_type codes used in SHL's catalog
# SHL test_type letter codes (from SHL convention):
#   A = Ability & Aptitude
#   B = Biodata & Situational Judgement
#   C = Competencies
#   D = Development & 360
#   E = Assessment Exercises
#   K = Knowledge & Skills
#   P = Personality & Behavior
#   S = Simulations
KEY_TO_LETTER = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Biodata & Situational Judgement": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Personality & Behaviour": "P",
    "Simulations": "S",
}


def normalize_test_types(keys: List[str]) -> List[str]:
    """Convert SHL 'keys' list to letter codes."""
    out = []
    for k in keys or []:
        letter = KEY_TO_LETTER.get(k.strip())
        if letter and letter not in out:
            out.append(letter)
    return out


def primary_test_type(keys: List[str]) -> str:
    """Return single primary test_type letter (first matched, or 'K' as fallback)."""
    letters = normalize_test_types(keys)
    return letters[0] if letters else "K"


_DURATION_RE = re.compile(r"(\d+)")


def parse_duration_minutes(duration: str) -> Optional[int]:
    if not duration:
        return None
    m = _DURATION_RE.search(duration)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


class CatalogItem:
    __slots__ = (
        "entity_id", "name", "url", "test_type", "test_types",
        "job_levels", "languages", "duration_minutes",
        "remote", "adaptive", "description", "search_text",
    )

    def __init__(self, raw: Dict[str, Any]):
        self.entity_id = str(raw.get("entity_id", ""))
        self.name = (raw.get("name") or "").strip()
        self.url = (raw.get("link") or "").strip()
        keys = raw.get("keys") or []
        self.test_types = normalize_test_types(keys)
        self.test_type = primary_test_type(keys)
        self.job_levels = raw.get("job_levels") or []
        self.languages = raw.get("languages") or []
        self.duration_minutes = parse_duration_minutes(raw.get("duration") or "")
        self.remote = (raw.get("remote") or "").lower() == "yes"
        self.adaptive = (raw.get("adaptive") or "").lower() == "yes"
        self.description = (raw.get("description") or "").strip()
        # Pre-built searchable text blob for BM25 / embedding
        parts = [
            self.name,
            self.description,
            " ".join(self.job_levels),
            " ".join(keys),
            " ".join(self.languages[:3]),
            f"duration {self.duration_minutes or 'unknown'} minutes",
            "remote" if self.remote else "",
            "adaptive" if self.adaptive else "",
        ]
        self.search_text = " ".join(p for p in parts if p)

    def to_recommendation(self) -> Dict[str, Any]:
        """Return the structured shape required by the API spec."""
        return {
            "name": self.name,
            "url": self.url,
            "test_type": self.test_type,
        }

    def to_brief(self) -> Dict[str, Any]:
        """Compact representation passed to the LLM for grounding."""
        return {
            "id": self.entity_id,
            "name": self.name,
            "test_type": self.test_type,
            "test_types_all": self.test_types,
            "job_levels": self.job_levels,
            "duration_minutes": self.duration_minutes,
            "languages": self.languages[:5],
            "remote": self.remote,
            "adaptive": self.adaptive,
            "description": (self.description[:280] + "…") if len(self.description) > 280 else self.description,
        }


class Catalog:
    def __init__(self, items: List[CatalogItem]):
        self.items = items
        self._by_id = {it.entity_id: it for it in items}
        self._by_name_lower = {it.name.lower(): it for it in items}
        self._valid_urls = {it.url for it in items}

    @classmethod
    def load(cls, path: str) -> "Catalog":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Catalog not found at {path}")
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        items = [CatalogItem(r) for r in raw if r.get("name")]
        return cls(items)

    def get(self, entity_id: str) -> Optional[CatalogItem]:
        return self._by_id.get(str(entity_id))

    def get_by_name(self, name: str) -> Optional[CatalogItem]:
        return self._by_name_lower.get(name.lower())

    def is_valid_url(self, url: str) -> bool:
        return url in self._valid_urls

    def __len__(self) -> int:
        return len(self.items)
