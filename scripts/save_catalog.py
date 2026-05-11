"""
Extract the SHL catalog JSON from a text source (e.g. the user's pasted dump
or a downloaded file) and save it to data/catalog.json.

Usage:
  python scripts/save_catalog.py path/to/catalog.json

If the source already parses as a list of dicts, we just normalize and write it.
"""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("usage: python scripts/save_catalog.py <source.json>")
        sys.exit(1)
    src = Path(sys.argv[1])
    if not src.exists():
        print(f"source not found: {src}")
        sys.exit(1)

    raw = src.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        print(f"expected JSON array, got {type(data).__name__}")
        sys.exit(1)

    # Light validation
    cleaned = []
    for r in data:
        if not isinstance(r, dict):
            continue
        if not r.get("name") or not r.get("link"):
            continue
        cleaned.append(r)

    out_path = Path(__file__).parent.parent / "data" / "catalog.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(cleaned)} items to {out_path}")


if __name__ == "__main__":
    main()
