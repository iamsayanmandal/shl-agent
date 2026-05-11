"""
Scrape SHL's Individual Test Solutions catalog into data/catalog.json.

The assignment specifies: "The catalog you build over is the SHL product catalog
at https://www.shl.com/solutions/products/product-catalog/, restricted to
Individual Test Solutions only."

How SHL's catalog is structured (verified against the live site):
- Listing URL: /solutions/products/product-catalog/?start=N&type=2
  - type=1 = Pre-packaged Job Solutions (out of scope)
  - type=2 = Individual Test Solutions (in scope)
  - 12 items per page; iterate start=0,12,24,...
- Each row in the listing has:
    a.product-catalogue__title  -> name + href to detail page
    span.product-catalogue__keys -> test_type icons (with title attrs)
    cells for "Remote Testing" and "Adaptive/IRT"
- Detail page has structured rows for "Description", "Job levels",
  "Languages", "Assessment length", and the same Test Type icons.

Run:
    python scripts/scrape_catalog.py             # writes data/catalog.json
    python scripts/scrape_catalog.py --max 50    # limit for testing
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup


BASE = "https://www.shl.com"
LISTING = "https://www.shl.com/solutions/products/product-catalog/"
PAGE_SIZE = 12

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Test type icon -> full name mapping. SHL uses the title attribute.
# These names match what we already normalize in app/catalog.py KEY_TO_LETTER.
ICON_TITLE_TO_KEY = {
    "Ability & Aptitude": "Ability & Aptitude",
    "Biodata & Situational Judgement": "Biodata & Situational Judgment",
    "Biodata & Situational Judgment": "Biodata & Situational Judgment",
    "Competencies": "Competencies",
    "Development & 360": "Development & 360",
    "Assessment Exercises": "Assessment Exercises",
    "Knowledge & Skills": "Knowledge & Skills",
    "Personality & Behaviour": "Personality & Behavior",
    "Personality & Behavior": "Personality & Behavior",
    "Simulations": "Simulations",
}


def make_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"},
        timeout=30.0,
        follow_redirects=True,
    )


def fetch(client: httpx.Client, url: str, retries: int = 2) -> Optional[str]:
    last = None
    for attempt in range(retries + 1):
        try:
            r = client.get(url)
            if r.status_code == 200:
                return r.text
            last = f"HTTP {r.status_code}"
        except Exception as e:
            last = str(e)
        time.sleep(0.5 * (attempt + 1))
    print(f"[scrape] giving up on {url}: {last}")
    return None


def parse_listing(html: str) -> List[Dict[str, str]]:
    """Return list of {name, link} from an Individual Test Solutions page."""
    soup = BeautifulSoup(html, "html.parser")
    out: List[Dict[str, str]] = []
    # Modern SHL structure: rows with anchors that include /view/<slug>/
    for a in soup.select("a[href*='/products/product-catalog/view/']"):
        name = a.get_text(strip=True)
        href = a.get("href", "").strip()
        if not name or not href:
            continue
        full = urljoin(BASE, href)
        out.append({"name": name, "link": full})
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for it in out:
        if it["link"] in seen:
            continue
        seen.add(it["link"])
        deduped.append(it)
    return deduped


def parse_detail(html: str, name: str, link: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")

    def section_text(label: str) -> str:
        # SHL uses <h4>Label</h4><p>...</p> blocks
        for tag in soup.find_all(["h4", "h3"]):
            t = tag.get_text(strip=True)
            if t.lower().startswith(label.lower()):
                sib = tag.find_next_sibling()
                if sib:
                    return sib.get_text(" ", strip=True)
        return ""

    description = section_text("Description")
    job_levels_raw = section_text("Job levels")
    languages_raw = section_text("Languages")
    duration_raw = section_text("Assessment length")

    # Test type icons — look for <span class="product-catalogue__key" title="...">
    keys: List[str] = []
    for span in soup.select("span.product-catalogue__key, span[title][class*='key']"):
        title = (span.get("title") or "").strip()
        norm = ICON_TITLE_TO_KEY.get(title)
        if norm and norm not in keys:
            keys.append(norm)

    page_text = soup.get_text(" ", strip=True).lower()
    remote = "yes" if re.search(r"remote\s*testing[:\s]*yes", page_text) else "no"
    adaptive = "yes" if re.search(r"adaptive\s*/?\s*irt[:\s]*yes", page_text) else "no"

    slug = urlparse(link).path.rstrip("/").split("/")[-1]
    # Try to match SHL's numeric entity_id if present in the page (e.g. data-id, hidden input)
    entity_id = slug
    m = soup.find(attrs={"data-entity-id": True})
    if m:
        entity_id = str(m["data-entity-id"]).strip()

    job_levels = [s.strip() for s in re.split(r",", job_levels_raw) if s.strip()]
    languages = [s.strip() for s in re.split(r",", languages_raw) if s.strip()]

    duration_match = re.search(r"(\d+)", duration_raw or "")
    duration = duration_match.group(0) + " minutes" if duration_match else (duration_raw or "")

    return {
        "entity_id": entity_id,
        "name": name,
        "link": link,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "job_levels": job_levels,
        "job_levels_raw": job_levels_raw,
        "languages": languages,
        "languages_raw": languages_raw,
        "duration": duration,
        "duration_raw": duration_raw,
        "status": "ok",
        "remote": remote,
        "adaptive": adaptive,
        "description": description,
        "keys": keys,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=0, help="cap items (0=all)")
    p.add_argument("--out", default="data/catalog.json")
    p.add_argument("--sleep", type=float, default=0.25)
    args = p.parse_args()

    items: Dict[str, Dict[str, str]] = {}
    with make_client() as client:
        start = 0
        empty_pages = 0
        while True:
            url = f"{LISTING}?start={start}&type=2"
            print(f"[scrape] listing start={start}")
            html = fetch(client, url)
            if not html:
                empty_pages += 1
                if empty_pages >= 2:
                    break
                start += PAGE_SIZE
                continue
            page_items = parse_listing(html)
            if not page_items:
                break
            new = 0
            for it in page_items:
                if it["link"] not in items:
                    items[it["link"]] = it
                    new += 1
            if new == 0:
                break
            if args.max and len(items) >= args.max:
                break
            start += PAGE_SIZE
            time.sleep(args.sleep)

        print(f"[scrape] found {len(items)} unique listing entries; fetching details...")
        details: List[Dict[str, Any]] = []
        for i, (link, brief) in enumerate(items.items()):
            if args.max and i >= args.max:
                break
            html = fetch(client, link)
            if not html:
                continue
            try:
                details.append(parse_detail(html, brief["name"], link))
            except Exception as e:
                print(f"[scrape] parse failed for {link}: {e}")
            if (i + 1) % 25 == 0:
                print(f"[scrape] {i + 1}/{len(items)} details done")
            time.sleep(args.sleep)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[scrape] wrote {len(details)} items to {out_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
