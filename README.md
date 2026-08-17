# 🤖 SHL Conversational Assessment Recommender

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi) ![Gemini](https://img.shields.io/badge/Gemini_2.0-8E75B2?style=for-the-badge&logo=googlebard&logoColor=white)

A FastAPI agent that takes a recruiter from a vague intent ("I'm hiring a Python or JAVA developer") to a grounded shortlist of SHL Individual Test Solutions through multi-turn dialogue.

## Endpoints

- `GET  /health` → `{"status": "ok"}`
- `POST /chat`   → stateless. Request body: `{"messages":[{"role":"user","content":"..."}]}`. Returns `{"reply", "recommendations":[{name,url,test_type}], "end_of_conversation"}`.
- `GET  /info`   → diagnostic: catalog size, whether LLM and dense retrieval are enabled.
- `GET  /`       → optional chat UI (frontend/index.html)

The schema of `/chat` exactly matches the assignment spec. `recommendations` is empty when clarifying or refusing, 1–10 items when committing to a shortlist.

## Conversational behaviors

The agent decides between four actions on every turn:

| action      | when                                                                 | recs returned |
|-------------|----------------------------------------------------------------------|---------------|
| `clarify`   | query too vague, e.g. "I need an assessment"                         | `[]`          |
| `recommend` | enough context (role, skills, seniority, or pasted JD)               | 1–10          |
| `compare`   | "What is the difference between OPQ and GSA?"                        | `[]`          |
| `refuse`    | off-topic, prompt-injection, legal/salary/general hiring advice      | `[]`          |

Refinements ("actually, add personality tests") update the shortlist instead of restarting.

## Architecture

1. **Catalog** (`app/catalog.py`) — loads `data/catalog.json`, normalizes SHL `keys` into the single-letter `test_type` codes the spec expects (`A/B/C/D/E/K/P/S`).
2. **Hybrid retrieval** (`app/retriever.py`) — BM25 over name+description+job-levels+keys, optional Gemini `text-embedding-004` dense retrieval, fused with Reciprocal Rank Fusion. Embeddings are cached to `data/embeddings.pkl`.
3. **LLM decision** (`app/agent.py`) — Gemini 2.0 Flash receives the conversation history plus 30 retrieved candidates, returns a structured JSON decision (`action`, `reply`, `selected_ids`).
4. **Strict validation** (`app/agent.py::validate_decision`) — every returned URL is re-validated against the catalog. Invalid IDs are dropped. If a "recommend" decision yields nothing valid, we fall back to "clarify".
5. **Heuristic fallback** — if the LLM call fails or no API key is set, the agent falls back to BM25 retrieval + heuristic action selection. The service never returns a malformed response.
6. **Turn cap** — at turn 8 (per spec) the agent is forced to commit to a shortlist regardless of the LLM's preference, so the harness never times out.

## Local run

```bash
git clone <repo>
cd shl-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit and add your GEMINI_API_KEY
export GEMINI_API_KEY="your_key"
uvicorn app.main:app --reload
```

Open https://shl-agent-1x9d.onrender.com/ for the chat UI.

```bash
# health
curl https://shl-agent-1x9d.onrender.com//health

# chat (the spec's example)
curl -X POST https://shl-agent-1x9d.onrender.com//chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[
    {"role":"user","content":"Hiring a Java developer who works with stakeholders"},
    {"role":"assistant","content":"Sure. What is seniority level?"},
    {"role":"user","content":"Mid-level, around 4 years"}
  ]}'
```

## Deploy to Render (free tier)

1. Push this repo to GitHub.
2. On Render: **New → Blueprint** and point at the repo. The `render.yaml` is pre-configured.
3. Add `GEMINI_API_KEY` as a secret env var.
4. First deploy takes ~3 min. The service cold-starts on the free tier; `/health` allows up to 2 minutes per the spec.
5. The first `/chat` call after a fresh deploy builds the embeddings (~30s for ~500 catalog items) and caches them on disk.

## Catalog

`data/catalog.json` is the SHL Individual Test Solutions catalog. The included file is a representative subset to keep the repo small and the smoke tests fast. To use the full ~500-item catalog:

```bash
# Option A: drop your own JSON file
cp /path/to/your/full_catalog.json data/catalog.json

# Option B: scrape it fresh
python scripts/scrape_catalog.py
```

The scraper restricts to the "Individual Test Solutions" table per the assignment.

## Failure modes guarded against

The assignment lists three common failure patterns. Here is how this implementation defends:

- **Weak programming foundations (happy-path-only).** Every external call (Gemini embed, generate, network) is wrapped in `try/except` with a graceful fallback. URL validation re-checks every recommended URL against the catalog. The `/chat` handler catches and returns a safe response on any unhandled error.
- - **Using components without understanding the reasoning behind them.** Hybrid retrieval (BM25 + dense + RRF) is justified in the approach doc. The 8-turn cap, heuristic fallback, and strict validation are all explicit design choices, not accidents.
- **Insufficient evaluation rigor.** The agent is tested against vague-query, prompt-injection, off-topic, and refinement scenarios; see `scripts/eval.py` (if included) and the example traces in `data/`.
