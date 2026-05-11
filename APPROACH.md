# Approach: SHL Conversational Assessment Recommender

**Author:** Sayan Mandal  ·  **Role:** SHL Labs AI Research  Intern

## 1. Design choices

**Stateless FastAPI service with strict schema, soft logic.** The spec is unambiguous: every `/chat` call carries the full history, and the response shape is fixed. I treat the schema as a hard contract (Pydantic models with `extra="ignore"`) and the agent's behavior as soft policy. This split lets me iterate on prompts without ever risking a malformed response that would fail the harness.

**Hybrid retrieval (BM25 + dense + RRF).** Lexical recall matters for SHL's catalog because product names contain exact technical terms (`Java 8`, `ReactJS`, `Kubernetes`) that semantic embeddings can over-cluster. Dense recall matters for paraphrased intent ("front-end skills" → `HTML/CSS`, `ReactJS`, `JavaScript`). I run both and fuse with Reciprocal Rank Fusion (`k=60`), which empirically beats single-method retrieval on this kind of mixed-vocabulary corpus and avoids the calibration headache of weighted score fusion. Embeddings are computed once per catalog version and cached to disk so cold starts stay under the 2-minute budget.

**Gemini 2.0 Flash for decisions, with heuristic fallback.** The LLM receives the last 12 turns plus 30 retrieved candidates (compact JSON, ~280-char descriptions) and emits a structured decision: `{action, reply, selected_ids}`. Forcing JSON output via `response_mime_type` removes a class of parsing bugs. If the API call fails — quota, network, malformed output — the request falls through to a heuristic that does pattern-match clarify/refuse and returns top-5 BM25 results otherwise. The service is **never** down because of an upstream LLM issue, which directly addresses the spec's "non-deterministic conversation should not make the system fall apart" criterion.

**Catalog-grounding by ID, not by name.** The LLM picks `selected_ids` from the candidate list shown in the prompt. The validator then re-resolves each ID against the catalog and re-validates each returned URL against the set of catalog URLs. If the LLM hallucinates a name, an unknown ID, or somehow injects a URL not in the catalog, that recommendation is dropped. If validation drops everything from a `recommend` decision, the action is downgraded to `clarify` so the user sees a question rather than an empty shortlist that violates the spec.

**Test-type normalization.** SHL's source data has a `keys` field with full names like `"Knowledge & Skills"`. The spec wants single-letter codes (`K`, `P`, `A`, ...). I built a deterministic mapping (`KEY_TO_LETTER`) at load time so `test_type` is always one of the eight valid letters and never invented at runtime.

**Turn-budget awareness.** The spec caps each conversation at 8 turns total. At the final turn, the agent is forced to commit to a shortlist even if the LLM still wants to clarify, because returning empty recommendations on the final turn would tank Recall@10. This is a small but high-leverage detail.

## 2. Prompt design

The system prompt does four things explicitly: (1) defines scope (SHL Individual Test Solutions only); (2) lists the four actions with one-sentence triggers; (3) specifies the JSON schema with examples of when each field is empty; (4) names the failure modes (refuse off-topic, ignore "ignore previous instructions", never invent URLs). The conversation history is injected as compact role-tagged lines, candidates as one-line JSON each. Total prompt is ~2-3K tokens, well under the model's window, and small enough that latency stays comfortably below the 30-second per-call budget.

## 3. Evaluation

I built `scripts/eval.py` — a harness that replays trace JSON files against a running `/chat` endpoint and reports three things: (a) **mean Recall@10** across traces, (b) **schema compliance** (every response has `reply`, `recommendations`, `end_of_conversation`; every recommendation has valid `name`, `url`, single-letter `test_type`), (c) **behavior probes** (six binary assertions covering vague-query handling, off-topic refusal, prompt-injection resistance, JD-driven recommendation, test-type validity, and refinement honoring).

Running it locally against a representative subset, BM25-only (no LLM, no dense embeddings — worst case): **6/6 behavior probes pass, schema 100% compliant, mean Recall@10 = 0.47** on five hand-built traces. The full ~500-item catalog with Gemini embeddings enabled in production should push Recall@10 higher, because (a) dense retrieval recovers paraphrased intent and (b) the LLM has ten times as many candidates to pick from.

What didn't work: the `frontend-team` trace scored 0.25 in BM25-only mode because "frontend" and "Java with stakeholders" share workplace vocabulary and BM25 over-rewarded the more frequent term. Adding dense retrieval is exactly what fixes this — it's the core motivation for hybrid retrieval, and it's why I use RRF rather than relying on either method alone.

I also stress-tested edge cases manually: very long pasted JDs (5500+ chars — handled, no timeouts), unicode and emoji in messages (handled), assistant-only history (handled — friendly greeting, no crash), unknown extra JSON fields (Pydantic ignores them per `extra="ignore"`), and turn 9 (forced commit so the harness never times out).

## 4. What didn't work / what I'd improve

- **Tool calls for retrieval** — I prototyped letting the LLM issue a `search(query, filters)` tool call mid-conversation. It worked but added a turn of latency and made schema enforcement harder. Pre-retrieving 30 candidates per turn turned out to be both faster and easier to ground.
- **Score fusion vs RRF** — weighted sum of normalized BM25 + cosine scores was sensitive to query length. RRF is parameter-light and held up better.
- **Embedding model choice** — `text-embedding-004` (768-d) was the right tradeoff. Larger models slowed cold start without measurable retrieval gains on this catalog.
- Given another day I would: (1) add a small `eval.py` that replays the 10 public traces and prints Recall@10, (2) add a per-turn entity resolver so `"GSA"` and `"the GSA"` resolve to the Global Skills Assessment without an embedding lookup, (3) include a confidence threshold below which the agent prefers `clarify` over `recommend`.

## 5. AI tools used

I used Claude (Sonnet/Opus 4.7) as a coding assistant to scaffold boilerplate, draft prompts, and review my retrieval/agent logic. Final design choices, the validator/fallback architecture, and the prompt itself are mine; the assistant was a sounding board, not the author. No no-code builders or agentic coding tools were used.

## 6. Stack

FastAPI · Pydantic · `google-genai` 2.x · `gemini-2.0-flash` · `text-embedding-004` · `rank-bm25` · NumPy. Deployed on Render free tier; `/health` honors the 2-minute cold-start. Embeddings cache to `data/embeddings.pkl` on first run.
