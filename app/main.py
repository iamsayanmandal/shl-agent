"""
main.py - FastAPI entry point for the SHL Assessment Recommender

I built this as a stateless REST service because the assignment spec
explicitly says each /chat call carries the full conversation history.
No session storage needed, which also makes deployment on Render much simpler.

Endpoints:
  GET  /health  -> {"status": "ok"}  (used by Render health checks)
  POST /chat    -> takes messages[], returns reply + recommendations + end_of_conversation
  GET  /info    -> diagnostic info: catalog size, LLM status, retrieval mode
  GET  /        -> serves the frontend chat UI
"""
from __future__ import annotations

import os
import time
from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ConfigDict

from .catalog import Catalog
from .retriever import HybridRetriever, load_or_build_embeddings
from .agent import Agent
from . import llm


# ------------------------------------------------------------------ #
# Request / Response models
# I'm using Pydantic with extra="ignore" so any extra fields the
# evaluation harness sends don't cause validation errors.
# ------------------------------------------------------------------ #

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    messages: List[ChatMessage] = Field(default_factory=list)


class Recommendation(BaseModel):
    # These three fields match the spec exactly
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


# ------------------------------------------------------------------ #
# Startup: load catalog and build/load embeddings
# I'm using FastAPI's lifespan context instead of @app.on_event because
# on_event is deprecated in newer FastAPI versions.
# ------------------------------------------------------------------ #

CATALOG_PATH = os.environ.get("SHL_CATALOG_PATH", "data/catalog.json")

# Simple dict to hold shared state across requests
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] loading catalog from {CATALOG_PATH}")
    catalog = Catalog.load(CATALOG_PATH)
    print(f"[startup] loaded {len(catalog)} catalog items")

    # Try to build embeddings using Gemini. If the key isn't set or
    # the API call fails, we fall back to BM25-only retrieval.
    # The service works either way, just with lower semantic recall.
    embed_fn = llm.embed_texts if llm.is_available() else None
    embeddings = load_or_build_embeddings(catalog, embed_fn)

    if embeddings is not None:
        print(f"[startup] embeddings ready: {embeddings.shape}")
    else:
        print("[startup] embeddings unavailable, BM25-only retrieval")

    retriever = HybridRetriever(catalog, embeddings=embeddings)
    _state["agent"] = Agent(catalog, retriever)
    _state["catalog_size"] = len(catalog)
    _state["llm_available"] = llm.is_available()
    _state["dense_retrieval"] = embeddings is not None
    yield


app = FastAPI(
    title="SHL Conversational Assessment Recommender",
    version="1.0.0",
    lifespan=lifespan,
)

# Allowing all origins so the frontend can talk to the API
# even when hosted on a different subdomain on Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@app.get("/health")
def health():
    # Render uses this to check if the service is alive
    return {"status": "ok"}


@app.get("/info")
def info():
    # Useful for debugging after deployment - shows catalog size,
    # whether Gemini is connected, and whether dense retrieval is active
    return {
        "status": "ok",
        "catalog_size": _state.get("catalog_size", 0),
        "llm_available": _state.get("llm_available", False),
        "dense_retrieval": _state.get("dense_retrieval", False),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    agent: Optional[Agent] = _state.get("agent")
    if agent is None:
        raise HTTPException(status_code=503, detail="agent not initialized")

    msgs = [m.model_dump() for m in req.messages]
    t0 = time.time()

    try:
        result = agent.respond(msgs)
    except Exception as e:
        # Never let an unhandled exception crash the API response.
        # Return a graceful fallback instead.
        print(f"[chat] error: {e}")
        return ChatResponse(
            reply="Something went wrong on my end. Could you rephrase what you're looking for?",
            recommendations=[],
            end_of_conversation=False,
        )

    elapsed = time.time() - t0
    action = "recommend" if result["recommendations"] else "no-recs"
    print(f"[chat] turns_in={len(msgs)} action={action} t={elapsed:.2f}s")

    return ChatResponse(**result)


# ------------------------------------------------------------------ #
# Serve the frontend if it exists
# I kept the frontend as a simple HTML file to avoid needing a separate
# build step. The API works fine without it (curl / evaluation harness).
# ------------------------------------------------------------------ #

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists() and (FRONTEND_DIR / "index.html").exists():
    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
else:
    @app.get("/")
    def index_fallback():
        return JSONResponse({
            "service": "SHL Conversational Assessment Recommender",
            "endpoints": ["/health", "/chat (POST)", "/info"],
        })
