from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import os, json, requests, logging

app = FastAPI(title="Property Matcher API", version="1.3.0")

# ---------- Env ----------
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
LISTINGS_AGENT_ID = os.getenv("LISTINGS_AGENT_ID", "")  # agt_* or wf_*
PUBLIC_CALLER_KEY = os.getenv("PUBLIC_CALLER_KEY", "")
OPENAI_BASE       = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")

# Optional (recommended if you see 404 Invalid URL from OpenAI)
OPENAI_PROJECT    = os.getenv("OPENAI_PROJECT", "")      # e.g. proj_abc123
OPENAI_ORG        = os.getenv("OPENAI_ORG", "")          # e.g. org_abc123

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("propertymatcher")

# ---------- Models ----------
class Criteria(BaseModel):
    location: Optional[str] = None
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    beds_min: Optional[int] = None
    baths_min: Optional[int] = None
    property_types: Optional[List[str]] = []
    keywords: Optional[str] = None

class AgentRequest(BaseModel):
    conversation_id: Optional[str] = "listing-run"
    limit: Optional[int] = 8
    criteria: Criteria

# ---------- Helpers ----------
def runs_url_for(identifier: str) -> str:
    """Route to the correct endpoint based on id prefix."""
    if identifier.startswith("wf_"):
        return f"{OPENAI_BASE}/workflows/{identifier}/runs"
    if identifier.startswith("agt_"):
        return f"{OPENAI_BASE}/agents/{identifier}/runs"
    # default: treat as agent id
    return f"{OPENAI_BASE}/agents/{identifier}/runs"

def require_env():
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
    if not LISTINGS_AGENT_ID:
        raise HTTPException(status_code=500, detail="LISTINGS_AGENT_ID is not set")

def build_openai_headers() -> dict:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        # Include both betas (workflows + agents)
        "OpenAI-Beta": "agents=v1,workflows=v1",
    }
    if OPENAI_PROJECT:
        headers["OpenAI-Project"] = OPENAI_PROJECT
    if OPENAI_ORG:
        headers["OpenAI-Organization"] = OPENAI_ORG
    return headers

# ---------- Routes ----------
@app.get("/")
def health():
    return {"status": "ok", "service": "propertymatcher", "docs": "/docs"}

@app.get("/debug/config")
def debug_config():
    url = runs_url_for(LISTINGS_AGENT_ID) if LISTINGS_AGENT_ID else None
    return {
        "api_version": "1.3.0",
        "id_prefix": LISTINGS_AGENT_ID[:3] if LISTINGS_AGENT_ID else None,
        "openai_runs_url": url,
        "has_openai_key": bool(OPENAI_API_KEY),
        "has_project_header": bool(OPENAI_PROJECT),
        "has_org_header": bool(OPENAI_ORG),
    }

@app.post("/agent/listings")
def run_listings_agent(payload: AgentRequest, request: Request):
    # Optional shared-secret to restrict callers (Bubble/Make send x-api-key)
    if PUBLIC_CALLER_KEY:
        provided = request.headers.get("x-api-key")
        if not provided or provided != PUBLIC_CALLER_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing x-api-key")

    require_env()

    # High-level instruction text; your OpenAI object will do the heavy lifting
    query_text = (
        f"Find up to {payload.limit} Québec real-estate listings based on: "
        f"{json.dumps(payload.criteria.dict(), ensure_ascii=False)}"
    )

    url = runs_url_for(LISTINGS_AGENT_ID)
    headers = build_openai_headers()
    body = {
        "input_as_text": query_text,
        "input_variables": payload.criteria.dict(),
    }

    log.info(f"Calling OpenAI runs endpoint: {url}")

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=120)
        if not resp.ok:
            # Bubble up the exact OpenAI error for visibility (helps debug 404/401/etc.)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent request failed: {e}")

    return JSONResponse({
        "conversation_id": payload.conversation_id,
        "criteria": payload.criteria.dict(),
        "agent_output": data
    })
