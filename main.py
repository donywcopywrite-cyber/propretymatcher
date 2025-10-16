from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import os, requests, json
from typing import Optional

app = FastAPI(title="Property Matcher API", version="1.0.0")

# --- Environment Variables ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
PUBLIC_CALLER_KEY = os.getenv("PUBLIC_CALLER_KEY")
USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (PropertyMatcher)")

# --- Data models for validation ---
class Criteria(BaseModel):
    location: Optional[str] = None
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    beds_min: Optional[int] = None
    baths_min: Optional[int] = None
    property_types: Optional[list[str]] = []
    keywords: Optional[str] = None

class AgentRequest(BaseModel):
    conversation_id: Optional[str] = "listing-run"
    limit: Optional[int] = 8
    criteria: Criteria

# --- Simple health check ---
@app.get("/")
async def root():
    return {"status": "ok", "service": "propertymatcher", "docs": "/docs"}

# --- Core listings endpoint ---
@app.post("/agent/listings")
async def get_listings(req: AgentRequest, request: Request):
    # Optional API key validation
    if PUBLIC_CALLER_KEY:
        auth_key = request.headers.get("x-api-key")
        if not auth_key or auth_key != PUBLIC_CALLER_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    # Build query text for your OpenAI Agent
    query = (
        f"Find up to {req.limit} real estate listings based on: "
        f"{json.dumps(req.criteria.dict(), ensure_ascii=False)}"
    )

    # Call your OpenAI Agent (replace wf_... with your agent ID)
    openai_agent_id = os.getenv("LISTINGS_AGENT_ID", "wf_your_agent_id_here")
    url = f"https://api.openai.com/v1/agents/{openai_agent_id}/runs"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "input_as_text": query,
        "input_variables": req.criteria.dict(),
    }

    try:
        response = requests.post(url, headers=headers, json=body, timeout=120)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent request failed: {str(e)}")

    return {
        "conversation_id": req.conversation_id,
        "criteria": req.criteria.dict(),
        "agent_output": data,
    }
