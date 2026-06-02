from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.agent.graph import run_agent

ROOT_DIR = Path(__file__).resolve().parent
INDEX_FILE = ROOT_DIR / "index.html"

app = FastAPI(title="OrderDesk Lab Test API")


class RunRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query to test the order agent.")
    provider: str = Field(default="openai", description="Provider to pass to run_agent.")
    model_name: str | None = Field(default=None, description="Optional model override.")
    today: str | None = Field(default="2026-06-01", description="Optional deterministic date.")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/run")
def run(payload: RunRequest) -> dict:
    result = run_agent(
        payload.query,
        provider=payload.provider,
        model_name=payload.model_name,
        today=payload.today,
    )
    return result.model_dump()
