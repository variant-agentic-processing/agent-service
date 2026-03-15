"""Agent service API — entry point."""

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv  # noqa: E402

load_dotenv()  # no-op in Cloud Run where env vars are injected directly  # noqa: E402

import anthropic  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src import agent, mcp_client  # noqa: E402
from src.secrets import get_secret  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-6"


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = get_secret("anthropic-api-key", "ANTHROPIC_API_KEY")
    model = os.environ.get("CLAUDE_MODEL", _DEFAULT_MODEL)

    app.state.anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
    app.state.model = model

    logger.info("Discovering MCP tools from %s", os.environ.get("MCP_SERVER_URL"))
    tools = await mcp_client.discover_tools()
    app.state.tools_schemas = mcp_client.tools_to_anthropic(tools)
    logger.info("Agent ready — model=%s tools=%d", model, len(tools))

    yield


app = FastAPI(title="agent-service", version="0.1.0", lifespan=lifespan)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    messages: list[Message]
    context: str | None = None


async def _stream_events(messages: list[dict], context: str | None, tools_schemas: list, anthropic_client, model: str) -> AsyncIterator[str]:
    try:
        async for event in agent.run(messages, tools_schemas, anthropic_client, model, context):
            yield f"data: {json.dumps(event)}\n\n"
    except Exception as exc:
        logger.exception("Agent run failed")
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
    finally:
        yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/query")
async def query(req: QueryRequest):
    if not req.messages or not any(m.role == "user" for m in req.messages):
        raise HTTPException(status_code=400, detail="messages must contain at least one user message")

    messages = [m.model_dump() for m in req.messages]

    return StreamingResponse(
        _stream_events(
            messages,
            req.context,
            app.state.tools_schemas,
            app.state.anthropic_client,
            app.state.model,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
def health():
    tools_count = len(getattr(app.state, "tools_schemas", []))
    return JSONResponse({"status": "ok", "name": agent.AGENT_NAME, "tools": tools_count})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
