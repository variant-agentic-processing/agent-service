"""MCP Streamable HTTP client with OIDC auth for Cloud Run → Cloud Run calls."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool

logger = logging.getLogger(__name__)

MAX_RESULT_CHARS = 8000


class _BearerAuth(httpx.Auth):
    """httpx Auth that re-applies a Bearer token on every request, including after redirects."""

    def __init__(self, token: str) -> None:
        self._token = token

    def auth_flow(self, request: httpx.Request):
        request.headers["Authorization"] = f"Bearer {self._token}"
        yield request


def _identity_token(audience: str) -> str | None:
    """Fetch an OIDC identity token for the given audience.

    Works on Cloud Run (GCE metadata server) and locally via ADC.
    Returns None if unavailable so callers can proceed without auth.
    """
    token_override = os.environ.get("MCP_IDENTITY_TOKEN")
    if token_override:
        return token_override

    try:
        import google.auth.transport.requests
        import google.oauth2.id_token

        auth_request = google.auth.transport.requests.Request()
        return google.oauth2.id_token.fetch_id_token(auth_request, audience)
    except Exception as exc:
        logger.warning("Could not fetch OIDC identity token: %s", exc)
        return None


@asynccontextmanager
async def mcp_session() -> AsyncIterator[ClientSession]:
    """Open an authenticated MCP session for the duration of the block."""
    base_url = os.environ.get("MCP_SERVER_URL", "").rstrip("/")
    url = f"{base_url}/mcp/"

    token = _identity_token(base_url)
    auth = _BearerAuth(token) if token else None

    async with streamablehttp_client(url, auth=auth) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def discover_tools() -> list[Tool]:
    """Open a one-shot session to list available tools."""
    async with mcp_session() as session:
        result = await session.list_tools()
        logger.info("Discovered %d MCP tools", len(result.tools))
        return result.tools


def tools_to_anthropic(tools: list[Tool]) -> list[dict]:
    """Convert MCP Tool objects to Anthropic API tool format."""
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema,
        }
        for t in tools
    ]


def truncate_result(text: str) -> str:
    if len(text) > MAX_RESULT_CHARS:
        return text[:MAX_RESULT_CHARS] + "\n[TRUNCATED — result exceeded 8000 chars]"
    return text
