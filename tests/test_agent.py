"""Unit tests for the agent reasoning loop."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mcp_client import tools_to_anthropic, truncate_result


# ---------------------------------------------------------------------------
# mcp_client helpers
# ---------------------------------------------------------------------------

def test_truncate_result_short():
    assert truncate_result("hello") == "hello"


def test_truncate_result_long():
    long = "x" * 9000
    result = truncate_result(long)
    assert len(result) <= 8100
    assert "[TRUNCATED" in result


def test_tools_to_anthropic():
    tool = MagicMock()
    tool.name = "search_variants"
    tool.description = "Search variants"
    tool.inputSchema = {"type": "object", "properties": {}}

    converted = tools_to_anthropic([tool])
    assert len(converted) == 1
    assert converted[0]["name"] == "search_variants"
    assert converted[0]["input_schema"] == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# agent run loop
# ---------------------------------------------------------------------------

def _make_text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(name: str, args: dict, tool_id: str = "tu_1"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = args
    block.id = tool_id
    return block


def _make_response(content):
    response = MagicMock()
    response.content = content
    return response


def _make_tool_result(text: str, is_error: bool = False):
    content_item = MagicMock()
    content_item.text = text
    result = MagicMock()
    result.content = [content_item]
    result.isError = is_error
    return result


@pytest.mark.asyncio
async def test_agent_no_tool_calls():
    """Agent returns answer immediately when Claude makes no tool calls."""
    from src import agent

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(
        return_value=_make_response([_make_text_block("HG002 has 5 million variants.")])
    )

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("src.agent.mcp_session", return_value=mock_session):
        events = []
        async for event in agent.run("How many variants?", [], mock_client, "claude-sonnet-4-6"):
            events.append(event)

    answer_events = [e for e in events if e["type"] == "answer"]
    assert len(answer_events) == 1
    assert "HG002" in answer_events[0]["text"]


@pytest.mark.asyncio
async def test_agent_one_tool_call():
    """Agent makes one tool call then returns a final answer."""
    from src import agent

    tool_response = _make_response([_make_text_block("Final answer after tool.")])
    tool_response.content = [_make_text_block("Final answer after tool.")]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(
        side_effect=[
            _make_response([_make_tool_use_block("get_individual_summary", {"individual_id": "HG002"})]),
            _make_response([_make_text_block("Final answer after tool.")]),
        ]
    )

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.call_tool = AsyncMock(return_value=_make_tool_result('{"total": 5000000}'))

    with patch("src.agent.mcp_session", return_value=mock_session):
        events = []
        async for event in agent.run("Summarise HG002", [], mock_client, "claude-sonnet-4-6"):
            events.append(event)

    types = [e["type"] for e in events]
    assert "tool_call" in types
    assert "tool_result" in types
    assert "answer" in types

    tool_call = next(e for e in events if e["type"] == "tool_call")
    assert tool_call["tool"] == "get_individual_summary"


@pytest.mark.asyncio
async def test_agent_tool_error_continues():
    """Agent continues reasoning when a tool call fails."""
    from src import agent

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(
        side_effect=[
            _make_response([_make_tool_use_block("search_variants", {"individual_id": "HG002", "gene_symbol": "BRCA1"})]),
            _make_response([_make_text_block("Could not retrieve results.")]),
        ]
    )

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.call_tool = AsyncMock(return_value=_make_tool_result("error", is_error=True))

    with patch("src.agent.mcp_session", return_value=mock_session):
        events = []
        async for event in agent.run("BRCA1 variants in HG002", [], mock_client, "claude-sonnet-4-6"):
            events.append(event)

    result_events = [e for e in events if e["type"] == "tool_result"]
    assert result_events[0]["is_error"] is True
    assert any(e["type"] == "answer" for e in events)
