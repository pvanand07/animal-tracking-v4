"""LangChain agent with OpenRouter and tracker tools (schema + read-only SQL)."""

import json
import logging
from typing import Optional, AsyncIterator

import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from config import config

logger = logging.getLogger(__name__)

# ── Tools (call tracker API) ────────────────────────────────────


@tool
def get_tracker_schema() -> str:
    """
    Fetch the database schema and API documentation for the animal tracker.
    Call this first when you need to know table names, columns, or available endpoints
    before writing SQL or answering questions about the data.
    """
    try:
        r = httpx.get(f"{config.api_base_url}/api/agent/schema", timeout=30.0)
        r.raise_for_status()
        return r.text
    except httpx.HTTPError as e:
        return f"Error fetching schema: {e!s}"


@tool
def run_tracker_sql(query: str) -> str:
    """
    Run a read-only SQL query (SELECT only) against the animal tracker database.
    Use get_tracker_schema first to see tables and columns. Only SELECT is allowed.
    """
    try:
        r = httpx.post(
            f"{config.api_base_url}/api/agent/query",
            json={"query": query.strip()},
            timeout=30.0,
        )
        data = r.json()
        if not data.get("ok"):
            return json.dumps({"ok": False, "error": data.get("error", "Unknown error")})
        return json.dumps({"ok": True, "rows": data.get("rows", [])})
    except httpx.HTTPError as e:
        return json.dumps({"ok": False, "error": str(e)})
    except json.JSONDecodeError as e:
        return json.dumps({"ok": False, "error": f"Invalid JSON: {e}"})


def get_tools():
    return [get_tracker_schema, run_tracker_sql]


# ── Model & agent ───────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant for the Animal Tracker system. You have access to:

1. get_tracker_schema – returns the database schema (tables, columns, indexes). Use this first when answering questions about detections, events, or animals.
2. run_tracker_sql – runs a read-only SELECT query against the tracker database.

When users ask about data (detections, events, animals, counts, etc.):
1. Call get_tracker_schema to see the schema.
2. Use run_tracker_sql with valid SELECT queries to get the data.
3. Summarize results clearly. If a query fails, suggest a correction or ask for clarification.

When you mention a specific detection by its tracking_id, include it in the format <tracking_id>id</tracking_id> so the UI can show a "View detection" button that opens the detection modal and allows video playback. Example: "The detection <tracking_id>abc-123-def</tracking_id> is a lion." Use the exact tracking_id value from the database (e.g. from detections or events).

Only SELECT queries are allowed; do not attempt INSERT/UPDATE/DELETE. Be concise and accurate."""


def create_model(model_id: Optional[str] = None) -> ChatOpenAI:
    """Create ChatOpenAI wired to OpenRouter."""
    return ChatOpenAI(
        model=model_id or config.llm_model,
        openai_api_key=config.openrouter_api_key,
        base_url=config.openrouter_base_url.rstrip("/"),
        temperature=0.2,
    )


def create_agent(model_id: Optional[str] = None):
    """Build ReAct agent with schema + SQL tools and OpenRouter."""
    model = create_model(model_id)
    tools = get_tools()
    return create_react_agent(
        model,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )


# ── Streaming chat ──────────────────────────────────────────────

async def process_message(
    message: str,
    thread_id: str = "default",
    model_id: Optional[str] = None,
) -> AsyncIterator[dict]:
    """
    Process a user message with the agent and stream events.
    Yields: {"type": "chunk", "content": str} | {"type": "tool_start", "name": str, "input": ...}
            | {"type": "tool_end", "name": str, "output": str} | {"type": "full_response", "content": str}
    """
    agent = create_agent(model_id)
    config = {"configurable": {"thread_id": thread_id}}

    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        version="v2",
    ):
        kind = event.get("event")
        data = event.get("data", {})

        if kind == "on_chat_model_stream":
            chunk = data.get("chunk")
            if chunk and getattr(chunk, "content", None):
                yield {"type": "chunk", "content": chunk.content}

        elif kind == "on_chain_end" and data.get("name") == "Agent":
            output = data.get("output", {})
            messages = output.get("messages", [])
            for msg in messages:
                if getattr(msg, "content", None) and not getattr(msg, "tool_calls", None):
                    yield {"type": "full_response", "content": msg.content}
                    break

        # Tool calls: LangGraph may emit tool_calls in different events
        elif kind == "on_tool_start":
            yield {
                "type": "tool_start",
                "name": data.get("name", ""),
                "input": data.get("input", {}),
            }
        elif kind == "on_tool_end":
            yield {
                "type": "tool_end",
                "name": data.get("name", ""),
                "output": str(data.get("output", "")),
            }
