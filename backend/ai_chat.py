"""LangChain agent with OpenRouter and tracker tools (schema + read-only SQL)."""

import json
import logging
import os
from datetime import datetime
from typing import Optional, AsyncIterator

import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from config import config

# Single shared checkpointer so the same thread_id keeps conversation history across requests.
_chat_checkpointer = MemorySaver()

# Log chat turns (user, tool calls/results, assistant) to backend/chat/<thread_id>.md
CHAT_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat")

logger = logging.getLogger(__name__)


def _safe_thread_filename(thread_id: str) -> str:
    """Sanitize thread_id for use as a filename (one .md file per thread)."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(thread_id))
    return f"{safe}.md" if safe else "default.md"


def _append_chat_log(content: str, thread_id: str) -> None:
    """Append content to the chat log markdown file for this thread in the chat folder."""
    try:
        os.makedirs(CHAT_LOG_DIR, exist_ok=True)
        path = os.path.join(CHAT_LOG_DIR, _safe_thread_filename(thread_id))
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        logger.warning("Could not write chat log: %s", e)

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
    """Create ChatOpenAI wired to OpenRouter; uses offline LLM model by default for chat."""
    return ChatOpenAI(
        model=model_id or config.llm_model_offline,
        openai_api_key=config.openrouter_api_key,
        base_url=config.openrouter_base_url.rstrip("/"),
        temperature=0.2,
    )


def create_agent(model_id: Optional[str] = None):
    """Build ReAct agent with schema + SQL tools, OpenRouter, and shared in-memory chat history."""
    model = create_model(model_id)
    tools = get_tools()
    return create_react_agent(
        model,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=_chat_checkpointer,
    )


# ── Streaming chat ──────────────────────────────────────────────

async def process_message(
    message: str,
    thread_id: str = "default",
    user_id: str = "default",
    model_id: Optional[str] = None,
) -> AsyncIterator[dict]:
    """
    Process a user message with the agent and stream events.
    Conversation history is persisted per thread_id (and user_id) via the checkpointer.
    Yields: {"type": "chunk", "content": str} | {"type": "tool_start", "name": str, "input": ...}
            | {"type": "tool_end", "name": str, "output": str} | {"type": "full_response", "content": str}
    """
    agent = create_agent(model_id)
    run_config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    # Log turn start (user message) to chat folder
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _append_chat_log(f"## Turn {ts} (thread: {thread_id})\n\n### User\n\n{message}\n\n", thread_id)

    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=message)]},
        config=run_config,
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
                    content = msg.content
                    _append_chat_log(f"### Assistant\n\n{content}\n\n---\n\n", thread_id)
                    yield {"type": "full_response", "content": content}
                    break

        # Tool calls: LangGraph may emit tool_calls in different events
        elif kind == "on_tool_start":
            name = data.get("name", "")
            inp = data.get("input", {})
            _append_chat_log(f"### Tool: `{name}`\n\nInput:\n```json\n{json.dumps(inp, indent=2)}\n```\n\n", thread_id)
            yield {
                "type": "tool_start",
                "name": name,
                "input": inp,
            }
        elif kind == "on_tool_end":
            name = data.get("name", "")
            output = str(data.get("output", ""))
            _append_chat_log(f"Result:\n```\n{output}\n```\n\n", thread_id)
            yield {
                "type": "tool_end",
                "name": name,
                "output": output,
            }
