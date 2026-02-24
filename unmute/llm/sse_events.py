"""SSE client for the orchestrator's /v1/events endpoint.

This endpoint is a long-lived SSE stream that fires when async agent
results arrive while the conversation is idle (waiting_for_user).  It
triggers a new LLM generation server-side and streams the output back
in OpenAI-compat chunk format.
"""

import json
from logging import getLogger
from typing import Any, AsyncIterator

import httpx

from unmute.llm.llm_utils import (
    AgentInjection,
    AgentProgress,
    LLMEvent,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
)

logger = getLogger(__name__)

# The endpoint blocks until async results arrive, so use a long timeout.
SSE_CONNECT_TIMEOUT = 10.0
SSE_READ_TIMEOUT = 300.0


async def stream_orchestrator_events(
    base_url: str,
    session_id: str,
) -> AsyncIterator[LLMEvent]:
    """Connect to the orchestrator SSE endpoint and yield LLMEvents.

    Connects to ``GET {base_url}/v1/events?session_id={session_id}`` and
    parses the SSE stream.  Content delta chunks are converted to
    ``TextDelta``; structured tool_calls become ``ToolCallStart`` /
    ``ToolCallEnd``.  Custom events (``agent.injection``,
    ``mcp.tool_result``) are logged but not yielded as text.

    The connection is kept alive until the server sends ``[DONE]`` or the
    iterator is closed (e.g. via cancellation).
    """
    url = f"{base_url}/v1/events"
    timeout = httpx.Timeout(
        connect=SSE_CONNECT_TIMEOUT,
        read=SSE_READ_TIMEOUT,
        write=10.0,
        pool=10.0,
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "GET", url, params={"session_id": session_id}
        ) as response:
            response.raise_for_status()

            event_type: str | None = None

            async for line in response.aiter_lines():
                line = line.rstrip("\r\n")

                # SSE blank line = end of event block
                if line == "":
                    event_type = None
                    continue

                # SSE event type line
                if line.startswith("event:"):
                    event_type = line[len("event:") :].strip()
                    continue

                # SSE data line
                if not line.startswith("data:"):
                    continue

                data_str = line[len("data:") :].strip()

                # Stream termination
                if data_str == "[DONE]":
                    logger.debug("SSE stream: [DONE]")
                    return

                # Custom events
                if event_type == "agent.injection":
                    logger.info("SSE agent injection: %s", data_str[:200])
                    try:
                        payload = json.loads(data_str)
                        agents = payload.get("agents", [])
                    except (json.JSONDecodeError, AttributeError):
                        agents = []
                    yield AgentInjection(agents=agents)
                    continue
                if event_type == "agent.progress":
                    logger.debug("SSE agent progress: %s", data_str[:200])
                    try:
                        payload = json.loads(data_str)
                        yield AgentProgress(
                            agent=payload.get("agent", ""),
                            progress_type=payload.get("progress_type", ""),
                            tool=payload.get("tool", ""),
                            arguments=payload.get("arguments"),
                            content=payload.get("content"),
                        )
                    except (json.JSONDecodeError, AttributeError):
                        pass
                    continue
                if event_type == "mcp.tool_result":
                    logger.info("SSE mcp tool result: %s", data_str[:200])
                    continue

                # Parse standard OpenAI-compat chunk
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning("SSE: unparseable data: %s", data_str[:200])
                    continue

                async for event in _parse_chunk(chunk):
                    yield event


async def _parse_chunk(chunk: dict[str, Any]) -> AsyncIterator[LLMEvent]:
    """Convert an OpenAI-compat streaming chunk to LLMEvents."""
    choices = chunk.get("choices", [])
    if not choices:
        return

    choice = choices[0]
    delta = choice.get("delta", {})
    finish_reason = choice.get("finish_reason")

    # Structured tool calls
    tool_calls = delta.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            args = fn.get("arguments", "")
            if not isinstance(args, str):
                args = json.dumps(args)
            yield ToolCallStart(
                tool_name=fn.get("name", ""),
                arguments_json=args,
                tool_call_id=tc.get("id", ""),
            )
        return

    if finish_reason == "tool_calls":
        yield ToolCallEnd()
        return

    # Text content
    content = delta.get("content")
    if content:
        yield TextDelta(text=content)
