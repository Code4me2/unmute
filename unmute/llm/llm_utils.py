import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from logging import getLogger
from typing import Any, AsyncIterator, Protocol, cast

from mistralai import Mistral
from openai import AsyncOpenAI, OpenAI

from unmute.kyutai_constants import LLM_SERVER

from ..kyutai_constants import KYUTAI_LLM_API_KEY, KYUTAI_LLM_MODEL

logger = getLogger(__name__)

INTERRUPTION_CHAR = "—"  # em-dash
USER_SILENCE_MARKER = "..."


# ---------------------------------------------------------------------------
# LLM streaming event types
# ---------------------------------------------------------------------------


@dataclass
class TextDelta:
    """Regular speakable text — route to TTS."""

    text: str


@dataclass
class ToolCallStart:
    """The LLM is invoking a tool — play a ping, don't speak."""

    tool_name: str
    arguments_json: str
    tool_call_id: str


@dataclass
class ToolCallEnd:
    """Tool call phase finished (finish_reason was 'tool_calls' or closing tag)."""

    pass


LLMEvent = TextDelta | ToolCallStart | ToolCallEnd


def preprocess_messages_for_llm(
    chat_history: list[dict[str, str]],
) -> list[dict[str, str]]:
    output = []

    for message in chat_history:
        message = deepcopy(message)

        # Sometimes, an interruption happens before the LLM can say anything at all.
        # In that case, we're left with a message with only INTERRUPTION_CHAR.
        # Simplify by removing.
        if message["content"].replace(INTERRUPTION_CHAR, "") == "":
            continue

        # If the llm was interrupted we don't want to insert the INTERRUPTION_CHAR
        # into the context, otherwise the LLM might want to repeat it.
        message["content"] = message["content"].strip().removesuffix(INTERRUPTION_CHAR)

        if output and message["role"] == output[-1]["role"]:
            output[-1]["content"] += " " + message["content"]
        else:
            output.append(message)

    def role_at(index: int) -> str | None:
        if index >= len(output):
            return None
        return output[index]["role"]

    if role_at(0) == "system" and role_at(1) in [None, "assistant"]:
        # Some LLMs, like Gemma, get confused if the assistant message goes before user
        # messages, so add a dummy user message.
        output = [output[0]] + [{"role": "user", "content": "Hello."}] + output[1:]

    for message in chat_history:
        if (
            message["role"] == "user"
            and message["content"].startswith(USER_SILENCE_MARKER)
            and message["content"] != USER_SILENCE_MARKER
        ):
            # This happens when the user is silent but then starts talking again after
            # the silence marker was inserted but before the LLM could respond.
            # There are special instructions in the system prompt about how to handle
            # the silence marker, so remove the marker from the message to not confuse
            # the LLM
            message["content"] = message["content"][len(USER_SILENCE_MARKER) :]

    return output


async def rechunk_to_words(iterator: AsyncIterator[str]) -> AsyncIterator[str]:
    """Rechunk the stream of text to whole words.

    Otherwise the TTS doesn't know where word boundaries are and will mispronounce
    split words.

    The spaces will be included with the next word, so "foo bar baz" will be split into
    "foo", " bar", " baz".
    Multiple space-like characters will be merged to a single space.
    """
    buffer = ""
    space_re = re.compile(r"\s+")
    prefix = ""
    async for delta in iterator:
        buffer = buffer + delta
        while True:
            match = space_re.search(buffer)
            if match is None:
                break
            chunk = buffer[: match.start()]
            buffer = buffer[match.end() :]
            if chunk != "":
                yield prefix + chunk
            prefix = " "

    if buffer != "":
        yield prefix + buffer


class LLMStream(Protocol):
    async def chat_completion(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Get a chat completion from the LLM."""
        ...


class MistralStream:
    def __init__(self):
        self.current_message_index = 0
        self.mistral = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    async def chat_completion(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        event_stream = await self.mistral.chat.stream_async(
            model="mistral-large-latest",
            messages=cast(Any, messages),  # It's too annoying to type this properly
            temperature=1.0,
        )

        async for event in event_stream:
            delta = event.data.choices[0].delta.content
            assert isinstance(delta, str)  # make Pyright happy
            yield delta


def get_openai_client(
    server_url: str = LLM_SERVER, api_key: str | None = KYUTAI_LLM_API_KEY
) -> AsyncOpenAI:
    # AsyncOpenAI() will complain if the API key is not set, so set a dummy string if it's None.
    # This still makes sense when using vLLM because it doesn't care about the API key.
    return AsyncOpenAI(api_key=api_key or "EMPTY", base_url=server_url + "/v1")


@cache
def autoselect_model() -> str:
    if KYUTAI_LLM_MODEL is not None:
        return KYUTAI_LLM_MODEL
    openai_client = get_openai_client()
    # OpenAI() will complain if the API key is not set, so set a dummy string if it's None.
    # This still makes sense when using vLLM because it doesn't care about the API key.
    client_sync = OpenAI(
        api_key=openai_client.api_key or "EMPTY", base_url=openai_client.base_url
    )
    models = client_sync.models.list()
    if len(models.data) != 1:
        raise ValueError("There are multiple models available. Please specify one.")
    return models.data[0].id


class VLLMStream:
    def __init__(
        self,
        client: AsyncOpenAI,
        temperature: float = 1.0,
    ):
        """
        If `model` is None, it will look at the available models, and if there is only
        one model, it will use that one. Otherwise, it will raise.
        """
        self.client = client
        self.model = autoselect_model()
        self.temperature = temperature

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        tools_path: str | None = None,
    ) -> AsyncIterator[LLMEvent]:
        """Stream a chat completion, yielding LLMEvent instances.

        Handles two tool-call paths:
        - Path A: structured delta.tool_calls (ministral, gpt-oss)
        - Path B: raw JSON/XML in delta.content (qwen3-coder-tools fallback)
        """
        create_kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=cast(Any, messages),
            stream=True,
            temperature=self.temperature,
        )
        if tools_path:
            create_kwargs["extra_body"] = {"tools_path": tools_path}

        stream = await self.client.chat.completions.create(**create_kwargs)

        # Dedup duplicate tool call chunks (Ollama quirk 1) using the
        # call_xxxxxxxx IDs generated by the Ollama fork.
        seen_tool_call_ids: set[str] = set()

        # Path B state: accumulate raw tool call JSON from content
        tool_call_buffer = ""
        in_tool_call = False

        async with stream:
            async for chunk in stream:
                choice = chunk.choices[0]
                delta = choice.delta

                # --- Path A: structured tool_calls on the delta ---
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc in delta.tool_calls:
                        tc_id = tc.id or ""
                        if tc_id and tc_id in seen_tool_call_ids:
                            continue  # duplicate chunk
                        if tc_id:
                            seen_tool_call_ids.add(tc_id)
                        yield ToolCallStart(
                            tool_name=tc.function.name or "",
                            arguments_json=tc.function.arguments or "",
                            tool_call_id=tc_id,
                        )
                    continue

                if choice.finish_reason == "tool_calls":
                    yield ToolCallEnd()
                    continue

                # --- Path B: raw JSON/XML in content ---
                content = delta.content or ""
                if not content:
                    continue

                if not in_tool_call:
                    stripped = content.lstrip()
                    if (
                        stripped.startswith("<tool_call>")
                        or stripped.startswith('{"name"')
                        or stripped.startswith("[TOOL_CALLS]")
                    ):
                        in_tool_call = True
                        tool_call_buffer = content
                        continue
                    yield TextDelta(text=content)
                else:
                    tool_call_buffer += content
                    if "</tool_call>" in tool_call_buffer or _is_complete_tool_json(
                        tool_call_buffer
                    ):
                        parsed = _parse_tool_call_json(tool_call_buffer)
                        yield ToolCallStart(
                            tool_name=parsed.get("name", "unknown"),
                            arguments_json=json.dumps(
                                parsed.get("arguments", {})
                            ),
                            tool_call_id="",
                        )
                        yield ToolCallEnd()
                        tool_call_buffer = ""
                        in_tool_call = False


# ---------------------------------------------------------------------------
# Path B helpers: parse raw tool call JSON from content stream
# ---------------------------------------------------------------------------


def _strip_xml_tags(text: str) -> str:
    """Remove <tool_call>, </tool_call>, and [TOOL_CALLS] markers."""
    text = text.replace("<tool_call>", "").replace("</tool_call>", "")
    text = text.replace("[TOOL_CALLS]", "")
    return text.strip()


def _is_complete_tool_json(buffer: str) -> bool:
    """Check if the buffer contains a complete JSON object."""
    cleaned = _strip_xml_tags(buffer)
    if not cleaned:
        return False
    try:
        json.loads(cleaned)
        return True
    except json.JSONDecodeError:
        return False


def _parse_tool_call_json(buffer: str) -> dict[str, Any]:
    """Parse a tool call JSON from the buffer, tolerating XML wrappers."""
    cleaned = _strip_xml_tags(buffer)
    try:
        return cast(dict[str, Any], json.loads(cleaned))
    except json.JSONDecodeError:
        logger.warning("Failed to parse tool call JSON: %s", cleaned[:200])
        return {"name": "unknown", "arguments": {}}


# ---------------------------------------------------------------------------
# Rechunking for mixed LLMEvent streams
# ---------------------------------------------------------------------------


async def rechunk_llm_events(
    events: AsyncIterator[LLMEvent],
) -> AsyncIterator[LLMEvent]:
    """Apply word-boundary rechunking to TextDelta events only.

    Text between tool calls is discarded — only the final text segment (after
    the last tool call) is yielded to TTS.  Tool call events pass through
    immediately so pings play in real-time.
    """
    text_buffer = ""
    saw_tool_call = False
    space_re = re.compile(r"\s+")

    async for event in events:
        if isinstance(event, ToolCallStart):
            # Discard any buffered intermediate text
            if text_buffer:
                logger.debug("Discarding intermediate text: %s", text_buffer[:80])
                text_buffer = ""
            saw_tool_call = True
            yield event
            continue

        if isinstance(event, ToolCallEnd):
            text_buffer = ""
            yield event
            continue

        # TextDelta
        if saw_tool_call:
            # We've seen at least one tool call — hold text back until
            # we're sure there are no more tool calls coming.
            text_buffer += event.text
        else:
            # No tool calls yet — stream text through with word rechunking
            text_buffer += event.text
            prefix = ""
            while True:
                match = space_re.search(text_buffer)
                if match is None:
                    break
                word = text_buffer[: match.start()]
                text_buffer = text_buffer[match.end() :]
                if word:
                    yield TextDelta(text=prefix + word)
                prefix = " "

    # Flush remaining text — this is either the only text (no tool calls)
    # or the final segment after the last tool call.
    if text_buffer:
        # Apply word-boundary rechunking to the final flush
        prefix = " " if saw_tool_call else ""
        remainder = text_buffer
        while True:
            match = space_re.search(remainder)
            if match is None:
                break
            word = remainder[: match.start()]
            remainder = remainder[match.end() :]
            if word:
                yield TextDelta(text=prefix + word)
            prefix = " "
        if remainder:
            yield TextDelta(text=prefix + remainder)
