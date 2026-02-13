# Feature: Audio Cues for LLM Tool Calls

## Overview

When the LLM invokes MCP tools via the custom Ollama fork, replace the
tool-call text in the audio stream with short synthesized audio cues ("pings")
so the user hears a sound effect instead of the TTS trying to pronounce JSON.
Different tones distinguish tool calls from sub-agent deployments. Tool call
events are surfaced in the text transcript (headless implementation).

---

## 1. The LLM Backend: Custom Ollama Fork

The workstation (`ssh -p 22222 velvetm@mb`) runs a **custom fork of Ollama**
at `/home/velvetm/Desktop/ollama/` — not stock Ollama. This fork implements a
full MCP subsystem in ~120KB of Go code across 13 source files in
`/home/velvetm/Desktop/ollama/server/`.

### 1a. Architecture of the Custom MCP System

```
Public API (mcp.go)
    |
    +-- MCPDefinitions (mcp_definitions.go)       -- Static server config
    +-- MCPSessionManager (mcp_sessions.go)       -- 30-min TTL sessions
            |
            +-- MCPManager (mcp_manager.go)       -- Multi-client orchestration
                    |
                    +-- MCPClient (mcp_client.go)          -- Stdio transport
                    +-- MCPWebSocketClient (mcp_client_ws.go)  -- WebSocket transport
                    |
                    (MCPClientInterface in mcp_client_interface.go)
```

Supporting modules:
- **`mcp_jit.go`** — JIT (Just-In-Time) tool discovery via `mcp_discover`
  meta-tool and glob pattern matching. Models start with only `mcp_discover`
  and lazily discover tools on demand, saving 500-5000 tokens.
- **`mcp_code_api.go`** — Context injection into model prompts (minimal + JIT)
- **`mcp_command_resolver.go`** — Cross-platform executable resolution
- **`mcp_security_config.go`** — Defense-in-depth: command blocklists, shell
  injection prevention, credential filtering, process group isolation
- **`routes_tools.go`** — HTTP API: `GET/POST /api/tools`,
  `POST /api/tools/search`
- **`mcp_test.go`** — 24KB test suite

### 1b. Key Custom Features

**JIT Tool Discovery**: Instead of loading all tool schemas upfront, the model
calls `mcp_discover("*file*")` and Ollama lazily connects to the MCP server
and injects matching schemas. Persistent across conversation rounds.

**Dual Transport**: Stdio (local MCP servers, JSON-RPC 2.0) and WebSocket
(remote servers via Tailscale). Both implement `MCPClientInterface`.

**Execution Planning**: `AnalyzeExecutionPlan()` detects read/write conflicts
on file paths and determines parallel vs sequential execution. Provides
`ExecuteWithPlan()`, `ExecuteToolsParallel()`, `ExecuteToolsSequential()`.

**Auto-Enable Modes**: `never`, `always`, `with_path`, `if_match` (conditional
on file existence or env vars).

**Session Management**: Singleton `MCPSessionManager` with 30-minute TTL,
automatic cleanup, session reuse via config matching.

**OpenAI Endpoint Compatibility**: The fork specifically enhanced
`/v1/chat/completions` for tool calling (commit `d614b4ce`).

### 1c. Known Issues (from `docs/mcp-jit-issues.md`)

| # | Issue | Status |
|---|-------|--------|
| 1 | Parser inconsistency on first tool call | FIXED |
| 2 | `maxToolsPerDiscovery=5` misses critical tools | OPEN (HIGH) |
| 3 | Wrong tool selected (consequence of #2) | OPEN (HIGH) |
| 4 | Raw format leaking on first attempt | PARTIALLY FIXED |
| 5 | Multi-pattern discovery | ADDRESSED |

### 1d. Remote MCP Bridge

A Node.js WebSocket bridge at
`/home/velvetm/Desktop/ollama/examples/remote-mcp/bridge.js` (164 lines)
wraps any stdio MCP server for access over Tailscale. Includes a systemd unit
with security hardening.

### 1e. MCP Server Configuration

File: `~/.ollama/mcp-servers.json`
```json
{
  "servers": [{
    "name": "filesystem",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem"],
    "requires_path": true,
    "auto_enable": "with_path"
  }]
}
```

---

## 2. How Tool Calls Appear in the Stream (Tested)

Tested with `ministral-3:14b` and `gpt-oss:20b` (the recommended models for
more consistent parsing) as well as `qwen3-coder-tools`.

### 2a. OpenAI-compatible `/v1/chat/completions` (streaming)

This is the endpoint the unmute backend uses via `AsyncOpenAI`.

**All three models produce structured `delta.tool_calls`:**
```json
{"choices":[{"delta":{
    "role":"assistant",
    "content":"",
    "tool_calls":[{
        "id":"",
        "index":0,
        "type":"function",
        "function":{"name":"list_directory","arguments":"{\"path\":\"/home/velvetm\"}"}
    }]
},"finish_reason":null}]}
```

**Then `finish_reason: "tool_calls"`:**
```json
{"choices":[{"delta":{"content":""},"finish_reason":"tool_calls"}]}
```

**Then Ollama internally attempts MCP execution.** If the MCP server isn't
configured for the request, it feeds the error back to the model, which then
generates fallback text content in subsequent chunks.

**Quirks observed and fix status** (Ollama fork commits `3d7baee5`,
`b7b59cad`, `8b66fc8f`):

| # | Quirk | Status | Impact on unmute |
|---|-------|--------|------------------|
| 1 | Duplicate tool call chunks (same call emitted twice) | OPEN — needs live reproduction | Still need client-side dedup |
| 2 | Stream continues after `finish_reason: "tool_calls"` | FIXED — `finish_reason` suppressed during `task_status: "working"` | Cleaner phase separation |
| 3 | `finish_reason: "tool_calls"` on text-only terminal chunks | FIXED — only set when chunk actually contains tool calls | Can trust `finish_reason` now |
| 4 | Tool call IDs are empty strings | FIXED — now generates `call_xxxxxxxx` format IDs | Can correlate tool results |
| 5 | gpt-oss non-standard `reasoning` field | DOCUMENTED — Ollama-specific extension | Ignore in unmute (don't speak) |
| 6 | Double terminal chunks before `[DONE]` | FIXED — `doneSent` flag prevents duplicates | Cleaner stream termination |

**With `qwen3-coder-tools` specifically, a second path also appears:**
- Raw JSON as `delta.content` text (Path B) — the model outputs
  `{"name": "read_file", ...}` as regular text tokens
- This is model-dependent; ministral and gpt-oss use the structured path
  more consistently

### 2b. Ollama native `/api/chat` (streaming)

More explicit — reveals Ollama's internal tool execution loop:

**ministral-3:14b:**
```
1. tool_call chunk + task_id + task_status: "working"
2. done_reason: "stop" + timing stats (188ms, 15 eval tokens)
3. tool_results with error: "Tool 'list_directory' is not available..."
4. Text content streamed: "It seems I don't have access..."
5. task_status: "completed", done: true
```

**gpt-oss:20b:**
```
1. thinking tokens: "We need to list files..."
2. tool_call chunk + timing stats (320ms, 47 eval tokens)
3. tool_results with error: "Tool 'list_directory' is not available..."
4. More thinking: "The tool is not available..."
5. Text content: "I'm sorry, but I don't have access..."
6. task_status: "completed"
```

Key fields unique to the native API:
- `task_id`: UUID tracking the multi-step tool execution
- `task_status`: `"working"` → `"completed"`
- `tool_results`: inline error/result from MCP execution
- `thinking`/`reasoning`: chain-of-thought (gpt-oss)

### 2c. Model Template Formats

**ministral-3:14b**: Uses `PARSER ministral` directive. Markers:
- `[AVAILABLE_TOOLS]...[/AVAILABLE_TOOLS]` — tool schema injection
- `[TOOL_CALLS]` prefix before tool call output
- `function_name[ARGS]{"key":"value"}` — name + `[ARGS]` separator + JSON
- `[TOOL_RESULTS]...[/TOOL_RESULTS]` — results

**gpt-oss:20b**: Channel-based format, no explicit PARSER directive:
- Tools defined as TypeScript namespaces: `namespace functions { type tool = ... }`
- Tool calls via `commentary to=functions.tool_name` channel
- Results via `functions.tool_name to=assistant` channel
- Thinking via `analysis` channel, final response via `final` channel

**qwen3-coder-tools**: XML-based:
- `<tool_call>{"name":...,"arguments":...}</tool_call>`
- `<tool_response>...</tool_response>`

---

## 3. Design Decision: Which API Endpoint

### Option A: Stay on OpenAI-compatible `/v1/chat/completions`

**Pros:**
- No code change to switch HTTP clients
- `AsyncOpenAI` library handles SSE parsing, retries, etc.
- The custom fork specifically enhanced this endpoint for tool calling
- ministral and gpt-oss produce consistent structured `delta.tool_calls`

**Cons:**
- No `task_id`/`task_status` tracking
- Duplicate tool call chunks must be deduplicated
- No visibility into Ollama's internal MCP execution (tool_results errors)
- `finish_reason` is unreliable as a state signal

### Option B: Switch to Ollama native `/api/chat`

**Pros:**
- `task_id` + `task_status` give precise tool execution state
- `tool_results` field shows what MCP returned
- `thinking`/`reasoning` fields available for gpt-oss
- Single consistent format (no Path A/B ambiguity)

**Cons:**
- Must replace `AsyncOpenAI` with raw `httpx`/`aiohttp` SSE parsing
- Different JSON schema from OpenAI — chatbot history format changes
- More coupling to Ollama-specific API

### Recommendation: Option A with targeted workarounds

Stay on `/v1/chat/completions` because:
1. The `openai` library is battle-tested for SSE streaming
2. ministral-3:14b and gpt-oss:20b produce consistent structured `tool_calls`
3. The custom fork's OpenAI endpoint was specifically enhanced for this
4. We only need to detect tool calls (not track execution state) since Ollama
   handles MCP internally — we just need to avoid speaking the artifacts
5. Deduplication of tool call chunks is trivial (track last seen tool call)

If we later need `task_id`/`task_status` (e.g., for long-running tools), we
can add a parallel native API connection for status polling without replacing
the main streaming path.

---

## 4. Current Code Path (What Happens Today)

### LLM streaming: `unmute/llm/llm_utils.py:162-183`

```python
async def chat_completion(self, messages) -> AsyncIterator[str]:
    stream = await self.client.chat.completions.create(
        model=self.model, messages=messages, stream=True, temperature=...
    )
    async with stream:
        async for chunk in stream:
            chunk_content = chunk.choices[0].delta.content
            if not chunk_content:   # ← tool_calls chunks skipped (empty content)
                continue
            yield chunk_content     # ← only yields text strings
```

- **Structured tool_calls**: Skipped (dead air during MCP execution, then
  fallback text spoken normally)
- **Raw JSON as content** (qwen3-coder-tools): TTS pronounces JSON

### Text → TTS: `unmute/unmute_handler.py:224-259`

```python
async for delta in rechunk_to_words(llm.chat_completion(messages)):
    await self.output_queue.put(ora.UnmuteResponseTextDeltaReady(delta=delta))
    await tts.send(delta)
```

### Audio output: `unmute/unmute_handler.py:540-543`

```python
audio = np.array(message.pcm, dtype=np.float32)
await output_queue.put((SAMPLE_RATE, audio))  # 24kHz float32
```

---

## 5. Proposed Implementation

### 5a. Streaming event types (new, in `llm_utils.py`)

```python
from dataclasses import dataclass

@dataclass
class TextDelta:
    """Regular speakable text — route to TTS."""
    text: str

@dataclass
class ToolCallStart:
    """The LLM is invoking a tool — play a ping, don't speak."""
    tool_name: str
    arguments_json: str
    tool_call_id: str  # call_xxxxxxxx from Ollama fork (quirk 4 fix)

@dataclass
class ToolCallEnd:
    """Tool execution finished. Now reliable — finish_reason only set on
    actual tool call chunks (quirk 3 fix)."""
    pass

LLMEvent = TextDelta | ToolCallStart | ToolCallEnd
```

### 5b. Detection in `VLLMStream.chat_completion()`

Handle both structured `tool_calls` (ministral, gpt-oss) and raw JSON content
(qwen3-coder-tools fallback). With the Ollama fork fixes:
- Quirk 3 FIXED: `finish_reason: "tool_calls"` is now reliable (only set on
  actual tool call chunks), so we can trust it as a phase signal
- Quirk 4 FIXED: Tool call IDs are now `call_xxxxxxxx` format
- Quirk 6 FIXED: No double terminal chunks
- Quirk 1 OPEN: Still need client-side dedup for duplicate tool call chunks

```python
async def chat_completion(self, messages, tools=None) -> AsyncIterator[LLMEvent]:
    create_kwargs = dict(
        model=self.model,
        messages=messages,
        stream=True,
        temperature=self.temperature,
    )
    if tools:
        create_kwargs["tools"] = tools

    stream = await self.client.chat.completions.create(**create_kwargs)

    seen_tool_call_ids: set[str] = set()  # dedup via call_xxxxxxxx IDs (quirk 1)
    tool_call_buffer = ""
    in_tool_call = False

    async with stream:
        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            # --- Path A: structured tool_calls (ministral, gpt-oss) ---
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tc in delta.tool_calls:
                    tc_id = tc.id or ""
                    if tc_id and tc_id in seen_tool_call_ids:
                        continue  # quirk 1 dedup
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

            # --- Path B: raw JSON/XML in content (qwen3-coder-tools) ---
            content = delta.content or ""
            if not content:
                continue

            if not in_tool_call:
                stripped = content.lstrip()
                if (stripped.startswith("<tool_call>")
                    or stripped.startswith('{"name"')
                    or stripped.startswith("[TOOL_CALLS]")):
                    in_tool_call = True
                    tool_call_buffer = content
                    continue
                yield TextDelta(text=content)
            else:
                tool_call_buffer += content
                if ("</tool_call>" in tool_call_buffer
                    or _is_complete_tool_json(tool_call_buffer)):
                    parsed = _parse_tool_call_json(tool_call_buffer)
                    yield ToolCallStart(
                        tool_name=parsed.get("name", "unknown"),
                        arguments_json=json.dumps(parsed.get("arguments", {})),
                        tool_call_id="",
                    )
                    yield ToolCallEnd()
                    tool_call_buffer = ""
                    in_tool_call = False
```

### 5c. `rechunk_llm_events()` — word-boundary rechunking for mixed streams

```python
async def rechunk_llm_events(
    events: AsyncIterator[LLMEvent],
) -> AsyncIterator[LLMEvent]:
    """Apply word-boundary rechunking to TextDelta events only.
    Non-text events flush the buffer and pass through immediately."""
    text_buffer = ""
    space_re = re.compile(r"\s+")

    async for event in events:
        if not isinstance(event, TextDelta):
            if text_buffer:
                yield TextDelta(text=text_buffer)
                text_buffer = ""
            yield event
            continue

        text_buffer += event.text
        prefix = ""
        while True:
            match = space_re.search(text_buffer)
            if match is None:
                break
            chunk = text_buffer[:match.start()]
            text_buffer = text_buffer[match.end():]
            if chunk:
                yield TextDelta(text=prefix + chunk)
            prefix = " "

    if text_buffer:
        yield TextDelta(text=text_buffer)
```

### 5d. Audio cue generation (new file: `unmute/audio_cues.py`)

Programmatic sine wave pings — no external audio files needed:

```python
import numpy as np

SAMPLE_RATE = 24000

def _envelope(t: np.ndarray, attack: float = 0.01, decay_rate: float = 12.0):
    """Smooth attack + exponential decay."""
    attack_samples = int(attack * SAMPLE_RATE)
    env = np.exp(-t * decay_rate)
    env[:attack_samples] *= np.linspace(0, 1, attack_samples)
    return env

def generate_tool_call_ping(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Rising two-tone chirp (~150ms). 'Something is happening.'"""
    dur = 0.15
    t = np.linspace(0, dur, int(sr * dur), dtype=np.float32)
    tone1 = np.sin(2 * np.pi * 660 * t) * _envelope(t, decay_rate=15)
    # Second tone starts at midpoint
    mid = len(t) // 2
    tone2 = np.zeros_like(t)
    t2 = t[:len(t) - mid]
    tone2[mid:] = np.sin(2 * np.pi * 880 * t2) * _envelope(t2, decay_rate=15)
    return 0.4 * (tone1 + tone2).astype(np.float32)

def generate_agent_ping(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Ascending arpeggio (~250ms). 'Spawning a sub-agent.'"""
    dur = 0.25
    freqs = [523, 659, 784]  # C5, E5, G5
    step = int(sr * dur / len(freqs))
    samples = np.zeros(int(sr * dur), dtype=np.float32)
    for i, freq in enumerate(freqs):
        start = i * step
        end = min(start + step, len(samples))
        seg_t = np.linspace(0, (end - start) / sr, end - start, dtype=np.float32)
        samples[start:end] = np.sin(2 * np.pi * freq * seg_t) * _envelope(seg_t, decay_rate=10)
    return 0.4 * samples

def generate_error_ping(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Descending tone (~150ms). 'Something went wrong.'"""
    dur = 0.15
    t = np.linspace(0, dur, int(sr * dur), dtype=np.float32)
    freq = 440 * np.exp(-t * 3)  # descending frequency
    return 0.35 * (np.sin(2 * np.pi * freq * t) * _envelope(t)).astype(np.float32)

# Module-level singletons — loaded once
PING_TOOL_CALL = generate_tool_call_ping()
PING_AGENT = generate_agent_ping()
PING_ERROR = generate_error_ping()
```

### 5e. Handler integration (`_generate_response_task()`)

```python
from unmute.audio_cues import PING_TOOL_CALL, PING_AGENT, PING_ERROR

# Configurable — will grow as agentic tools are defined
AGENT_TOOL_NAMES = {"spawn_agent", "delegate", "run_sub_agent"}

async def _generate_response_task(self):
    generating_message_i = len(self.chatbot.chat_history)
    # ... existing setup (ResponseCreated, TTS startup, etc.) ...

    llm = VLLMStream(self.openai_client, temperature=...)
    messages = self.chatbot.preprocessed_messages()

    raw_events = llm.chat_completion(messages, tools=self.tool_definitions)
    events = rechunk_llm_events(raw_events)

    async for event in events:
        if isinstance(event, TextDelta):
            await self.output_queue.put(
                ora.UnmuteResponseTextDeltaReady(delta=event.text)
            )
            await tts.send(event.text)
            await self.add_chat_message_delta(
                event.text, "assistant",
                generating_message_i=generating_message_i,
            )

        elif isinstance(event, ToolCallStart):
            # Pick ping based on tool type
            if event.tool_name in AGENT_TOOL_NAMES:
                ping = PING_AGENT
            else:
                ping = PING_TOOL_CALL

            # Inject audio cue directly, bypassing TTS
            await self.output_queue.put((SAMPLE_RATE, ping))

            # Surface to text transcript (headless)
            await self.output_queue.put(
                ora.UnmuteToolCallEvent(
                    tool_name=event.tool_name,
                    arguments=event.arguments_json,
                )
            )
            logger.info("Tool call: %s(%s)", event.tool_name,
                        event.arguments_json)

        elif isinstance(event, ToolCallEnd):
            pass  # Could play a completion chime later

        if len(self.chatbot.chat_history) > generating_message_i:
            break  # interrupted
```

### 5f. New event type for transcript (`openai_realtime_api_events.py`)

```python
class UnmuteToolCallEvent(BaseEvent[Literal["unmute.tool_call"]]):
    """Tool call detected — surfaced in text transcript, audio cue played."""
    tool_name: str
    arguments: str
```

Add to `ServerEvent` union.

### 5g. Tool definitions (initial — filesystem only for testing)

```python
# In unmute_handler.py or a new unmute/llm/tool_definitions.py
FILESYSTEM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List contents of a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"],
            },
        },
    },
]
```

---

## 6. Chat History Considerations

Tool call text must NOT be added to `chatbot.chat_history` as assistant
content. Since Ollama handles MCP execution internally (the custom fork
executes tools and feeds results back to the model within the same API call),
the tool call is transparent to the chat history. We only need to:

1. Avoid speaking it (done via event type routing)
2. Avoid recording it as assistant content (done by only calling
   `add_chat_message_delta` for `TextDelta` events)
3. Surface it in the transcript (done via `UnmuteToolCallEvent`)

---

## 7. Edge Cases

### 7a. Duplicate tool call chunks (Quirk 1 — still open)
Both ministral and gpt-oss may emit the same tool call twice in the stream.
The `seen_tool_call_ids` set in section 5b deduplicates by the `call_xxxxxxxx`
ID (now generated by the Ollama fork, quirk 4 fix). If the duplicate shares
the same ID, it's filtered. If IDs differ (or are empty in Path B), we fall
back to `name:args` dedup.

### 7b. Partial/interrupted tool calls
VAD interruption during a tool call: the ping is a one-shot push to the
output queue, so `interrupt_bot()` works unchanged.

### 7c. Mixed content (text + tool calls)
The model may say "Let me check that" before calling a tool.
`rechunk_llm_events` flushes buffered text before yielding tool events.

### 7d. gpt-oss `reasoning` field
The `reasoning`/`thinking` field appears on the delta alongside content.
Current proposal: ignore it (don't speak or display internal reasoning).
Can be surfaced later if useful for debugging.

### 7e. Long tool execution
Ollama handles MCP internally, so from our streaming perspective there's
just a pause in chunks. If >3s of silence during a tool call:
- Option 1: Periodic "still working" subtle tone
- Option 2: Accept the pause (simpler)

### 7f. TTS connection during tool pause
TTS WebSocket is open but receiving no text. Short pauses (<5s) are fine.
For longer pauses, may need keepalive frames or lazy TTS connection.

### 7g. `finish_reason` behavior (Quirk 3 — FIXED)
After the Ollama fork fix (`8b66fc8f`), `finish_reason: "tool_calls"` is only
set on chunks that actually contain tool calls. Text-only terminal chunks now
use `finish_reason: "stop"`. We can rely on `finish_reason` as a phase signal.

---

## 8. Files to Modify

| File | Change |
|------|--------|
| `unmute/llm/llm_utils.py` | `LLMEvent` types, modify `VLLMStream.chat_completion()`, add `rechunk_llm_events()`, `_is_complete_tool_json()`, `_parse_tool_call_json()` |
| `unmute/unmute_handler.py` | Modify `_generate_response_task()` for `LLMEvent` routing, add `tool_definitions` |
| `unmute/audio_cues.py` (new) | `generate_tool_call_ping()`, `generate_agent_ping()`, module-level singletons |
| `unmute/openai_realtime_api_events.py` | Add `UnmuteToolCallEvent` to `ServerEvent` union |
| `unmute/llm/tool_definitions.py` (new, optional) | Tool schemas, `AGENT_TOOL_NAMES` set |

Files **not** modified:
- `unmute/llm/chatbot.py` — no change (Ollama handles MCP internally)
- `unmute/tts/text_to_speech.py` — no change (TTS unaware of tool calls)
- `unmute/llm/system_prompt.py` — no change for now (tool instructions added
  later when agentic tools are defined)

---

## 9. Testing Strategy

### Unit tests
- `rechunk_llm_events` correctly interleaves text and tool events
- `_is_complete_tool_json` / `_parse_tool_call_json` handle edge cases
- Audio cue generation produces valid float32 arrays at 24kHz
- Deduplication filters duplicate tool call chunks

### Integration test
- Point unmute backend at workstation Ollama with `ministral-3:14b`
- Send a prompt that triggers a tool call (e.g., "list files in /tmp")
- Verify: tool call text not sent to TTS, ping audio appears in output queue,
  `UnmuteToolCallEvent` appears in transcript, spoken response heard after tool

### Models to test with
- **`ministral-3:14b`** — recommended, consistent structured parser
- **`gpt-oss:20b`** — recommended, consistent but has `reasoning` field
- `qwen3-coder-tools` — use for Path B (raw JSON) regression testing only

---

## 10. Resolved Design Decisions

| # | Question | Decision |
|---|----------|----------|
| 1 | Which API endpoint? | Stay on `/v1/chat/completions` (see section 3) |
| 2 | Pass tool schemas? | Yes, explicitly. Use filesystem tools for now, extend later |
| 3 | Programmatic vs WAV? | Programmatic sine waves for now, WAV overrides later |
| 4 | Surface in frontend? | Yes — `UnmuteToolCallEvent` in text transcript (headless) |
| 5 | Preferred models? | `ministral-3:14b` or `gpt-oss:20b` for consistent parsing |

## 11. Ollama Fork Fix Status

Commits on workstation (`/home/velvetm/Desktop/ollama/`):

| Commit | Fix |
|--------|-----|
| `3d7baee5` | Quirk 4: Generate `call_xxxxxxxx` tool call IDs |
| `b7b59cad` | Quirk 2: Suppress `finish_reason` during MCP internal execution (`task_status: "working"`) |
| `8b66fc8f` | Quirk 3 + 6: `finish_reason: "tool_calls"` only on actual tool call chunks; no double terminal chunks |

**Remaining on Ollama side:**
- Quirk 1 (duplicate tool call chunks): Needs live test case reproduction.
  Unmute handles this via `seen_tool_call_ids` dedup.
- Quirk 5 (`reasoning` field): Documented as Ollama-specific extension.
  Unmute ignores it (not spoken, not displayed).
