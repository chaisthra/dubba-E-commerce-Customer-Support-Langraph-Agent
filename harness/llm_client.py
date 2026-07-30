"""
Small provider abstraction, per the assignment's own recommendation (section 7): one
function that takes a system prompt, messages, and tools, and returns a normalized
response. Swapping providers later means touching this file only.

Every call uses the same 2-model chain: PRIMARY_MODEL is tried first; on a hard
failure (API error, timeout, empty response) the harness falls back to
FALLBACK_MODEL. Fallback never triggers on "the answer looks bad" -- only on the call
itself failing. Chosen live against /v1/models on 2026-07-30 (see log/DECISIONS.md):
Sonnet for tool-calling reliability (the harness's correctness depends on clean
structured output), Haiku as a cost/rate-limit-relief fallback.
"""

from dataclasses import dataclass, field

import anthropic
from anthropic import Anthropic

_client = Anthropic()

PRIMARY_MODEL = "claude-sonnet-4-5-20250929"
FALLBACK_MODEL = "claude-haiku-4-5-20251001"
MODEL_CHAIN = (PRIMARY_MODEL, FALLBACK_MODEL)


@dataclass
class ToolCall:
    name: str
    input: dict


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = PRIMARY_MODEL


def complete(
    system: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    force_tool: str | None = None,
    any_tool: bool = False,
) -> LLMResponse:
    """One normalized call to the LLM. `force_tool` pins the response to a specific
    tool name (used for single-schema structured-output steps like classify).
    `any_tool` forces the model to call exactly one of the provided `tools`, its
    choice which (used by graph.py's propose node, which offers a real multi-way
    choice: respond / ask_clarification / lookup_order / check_account_status).
    Tries PRIMARY_MODEL, then FALLBACK_MODEL on a hard failure -- raises the last
    error if both fail."""
    kwargs = {}
    if tools:
        kwargs["tools"] = tools
        if force_tool:
            kwargs["tool_choice"] = {"type": "tool", "name": force_tool}
        elif any_tool:
            kwargs["tool_choice"] = {"type": "any"}
        else:
            kwargs["tool_choice"] = {"type": "auto"}

    last_error: Exception | None = None
    for model in MODEL_CHAIN:
        try:
            response = _client.messages.create(
                model=model,
                max_tokens=1024,
                system=system,
                messages=messages,
                **kwargs,
            )
            return _normalize(response, model)
        except (anthropic.APIStatusError, anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"All models in the fallback chain failed ({MODEL_CHAIN}); last error: {last_error}"
    ) from last_error


def _normalize(response, model: str) -> LLMResponse:
    text = None
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            text = block.text
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(name=block.name, input=block.input))

    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        stop_reason=response.stop_reason,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        model=model,
    )
