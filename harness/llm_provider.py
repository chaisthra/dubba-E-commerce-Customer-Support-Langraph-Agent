"""
Multi-provider LLM abstraction with automatic fallback. Anthropic first (tool-calling
reliability -- the harness's correctness depends on clean structured output, see
log/DECISIONS.md), Groq second -- reached only once the ENTIRE Anthropic provider is
exhausted (both PRIMARY_MODEL and FALLBACK_MODEL failed), not on a single model's
hiccup. Every Provider.create() returns the same LLMResponse shape regardless of which
provider/model actually served the call, so callers (classify_node, propose_node, ...)
read the response identically either way -- `response.provider`/`response.model` say
which one it actually was, for tracing.

Callers needing ONLY Groq, never Anthropic (harness/summarizer.py's rolling
summarization -- deliberately cheap background housekeeping, never the primary model)
instantiate GroqProvider() directly instead of going through get_llm_client().
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import anthropic
import groq

from harness.llm_client import MODEL_CHAIN


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
    model: str = ""
    provider: str = ""


class Provider(ABC):
    name: str

    @abstractmethod
    def create(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        force_tool: str | None = None,
        any_tool: bool = False,
    ) -> LLMResponse:
        """`force_tool` pins the response to one specific tool name (structured-output
        steps like classify/evaluate/summarize). `any_tool` forces exactly one of
        `tools`, model's choice which (graph.py's propose node). Neither set means a
        plain/auto call. Raises on total failure (every model in this provider's own
        chain failed) -- callers (ProviderChain) decide what to do with that."""


class AnthropicProvider(Provider):
    name = "anthropic"
    MODELS = MODEL_CHAIN  # (PRIMARY_MODEL, FALLBACK_MODEL) -- see harness/llm_client.py

    def __init__(self) -> None:
        self._client = anthropic.Anthropic()

    def create(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        force_tool: str | None = None,
        any_tool: bool = False,
    ) -> LLMResponse:
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
        for model in self.MODELS:
            try:
                response = self._client.messages.create(
                    model=model, max_tokens=1024, system=system, messages=messages, temperature=0, **kwargs
                )
                return _normalize_anthropic(response, model)
            except (anthropic.APIStatusError, anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
                last_error = exc
                continue

        raise RuntimeError(
            f"AnthropicProvider: all models failed ({self.MODELS}); last error: {last_error}"
        ) from last_error


class GroqProvider(Provider):
    name = "groq"

    # Order chosen by actually running each model against our real tool-calling loop
    # (forced single-tool AND any-of-N-tools, both used by this app) and verifying it
    # produces tool calls this SDK path can parse against tools we actually offered --
    # NOT ordered by release recency or headline benchmark. Some Groq-hosted models
    # emit tool-call arguments in shapes this code rejects, or hallucinate calls to
    # tools that were never in the `tools` list; those failure modes only show up by
    # actually exercising the loop, not by reading a spec sheet.
    MODELS = [
        "openai/gpt-oss-20b",
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-safeguard-20b",
        "openai/gpt-oss-120b",
    ]

    def __init__(self) -> None:
        self._client = groq.Groq()

    def create(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        force_tool: str | None = None,
        any_tool: bool = False,
    ) -> LLMResponse:
        groq_messages = [{"role": "system", "content": system}, *messages]

        kwargs = {}
        if tools:
            kwargs["tools"] = _to_groq_tools(tools)
            if force_tool:
                kwargs["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
            elif any_tool:
                kwargs["tool_choice"] = "required"
            else:
                kwargs["tool_choice"] = "auto"

        last_error: Exception | None = None
        for model in self.MODELS:
            try:
                response = self._client.chat.completions.create(
                    model=model, messages=groq_messages, temperature=0, **kwargs
                )
                return _normalize_groq(response, model)
            # Broad on purpose: a rate limit is one way a model can be unusable for
            # this call, but so is a model disobeying a forced tool_choice (Groq
            # validates that server-side and raises BadRequestError, a sibling
            # APIStatusError subclass, not RateLimitError) -- either way, the fix is
            # the same: move on to the next model in the chain rather than crash.
            except groq.APIStatusError as exc:
                last_error = exc
                continue

        raise RuntimeError(
            f"GroqProvider: all models failed ({self.MODELS}); last error: {last_error}"
        ) from last_error


class ProviderChain:
    """Tries providers in order. Falls through to the next provider only once the
    current one raises (i.e. its OWN internal per-model chain is already exhausted --
    AnthropicProvider/GroqProvider never raise on a single model failing, only when
    every model they know about has failed), not on any single model's failure."""

    def __init__(self, providers: list[Provider]) -> None:
        self.providers = providers

    def create(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        force_tool: str | None = None,
        any_tool: bool = False,
    ) -> LLMResponse:
        last_error: Exception | None = None
        for provider in self.providers:
            try:
                return provider.create(system, messages, tools=tools, force_tool=force_tool, any_tool=any_tool)
            except Exception as e:
                last_error = e
                continue
        raise last_error


def _to_groq_tools(tools: list[dict]) -> list[dict]:
    """Anthropic tool shape ({name, description, input_schema}) -> OpenAI/Groq shape
    ({type: "function", function: {name, description, parameters}}). Same JSON Schema
    content either way, just renamed/nested differently."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _normalize_anthropic(response, model: str) -> LLMResponse:
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
        provider="anthropic",
    )


def _normalize_groq(response, model: str) -> LLMResponse:
    message = response.choices[0].message
    tool_calls = [
        ToolCall(name=tc.function.name, input=json.loads(tc.function.arguments))
        for tc in (message.tool_calls or [])
    ]

    return LLMResponse(
        text=message.content,
        tool_calls=tool_calls,
        stop_reason=response.choices[0].finish_reason or "",
        input_tokens=response.usage.prompt_tokens if response.usage else 0,
        output_tokens=response.usage.completion_tokens if response.usage else 0,
        model=model,
        provider="groq",
    )


_chain: ProviderChain | None = None


def get_llm_client() -> ProviderChain:
    """The app-wide entry point for LLM calls that should try Anthropic first and
    only fall back to Groq if Anthropic is entirely down. Cached -- providers just
    wrap stateless API clients, no reason to reconstruct them per call."""
    global _chain
    if _chain is None:
        _chain = ProviderChain([AnthropicProvider(), GroqProvider()])
    return _chain
