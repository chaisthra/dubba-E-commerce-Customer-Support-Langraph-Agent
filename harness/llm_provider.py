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
import re
import time
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
                # No temperature= here on purpose -- removed from anthropic-sdk-python's
                # Messages.create() signature entirely as of a release that shipped
                # mid-session (confirmed live via inspect.signature() in the deployed
                # container: SDK went 1.2.0 -> 1.3.0 between two of our own tests a few
                # minutes apart, both on requirements.txt's unpinned `anthropic`).
                # output_config (the new param that appeared) is unrelated -- reasoning
                # effort + structured-output schema, not sampling. Anthropic's own
                # stated fix is "omit entirely, control determinism via prompt" -- no
                # replacement kwarg exists. See log/DECISIONS.md.
                response = self._client.messages.create(
                    model=model, max_tokens=1024, system=system, messages=messages, **kwargs
                )
                return _normalize_anthropic(response, model)
            except (anthropic.APIStatusError, anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
                last_error = exc
                continue

        raise RuntimeError(
            f"AnthropicProvider: all models failed ({self.MODELS}); last error: {last_error}"
        ) from last_error


# Groq's 429 body includes a human-readable hint like "Please try again in 60ms"
# or "in 1.234s" -- parsed as a floor for the backoff sleep when present, since
# it's a real, model-specific signal from the account's own per-model TPM window
# (confirmed empirically: a real 429 named one specific model, "Limit 8000, Used
# 6372" -- Groq's rate limits are per-model, not account-wide, which is why
# retrying the SAME model after a short wait is worth doing before moving on).
_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)(ms|s)\b")


def _parse_retry_after_seconds(message: str) -> float | None:
    match = _RETRY_AFTER_RE.search(message)
    if not match:
        return None
    value, unit = match.groups()
    seconds = float(value) / 1000 if unit == "ms" else float(value)
    return seconds


class GroqProvider(Provider):
    name = "groq"

    # Rate-limit-specific retry knobs -- separate from the per-model fallback
    # loop below. A 429 is worth retrying the SAME model after a short wait
    # (the limit is per-model and per-minute, so it often clears fast).
    MAX_RATE_LIMIT_RETRIES = 3
    BASE_BACKOFF_SECONDS = 1.0
    # Groq's 429 body can suggest a genuinely long wait -- a real one seen this
    # session said "try again in 20m45.888s" (a per-model DAILY token cap, not a
    # per-minute one). _parse_retry_after_seconds honors that value as a floor
    # with no ceiling, so a single rate-limited model could block this whole
    # request for ~20 minutes before ever falling through to the next model in
    # MODELS -- confirmed live: a CI eval-gate run stalled ~23 minutes on exactly
    # this path. Capped here so MAX_RATE_LIMIT_RETRIES worth of waits on one
    # model costs at most MAX_BACKOFF_SECONDS * MAX_RATE_LIMIT_RETRIES (90s),
    # then falls through to the next model instead of blocking on a cap that
    # won't clear within this request's lifetime regardless of how long we wait.
    MAX_BACKOFF_SECONDS = 30.0

    # A 400 from disobeying a forced/required tool_choice is a genuinely
    # different failure than a rate limit -- not capacity-related, so no
    # backoff delay needed -- but it's also not necessarily permanent: the
    # SAME model, asked again, sometimes just complies the second time (this
    # is a real, observed instructor-flagged case: their harness gets this
    # same "wrong tool name" failure mode a genuine second attempt via an
    # MCP-level error-and-retry loop; ours can't do that -- Groq's own API
    # rejects the call server-side before we ever get a response object with
    # a tool-call to feed back as an error -- so this is the closest
    # equivalent: retry the model itself, not the tool-execution step).
    MAX_TOOL_VALIDATION_RETRIES = 2

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
            rate_limit_attempt = 0
            tool_validation_attempt = 0
            while True:
                try:
                    response = self._client.chat.completions.create(
                        model=model, messages=groq_messages, temperature=0, **kwargs
                    )
                    return _normalize_groq(response, model)
                except groq.RateLimitError as exc:
                    last_error = exc
                    rate_limit_attempt += 1
                    if rate_limit_attempt > self.MAX_RATE_LIMIT_RETRIES:
                        break  # exhausted retries on this model -- fall through to the next one
                    delay = min(
                        _parse_retry_after_seconds(str(exc))
                        or (self.BASE_BACKOFF_SECONDS * (2 ** (rate_limit_attempt - 1))),
                        self.MAX_BACKOFF_SECONDS,
                    )
                    time.sleep(delay)
                    continue  # retry the SAME model, not the next one -- per-model limit, likely to clear
                # Broad on purpose (and deliberately separate from RateLimitError
                # above, which needs a delay, not an immediate reattempt): a
                # model disobeying a forced/required tool_choice raises
                # BadRequestError, a sibling APIStatusError subclass, server-side
                # -- not capacity-related, so no delay, but worth a couple of
                # immediate reattempts on the SAME model before giving up on it,
                # since compliance is a per-call roll of the dice, not a fixed
                # property of the model (see MAX_TOOL_VALIDATION_RETRIES above).
                except groq.APIStatusError as exc:
                    last_error = exc
                    tool_validation_attempt += 1
                    if tool_validation_attempt > self.MAX_TOOL_VALIDATION_RETRIES:
                        break  # gave this model its chances -- fall through to the next one
                    continue  # retry the SAME model immediately, no backoff needed

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
        errors: list[tuple[str, Exception]] = []
        for provider in self.providers:
            try:
                return provider.create(system, messages, tools=tools, force_tool=force_tool, any_tool=any_tool)
            except Exception as e:
                errors.append((provider.name, e))
                continue

        # Every provider's failure preserved, not just the last one -- a bare
        # `raise last_error` here previously discarded every earlier provider's
        # error the moment a later one also failed, making it structurally
        # impossible to see WHY the primary (Anthropic) provider fell through,
        # only ever showing the final (Groq) failure. Chained via `from` onto
        # the last real exception so the traceback still has a real cause, not
        # just this summary string.
        summary = "; ".join(f"{name}: {err}" for name, err in errors)
        raise RuntimeError(f"ProviderChain: every provider failed -- {summary}") from errors[-1][1]


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


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_reasoning(text: str | None) -> str | None:
    """Groq's reasoning-capable models (the openai/gpt-oss family, Qwen) emit
    their chain-of-thought inline as literal <think>...</think> text in the
    completion by default -- `reasoning_format` (the API's own suppression
    knob) isn't even supported for gpt-oss ("reasons always-on and cannot be
    disabled" per Groq's own docs), so this can't be fixed by a request
    parameter for our model list; has to be stripped from the text itself.
    Confirmed necessary by a real leaked reply -- see log/DECISIONS.md."""
    if text is None:
        return None
    return _THINK_TAG_RE.sub("", text).strip()


def _normalize_groq(response, model: str) -> LLMResponse:
    message = response.choices[0].message
    tool_calls = [
        ToolCall(name=tc.function.name, input=json.loads(tc.function.arguments))
        for tc in (message.tool_calls or [])
    ]

    return LLMResponse(
        text=_strip_reasoning(message.content),
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
