"""
Anthropic-specific pieces that aren't part of the multi-provider abstraction
(harness/llm_provider.py): the real-tokenizer token count used by graph.py's
session-level token cap (Anthropic-specific by nature -- it has to match what
Anthropic's own context window actually counts, not an approximation), and the
PRIMARY_MODEL/FALLBACK_MODEL/MODEL_CHAIN constants AnthropicProvider is built from
(single source of truth so the "which 2 Anthropic models" answer lives in one place).

The actual multi-provider `create()` call path (Anthropic -> Groq fallback) lives in
harness/llm_provider.py now -- this file used to also hold that (as `complete()`),
before Groq was added as a fallback provider.
"""

from anthropic import Anthropic

_client = Anthropic()

PRIMARY_MODEL = "claude-sonnet-4-5-20250929"
FALLBACK_MODEL = "claude-haiku-4-5-20251001"
MODEL_CHAIN = (PRIMARY_MODEL, FALLBACK_MODEL)


def count_tokens(system: str, messages: list[dict], model: str = PRIMARY_MODEL) -> int:
    """Real tokenizer count via the Messages API (not a heuristic) -- used by
    graph.py's session-level token cap, which needs to match what actually drives
    cost and context-window limits, not an approximation of it."""
    if not messages:
        messages = [{"role": "user", "content": " "}]  # API requires >=1 message
    result = _client.messages.count_tokens(model=model, system=system, messages=messages)
    return result.input_tokens
