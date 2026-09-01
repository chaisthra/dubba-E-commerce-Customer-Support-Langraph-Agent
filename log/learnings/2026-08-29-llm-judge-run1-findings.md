# LLM-as-judge run 1 — two findings, different kinds

Real run of `evals/llm_judge.py` (3 runs/ticket) against the 15-ticket golden
dataset, 2026-08-29. Overall average 2.96/5 on the first pass — looked alarming
until actually reading the judge's reasoning per ticket, which split cleanly into
two very different causes. Documenting both, since only one of them is an agent
defect.

## Finding A — judge-design bug, not an agent bug (fixed)

Several order-status replies scored low (2-3) for including real, accurate details
— e.g. *"Mona Lisa's Smirk (**Vanilla Musk**)"* — that the judge flagged as
"unsupported by the reference." They weren't unsupported; they're the real item
names from `mock_data.py`'s `_ORDERS`. The problem was the reference material
itself: `llm_judge.py` only gave the judge my own hand-written `expected_answer`
text for order-status tickets (no policy doc applies), and my summaries had
abbreviated the scent/variant names away. The judge was doing its job correctly —
checking claims against what it was given — the reference was just incomplete
relative to the real data the agent legitimately had access to.

**Fix**: added an `order_id` field to the 10 golden-dataset tickets that concern a
specific order, and `llm_judge.py`'s `_build_reference()` now fetches the real
order record fresh via `mcp_server.mock_data.get_order()` and includes it
verbatim as reference, alongside (not instead of) the hand-written expected answer.
Re-running should raise several of the affected scores without the agent's actual
output changing at all — the fix is entirely on the judge/reference side.

## Finding B — a real, different agent-adjacent bug: retrieval recall miss on a covered topic

`membership_cancel_cost` (*"what is the membership cost if a member cancels their
account"*) scored **1/5**, consistently across all 3 runs. The reply:

> *"I don't see anything in our policies about membership costs or cancellation
> fees. This isn't something I have information on... please reach out to our team..."*

This directly contradicts `06_subscription_cancellation.md`, which states plainly:
*"There is no minimum commitment period, no cancellation fee..."* The judge (which
had the real doc as reference) caught this correctly.

**This is not the same failure as the customs-duties case** (which correctly
scored 5 for an honest "not covered" answer). Customs duties are genuinely absent
from all 6 policy docs. Cancellation fees are explicitly addressed. Checking the
actual `search_policy` call from the trajectory-eval run that produced this reply:
the query scored the relevant "How to Cancel" chunk at **0.51-0.54** — all below
`MIN_SIMILARITY=0.6` (`rag/retriever.py`). `search_policy` returned zero chunks,
and the agent correctly applied the "never fabricate on an empty retrieval"
behavior we deliberately built into `RESPOND_SYSTEM_PROMPT` earlier this session
(the customs-honest-gap fix) — except here, that behavior produces a **wrong**
answer, because the topic isn't actually uncovered, retrieval just missed it for
this specific phrasing.

**The real problem**: the agent (and the honest-gap instruction) has no way to
distinguish "genuinely uncovered" from "covered, but this query's embedding didn't
clear the similarity threshold" — both look identical from inside the turn (an
empty `search_policy` result). This is the same underlying RAG-quality gap already
flagged in `log/loophole.md` ("RAG chunking dilutes multi-criterion sections"), now
with a second, concrete real-world consequence beyond the "already lit" case:
retrieval misses don't just return the wrong chunk, they can make a correctly-
designed honesty behavior actively wrong.

**Not fixed as part of this eval run** — this is a retrieval-quality problem
(chunking granularity and/or `MIN_SIMILARITY` tuning), not a prompt-instruction
problem, and touching either risks the same kind of regression this session already
saw once with `MIN_SIMILARITY`. Flagged in `log/loophole.md` for the next RAG-focused
session rather than patched reactively here.
