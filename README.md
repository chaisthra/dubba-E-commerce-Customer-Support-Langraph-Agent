# Dubba — E-Commerce Order-Support Agent

Dubba sells hand-poured, painting-inspired candles (Mona Lisa's Smirk, Starry Night
Swirl, The Scream Queen, and similar). This is a harness-controlled, tool-connected,
RAG-grounded support agent for it: customers log in, then ask about order status,
delivery issues, refunds, or their subscription/account, across a real multi-turn
conversation. The agent classifies each message, looks up real (mock) order and
account data through permission-scoped MCP tools, retrieves grounded policy answers
from a small RAG index, and remembers prior tickets across sessions — but at every
step, a proposed action only executes if the harness's own code approves it. The
model never decides that on its own.

## Prerequisites

- Python 3.10+ (developed against 3.13/3.14)
- An [Anthropic API key](https://console.anthropic.com/)
- A [Langfuse](https://cloud.langfuse.com) account (free tier is fine) for tracing
- No Docker, no external DB server — Chroma runs in-memory and long-term memory is a
  local SQLite file, both created automatically on first run.

## Setup

**macOS / Linux:**
```bash
git clone https://github.com/chaisthra/dubba-E-commerce-Customer-Support-Langraph-Agent.git
cd dubba-E-commerce-Customer-Support-Langraph-Agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# now edit .env and fill in ANTHROPIC_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/chaisthra/dubba-E-commerce-Customer-Support-Langraph-Agent.git
cd dubba-E-commerce-Customer-Support-Langraph-Agent
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# now edit .env and fill in ANTHROPIC_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
```

`.env` expects exactly these variable names:
```
ANTHROPIC_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

## Running it

```bash
python main.py                # LangGraph mode (default) -- real MCP tool calls + RAG
python main.py --mode=loop    # while-loop mode -- harness-pattern demo, no tools
```

Start with plain `python main.py` — it's the functional agent. `--mode=loop` is kept
in the repo as a from-scratch demonstration of the harness pattern itself (see
"Why two harness implementations" below), not a second product mode.

Log in with any mock account from `mcp_server/mock_data.py`, e.g.:
```
Email: asha.rao@example.com
Access code: 4821
```
Type `exit`, `quit`, or `bye` to end a session (this is also when long-term memory
gets written).

**To see the required rejection demo live**: ask about an order ID that belongs to a
*different* mock customer (e.g. log in as `asha.rao@example.com` but ask about
`ORD1003`, which belongs to `ben.oliver@example.com`) — the harness rejects the tool
call before it runs, not the model.

**To see the RAG "honest gap" case**: ask something like "do I have to pay customs
fees on an international order" — none of the 6 policy docs cover customs, so the
agent should say so rather than stretch the pricing doc to answer it.

## Project structure

```
main.py              entry point -- login, then the turn loop (--mode=loop|graph)
harness/
  loop.py             stage-1 hand-rolled while-loop harness (no tools)
  graph.py             LangGraph harness -- real MCP tool calls, permission-scoped
  auth.py               mock login (email + access code)
  permissions.py        the permission-scoping check -- one clear function per action type
  schema.py              shape/type validation for a proposed action
  llm_client.py           Anthropic provider abstraction, primary+fallback model chain
  prompts.py               versioned system prompts
  mcp_client.py             async stdio client wrapper for the MCP server subprocess
mcp_server/
  server.py            MCP server exposing lookup_order, check_account_status, search_policy
  mock_data.py           mock order + account DB, behind get_order()/get_account()
rag/
  retriever.py          Chroma + sentence-transformers retrieval over policy_docs/
  policy_docs/            6 policy docs (refunds, returns, delays, pricing, suspension, subscriptions)
memory/
  store.py              long-term ticket history (SQLite this phase -- see below)
requirements.txt     Python dependencies
.env.example          template for .env -- exact variable names the code expects
```

## Why I built the harness this way

**The core rule, unchanged everywhere in this repo**: the LLM proposes an action via
structured output; harness code checks it against real state before anything
executes. A system prompt telling the model "only look up the customer's own order"
is a request, not an enforcement mechanism — it can be ignored or bypassed under
adversarial input. `harness/permissions.py`'s `check_action_permission()` (and the
per-tool `check_order_permission()` / `check_account_permission()`) run against real
session state every single time, independent of what the model claims. That's also
why the check lives in the harness, not inside `mcp_server/server.py`'s tool
functions themselves — bolting a permission check inside the tool function, rather
than enforcing it as a real boundary before the tool is ever called, is the specific
failure mode the assignment flags as the most common way to lose points on this.
In the LangGraph implementation this is architecturally a **conditional edge**
(`route_after_propose`) between the node that proposes a tool call and the node that
executes it — not logic buried inside either node.

**Two harness implementations, on purpose.** `harness/loop.py` is a hand-rolled
while-loop; `harness/graph.py` is the same principle rebuilt in LangGraph once real
MCP tool calls were needed. Building the while-loop first made the
harness-controls-execution boundary maximally visible (one exact line, one file,
nothing framework-shaped in between) before rebuilding the identical enforcement
boundary as a graph edge — so both shapes of the same pattern are demonstrated, not
just one.

**How ticket types were scoped.** Five categories: `order_status`, `delivery_issue`,
`refund_request`, `subscription_account`, and a deliberate `other` bucket for
anything that doesn't cleanly fit. A single message can touch more than one category
(e.g. "my order is late AND I want a refund") — `classify` returns every category
it detects, and the harness processes them one at a time, in the same turn,
composing one final reply. The `other` bucket exists so the agent can be honest about
being out of scope rather than forcing a bad-fit classification, the same "honest
gap" principle applied to RAG retrieval below.

**One specific permission-boundary decision.** `check_account_permission()` rejects
any `check_account_status` call for a `customer_id` other than the authenticated
session's own — even though a well-behaved model would only ever ask about itself.
This scoping exists in the codebase from the moment the tool was written; there was
never an unscoped version of it, even temporarily, because correctness can't be
allowed to depend on the model happening to behave.

**RAG's honest gap is deliberate, not missing coverage.** None of the 6 policy docs
mention international customs fees or import duties — verified empirically (only
weak, topically-adjacent matches surface for that query, no strong hit). The system
prompt explicitly forbids stretching an adjacent doc (e.g. the general shipping-fee
doc) to sound like it answers something more specific it doesn't cover.

**Long-term memory: SQLite now, Postgres and AWS RDS later.** `memory/store.py`
exposes `get_ticket_history()` / `save_ticket_summary()` as the interface every
caller uses — swapping the backing store later only touches this one file. The same
principle applies to the mock order/account data in `mcp_server/mock_data.py`:
everything above it calls `get_order()` / `get_account()`, never the underlying
dicts directly.

**Every LLM call has a fallback**, not a single hard-coded model:
`claude-sonnet-4-5-20250929` primary (chosen for tool-calling reliability, since the
harness's correctness depends on clean structured output), `claude-haiku-4-5-20251001`
fallback on a hard call failure only (timeout, API error, malformed response) — never
because the harness disliked a valid answer's quality.

**Every step is traced in Langfuse**: one trace per user turn, spans per category and
per tool call, so grounding and permission decisions are checkable, not just claimed.
