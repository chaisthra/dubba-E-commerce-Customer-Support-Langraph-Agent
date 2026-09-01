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

## Architecture

![Dubba architecture: main.py, the LangGraph harness (classify, propose, permission check, execute_tool, evaluate, respond, escalate, reject_tool), the MCP server and its three tools, RAG, memory, and Langfuse tracing](assets/architecture.png)

### LangGraph nodes (`harness/graph.py`, `--mode=graph`, the default/functional agent)

| Node | What it does |
|---|---|
| `classify` | LLM call. Detects every ticket category the message touches (a message can span more than one), upfront. |
| `propose` | LLM call, forced to pick exactly one of: `respond`, `ask_clarification`, `lookup_order`, `check_account_status`, `search_policy`. Only proposes — never executes anything itself. |
| *(permission check)* | Not a node — a **conditional edge** (`route_after_propose`) between `propose` and `execute_tool`. Runs `harness/permissions.py` against the real session (does this order/customer ID belong to the logged-in customer?) before any tool call happens. |
| `execute_tool` | Only reached if the permission check passed. Awaits the actual MCP tool call over stdio and records the result. |
| `evaluate` | LLM-as-judge call. Given what's been gathered so far, decides whether there's enough to answer the customer, or whether another tool call is needed (capped at 3 tool calls total per turn, shared across all categories — not a per-category budget). |
| `refund` | **No LLM call.** Assignment 3's HITL entry point — reached from `evaluate` the moment a `refund_request` category has an order looked up, ahead of the `sufficient`/cap/propose branches. Computes eligibility deterministically from real order/account data (`mcp_server/mock_data.py`), creates a `PendingAction` for human review when warranted, and writes its own customer-facing message directly (same bypass-`finalize` mechanism as `ask_clarification`). See `refunds/`. |
| `respond` | Does **not** write customer-facing prose. Marks the category as answer-ready (deferred to `finalize`) — except `ask_clarification`, which is passed through as-is immediately (short, category-specific, no duplication risk). |
| `escalate` | Reached when `evaluate` still isn't satisfied and the tool-call cap is hit. Deterministic HITL message: a ticket is created, a human follows up from `dubba.support@dubba.com`, details arrive by email. |
| `reject_tool` | Reached when the permission check denies a proposed tool call. Deterministic rejection message — no retry of a denied action within that category, ever. |
| `finalize` | The **one** LLM call that writes customer-facing prose, once every category has been through the loop — covering every answer-ready category together in a single, non-repetitive reply. |

Categories are processed one at a time: after `respond` / `escalate` / `reject_tool`,
the graph loops back to `propose` for the next detected category. Once every category
is done, `finalize` runs (if any are answer-ready) and the turn ends. All
evaluate/tool-call work for every category finishes before any prose is written —
this is what stops two categories about the same underlying issue (e.g. a damaged
candle raising both `delivery_issue` and `refund_request`) from each independently
writing near-identical answers.

## Prerequisites

- Python 3.10+ (developed against 3.13/3.14)
- An [Anthropic API key](https://console.anthropic.com/)
- A [Langfuse](https://cloud.langfuse.com) account (free tier is fine) for tracing
- Docker (for local Postgres+pgvector — `docker compose up -d`) — see
  [Assignment 2](#assignment-2--observability-evaluation-regression-gate-aws-deployment)
  below for the full setup including tracing, evals, and AWS deployment.

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

`.env` expects exactly these variable names (see `.env.example`):
```
ANTHROPIC_API_KEY=
GROQ_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=http://localhost:3000
DATABASE_URL=postgresql://dubba:dubba@localhost:5433/dubba
```

Start Postgres+pgvector before running the app:
```bash
docker compose up -d
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
**Ending a session** — type `exit`, `quit`, or `bye`. You'll be asked "was your issue
resolved today?" — the answer is recorded as `closure_reason` (`resolved` /
`abandoned`) when the session is written to long-term memory (the `tickets` table
in Postgres). Closing the terminal or hitting Ctrl+C also works —
that's caught explicitly too, saved as `abandoned` rather than silently dropped, and
the MCP server subprocess is shut down cleanly either way.

**To see the required rejection demo live**: ask about an order ID that belongs to a
*different* mock customer (e.g. log in as `asha.rao@example.com` but ask about
`ORD1003`, which belongs to `ben.oliver@example.com`) — the harness rejects the tool
call before it runs, not the model.

**To see the RAG "honest gap" case**: ask something like "do I have to pay customs
fees on an international order" — none of the 6 policy docs cover customs, so the
agent should say so rather than stretch the pricing doc to answer it.

## Project structure

```
main.py              CLI entry point -- login, then the turn loop (--mode=loop|graph)
api.py               HTTP entry point -- FastAPI, what runs in the ECS/Fargate deployment
Dockerfile           builds api.py into the image pushed to ECR
harness/
  loop.py             stage-1 hand-rolled while-loop harness (no tools)
  graph.py             LangGraph harness -- real MCP tool calls, permission-scoped
  auth.py               mock login (email + access code)
  permissions.py        the permission-scoping check -- one clear function per action type
  schema.py              shape/type validation for a proposed action
  llm_client.py           token counting, PRIMARY_MODEL/FALLBACK_MODEL constants
  llm_provider.py           Anthropic+Groq provider abstraction with automatic fallback chain
  summarizer.py             rolling short-term-memory summarization (harness-computed structured fields + one LLM call for prose)
  prompts.py               versioned system prompts (log/prompts/CHANGELOG.md)
  mcp_client.py             async stdio client wrapper for the MCP server subprocess
mcp_server/
  server.py            MCP server exposing lookup_order, check_account_status, search_policy
  mock_data.py           mock order + account DB, behind get_order()/get_account()
rag/
  store.py               Postgres+pgvector backend (policy_chunks table)
  retriever.py            sentence-transformers embeddings + chunking over policy_docs/
  policy_docs/              6 policy docs (refunds, returns, delays, pricing, suspension, subscriptions)
memory/
  checkpointer.py        LangGraph's Postgres checkpointer (short-term, per-thread_id state)
  store.py                long-term ticket history (Postgres `tickets` table)
evals/
  golden_dataset.json     15 test tickets, expected_trajectory + expected_answer per ticket
  trajectory_eval.py       rule-based trajectory eval (section 2.2)
  llm_judge.py              fact-checking LLM-as-judge (section 2.3)
  combined_score.py          combines both into one score, baseline save/compare, the CI gate's actual pass/fail logic
  baseline.json               stored known-good score, read by CI
.github/workflows/
  ci-cd.yml               eval-gate -> build-and-push -> deploy pipeline
  deploy-no-gate.yml        same pipeline minus eval-gate, for the "what a missing gate costs" demo
infra/
  aws-network-rds-ecr.yaml   private subnets, RDS PostgreSQL+pgvector, ECR (deploy first)
  aws-ecs-alb.yaml             ALB, ECS Fargate service/task, autoscaling (deploy second)
  langfuse-ec2.yaml             self-hosted Langfuse on EC2 (auxiliary tracing infra, separate from the graded deployment)
  teardown-aws-deploy.sh          tears down the two AWS deploy stacks
  teardown-langfuse-ec2.sh          tears down the Langfuse EC2 stack
  push-secrets-to-ssm.sh             pushes ANTHROPIC_API_KEY/GROQ_API_KEY to SSM
log/
  DEV_LOG.md, DECISIONS.md, todos.md, loophole.md, learnings/   dated project history, decisions, open items, and the required "confident wrong path" writeups
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

**Long-term and short-term memory: Postgres, RDS in production.** `memory/store.py`
exposes `get_customer_history()` / `save_ticket_summary()` as the interface every
caller uses; `memory/checkpointer.py` wraps LangGraph's own `AsyncPostgresSaver` for
short-term, per-`thread_id` conversation state. Both read `DATABASE_URL` alone —
local Docker Postgres today, AWS RDS in the Assignment 2 deployment, no code change
between them. The same principle applies to the mock order/account data in
`mcp_server/mock_data.py`: everything above it calls `get_order()` / `get_account()`,
never the underlying dicts directly.

**Every LLM call has a fallback, across two providers, not a single hard-coded
model.** `harness/llm_provider.py`'s `ProviderChain`: Anthropic first
(`claude-sonnet-4-5-20250929` primary, `claude-haiku-4-5-20251001` fallback — chosen
for tool-calling reliability, since the harness's correctness depends on clean
structured output), Groq second, reached only once the entire Anthropic provider is
exhausted, not on a single model's hiccup. `temperature=0` on every call, both
providers. The rolling short-term-memory summarizer (`harness/summarizer.py`)
deliberately stays on Groq alone, regardless of Anthropic's health — background
housekeeping, not a customer-facing answer.

**RAG runs against Postgres+pgvector**, not a separate vector-store process —
`rag/store.py`'s `policy_chunks` table lives in the same Postgres instance as the
checkpointer and `tickets` table (local Docker today, the same RDS instance in the
Assignment 2 deployment). Chunked by markdown `## ` section, embedded with
`sentence-transformers/all-MiniLM-L6-v2`, retrieved via pgvector's `<=>` cosine-
distance operator with a `MIN_SIMILARITY` floor enforced in code, not left to a
prompting instruction.

**Every step is traced in Langfuse**: one trace per user turn, spans per category and
per tool call, so grounding and permission decisions are checkable, not just claimed.
See [Assignment 2](#assignment-2--observability-evaluation-regression-gate-aws-deployment)
below for the full tracing/eval/gate/deployment setup.

---

## Assignment 2 — Observability, Evaluation, Regression Gate, AWS Deployment

### 1. Tracing setup

Every LangGraph node (`harness/graph.py`) and MCP tool call produces a real
Langfuse span — `ticket-turn` (the whole turn) as the parent, with
`classify-intent`, `propose-action`, `execute-tool:*`, `evaluate-sufficiency`,
`respond-decision`, `finalize-response`, `rolling-summarize`, and the HITL/
rejection spans nested underneath. Nothing hand-rolled — the `langfuse` SDK's
`start_as_current_observation()` context manager, `get_client()` once at import
time, per `harness/graph.py`.

**Two Langfuse deployments exist, for different purposes — don't confuse them:**

| | Purpose | `.env` values |
|---|---|---|
| **Local self-hosted** (`docker compose`, cloned sibling repo — see `log/DECISIONS.md`) | Local dev, `python main.py`, this repo's own eval scripts | `LANGFUSE_BASE_URL=http://localhost:3000` + that instance's project keys |
| **EC2** (`infra/langfuse-ec2.yaml`) | Auxiliary — a real, internet-reachable instance for manual inspection; **not** used by CI (see below) | Fetch from SSM: `aws ssm get-parameter --name /langfuse-ec2/base-url ...` etc. |

Exact env vars the code reads (`.env.example`):
```
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=http://localhost:3000
```

**One-time setup for local self-hosted Langfuse**: clone `langfuse/langfuse` as a
sibling directory, `docker compose up -d` there (6 containers: Postgres,
ClickHouse, Redis, MinIO, langfuse-web, langfuse-worker), then copy the
headless-init project keys from that clone's own `.env` into this repo's `.env`.

**Why CI doesn't need Langfuse reachable at all**: the SDK degrades gracefully
with no credentials set — logs a warning, swaps in a no-op OTel tracer, never
raises (verified directly against the installed SDK, `langfuse/_client/client.py`).
`eval-gate` (below) never sets `LANGFUSE_*`, so it runs identically whether or not
Langfuse is reachable from GitHub's runners — which it deliberately isn't, since
Langfuse EC2 is kept private (see `log/DECISIONS.md`).

### 2. Trajectory eval

```bash
python evals/trajectory_eval.py
```
Runs all 15 test tickets in `evals/golden_dataset.json` (5 ticket categories:
`order_status`, `delivery_issue`, `refund_request`, `subscription_account`,
`other`) through the real agent, checks `session_tool_log` against each ticket's
`expected_trajectory` (superset check — extra tool calls are fine, missing
required ones fail), writes full results to `evals/trajectory_eval_results.json`.
Needs Postgres up (`docker compose up -d`) and real `ANTHROPIC_API_KEY`/
`GROQ_API_KEY` — same requirements as `python main.py`.

The dataset also carries `expected_answer`/`expected_source`/`expected_escalation`
per ticket (used by the LLM judge, next) and `notes` explaining what each ticket
specifically tests — see the file directly for the full picture, including two
tickets deliberately checking the same lit-candle-disqualification and honest-gap
behaviors documented in `log/learnings/`.

```bash
python evals/llm_judge.py --runs 5
```
Fact-checking rubric (1–5, `temperature=0`), run against `evals/golden_dataset.json`
+ `evals/trajectory_eval_results.json`'s actual replies + the real policy doc text
+ the real order record (`mcp_server/mock_data.get_order()`) as reference — never
the agent's own reply used as its own reference. Multiple runs per ticket
(default 3, `--runs 5` for a closer look) because judge scores are genuinely
non-deterministic even at `temperature=0` — confirmed directly, several tickets
showed real run-to-run score variance (see `log/learnings/2026-08-29-llm-judge-run2-findings.md`).

```bash
python evals/combined_score.py
```
Runs both of the above fresh and combines them (50% trajectory pass rate + 50%
judge average, both rescaled to 0–100) into one number — this combined score is
what the baseline and the CI gate actually track, not either signal alone (see
the Defensible justifications below for why).

**Where the baseline lives**: `evals/baseline.json`, a committed JSON file (current
value: `combined_score: 71.92`). To establish a new one after a real, reviewed
improvement:
```bash
python evals/combined_score.py --save-baseline
```

### 3. CI gate configuration

**File**: `.github/workflows/ci-cd.yml`, job `eval-gate`. **Pass/fail logic lives
in Python** (`evals/combined_score.py --gate`), not the YAML, per the assignment's
own build guide — the workflow just runs the script and lets its exit code decide.

**What it checks**: the combined score (trajectory pass rate + LLM-judge average,
50/50) against `evals/baseline.json`, using 3 judge runs per ticket (not 5 — kept
lower than the manual-inspection default to bound CI time/cost, still satisfies
"more than once" per Session 3's own non-determinism finding).

**Threshold**: fails if the combined score drops **more than 10 points** from
baseline (`REGRESSION_THRESHOLD_POINTS = 10` in `evals/combined_score.py`) — a
**relative** check against the stored baseline, not a fixed pass/fail bar. See
Defensible justification #1 below for why 10.

**The `needs:` chain is the entire mechanism**: `build-and-push` needs `eval-gate`,
`deploy` needs `build-and-push`. Nothing downstream runs if the gate script exits
non-zero — no conditional logic anywhere in the YAML.

**Authentication**: OIDC only. `eval-gate` assumes `dubba-github-actions-deploy`
(trust policy scoped to this exact repo, `infra/github-oidc-trust-policy.json`)
via `aws-actions/configure-aws-credentials@v4`, then fetches
`ANTHROPIC_API_KEY`/`GROQ_API_KEY` from AWS SSM Parameter Store
(`/dubba/anthropic-api-key`, `/dubba/groq-api-key`) at runtime — **no static AWS
keys anywhere in this repo**, confirmed by grep (`AWS_ACCESS_KEY_ID`/
`AWS_SECRET_ACCESS_KEY` appear nowhere in `.github/workflows/`).

**Demoing a live regression** (the required "watch it actually block" evidence):
add `AGENT_REGRESSED: "true"` to `eval-gate`'s `env:` block in `ci-cd.yml` (one
line), push, watch the run fail on GitHub, then revert. `harness/graph.py` reads
this exact env var and strips `lookup_order` out of the tools the agent can call
entirely — not discouraged in prose, structurally absent from the tool-calling API
call itself — so every ticket needing an order lookup fails the trajectory check.
`workflow_dispatch`'s `agent_regressed` input does the same thing without a
commit, for quick manual testing. `.github/workflows/deploy-no-gate.yml` is the
same pipeline with `eval-gate` entirely removed (triggers only on a
`no-gate-demo` branch or manual dispatch, never `main`/`w1a-harness`) — the "what
does a missing gate actually cost" half of the demo.

### 4. Reproducing the before/after comparison

```bash
python evals/before_after.py
```
Runs a fresh eval, compares it against the saved baseline
(`evals/baseline.json` + the per-ticket snapshots `evals/baseline_trajectory_results.json`/
`evals/baseline_judge_results.json`, all written together by
`python evals/combined_score.py --save-baseline`), and prints **both** the
aggregate before/after numbers (`BEFORE: combined=71.92 (...) / AFTER:
combined=52.1 (...) / DROP: combined score dropped 19.82 points`) **and** a
per-ticket diff — which specific tickets' trajectory pass/fail status flipped,
and which tickets' judge average moved by a full point or more (a full point,
not any movement at all, to distinguish a real change from ordinary judge
run-to-run noise — see the non-determinism finding in justification #1 below).

To demo the regressed case specifically:
```bash
AGENT_REGRESSED=true python evals/before_after.py
```

To diff two already-saved result sets without re-running anything (e.g. CI
artifacts from two different pipeline runs):
```bash
python evals/before_after.py --after-only --after-trajectory <path> --after-judge <path>
```

### 5. Defensible justifications

**Why the regression threshold is 10 points, on a 50/50 trajectory+judge combined
score:**
A tighter threshold (e.g. 2–3 points) would fail the gate on ordinary LLM-judge
noise alone — this session's own judge runs showed real tickets swinging 2–3
points between identical, unchanged runs (`log/learnings/2026-08-29-llm-judge-run2-findings.md`:
several tickets scored anywhere from 2 to 5 across 5 runs with nothing about the
agent changed). A threshold that tight would make the gate cry wolf on noise, not
signal — exactly the kind of gate nobody trusts after the third false alarm. A
much looser threshold (e.g. 30+ points) would let a real, meaningful regression
through — our own `lit_candle_return_denied` finding (a genuine policy-compliance
failure, not noise) accounts for real, multi-point swings in the judge component
on its own. 10 points sits above the observed noise floor for a single ticket's
judge variance, but well below what one or two genuinely broken tickets would
cost the aggregate score — tight enough to catch a real regression, loose enough
not to flap on judge noise.

**What the gate actually checks, and what that means it can't catch:**
It checks a **combined** trajectory (rule-based, did the required tools get
called) and LLM-judge (fact-checking, is the answer grounded) score — deliberately
both, not one alone, because this session found real cases where each one misses
what the other catches: `lit_candle_return_denied` had a perfect trajectory
(exactly the required tools, in order) but a wrong, policy-contradicting answer —
a trajectory-only gate would have waved it through clean. Conversely, a
judge-only gate has no way to catch a skipped or wrong-argument tool call whose
final answer still happens to read plausibly — the whole "confident wrong path"
premise this assignment is built around. What it still can't catch: cost/latency
regressions (a correct but 10x-slower response scores the same), UI/UX
regressions (there is none to check), anything the 15-ticket golden dataset
doesn't cover (a genuinely novel failure mode outside these 15 scenarios), and —
inherent to any LLM-judge — some irreducible run-to-run noise even after
averaging 3 runs.

### 6. AWS architecture

![AWS deployment architecture: GitHub Actions authenticating via OIDC, pushing to ECR and deploying to ECS Fargate behind an ALB, with RDS PostgreSQL+pgvector in private subnets, secrets from SSM Parameter Store, and Application Auto Scaling on ECS CPU utilization](assets/aws-deployment-architecture.png)

Deployed in two ordered CloudFormation stacks (ECS needs a real image already in
ECR before it can report healthy — ECR has to exist, and be populated, before the
ECS stack can succeed):

| Resource | Name / how to fetch |
|---|---|
| VPC | `vpc-01bf8f177b6df0109` (existing default VPC — also hosts Langfuse EC2) |
| ECR repository | `dubba` (SHA-tagged, `IMMUTABLE`) |
| RDS instance identifier | `dubba-rds` (PostgreSQL 16.15, `db.t4g.micro`, single-AZ, private subnet, `pgvector` extension) |
| ECS cluster | `dubba-cluster` |
| ECS service | `dubba-service` |
| ECS task definition family | `dubba-task` (Fargate, 1 vCPU / 3 GB) |
| Container name | `dubba` |
| ALB DNS name | `aws cloudformation describe-stacks --stack-name dubba-ecs-alb --query "Stacks[0].Outputs[?OutputKey=='ALBDNSName'].OutputValue" --output text` |
| Autoscaling | Target-tracking, `ECSServiceAverageCPUUtilization`, target **60%**, min **1** / max **4** tasks (`infra/aws-ecs-alb.yaml`'s `MinTaskCount`/`MaxTaskCount`/`CPUTargetValue` parameters) |

**pgvector wiring**: `rag/store.py`'s `connect()` runs `CREATE EXTENSION IF NOT
EXISTS vector` and creates `policy_chunks` (an HNSW-indexed `vector(384)` column)
against whatever `DATABASE_URL` points at — local Docker Postgres today, `dubba-rds`
in this deployment, no code difference between them. The ECS task definition
injects `DATABASE_URL` via the `secrets` field (SSM parameter `/dubba/database-url`,
built from the RDS endpoint output + the master password supplied at RDS-stack
deploy time — never baked into the task definition as plaintext).

**Networking**: ECS tasks sit in the VPC's existing **public** subnets with a
public IP, but are reachable only through the ALB's security group — not
`0.0.0.0/0`, not even a specific IP (`infra/aws-ecs-alb.yaml`'s
`DubbaTaskSecurityGroup`). Chosen over a private subnet + NAT Gateway to avoid
~$32–45/month in NAT Gateway cost for a teaching deployment while still meeting
the "only the ALB can reach the task directly" requirement via security groups
rather than subnet privacy. RDS, by contrast, sits in genuinely private subnets
with no internet route at all (`infra/aws-network-rds-ecr.yaml`) — required, not
optional, since RDS itself needs no outbound internet access the way ECS does.

### 7. Deploy pipeline

`.github/workflows/ci-cd.yml` — `eval-gate` → `build-and-push` → `deploy`.
`aws-actions/configure-aws-credentials@v4` + OIDC (`role-to-assume:
${{ vars.AWS_DEPLOY_ROLE_ARN }}`) in every AWS-touching job; **no
`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` anywhere in this repository** —
verified directly, not just claimed:
```bash
grep -rn "AWS_ACCESS_KEY_ID\|AWS_SECRET_ACCESS_KEY" .github/ infra/
# (no matches)
```
Images tagged by `${{ github.sha }}`, never `latest`-only (§2.6.1's audit-trail
requirement).

**Required GitHub repo variables** (Settings → Secrets and variables → Actions →
Variables — set once the AWS stacks above exist):
```
AWS_DEPLOY_ROLE_ARN, AWS_REGION, ECR_REPOSITORY, ECS_CLUSTER, ECS_SERVICE, ECS_TASK_FAMILY, CONTAINER_NAME
```
No GitHub Actions *Secrets* are used for AWS or the LLM provider keys — everything
credential-shaped is fetched from SSM Parameter Store at runtime via the OIDC
role (`infra/push-secrets-to-ssm.sh` pushes `ANTHROPIC_API_KEY`/`GROQ_API_KEY`;
the RDS stack's own deploy step pushes `DATABASE_URL`).

### 8. Teardown

```bash
./infra/teardown-aws-deploy.sh       # ALB, ECS, autoscaling, then private subnets, RDS, ECR (in that order -- ECS stack depends on the network stack's resources)
./infra/teardown-langfuse-ec2.sh     # separate stack -- Langfuse EC2, not touched by the above
```
Both are plain `aws cloudformation delete-stack` + `wait stack-delete-complete` —
tested against a real deployed stack, not just described (see this repo's own
`log/DEV_LOG.md` for the Langfuse EC2 deploy/redeploy history this was exercised
against). `EmptyOnDelete: true` on the ECR repository (`infra/aws-network-rds-ecr.yaml`)
means teardown succeeds even with images still pushed to it — no manual
`aws ecr batch-delete-image` step needed first.

### 9. Bedrock-ready path

Not attempted this assignment — explicitly deferred, not forgotten. §2.6.4 is
optional and out of scope for this submission; `harness/llm_provider.py`'s
existing `Provider`/`ProviderChain` abstraction (built for the Anthropic/Groq
fallback chain) is the natural seam a future `BedrockProvider` stub would slot
into, whenever that gets picked up.

## Assignment 3 — Refund Node + Human-in-the-Loop Approval

### 1. Node, not a second agent

`refund` is a node inside the same LangGraph, not an A2A handoff to a separate
service. Walked the four questions honestly (different tools/data/authority?
one-sentence job? real current bottleneck? can we afford the failure mode?) — the
failure mode is what settles it: a network hop that can silently retry is real
risk taken on for architectural tidiness, on a flow whose entire purpose is not
refunding twice. See `refund_node`'s own docstring in `harness/graph.py`.

### 2. Graph changes

One new node (`refund`), one new branch on the existing `route_after_evaluate`
edge, checked first — ahead of `sufficient`/cap/propose, since refund eligibility
is a deterministic computation on data already in state, not something worth
asking `evaluate`'s LLM to judge:

```
classify -> propose <-> evaluate -> [route_after_evaluate] -> refund   (no LLM call)
                                                             -> propose (under cap)
                                                             -> respond
                                                             -> escalate
```

`refund_node` has **no LLM call**. The model already did its reasoning at
`propose_node` (deciding to call `lookup_order`); this node just computes
eligibility from real data (`mcp_server/mock_data.py`'s `refund_window_eligible`/
`account_standing_ok`/`refund_window_days_remaining`, already derived from real
dates before this assignment even started) and writes its own customer-facing
text directly via the same `_advance()` helper `ask_clarification`/`escalate`/
`reject_tool` already use to bypass `finalize_node`'s LLM rephrasing entirely —
money-related text a human should be able to trust matches the actual decision,
not an LLM's paraphrase of it.

`refund_type` (standard / damaged-in-transit / non-delivery) has one honest
limitation: order data alone can only detect non-delivery (missing shipped/
delivery dates). Distinguishing a damaged item from a plain return needs some
signal beyond dates, and the only place "damage" is expressed today is the
customer's own message — so `refund_node` does a plain keyword check on
`state["user_message"]` (`_DAMAGE_KEYWORDS` in `harness/graph.py`). A missed
phrasing just means a real damage claim falls through to the `standard` path
(still creates a `PendingAction` for human review) rather than getting the more
permissive window-check exemption `damaged_in_transit` gets.

### 3. The `refunds/` package

- `refunds/schemas.py` — `PendingAction`, `RefundDecision`, and this app's own
  `RefundEligibilityResult` (refund_node's internal eligibility computation,
  deliberately named apart from `RefundDecision` — see that file's own
  docstring for why the two aren't the same thing despite the original design
  doc using one name for both).
- `refunds/approval_gate.py` — the `ApprovalStore` interface (`save`/`get`/
  `get_active_for_resource`/`all_pending`/`transition`) plus two backends:
  `JsonFileStore` (local dev, `fcntl`-locked read-modify-write) and
  `DynamoDBStore` (AWS, `ConditionExpression`-based). No separate Postgres
  `refund_tickets` table — this store *is* the durable record, including
  duplicate detection (`get_active_for_resource`, correlated on `order_id`,
  never ticket/session id). Selected by `HITL_STORE_BACKEND` (unset/`json`
  locally, `dynamodb` in the deployed task).
- `refunds/execute.py` — `execute_refund()`, the re-validation step between
  "approved" and "executed." Approving is not executing; state can change in
  between, so this re-fetches live order/account data and refuses on drift
  (`account_standing`, `order_status`) rather than trusting a stale snapshot.

`PendingAction`/`RefundDecision`'s field shapes are a **fixed external
contract** (an instructor-provided `hitl_cli.py`/`approval_gate.py` interface
this app interoperates with), not something redesigned here.

### 4. Reviewer CLI

```bash
python hitl_cli.py list
python hitl_cli.py approve <action_id> --reviewer alice
python hitl_cli.py execute <action_id> --reviewer alice   # re-validates against live state
python hitl_cli.py reject  <action_id> --reviewer alice
```

`approve` and `execute` are deliberately separate steps, not one combined
action — the graded bug scenario below (approve, then the account gets
suspended, then execute) only makes sense with a real window between them for
state to change.

### 5. Graded bug scenarios

`python evals/refund_bug_scenarios.py` — four real repros, not descriptions of
intended behavior, run against the actual `JsonFileStore`/`execute_refund`:

1. Two refund requests on the same order — the second finds the first via
   `get_active_for_resource`, never creates a duplicate.
2. Two reviewers approving the same `PendingAction` simultaneously — 10 real
   concurrent subprocesses race `transition()`; exactly one wins, verified
   directly, not assumed from the locking design.
3. Approve, then suspend the account, then execute — the drift check refuses,
   status stays `APPROVED`, never reaches `EXECUTED`.
4. Replay an approval call (a network retry) — the second call is refused, not
   double-applied; `resolved_by` stays the first reviewer's.

### 6. AWS backend

`infra/aws-network-rds-ecr.yaml` provisions `HitlApprovalsTable` (DynamoDB,
`PAY_PER_REQUEST`, partition key `action_id`). `infra/aws-ecs-alb.yaml` adds a
`TaskRole` distinct from the existing `ExecutionRole` — the execution role is
assumed by the ECS agent (image pull, secrets fetch) before the container
starts; DynamoDB calls happen from the *running application code itself*
(`refunds/approval_gate.py`'s `boto3` calls), a different trust boundary, so a
different role, scoped only to `dynamodb:GetItem`/`PutItem`/`UpdateItem`/`Scan`
on `HitlApprovalsTable`'s ARN. Template-level only as of this writing — not yet
applied to the live stack (same treatment as other infra changes made this
week without a live redeploy each time: the template is the source of truth
for the next deploy, not something that triggers one on its own).

### What stays out

Actual email sending — `PendingAction` has nothing that emails a reviewer or a
customer; `hitl_cli.py list`/the eventual review is manual. AgentCore migration
— the design this was built against notes where the graph/node/Pydantic models
would move unchanged, and flags that the admin surface's `hitl_cli.py` may need
to stay a small separate process if AgentCore's invocation surface doesn't
expose custom entry points; not attempted this week.
