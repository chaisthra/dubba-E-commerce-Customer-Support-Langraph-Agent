# A day of real production bugs — deployment reliability, not agent behavior

Assignment 2's AWS deployment (§2.6), found and fixed live against the real deployed
stack, not simulated. Five real incidents in one debugging session, each with a
concrete symptom, a root cause found through live evidence (not guessed), and a fix.
Documented together because they compounded — several of these were live
simultaneously, masking each other.

## 1. Anthropic SDK removed `temperature` mid-session — every call silently fell through to Groq

**Symptom**: `/chat` worked (replies looked fine), but Langfuse traces showed
`provider=groq` on every single call. Nothing in the traces explained why — Langfuse
only ever records the *winning* provider, never a failed one's error.

**Found via**: live `inspect.signature()` introspection run inside the actual deployed
container (`aws ecs run-task` with an inline Python one-liner), not guessed and not
found by reading changelogs. Confirmed `Messages.create()`'s parameter list had no
`temperature` at all — a genuine SDK behavior change, not our bug.

**Root cause**: `requirements.txt` pinned `anthropic` with no version constraint.
Between two of our own test runs a few minutes apart, `pip install` resolved a
different version (`1.2.0` → `1.3.0`) — both already lacked `temperature`. Anthropic's
own stated guidance: omit the parameter entirely, control determinism via prompt
instead; no replacement kwarg exists. The new `output_config` param that appeared
alongside it is unrelated (reasoning effort + structured-output schema — confirmed by
reading the SDK's own type definitions on GitHub, not assumed).

**Compounding factor**: this had likely been silently broken for a while before it was
noticed — Groq masked it completely by design (`ProviderChain` falls through silently
to the next provider), so there's no way to know exactly when it started.

**Fix**: removed `temperature=0` from `AnthropicProvider.create()`'s call. Pinned
`anthropic>=1.3.0,<2.0.0` so a future release can't silently break this the same way
again — this is the second incident this session caused by an unpinned `anthropic`
version (see #5).

## 2. Database connection pool exhaustion — a query that finished, sitting stuck forever

**Symptom**: an 8+ minute hung `/chat` request, ending in a 504, with **zero**
application-level error output — no traceback, no exception, nothing.

**Found via**: `pg_stat_activity` on the live RDS instance, showing a real connection
sitting `idle in transaction` for 1102 seconds — mid the exact `policy_chunks`
similarity query the hung request's own logs showed last.

**Root cause**: `rag/store.py`'s `search()` ran a raw `SELECT` and never called
`.commit()`/`.rollback()`; `store.connect()` never set `autocommit=True` either. Since
psycopg defaults to `autocommit=False`, every retrieval implicitly opened a
transaction that never closed. Worse: `rag/retriever.py` held one single
module-level connection, reused across every call for the process's entire
lifetime — and that same synchronous connection object was almost certainly being
shared across concurrent requests from different threads (the MCP framework likely
dispatches sync tool functions to worker threads), which psycopg doesn't support
safely. One request's stuck connection could block a completely different request
that needed the same connection.

**Fix**: replaced the single shared connection with `psycopg_pool.ConnectionPool`
(`autocommit=True`, each call checks out and returns its own connection). Hit a
second, self-inflicted bug applying this fix: the pool's eager background
connection-opening raced `CREATE EXTENSION vector` on a fresh database (CI's Postgres
container starts empty every run) — fixed by deferring `_pool.open()` until after the
extension/table DDL runs on a bare connection first.

## 3. Langfuse unreachable from ECS — a public URL doesn't mean what it sounds like

**Symptom**: `Failed to export span batch due to timeout, max retries or shutdown` on
every turn. Ruled out first: bad credentials (task def had them), Langfuse itself
being down (confirmed healthy via direct SSM exec into the instance), the security
group rule being missing (confirmed present, correctly scoped).

**Found via**: a live reachability test — a one-off ECS task, same real network
config as the actual service, doing nothing but `curl -v` to Langfuse. The public EIP
timed out completely after 10s (`Connection timed out`, not refused). The exact same
`curl` to Langfuse's *private* IP, from the same task, connected instantly and
returned a real `200 OK`.

**Root cause**: a real, known AWS behavior, not obvious until hit — security-group
rules that reference another security group by ID (`UserIdGroupPairs`, as opposed to
a plain CIDR range) only match traffic that stays on the VPC's *private* network path.
Same-VPC traffic addressed to a *public* IP hairpins out through the Internet Gateway
and back in, and at that point it no longer carries the source instance's
security-group identity — so the rule silently never matches, and the connection is
dropped rather than rejected (hence a timeout, not a fast failure).

**Fix**: split the single `LANGFUSE_BASE_URL` into two SSM parameters —
`/langfuse-ec2/base-url` (public EIP, kept for external/local callers) and a new
`/langfuse-ec2/internal-base-url` (private IP, for same-VPC callers). The deployed ECS
task now reads the private one.

## 4. Groq: rate limits, tool-call disobedience, and an unbounded retry delay

Three separate but related problems with the fallback provider:

- **Rate limits**: Groq's free tier has real per-model, per-minute AND per-day caps.
  A burst of judge-eval calls hit a 429 with a daily cap message suggesting a
  **20+ minute** wait.
- **Malformed/disobedient tool calls**: unlike Anthropic's genuine constrained
  decoding (a tool call structurally cannot name a tool outside the offered list),
  Groq's `tool_choice` is closer to a strong instruction the model can still violate —
  confirmed empirically multiple times this project (a forced/required tool call
  sometimes returning plain text, or naming a tool that was never offered).
- **The delays got huge**: the rate-limit backoff logic honored Groq's own suggested
  wait time with no upper bound. When that suggestion was ~20 minutes (a daily cap,
  which waiting doesn't fix any faster), the code would actually sleep the full
  duration on a single model before ever trying the next one — confirmed live: one
  CI `eval-gate` run stalled ~23 minutes on exactly this path.

**Fix**: retry logic added for both failure classes (a few immediate retries for
tool-call disobedience, exponential backoff with Groq's suggested delay as a floor
for rate limits) — capped at `MAX_BACKOFF_SECONDS = 30s` so a long suggested wait
can't stall an entire request; it falls through to the next model instead, since
waiting longer wouldn't clear a daily cap any faster anyway.

## 5. And then everything else — a missing IAM permission and a self-inflicted CI race

- **CI couldn't push images at all**: `dubba-github-actions-deploy`'s only policy was
  SSM read access — nothing had ever granted ECR push or ECS deploy permissions, so
  `build-and-push` had never actually gotten past `ecr:GetAuthorizationToken`. Fixed
  with a scoped policy (ECR push to the one repo, ECS task-def register/describe,
  service update scoped to one service, `iam:PassRole` scoped to one execution role).
- **CI raced against itself**: with a PR open, every push fired both a `push` event
  AND a `pull_request:synchronize` event — two full pipelines, both trying to deploy
  to the same ECS service at once. One run's deploy step reported a false
  "circuit breaker rollback" because the other run's deployment superseded it
  mid-stabilization — the app itself never actually failed a health check. Fixed by
  scoping `build-and-push`/`deploy` to `push` events only.

## The pattern underneath all five

Every one of these was found by getting **real, live evidence** from inside the
actual deployed environment — `pg_stat_activity`, `inspect.signature()`, a `curl -v`
from the real network path, actual CloudWatch tracebacks — never by guessing from a
plausible-sounding theory and moving on. Three of the five (#1, #2 partially, #5's
first half) produced **zero application-level error output** on their own; the bug
only became visible by asking a more specific question of the live system than
"is it broken," and in every case the first theory (the assumption walking in) turned
out to be wrong or incomplete once actually checked.
