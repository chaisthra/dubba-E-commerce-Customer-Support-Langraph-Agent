# Trajectory eval run 1 — failure analysis (13/15, 86.7%)

Real run of `evals/trajectory_eval.py` against `evals/golden_dataset.json`, 2026-08-29.
Full results in `evals/trajectory_eval_results.json`. Two failures — one is a real
agent defect (a confident wrong path, worse than the missing-tool-call trajectory
check alone reveals), the other turned out to be a mistake in the golden dataset
itself, not the agent. Documenting both honestly rather than only the flattering one.

## `delivery_taking_forever` — real bug: wrong order looked up, confidently wrong answer

**Query**: *"my order hasn't arrived yet and it's taking forever"* — customer
`asha.rao@example.com` (`CUST001`), designed against `ORD1002` (11 days in transit,
genuinely delayed, still undelivered).

**Trajectory check result**: `search_policy` never got called — that's what the
rule-based eval flagged. But reading the actual reply shows something worse:

> *"I've checked on your order (ORD1001), and I can see it was actually delivered on
> August 24th, just 2 days after shipping... If you haven't received it, a few things
> might have happened: It was left in an unexpected spot... Someone else accepted
> it... It was misdelivered..."*

**The agent looked up the wrong order.** `CUST001` has two orders: `ORD1002`
(undelivered, matches the complaint) and `ORD1001` (delivered 5 days ago). The agent
called `lookup_order(ORD1001)` — the *other* one — got back `status: delivered`, and
instead of recognizing the mismatch, confidently wrote a whole answer about a
delivered package possibly being misplaced by a neighbor or building staff. None of
that troubleshooting content comes from any tool result or policy doc — it's
generic customer-service boilerplate invented to explain a contradiction the agent
never actually noticed.

**Why the wrong order got picked**: `PROPOSE_SYSTEM_PROMPT`'s ambiguous-order logic
says to check the most-recently-placed order first, and — critically — has an
explicit fallback for exactly this situation: *"If that order is already delivered
and the question implies something still in progress, check the next most recent
one instead of guessing."* The agent called `lookup_order` once, got `delivered`,
and did **not** follow that fallback despite the query ("hasn't arrived yet, taking
forever") being about as clear an "implies something still in progress" signal as
a query can give. It just ran with the mismatched result.

**Compounding factor**: `CUST001`'s `order_ids` list was reordered from
`["ORD1001", "ORD1002"]` to `["ORD1002", "ORD1001"]` earlier this session (fixing a
genuine oldest-first bug — see `log/DECISIONS.md`, 2026-08-29). That fix was correct
on its own terms, but it means "most recently placed" (now correctly `ORD1001`) and
"the order this complaint is actually about" (`ORD1002`) are now different orders
for this specific customer — which is exactly the case the prompt's fallback clause
exists to handle, and exactly the case where the agent didn't execute it.

**A second, likely contributing failure**: `evaluate_node` accepted the
`lookup_order(ORD1001)` result as sufficient without ever flagging the contradiction
between "tool says delivered" and "customer says not arrived." That's the same
sufficiency-judgment class of bug documented in
`log/learnings/2026-08-28-confident-wrong-path-return-request.md`, showing up again
in a different shape — evaluate_node has no criterion for "the data I have
contradicts what the customer described," only for "do I have enough data."

**Category (assignment's own examples)**: *"a tool called with the wrong
argument"* — `lookup_order` was called with `order_id=ORD1001` when the ticket was
about `ORD1002`. Not fixed as part of this eval run — documented for the next
harness-work session.

## `suspended_account_appeal` — not an agent bug, a golden-dataset error

**Query**: *"why is my account suspended and how do i appeal"* — `diego.mora@example.com`
(`CUST004`, genuinely suspended).

**Trajectory check result**: `check_account_status` was expected but never called.

**Why this isn't actually wrong**: `PROPOSE_SYSTEM_PROMPT` already tells the model
account standing is provided directly — `check_account_status(customer_id): ...
You will rarely need this for the current customer -- you're already given their
order_ids and standing below.` And `propose_node`'s own system prompt construction
(`harness/graph.py`) includes `f"Account standing: {account['standing']}"` in every
call. The agent already knew the account was suspended without calling the tool,
went straight to `search_policy` for the appeals doc, and produced a complete,
accurate, policy-grounded answer (correctly citing the 30-day appeal window, 5
business day review, and what remains accessible during suspension).

**The actual mistake was in `evals/golden_dataset.json`**: I designed this ticket's
`expected_trajectory` assuming `check_account_status` would be needed, without
checking that the harness already surfaces standing for free. Fixed in the dataset
(removed `check_account_status` from this entry's expected trajectory) rather than
left in as a permanently-failing false positive that would corrupt future eval runs
and the §2.4 regression gate.

## Corrected score

With the dataset fix applied, re-running would be expected to land at **14/15
(93.3%)** — one real, documented defect remaining (`delivery_taking_forever`), zero
false positives from bad test expectations.
