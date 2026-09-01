# Confident wrong path — plain return request escalated despite having a complete answer

**Assignment 2, section 2.2** — real trace, not a constructed example.

- **Session ID**: `d7b5f264-20b9-4d32-a6c3-f850085a0469`
- **Customer message**: *"i want to return order 1001"*
- **Account**: `asha.rao@example.com`, order `ORD1001` (delivered, refund window open, account active — no damage mentioned anywhere in the message)

## The final answer (what it looks like on its own)

> I wasn't able to fully resolve this after checking the information available to me, so I'm creating a support ticket for you now. One of our team will reach out shortly from dubba.support@dubba.com -- you'll get the full details by email.

Read in isolation, this is a perfectly reasonable message. Polite, honest about a limitation, gives a real next step. Nothing about it signals a bug.

## The actual trajectory

| Step | Node | Action | Result |
|---|---|---|---|
| 1 | `propose_node` | `lookup_order(ORD1001)` | order status, `refund_window_eligible=true`, `account_standing_ok=true` |
| 2 | `evaluate_node` | judge sufficiency | **insufficient** — knows eligibility, not the return *process* |
| 3 | `propose_node` | `search_policy("return refund process eligibility requirements standard refund damaged item")` | **correct, relevant hits**: `02_return_policy#0 Overview` (0.6304), `02_return_policy#4 Inspection Process` (0.6077) |
| 4 | `evaluate_node` | judge sufficiency | **insufficient** — again, despite having the actual return-policy conditions (unlit requirement, original packaging, invoice, warehouse inspection) already in hand |
| 5 | `propose_node` | `search_policy("damaged in transit refund photo evidence upload process requirements")` | `01_refund_policy#4 Damaged-in-Transit Refunds` (0.6233) — **retrieved correctly for the query as phrased, but the query itself has no basis in this ticket** — the customer never said anything was damaged |
| 6 | `evaluate_node` | judge sufficiency | insufficient a third time |
| 7 | routing | `tool_call_count` (3) `>= TOOL_CALL_CAP` (3) | forced to `escalate_node` instead of another `propose` cycle |

## What the trajectory reveals that the final answer hides

A **complete, policy-grounded answer was available after step 3** — the return policy's actual conditions (unlit, original packaging + invoice, physical inspection on arrival) are exactly what a customer asking "I want to return order 1001" needs. Instead:

1. **`evaluate_node` rejected a correct, sufficient answer as insufficient.** Its own scope note treats "no confirmed yes/no on final eligibility" as missing information — but for a *return* (as opposed to a delay-compensation or documented refund case), no such confirmation is obtainable at this stage by design: `02_return_policy.md` states final approval only happens after the item is physically shipped back and inspected at the warehouse. `evaluate_node` was holding out for a confirmation that the real business process doesn't produce yet, and had no concept that "here are the requirements, approval is pending inspection" *is* a complete answer.
2. **That false expectation of a missing confirmation drove `propose_node` toward an ungrounded retrieval.** Search 2's query already contains "damaged item" — a premise the customer never stated. The model was hunting for *some* verification mechanism, and the only "evidence/verification" concept anywhere in its own tool instructions was the damage-evidence flow — so it reached for that framing even though nothing about this ticket warranted it.
3. **That wasted the tool-call budget.** Two of three tool calls went to a real but irrelevant policy chunk instead of the agent simply using what it already had after call #1. The escalation wasn't caused by the docs lacking an answer — it was caused by the harness discarding a correct answer and burning its budget chasing a premise that was never true.

## Failure category (per the assignment's own examples)

Closest fit: **retrieval that got used on the wrong premise, downstream of a sufficiency-judgment bug** — not a bad retrieval in isolation (the damage chunk genuinely matches its own query), but a retrieval built on an unstated, invented assumption, only reachable because a correct answer was wrongly marked insufficient one step earlier.

## Root cause

- `EVALUATE_TOOL`'s schema had `sufficient` listed *before* `reasoning` in both `properties` and `required` — tool-call fields generate in schema order, so the model was committing to the boolean before actually reasoning through it. (Separately confirmed and fixed this same session, prior to this trace's underlying prompt-content bug.)
- `EVALUATE_SYSTEM_PROMPT` had no criterion recognizing "requirements stated + outcome conditional on a downstream process (inspection/verification)" as a complete answer — only "eligibility confirmed" counted as sufficient.
- `PROPOSE_SYSTEM_PROMPT`'s damage-evidence-upload tool section was the longest, most procedurally detailed block in the entire tool list (a 4-step numbered procedure vs. one line for every other tool) — its prominence plausibly biased the model toward damage framing on tickets that never mentioned damage.

## Fix applied

1. `EVALUATE_TOOL` field order swapped (`reasoning` before `sufficient`) — same session, prior to this specific incident being fully diagnosed.
2. `EVALUATE_SYSTEM_PROMPT` updated to explicitly count "requirements stated, outcome conditional on a downstream process" as sufficient (this change).
3. `request_damage_evidence_upload`/`confirm_damage_evidence_upload` removed entirely from the general support agent's tool set — that capability belongs to a refund-specialist agent, not this one, and its outsized prompt presence was actively causing this failure mode on unrelated tickets (this change).

Verification: re-running "i want to return order 1001" post-fix should resolve in 1-2 tool calls (`lookup_order` + one `search_policy`) with a policy-grounded answer about packaging, invoice, and unlit condition, instead of escalating at the tool-call cap.
