# LLM-as-judge run 2 (5 runs/ticket, post order-record fix) — three findings

Re-run of `evals/llm_judge.py --runs 5` after Finding A's fix (real order records
now included as reference). Overall average rose to 3.61/5 (from 2.96), and the
order-status false positives from run 1 are gone (`where_is_ord1001`,
`most_recent_order_status`, `refund_flagged_account` all now clean 5/5 across all 5
runs). Three things left worth documenting, in order of how much they actually
matter.

## Finding 1 — real, serious agent bug: lit-candle disqualification not applied

`lit_candle_return_denied` scored **1/5, consistently, across all 8 runs so far**
(3 in run 1, 5 in run 2 — zero variance). This ticket exists specifically to test
whether the agent applies the return policy's physical-condition disqualification
even when date/account eligibility checks pass (see `evals/golden_dataset.json`'s
notes for this ticket). It does not. The actual reply:

> *"I don't have specific policy information about returns for candles that have
> already been lit or used... please reach out directly to dubba.support@dubba.com
> ... You're still within the general return window (10 days remaining), so it's
> worth asking!"*

This is wrong in a specific, checkable way: `02_return_policy.md` states plainly
that a lit candle is disqualified from return, full stop, regardless of the refund
window. The agent's answer implies the opposite — that being within the window
means it's still worth asking. The trajectory check alone would have scored this a
clean PASS (`lookup_order` + `search_policy` were both called, exactly as
required) — this is the clearest demonstration in this whole session of why
section 2.4's gate has to combine both signals: a passing trajectory here hides a
policy-compliance failure the judge is the only thing that catches.

**Open question, not yet resolved**: `search_policy` was called (confirmed via the
trajectory result), but the reply reads as if no relevant policy content was
found at all. Can't tell from the saved results alone whether retrieval missed the
disqualification chunk (same failure class as `membership_cancel_cost` --
similarity below `MIN_SIMILARITY`, see `log/loophole.md`) or surfaced it but the
model failed to apply it. Needs the raw `[RAG]` stderr output from a fresh run to
distinguish -- worth checking before deciding which layer to fix.

## Finding 2 — minor real bug: wrong refund-window day count

`broken_candle_refund_ord1001` scored a consistent **2/5**. The reply states *"you're
well within our 14-day refund window"* -- the real policy window is **15 days**
(`01_refund_policy.md`), and `REFUND_WINDOW_DAYS = 15` in `mock_data.py`. A small,
clean, one-digit factual error, correctly caught by the judge, correctly ignored by
the trajectory check (both tools were called, this isn't a trajectory problem).
Worth a quick manual double-check on a fresh run to see if this is a one-off model
slip or a stable pattern before treating it as worth fixing.

## Finding 3 — judge over-strictness, not an agent bug

`suspended_account_appeal` scored 2-3 (avg 2.6). The judge's stated reason:
penalizing the agent for citing `dubba.support@dubba.com` as the appeal-submission
address, calling it *"not mentioned in the reference policy... unsupported
detail."* But that email is real -- it's the actual support contact used
throughout this whole system (the escalation message, the honest-gap fix, etc.).
`05_account_suspension_appeals.md` just says "submit through support" without
naming a channel; the agent correctly filled in the real one. This is the same
underlying issue as Finding A from run 1 (judge reference material incomplete
relative to real system facts), in a new shape -- this time it's a real contact
address rather than an order record. Not fixed as part of this round -- flagging
it rather than patching reactively again; if this pattern shows up a third time
across different tickets, the judge's reference-building probably needs a general
"known real system facts" section (support email, escalation message, etc.) rather
than another one-off fix per finding.

## Non-determinism, demonstrated (not just described)

4 tickets showed real run-to-run variance at temperature=0
(`refund_policy_ord1001`: 2,5,5,5,5; `not_delivered_20_days`: 2,5,2,5,5;
`refund_window_closed`: 5,3,5,5,5; `cancelled_order_status`: 3,3,3,5,5). This is
concrete evidence for Session 3's own flagged pitfall -- a single judge run on any
of these four tickets could have reported a materially different score depending
on which run happened to land. Directly why `combined_score.py` (the new
baseline/gate script) uses multiple judge runs per ticket rather than one, and why
a single judge score should never be reported as if it were deterministic ground
truth.
