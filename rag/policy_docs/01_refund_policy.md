# Refund Eligibility

## Overview

Dubba issues refunds to customers whose orders meet specific eligibility conditions
related to delivery status, item condition, and account standing. This document
outlines when a refund applies, what evidence is required, how refunds are processed,
and how refund requests interact with account status.

## Eligibility Window

A refund request is eligible if it is submitted within **15 days** of the order being
marked as delivered in the system. The 15-day window begins from the delivery
timestamp recorded by the shipping carrier, not from the date the order was placed or
shipped. Requests submitted after the 15-day window has closed are not eligible for a
standard refund and will require escalation to a support specialist for manual review.

## Account Standing Requirement

The customer's account must be **active** (not flagged or suspended) at the time the
refund request is submitted. If an account is flagged for review or suspended, any
pending refund request is placed on hold until the account status is resolved. Refunds
cannot be processed against a suspended account even if the underlying order otherwise
meets all eligibility criteria, since account standing is checked as a precondition
before any refund logic runs.

## Non-Delivery Refunds

If a customer reports that an order was never delivered despite tracking showing a
"delivered" status, a refund is issued only after verification with the shipping
carrier confirms non-delivery. This verification step exists to prevent refund fraud
in cases of package theft after legitimate delivery, which is a carrier and property
matter rather than a Dubba fulfillment issue. Verification typically involves checking
the carrier's delivery confirmation (signature, photo, or GPS drop location) against
the customer's claim. If the carrier cannot confirm delivery occurred, the refund is
approved.

## Damaged-in-Transit Refunds

Items damaged during shipping qualify for a full refund once photo evidence of the
damage is provided by the customer. For candle products specifically, common
transit-damage scenarios include a broken, cracked, or missing wick, a cracked or
shattered vessel, or wax that has melted and re-solidified in a way that has visibly
deformed the product. Photo evidence should clearly show the damage and, where
possible, the shipping packaging the item arrived in, since this helps distinguish
transit damage from damage that occurred after delivery.

## Refund Processing

Approved refunds are issued in USD to the customer's original payment method. Refunds
typically process within 5-7 business days after approval, though the exact time
funds appear in the customer's account depends on their bank or payment provider.
Customers are not required to return the damaged or non-delivered item before the
refund is issued for damaged-in-transit or non-delivery cases; however, for standard
returns-based refunds, the item must be received back and inspected before the refund
is finalized (see the separate Return Eligibility document for physical condition
requirements).

## Interaction With Other Policies

Refund eligibility is evaluated independently of, but often alongside, return
eligibility and shipping delay compensation. A single order issue may qualify a
customer for more than one type of resolution — for example, a severely delayed order
may qualify for shipping delay compensation in addition to a refund if the customer
ultimately chooses not to keep the item. Support agents should check all three
policies when handling a complex ticket rather than assuming only one applies.
