"""
Mock order + account data for Dubba.

Nothing above this file (harness, MCP tools, auth) touches the dicts below
directly -- everything goes through the functions at the bottom, so a real
DB can be swapped in later without touching callers.

get_order() computes shipping-delay facts (days_in_transit, delay_compensation_eligible)
AND standard refund-window facts (refund_window_eligible, refund_window_days_remaining,
account_standing_ok) from real dates/account standing itself -- never left for the LLM
to eyeball from what the customer claims. See log/DECISIONS.md: a customer saying
"it's been 10 days" or "my account is fine" is conversational context, never the
source of truth for either calculation.

Country determines the shipping-delay threshold (rag/policy_docs/03_shipping_delay_compensation.md):
domestic = account's country matches HOME_COUNTRY ("India"), 8-day threshold.
international = any other country, 10-day threshold.

Order dates are defined as offsets from date.today() (via _days_ago()), not hardcoded
literal dates -- evergreen, so e.g. "shipped 11 days ago, past the domestic delay
threshold" is still true whenever this app actually runs, instead of silently rotting
the moment real time passes the date it was written on (as the previous hardcoded-July
dates did -- see log/DECISIONS.md).
"""

from datetime import date, timedelta

HOME_COUNTRY = "India"
DELAY_THRESHOLD_DOMESTIC_DAYS = 8
DELAY_THRESHOLD_INTERNATIONAL_DAYS = 10
REFUND_WINDOW_DAYS = 15  # rag/policy_docs/01_refund_policy.md: 15 days from delivery


def _days_ago(n: int) -> str:
    return (date.today() - timedelta(days=n)).isoformat()


_ACCOUNTS = {
    "CUST001": {
        "customer_id": "CUST001",
        "email": "asha.rao@example.com",
        "access_code": "4821",
        "standing": "active",
        "country": "India",
        # oldest first, per PROPOSE_SYSTEM_PROMPT's assumption -- ORD1002 (ordered
        # 13 days ago) predates ORD1001 (9 days ago), despite the lower order_id.
        "order_ids": ["ORD1002", "ORD1001"],
    },
    "CUST002": {
        "customer_id": "CUST002",
        "email": "ben.oliver@example.com",
        "access_code": "5539",
        "standing": "active",
        "country": "India",
        "order_ids": ["ORD1003"],
    },
    "CUST003": {
        "customer_id": "CUST003",
        "email": "chioma.eze@example.com",
        "access_code": "1190",
        "standing": "flagged",
        "country": "India",
        # oldest first, same reason -- ORD1005 (15 days ago) predates ORD1004 (10 days ago).
        "order_ids": ["ORD1005", "ORD1004"],
    },
    "CUST004": {
        "customer_id": "CUST004",
        "email": "diego.mora@example.com",
        "access_code": "7702",
        "standing": "suspended",
        "country": "India",
        "order_ids": ["ORD1006"],
    },
    "CUST005": {
        "customer_id": "CUST005",
        "email": "erin.walsh@example.com",
        "access_code": "3364",
        "standing": "active",
        "country": "United States",  # outside HOME_COUNTRY -- international threshold
        "order_ids": ["ORD1007", "ORD1008"],
    },
}

# ordered_date/shipped_date/delivery_date are all offsets from today (_days_ago), so
# every scenario comment below ("N days ago") stays true regardless of when this app
# actually runs. delivery_date is null until the order is ACTUALLY delivered -- it is
# never an "expected" date. shipped_date is null until the carrier has actually
# shipped it (i.e. "processing" orders have no shipped_date yet).
_ORDERS = {
    "ORD1001": {
        "order_id": "ORD1001",
        "customer_id": "CUST001",
        "status": "delivered",
        "items": ["Mona Lisa's Smirk (Vanilla Musk)", "Starry Night Swirl (Lavender Dusk)"],
        "ordered_date": _days_ago(9),
        "shipped_date": _days_ago(7),
        # 2-day transit, well under the 8-day domestic threshold -- not delayed.
        # Delivered 5 days ago -- comfortably inside the 15-day refund window, and
        # account is active: the clean "fully eligible" refund scenario.
        "delivery_date": _days_ago(5),
    },
    "ORD1002": {
        "order_id": "ORD1002",
        "customer_id": "CUST001",
        "status": "in_transit",
        "items": ["The Scream Queen (Espresso & Ash)"],
        "ordered_date": _days_ago(13),
        # 11 days in transit and counting, past the 8-day domestic threshold --
        # genuinely delayed. Not yet delivered, so refund-window facts don't apply.
        "shipped_date": _days_ago(11),
        "delivery_date": None,
    },
    "ORD1003": {
        "order_id": "ORD1003",
        "customer_id": "CUST002",
        "status": "delayed",
        "items": ["Versailles Vanity (Rose Gold Amber)"],
        "ordered_date": _days_ago(22),
        # 20 days in transit -- clearly, unambiguously past the 8-day domestic
        # threshold. Not yet delivered.
        "shipped_date": _days_ago(20),
        "delivery_date": None,
    },
    "ORD1004": {
        "order_id": "ORD1004",
        "customer_id": "CUST003",  # account standing: flagged
        "status": "delivered",
        "items": ["Girl with a Pearl Earring (Sea Salt Bergamot)"],
        "ordered_date": _days_ago(10),
        "shipped_date": _days_ago(8),
        # 2-day transit, not delayed. Delivered 6 days ago -- the refund WINDOW is
        # open (refund_window_eligible=True), but this customer's account is
        # "flagged", so account_standing_ok=False blocks it anyway. Deliberately
        # exercises the two refund gates independently -- window-open-but-blocked,
        # not window-closed.
        "delivery_date": _days_ago(6),
    },
    "ORD1005": {
        "order_id": "ORD1005",
        "customer_id": "CUST003",
        "status": "cancelled",
        "items": ["Venus on the Half Shell (Coconut Sandalwood)"],
        "ordered_date": _days_ago(15),
        "shipped_date": None,
        "delivery_date": None,
    },
    "ORD1006": {
        "order_id": "ORD1006",
        "customer_id": "CUST004",  # account standing: suspended
        "status": "processing",
        "items": ["Melting Clocks & Musk (Dali's Dream)"],
        "ordered_date": _days_ago(2),
        "shipped_date": None,  # hasn't shipped yet -- still processing
        "delivery_date": None,
    },
    "ORD1007": {
        "order_id": "ORD1007",
        "customer_id": "CUST005",  # international (United States)
        "status": "delivered",
        "items": ["American Gothic Pitchfork Pine"],
        "ordered_date": _days_ago(19),
        "shipped_date": _days_ago(17),
        # 1-day transit, well under the 10-day international threshold -- not
        # delayed. Delivered 16 days ago -- one day PAST the 15-day refund window:
        # refund_window_eligible=False, refund_window_days_remaining=-1. Account is
        # active, so this is purely a window-closed case, isolated from the
        # account-standing gate.
        "delivery_date": _days_ago(16),
    },
    "ORD1008": {
        "order_id": "ORD1008",
        "customer_id": "CUST005",
        "status": "in_transit",
        "items": ["Starry Night Swirl (Lavender Dusk)", "Mona Lisa's Smirk (Vanilla Musk)"],
        # 3 days in transit, under the 10-day international threshold -- normal,
        # not yet delayed. Not yet delivered.
        "ordered_date": _days_ago(5),
        "shipped_date": _days_ago(3),
        "delivery_date": None,
    },
}


def _with_delay_status(order: dict) -> dict:
    """Computes days_in_transit / delay_compensation_eligible from real dates.
    Never derived from what a customer claims."""
    order = dict(order)
    shipped = order.get("shipped_date")

    if shipped is None:
        order["days_in_transit"] = None
        order["delay_compensation_eligible"] = False
        return order

    account = _ACCOUNTS.get(order["customer_id"])
    is_domestic = account is not None and account.get("country") == HOME_COUNTRY
    threshold = DELAY_THRESHOLD_DOMESTIC_DAYS if is_domestic else DELAY_THRESHOLD_INTERNATIONAL_DAYS

    end = order["delivery_date"] or date.today().isoformat()
    days = (date.fromisoformat(end) - date.fromisoformat(shipped)).days
    order["days_in_transit"] = days
    order["delay_threshold_days"] = threshold
    order["delay_compensation_eligible"] = days > threshold
    return order


def _with_eligibility_status(order: dict) -> dict:
    """Computes the STANDARD refund-window facts (rag/policy_docs/01_refund_policy.md's
    "Eligibility Window" + "Account Standing Requirement" sections) from real dates and
    account standing. Never derived from what a customer claims about either.

    refund_window_eligible and account_standing_ok are deliberately independent
    signals, not pre-collapsed into one bool -- a refund needs BOTH true, but keeping
    them separate lets the harness/model explain WHICH gate is the actual blocker
    (e.g. "your window is open, but your account is flagged" vs. "your window has
    closed"). See harness/prompts.py's PROPOSE_SYSTEM_PROMPT for how these are used.

    This does NOT cover the refund policy's two evidence-based paths -- non-delivery
    (needs carrier verification) and damaged-in-transit (needs photo evidence) --
    neither is a pure date computation, so those still require judgment even when
    this check passes."""
    order = dict(order)
    account = _ACCOUNTS.get(order["customer_id"])
    order["account_standing_ok"] = account is not None and account["standing"] == "active"

    delivered = order.get("delivery_date")
    if delivered is None:
        order["refund_window_eligible"] = False
        order["refund_window_days_remaining"] = None
        return order

    days_remaining = REFUND_WINDOW_DAYS - (date.today() - date.fromisoformat(delivered)).days
    order["refund_window_days_remaining"] = days_remaining
    order["refund_window_eligible"] = days_remaining >= 0
    return order


def get_order(order_id: str) -> dict | None:
    order = _ORDERS.get(order_id)
    if order is None:
        return None
    return _with_eligibility_status(_with_delay_status(order))


def get_account(customer_id: str) -> dict | None:
    return _ACCOUNTS.get(customer_id)


def find_account_by_email(email: str) -> dict | None:
    for account in _ACCOUNTS.values():
        if account["email"].lower() == email.lower():
            return account
    return None
