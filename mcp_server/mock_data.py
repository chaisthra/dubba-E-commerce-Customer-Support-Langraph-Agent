"""
Mock order + account data for Dubba.

Nothing above this file (harness, MCP tools, auth) touches the dicts below
directly -- everything goes through the functions at the bottom, so a real
DB can be swapped in later without touching callers.

get_order() computes shipping-delay facts (days_in_transit, delay_compensation_eligible)
from shipped_date/delivery_date itself -- never left for the LLM to eyeball from
what the customer claims. See log/DECISIONS.md: a customer saying "it's been 10
days" is conversational context, never the source of truth for this calculation.

Country determines the shipping-delay threshold (rag/policy_docs/03_shipping_delay_compensation.md):
domestic = account's country matches HOME_COUNTRY ("India"), 8-day threshold.
international = any other country, 10-day threshold.
"""

from datetime import date

HOME_COUNTRY = "India"
DELAY_THRESHOLD_DOMESTIC_DAYS = 8
DELAY_THRESHOLD_INTERNATIONAL_DAYS = 10

_ACCOUNTS = {
    "CUST001": {
        "customer_id": "CUST001",
        "email": "asha.rao@example.com",
        "access_code": "4821",
        "standing": "active",
        "country": "India",
        "order_ids": ["ORD1001", "ORD1002"],
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
        "order_ids": ["ORD1004", "ORD1005"],
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

# delivery_date is null until the order is ACTUALLY delivered -- it is never an
# "expected" date. shipped_date is null until the carrier has actually shipped it
# (i.e. "processing" orders have no shipped_date yet).
_ORDERS = {
    "ORD1001": {
        "order_id": "ORD1001",
        "customer_id": "CUST001",
        "status": "delivered",
        "items": ["Mona Lisa's Smirk (Vanilla Musk)", "Starry Night Swirl (Lavender Dusk)"],
        "shipped_date": "2026-07-14",
        "delivery_date": "2026-07-18",
    },
    "ORD1002": {
        "order_id": "ORD1002",
        "customer_id": "CUST001",
        "status": "in_transit",
        "items": ["The Scream Queen (Espresso & Ash)"],
        "shipped_date": "2026-07-19",  # 11 days ago as of 2026-07-30 -- genuinely delayed
        "delivery_date": None,
    },
    "ORD1003": {
        "order_id": "ORD1003",
        "customer_id": "CUST002",
        "status": "delayed",
        "items": ["Versailles Vanity (Rose Gold Amber)"],
        "shipped_date": "2026-07-10",  # 20 days ago -- clearly delayed
        "delivery_date": None,
    },
    "ORD1004": {
        "order_id": "ORD1004",
        "customer_id": "CUST003",
        "status": "delivered",
        "items": ["Girl with a Pearl Earring (Sea Salt Bergamot)"],
        "shipped_date": "2026-07-06",
        "delivery_date": "2026-07-10",
    },
    "ORD1005": {
        "order_id": "ORD1005",
        "customer_id": "CUST003",
        "status": "cancelled",
        "items": ["Venus on the Half Shell (Coconut Sandalwood)"],
        "shipped_date": None,
        "delivery_date": None,
    },
    "ORD1006": {
        "order_id": "ORD1006",
        "customer_id": "CUST004",
        "status": "processing",
        "items": ["Melting Clocks & Musk (Dali's Dream)"],
        "shipped_date": None,  # hasn't shipped yet -- still processing
        "delivery_date": None,
    },
    "ORD1007": {
        "order_id": "ORD1007",
        "customer_id": "CUST005",
        "status": "delivered",
        "items": ["American Gothic Pitchfork Pine"],
        "shipped_date": "2026-07-18",
        "delivery_date": "2026-07-22",
    },
    "ORD1008": {
        "order_id": "ORD1008",
        "customer_id": "CUST005",
        "status": "in_transit",
        "items": ["Starry Night Swirl (Lavender Dusk)", "Mona Lisa's Smirk (Vanilla Musk)"],
        "shipped_date": "2026-07-27",  # 3 days ago -- normal transit, not yet delayed
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


def get_order(order_id: str) -> dict | None:
    order = _ORDERS.get(order_id)
    return _with_delay_status(order) if order else None


def get_account(customer_id: str) -> dict | None:
    return _ACCOUNTS.get(customer_id)


def find_account_by_email(email: str) -> dict | None:
    for account in _ACCOUNTS.values():
        if account["email"].lower() == email.lower():
            return account
    return None
