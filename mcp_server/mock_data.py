"""
Mock order + account data for Dubba.

Nothing above this file (harness, MCP tools, auth) touches the dicts below
directly -- everything goes through the functions at the bottom, so a real
DB can be swapped in later without touching callers.
"""

_ACCOUNTS = {
    "CUST001": {
        "customer_id": "CUST001",
        "email": "asha.rao@example.com",
        "access_code": "4821",
        "standing": "active",
        "order_ids": ["ORD1001", "ORD1002"],
    },
    "CUST002": {
        "customer_id": "CUST002",
        "email": "ben.oliver@example.com",
        "access_code": "5539",
        "standing": "active",
        "order_ids": ["ORD1003"],
    },
    "CUST003": {
        "customer_id": "CUST003",
        "email": "chioma.eze@example.com",
        "access_code": "1190",
        "standing": "flagged",
        "order_ids": ["ORD1004", "ORD1005"],
    },
    "CUST004": {
        "customer_id": "CUST004",
        "email": "diego.mora@example.com",
        "access_code": "7702",
        "standing": "suspended",
        "order_ids": ["ORD1006"],
    },
    "CUST005": {
        "customer_id": "CUST005",
        "email": "erin.walsh@example.com",
        "access_code": "3364",
        "standing": "active",
        "order_ids": ["ORD1007", "ORD1008"],
    },
}

_ORDERS = {
    "ORD1001": {
        "order_id": "ORD1001",
        "customer_id": "CUST001",
        "status": "delivered",
        "items": ["Mona Lisa's Smirk (Vanilla Musk)", "Starry Night Swirl (Lavender Dusk)"],
        "delivery_date": "2026-07-18",
    },
    "ORD1002": {
        "order_id": "ORD1002",
        "customer_id": "CUST001",
        "status": "in_transit",
        "items": ["The Scream Queen (Espresso & Ash)"],
        "delivery_date": "2026-08-02",
    },
    "ORD1003": {
        "order_id": "ORD1003",
        "customer_id": "CUST002",
        "status": "delayed",
        "items": ["Versailles Vanity (Rose Gold Amber)"],
        "delivery_date": "2026-08-05",
    },
    "ORD1004": {
        "order_id": "ORD1004",
        "customer_id": "CUST003",
        "status": "delivered",
        "items": ["Girl with a Pearl Earring (Sea Salt Bergamot)"],
        "delivery_date": "2026-07-10",
    },
    "ORD1005": {
        "order_id": "ORD1005",
        "customer_id": "CUST003",
        "status": "cancelled",
        "items": ["Venus on the Half Shell (Coconut Sandalwood)"],
        "delivery_date": None,
    },
    "ORD1006": {
        "order_id": "ORD1006",
        "customer_id": "CUST004",
        "status": "processing",
        "items": ["Melting Clocks & Musk (Dali's Dream)"],
        "delivery_date": "2026-08-10",
    },
    "ORD1007": {
        "order_id": "ORD1007",
        "customer_id": "CUST005",
        "status": "delivered",
        "items": ["American Gothic Pitchfork Pine"],
        "delivery_date": "2026-07-22",
    },
    "ORD1008": {
        "order_id": "ORD1008",
        "customer_id": "CUST005",
        "status": "in_transit",
        "items": ["Starry Night Swirl (Lavender Dusk)", "Mona Lisa's Smirk (Vanilla Musk)"],
        "delivery_date": "2026-08-01",
    },
}


def get_order(order_id: str) -> dict | None:
    return _ORDERS.get(order_id)


def get_account(customer_id: str) -> dict | None:
    return _ACCOUNTS.get(customer_id)


def find_account_by_email(email: str) -> dict | None:
    for account in _ACCOUNTS.values():
        if account["email"].lower() == email.lower():
            return account
    return None
