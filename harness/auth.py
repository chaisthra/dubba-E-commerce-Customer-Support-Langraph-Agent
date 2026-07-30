"""
Mock login: email + access code, checked against mcp_server.mock_data.

This is intentionally fake auth (plaintext access code, no hashing) --
acceptable for a mock assignment. Week 2 replaces this with real,
encrypted authentication.
"""

from mcp_server.mock_data import find_account_by_email

MAX_ATTEMPTS = 3


def authenticate() -> dict | None:
    """Prompts for email + access code. Returns the account dict on
    success, or None if all attempts are exhausted."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        email = input("Email: ").strip()
        access_code = input("Access code: ").strip()

        account = find_account_by_email(email)
        if account and account["access_code"] == access_code:
            return account

        remaining = MAX_ATTEMPTS - attempt
        if remaining > 0:
            print(f"Email/access code didn't match. {remaining} attempt(s) left.")

    return None
