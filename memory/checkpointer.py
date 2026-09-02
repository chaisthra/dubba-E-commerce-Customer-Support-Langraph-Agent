"""
Owns the Postgres checkpointer lifecycle -- LangGraph's own short-term-state tables
(checkpoints / checkpoint_writes / checkpoint_blobs), keyed by thread_id = our
session_id. Separate from memory/store.py's `tickets` table (long-term, across-session
history) -- same Postgres instance, but the checkpointer's own setup() never touches
it and vice versa.

DATABASE_URL is the single source of truth for which Postgres this points at -- local
Docker now (docker-compose.yml), AWS RDS later (log/DECISIONS.md roadmap). Only the
env var value changes; nothing here or in memory/store.py hardcodes an environment.
"""

import os
from contextlib import AsyncExitStack

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set -- see .env.example")
    return url


class PostgresCheckpointer:
    """Same AsyncExitStack lifecycle pattern as harness/mcp_client.py's MCPClient --
    one long-lived connection for the whole process, started once at app startup and
    closed once at shutdown, never opened/closed per turn."""

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self.saver: AsyncPostgresSaver | None = None

    async def start(self) -> None:
        self.saver = await self._stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(database_url())
        )
        await self.saver.setup()  # idempotent (IF NOT EXISTS-style) -- safe every startup

    async def close(self) -> None:
        await self._stack.aclose()
