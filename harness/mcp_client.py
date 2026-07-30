"""
Thin async wrapper around the MCP stdio client. Starts mcp_server/server.py as a
subprocess once, at app startup (see main.py), and reuses the same session for the
whole process lifetime -- not spun up per turn.
"""

import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MCPClient:
    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def start(self) -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            cwd=str(PROJECT_ROOT),
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    async def call_tool(self, name: str, arguments: dict) -> dict:
        result = await self._session.call_tool(name, arguments)
        text = " ".join(block.text for block in result.content if hasattr(block, "text"))

        if result.is_error:
            return {"error": text or "tool call failed"}
        if result.structured_content:
            return result.structured_content
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return {"error": f"could not parse tool result: {text!r}"}

    async def close(self) -> None:
        await self._stack.aclose()
