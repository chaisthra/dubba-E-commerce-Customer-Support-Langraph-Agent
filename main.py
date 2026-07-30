"""
Entry point. Load environment before anything that reads it (Langfuse, Anthropic
client) gets imported -- see harness/llm_client.py and log/WHY.md.

Two harness implementations, both built on the same principle (harness code checks a
proposed action before it executes) -- pick which one runs with --mode:
  loop  : harness/loop.py, hand-rolled while-loop. respond/ask_clarification only,
          no real tool calls -- the stage-1 harness-pattern demonstration.
  graph : harness/graph.py, LangGraph. Real MCP tool calls (lookup_order,
          check_account_status), permission-scoped in a conditional edge. Default --
          this is the functional agent for the assignment's 5 ticket scenarios.
"""

import argparse
import asyncio

from dotenv import load_dotenv

load_dotenv()

from harness import auth, loop  # noqa: E402  (must follow load_dotenv())
from harness.mcp_client import MCPClient  # noqa: E402
from memory.store import save_ticket_summary  # noqa: E402

EXIT_COMMANDS = {"exit", "quit", "bye"}

WELCOME = """
==========================================
  Welcome to Dubba
  Hand-poured candles inspired by famous
  paintings -- Mona Lisa, Starry Night,
  The Scream, Palace of Versailles & more.
==========================================
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dubba order-support agent")
    parser.add_argument(
        "--mode",
        choices=["loop", "graph"],
        default="graph",
        help="loop = while-loop harness (no tools). graph = LangGraph harness with real MCP tool calls (default).",
    )
    return parser.parse_args()


async def run_loop_mode(session: dict) -> None:
    while True:
        user_message = input("\nYou: ").strip()
        if not user_message:
            continue
        if user_message.lower() in EXIT_COMMANDS:
            session["closure_reason"] = "resolved"
            print("Dubba: Thanks for reaching out -- take care!")
            return
        reply = loop.handle_turn(session, user_message)
        print(f"\nDubba: {reply}")


async def run_graph_mode(session: dict) -> None:
    from harness import graph

    mcp_client = MCPClient()
    await mcp_client.start()
    graph.set_mcp_client(mcp_client)

    try:
        while True:
            user_message = input("\nYou: ").strip()
            if not user_message:
                continue
            if user_message.lower() in EXIT_COMMANDS:
                session["closure_reason"] = "resolved"
                print("Dubba: Thanks for reaching out -- take care!")
                return
            reply = await graph.run_turn(session, user_message)
            print(f"\nDubba: {reply}")
    finally:
        await mcp_client.close()


async def main() -> None:
    args = parse_args()

    print(WELCOME)
    account = auth.authenticate()
    if account is None:
        print("Too many failed attempts. Goodbye.")
        return

    print(f"\nWelcome back, {account['email']}. How can I help with your order today?")
    session = loop.new_session(account["customer_id"], account)

    if args.mode == "loop":
        await run_loop_mode(session)
    else:
        await run_graph_mode(session)

    if session["short_term_buffer"]:
        save_ticket_summary(session)


if __name__ == "__main__":
    asyncio.run(main())
