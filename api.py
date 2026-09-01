"""
HTTP entry point (Assignment 2, section 2.6.1) -- the container that actually runs
behind the ALB. main.py (the CLI) stays as-is for local dev; this wraps the same
harness/graph.py agent in a stateless HTTP surface instead of an interactive loop.

Stateless by design: LangGraph's Postgres checkpointer already persists full
conversation state per thread_id (= session_id) -- see harness/graph.py's
run_turn(). Every /chat call reconstructs a minimal session scaffold from scratch
(customer_id -> account, prior_tickets) rather than holding anything in server
memory between requests; the checkpointer, not this process, is what actually
carries a conversation forward across calls, exactly as it already does for the
CLI across turns within one run.

Auth is a separate step (POST /auth), not folded into /chat -- mirrors
harness/auth.py's real check (email + access code against mock_data) without its
CLI-specific input()/retry-loop shape, which doesn't make sense for a stateless API.
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from harness import graph  # noqa: E402
from harness.mcp_client import MCPClient  # noqa: E402
from mcp_server.mock_data import find_account_by_email, get_account  # noqa: E402
from memory.checkpointer import PostgresCheckpointer  # noqa: E402
from memory.store import get_customer_history, save_ticket_summary  # noqa: E402

_mcp_client: MCPClient | None = None
_checkpointer: PostgresCheckpointer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _mcp_client, _checkpointer
    _mcp_client = MCPClient()
    await _mcp_client.start()
    graph.set_mcp_client(_mcp_client)

    _checkpointer = PostgresCheckpointer()
    await _checkpointer.start()
    graph.init_graph(_checkpointer.saver)

    yield

    await _mcp_client.close()
    await _checkpointer.close()


app = FastAPI(title="Dubba", lifespan=lifespan)


class AuthRequest(BaseModel):
    email: str
    access_code: str


class AuthResponse(BaseModel):
    customer_id: str
    email: str


class ChatRequest(BaseModel):
    customer_id: str
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


class CloseRequest(BaseModel):
    customer_id: str
    session_id: str
    closure_reason: str = "resolved"  # resolved|abandoned|timeout, see log/SESSION_DESIGN.md


@app.get("/health")
def health() -> dict:
    # Liveness only, deliberately -- no DB/MCP round-trip here. A transient
    # Postgres hiccup shouldn't flap the ALB target group's health status; the
    # actual work (lookup_order, search_policy, etc.) will surface a real error
    # through /chat itself if something downstream is actually broken.
    return {"status": "ok"}


@app.post("/auth", response_model=AuthResponse)
def authenticate(req: AuthRequest) -> AuthResponse:
    account = find_account_by_email(req.email)
    if account is None or account["access_code"] != req.access_code:
        raise HTTPException(status_code=401, detail="email/access code didn't match")
    return AuthResponse(customer_id=account["customer_id"], email=account["email"])


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    account = get_account(req.customer_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"no account found for customer_id={req.customer_id!r}")

    # Minimal scaffold -- run_turn() only reads these 4 fields when this is the
    # thread's first-ever turn (no existing checkpoint); every subsequent call
    # for the same session_id is carried forward by the checkpointer itself, not
    # by anything reconstructed here. prior_tickets is re-fetched fresh every
    # call (cheap, always current) rather than cached anywhere in this process.
    session = {
        "session_id": req.session_id,
        "customer_id": req.customer_id,
        "account": account,
        "prior_tickets": await get_customer_history(req.customer_id),
    }

    reply = await graph.run_turn(session, req.message)
    return ChatResponse(reply=reply)


@app.post("/close")
async def close_session(req: CloseRequest) -> dict:
    """Explicit session-close, since HTTP has no natural 'the user hung up' event
    the way Ctrl+D does for the CLI -- the client calls this when a conversation
    is genuinely done. Writes the long-term `tickets` row exactly like main.py's
    own close-time save does."""
    account = get_account(req.customer_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"no account found for customer_id={req.customer_id!r}")

    state = await graph.get_session_state(req.session_id)
    if not state.get("short_term_buffer"):
        return {"saved": False, "reason": "no conversation to save for this session_id"}

    session = {
        "session_id": req.session_id,
        "customer_id": req.customer_id,
        "closure_reason": req.closure_reason,
        "short_term_buffer": state.get("short_term_buffer", []),
        "conversation_summary_xml": state.get("conversation_summary_xml", ""),
        "session_tool_log": state.get("session_tool_log", []),
        "session_permission_denials": state.get("session_permission_denials", []),
    }
    await save_ticket_summary(session)
    return {"saved": True, "closure_reason": req.closure_reason}
