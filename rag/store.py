"""
Postgres + pgvector backend for RAG's policy chunks -- same Postgres instance as
memory/store.py's `tickets` table and memory/checkpointer.py's checkpoint tables
(DATABASE_URL), but its own `policy_chunks` table, untouched by either. Replaces the
earlier in-memory ChromaDB collection (log/DECISIONS.md) -- same chunking, same
embedding model, same cosine-similarity math (pgvector's `<=>` operator IS cosine
distance, so `1 - distance` is exactly the same similarity score Chroma reported),
different storage.

Runs inside the MCP server subprocess (mcp_server/server.py), a separate process from
the main app -- uses psycopg's SYNC API (not memory/store.py's async one), because
mcp_server/server.py's tool functions are plain `def`, not `async def`. DATABASE_URL
has to be explicitly passed into that subprocess's env (harness/mcp_client.py) since
it doesn't inherit the parent process's environment by default (see rag/retriever.py).

Uses a ConnectionPool, not a single shared connection -- a sync function inside an
otherwise-async framework (mcp_server/server.py's tools) is a strong hint the MCP
framework dispatches concurrent tool calls to worker threads, and psycopg's
Connection/Cursor objects are not safe for concurrent use across threads. Each
call here checks a connection out of the pool for just its own duration and returns
it automatically (`with _pool.connection()`), instead of every caller sharing one
connection object that could be mid-transaction on another thread when the next
call arrives.
"""

import os

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2's output dimension


def _configure(conn: psycopg.Connection) -> None:
    """Called by the pool on every new physical connection it opens. autocommit=True
    because every caller here is a read (or an idempotent upsert) -- no query should
    ever hold a transaction open past its own statement. register_vector is
    per-connection state (the pgvector type adapter), so it has to run here, not
    once at module import -- a pool can open more than one physical connection.
    See log/loophole.md for the incident this replaced (a single shared,
    never-committed connection reused across concurrent threads -- caused a real
    stuck transaction and, almost certainly, the request hang that surfaced it)."""
    conn.autocommit = True
    register_vector(conn)


_pool = ConnectionPool(os.environ["DATABASE_URL"], configure=_configure, open=True)


def ensure_schema() -> None:
    with _pool.connection() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS policy_chunks (
                id SERIAL PRIMARY KEY,
                doc_name TEXT NOT NULL,
                chunk_index INT NOT NULL,
                heading TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding vector({EMBEDDING_DIM}) NOT NULL,
                UNIQUE (doc_name, chunk_index)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_policy_chunks_embedding ON policy_chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )


def is_empty() -> bool:
    with _pool.connection() as conn:
        return conn.execute("SELECT count(*) FROM policy_chunks").fetchone()[0] == 0


def ingest(chunks: list[dict], embeddings: list) -> None:
    with _pool.connection() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO policy_chunks (doc_name, chunk_index, heading, content, embedding) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (doc_name, chunk_index) DO NOTHING",
            [
                (c["doc_name"], c["chunk_index"], c["heading"], c["content"], emb)
                for c, emb in zip(chunks, embeddings)
            ],
        )


def search(query_embedding, top_k: int) -> list[dict]:
    with _pool.connection() as conn:
        rows = conn.execute(
            "SELECT doc_name, chunk_index, heading, content, 1 - (embedding <=> %s) AS similarity "
            "FROM policy_chunks ORDER BY embedding <=> %s LIMIT %s",
            (query_embedding, query_embedding, top_k),
        ).fetchall()
    return [
        {"doc_name": r[0], "chunk_index": r[1], "heading": r[2], "content": r[3], "similarity": round(r[4], 4)}
        for r in rows
    ]
