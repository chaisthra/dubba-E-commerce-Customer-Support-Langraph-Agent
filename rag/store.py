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
"""

import os

import psycopg
from pgvector.psycopg import register_vector

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2's output dimension


def connect() -> psycopg.Connection:
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
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
    conn.commit()
    return conn


def is_empty(conn: psycopg.Connection) -> bool:
    return conn.execute("SELECT count(*) FROM policy_chunks").fetchone()[0] == 0


def ingest(conn: psycopg.Connection, chunks: list[dict], embeddings: list) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO policy_chunks (doc_name, chunk_index, heading, content, embedding) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (doc_name, chunk_index) DO NOTHING",
            [
                (c["doc_name"], c["chunk_index"], c["heading"], c["content"], emb)
                for c, emb in zip(chunks, embeddings)
            ],
        )
    conn.commit()


def search(conn: psycopg.Connection, query_embedding, top_k: int) -> list[dict]:
    rows = conn.execute(
        "SELECT doc_name, chunk_index, heading, content, 1 - (embedding <=> %s) AS similarity "
        "FROM policy_chunks ORDER BY embedding <=> %s LIMIT %s",
        (query_embedding, query_embedding, top_k),
    ).fetchall()
    return [
        {"doc_name": r[0], "chunk_index": r[1], "heading": r[2], "content": r[3], "similarity": round(r[4], 4)}
        for r in rows
    ]
