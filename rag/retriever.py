"""
RAG over Dubba's policy docs. Chroma (in-memory, no server process) + sentence-
transformers embeddings. Chunked by markdown "## " sections -- the natural semantic
boundary these docs already use.

No Langfuse calls in this file on purpose: this module runs inside the MCP server
subprocess (mcp_server/server.py), which only inherits a minimal env allowlist
(HOME/LOGNAME/PATH/SHELL/TERM/USER) -- no LANGFUSE_* credentials -- so a span opened
here would silently go nowhere. Traceability instead comes from (a) the stderr log
below (the assignment's literal "log or print which chunk(s) got used" requirement --
stderr, never stdout: stdio MCP servers use stdout as the actual JSON-RPC protocol
channel, so printing there corrupts the message stream) and (b) the existing Langfuse
"tool" span in harness/graph.py's execute_tool_node, which already captures this
function's full output (including chunk ids + similarity) in the parent process,
which does have credentials.

Deliberately uncovered: none of the 6 policy docs mention international customs fees
or import duties -- see log/DECISIONS.md. That's the "honest gap" case (assignment
2.3) -- prompts.py instructs the model not to stretch an adjacent doc to cover it.
"""

import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR = Path(__file__).resolve().parent / "policy_docs"
# 3 sometimes missed the right chunk for candle-specific condition questions (e.g.
# "already lit" competing against the subscription doc's "candle" mention) --
# verified empirically that 4 reliably surfaces it across phrasings; bumped to 5 to
# give MIN_SIMILARITY (below) more headroom. See log/DECISIONS.md.
TOP_K = 5
# Placeholder value, explicitly provisional (log/DECISIONS.md) -- known tradeoff:
# the "already lit" return-eligibility chunk scores ~0.41 for its correct answer, so
# it will now be filtered out below this threshold even though it's genuinely
# relevant. Chosen anyway to start enforcing a real code-level cutoff rather than
# relying only on the prompt; will be tuned once real query volume exists.
MIN_SIMILARITY = 0.6

_client = chromadb.Client()
_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def _chunk_doc(path: Path) -> list[dict]:
    """Split one doc into chunks by '## ' section headers. The intro text before the
    first '## ' (every doc here opens with '## Overview') becomes the first chunk."""
    lines = path.read_text().splitlines()
    title = lines[0].lstrip("#").strip() if lines and lines[0].startswith("#") else path.stem

    chunks: list[dict] = []
    heading = "Overview"
    body_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(body_lines).strip()
        if body:
            chunks.append({
                "doc_name": path.stem,
                "heading": heading,
                "content": f"# {title}\n## {heading}\n{body}",
            })

    for line in lines[1:]:
        if line.startswith("## "):
            flush()
            heading = line[3:].strip()
            body_lines = []
        else:
            body_lines.append(line)
    flush()

    for i, chunk in enumerate(chunks):
        chunk["chunk_index"] = i
    return chunks


def _build_collection():
    collection = _client.get_or_create_collection(name="policy_docs", embedding_function=_embed_fn)
    if collection.count() > 0:
        return collection

    all_chunks = [chunk for path in sorted(DOCS_DIR.glob("*.md")) for chunk in _chunk_doc(path)]
    collection.add(
        ids=[f"{c['doc_name']}_{c['chunk_index']}" for c in all_chunks],
        documents=[c["content"] for c in all_chunks],
        metadatas=[
            {"doc_name": c["doc_name"], "chunk_index": c["chunk_index"], "heading": c["heading"]}
            for c in all_chunks
        ],
    )
    return collection


_collection = _build_collection()


def retrieve(query: str, top_k: int = TOP_K, min_similarity: float = MIN_SIMILARITY) -> list[dict]:
    results = _collection.query(query_texts=[query], n_results=top_k)

    all_chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        all_chunks.append({
            "doc_name": meta["doc_name"],
            "chunk_index": meta["chunk_index"],
            "heading": meta["heading"],
            "content": doc,
            "similarity": round(1 - dist, 4),
        })

    chunks = [c for c in all_chunks if c["similarity"] >= min_similarity]
    dropped = [c for c in all_chunks if c["similarity"] < min_similarity]

    # stderr, not stdout -- stdout is the MCP stdio protocol channel; printing there
    # corrupts the JSON-RPC message stream.
    print(f"[RAG] query={query!r} -> chunks used: "
          f"{[(c['doc_name'], c['chunk_index'], c['heading'], c['similarity']) for c in chunks]}"
          f" | dropped below min_similarity={min_similarity}: "
          f"{[(c['doc_name'], c['chunk_index'], c['similarity']) for c in dropped]}",
          file=sys.stderr)

    return chunks
