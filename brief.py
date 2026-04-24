"""
engine/rag.py
Optional project indexing using ChromaDB and Ollama embeddings.

Gracefully degrades if chromadb is not installed — is_available()
returns False and all other functions become safe no-ops or raise
ImportError with a clear message.

Embedding endpoint (v3.2 spec):
  POST http://localhost:11434/api/embeddings
  body: {"model": "nomic-embed-text", "prompt": <chunk_text>}
  One request per chunk — /api/embeddings does not support batching.

Storage: ~/.vibrestudio/chroma_db/
"""

import re
import requests
from pathlib import Path
from engine.logger import log
from core.config import OLLAMA_URL

# ---------------------------------------------------------------------------
# File types to index
# ---------------------------------------------------------------------------
INDEXED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".cpp", ".c", ".h", ".cs",
    ".go", ".rs", ".sql",
    ".md", ".yaml", ".yml", ".toml", ".json",
}

# Directories to skip entirely during walk
SKIP_DIRS = {
    "node_modules", "__pycache__", ".git",
    "venv", ".venv", "dist", "build",
}

# Symbol patterns for reranking boost
SYMBOL_PATTERNS = [
    r"^(?:async )?def\s+(\w+)",
    r"^class\s+(\w+)",
    r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)",
    r"^(?:export\s+)?const\s+(\w+)\s*=",
    r"^func\s+(\w+)",
]

CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 200
EMBED_MODEL   = "nomic-embed-text"
CHROMA_DIR    = Path.home() / ".vibrestudio" / "chroma_db"


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """
    Returns True if chromadb is installed and importable.
    Called by the sidebar to decide whether to show the RAG section.
    Never raises.
    """
    try:
        import chromadb  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_client():
    """Returns a persistent ChromaDB client. Raises ImportError if unavailable."""
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _get_or_create_collection(client, repo_path: str):
    """Returns (or creates) a ChromaDB collection keyed to the repo path."""
    # Collection name: sanitised path — alphanumeric + underscores only
    name = re.sub(r"[^a-zA-Z0-9_]", "_", str(repo_path))[-60:]
    return client.get_or_create_collection(name=name)


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Splits text into overlapping chunks of approximately `size` characters."""
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def _embed(text: str) -> list[float]:
    """
    Embeds a single text string via Ollama /api/embeddings.
    Returns the embedding vector.
    Raises on network failure or unexpected response shape.
    """
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def _extract_symbols(text: str) -> set[str]:
    """Extracts function/class/const names from a code chunk for reranking."""
    symbols = set()
    for line in text.splitlines():
        for pattern in SYMBOL_PATTERNS:
            m = re.match(pattern, line.strip())
            if m:
                symbols.add(m.group(1))
    return symbols


def _extract_imports(text: str) -> set[str]:
    """Extracts imported module/symbol names for reranking."""
    imports = set()
    for line in text.splitlines():
        # Python: import x, from x import y
        m = re.match(r"^(?:import|from)\s+([\w.]+)", line.strip())
        if m:
            imports.add(m.group(1).split(".")[0])
        # JS/TS: import x from 'y'
        m2 = re.match(r"^import\s+.*\s+from\s+['\"](.+?)['\"]", line.strip())
        if m2:
            imports.add(Path(m2.group(1)).stem)
    return imports


def _collect_files(repo_path: str) -> list[Path]:
    """Walks repo_path, skipping excluded dirs, returning indexable files."""
    root = Path(repo_path)
    files = []
    for p in root.rglob("*"):
        # Skip excluded directories anywhere in the path
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_file() and p.suffix.lower() in INDEXED_EXTENSIONS:
            files.append(p)
    return files


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def index_project(repo_path: str, progress_callback=None) -> int:
    """
    Indexes all code files in repo_path into ChromaDB.

    Each file is split into overlapping chunks of ~1200 chars.
    Each chunk is embedded via Ollama /api/embeddings (one request per chunk).
    Progress is logged every 50 chunks and reported via progress_callback.

    Args:
        repo_path:         Absolute path to the project root.
        progress_callback: Optional callable(msg: str, pct: float).

    Returns:
        Number of chunks indexed.
    """
    client     = _get_client()
    collection = _get_or_create_collection(client, repo_path)
    files      = _collect_files(repo_path)

    log(f"[rag] index_project: {len(files)} files found in {repo_path}")

    chunk_count = 0
    doc_ids     = []
    embeddings  = []
    documents   = []
    metadatas   = []

    for file_idx, filepath in enumerate(files):
        try:
            text   = filepath.read_text(encoding="utf-8", errors="ignore")
            chunks = _chunk_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                doc_id = f"{filepath}::{chunk_idx}"
                try:
                    embedding = _embed(chunk)
                except Exception as e:
                    log(f"[rag] embed failed for {filepath}:{chunk_idx}: {e}")
                    continue

                doc_ids.append(doc_id)
                embeddings.append(embedding)
                documents.append(chunk)
                metadatas.append({
                    "file":    str(filepath),
                    "chunk":   chunk_idx,
                    "symbols": ",".join(_extract_symbols(chunk)),
                    "imports": ",".join(_extract_imports(chunk)),
                })
                chunk_count += 1

                if chunk_count % 50 == 0:
                    msg = f"Indexed {chunk_count} chunks ({file_idx + 1}/{len(files)} files)…"
                    log(f"[rag] {msg}")
                    if progress_callback:
                        progress_callback(msg, chunk_count)

                    # Flush batch to ChromaDB
                    collection.upsert(
                        ids=doc_ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas,
                    )
                    doc_ids, embeddings, documents, metadatas = [], [], [], []

        except Exception as e:
            log(f"[rag] skipped {filepath}: {e}")
            continue

    # Flush remaining
    if doc_ids:
        collection.upsert(
            ids=doc_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    log(f"[rag] index_project complete: {chunk_count} chunks indexed")
    return chunk_count


def query_project(repo_path: str, question: str, top_k: int = 6) -> str:
    """
    Embeds the question and queries ChromaDB for the most relevant chunks.

    Applies symbol-boosted reranking:
      +0.05 per symbol shared between question tokens and chunk symbols
      +0.03 per import shared between question tokens and chunk imports

    Returns a formatted context string ready to inject into a prompt,
    or an empty string if nothing relevant is found.
    """
    client     = _get_client()
    collection = _get_or_create_collection(client, repo_path)

    if collection.count() == 0:
        log("[rag] query_project: collection is empty — index the project first")
        return ""

    try:
        question_embedding = _embed(question)
    except Exception as e:
        log(f"[rag] embed question failed: {e}")
        return ""

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k * 2, collection.count()),  # fetch extra for reranking
        include=["documents", "metadatas", "distances"],
    )

    if not results or not results["documents"] or not results["documents"][0]:
        return ""

    # Symbol-boosted reranking
    question_tokens = set(question.lower().split())
    scored = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        base_score = 1.0 - dist  # ChromaDB returns L2 distance; lower = better
        boost = 0.0
        chunk_symbols = set(s for s in meta.get("symbols", "").split(",") if s)
        chunk_imports = set(i for i in meta.get("imports", "").split(",") if i)
        boost += sum(0.05 for s in chunk_symbols if s.lower() in question_tokens)
        boost += sum(0.03 for i in chunk_imports if i.lower() in question_tokens)
        scored.append((base_score + boost, doc, meta))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_results = scored[:top_k]

    if not top_results:
        return ""

    parts = []
    for score, doc, meta in top_results:
        file_name = Path(meta.get("file", "unknown")).name
        parts.append(f"--- {file_name} (relevance: {score:.2f}) ---\n{doc}")

    context = "\n\n".join(parts)
    log(f"[rag] query_project: returned {len(top_results)} chunks for '{question[:60]}'")
    return context
