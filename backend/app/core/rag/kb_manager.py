from functools import wraps
from typing import Any, Dict, Optional, TypedDict
import logging
import threading
import time
import traceback

import numpy as np

try:
    from .bm25_index import build_bm25_corpus, build_bm25_index
    from .chunker import build_chunks
    from .embedder import embed_texts
    from .graph_builder import (
        build_knowledge_graph,
        load_graph_from_disk,
        save_graph_to_disk,
    )
    from .vector_store import (
        build_cluster_indexes,
        build_faiss_index,
        load_kb_from_disk,
        save_kb_to_disk,
    )
except ImportError:
    from bm25_index import build_bm25_corpus, build_bm25_index
    from chunker import build_chunks
    from embedder import embed_texts
    from graph_builder import (
        build_knowledge_graph,
        load_graph_from_disk,
        save_graph_to_disk,
    )
    from vector_store import (
        build_cluster_indexes,
        build_faiss_index,
        load_kb_from_disk,
        save_kb_to_disk,
    )

class VersionedKB(TypedDict):
    versions: Dict[str, Dict[str, Any]]
    active_version: Optional[str]
    locks: Dict[str, threading.Lock]

KB_CACHE: Dict[str, VersionedKB] = {}

#KB_CACHE: Dict[str, Dict[str, Any]] = {}
KB_CACHE_LOCKS: Dict[str, threading.Lock] = {}
KB_CACHE_LOCK = threading.Lock()

logger = logging.getLogger(__name__)


def track_node_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[{func.__name__}] Выполнено за {elapsed:.3f} сек.")
        return result

    return wrapper


@track_node_time
def ingest_request(state: Dict[str, Any]) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    warnings = list(state.get("warnings", []))

    if not state.get("query", "").strip():
        errors.append("Не передан query")

    if not state.get("guideline_paths"):
        errors.append("Не переданы пути к гайдлайнам")

    return {
        **state,
        "errors": errors,
        "warnings": warnings,
    }


def load_pdf_documents(folder_path: str):
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

    pdf_loader = DirectoryLoader(
        folder_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    return pdf_loader.load()


def build_kb(folder_path: str, use_bm25: bool = True) -> Dict[str, Any]:
    docs = load_pdf_documents(folder_path)
    chunks = build_chunks(docs)
    texts = [chunk["text"] for chunk in chunks]

    #Считаем эмбеддинги ОДИН раз для всех чанков
    embeddings = embed_texts(texts) if texts else np.empty((0, 0), dtype=np.float32)
    text_to_emb = {t: e for t, e in zip(texts, embeddings)}
    faiss_index, dim = build_faiss_index(embeddings)
    cluster_indexes = build_cluster_indexes(chunks, text_to_emb=text_to_emb)

    graph = build_knowledge_graph(chunks)

    kb: Dict[str, Any] = {
        "docs": docs,
        "chunks": chunks,
        "faiss_index": faiss_index,
        "dim": dim,
        "knowledge_graph": graph,
        "cluster_indexes": cluster_indexes,
    }

    if use_bm25 and texts:
        corpus = build_bm25_corpus(texts)
        kb["bm25_corpus"] = corpus
        kb["bm25_index"] = build_bm25_index(corpus)

    return kb


def _get_or_create_lock(docs_path: str) -> threading.Lock:
    """Get or create a lock for a specific docs_path."""
    with KB_CACHE_LOCK:
        if docs_path not in KB_CACHE_LOCKS:
            KB_CACHE_LOCKS[docs_path] = threading.Lock()
        return KB_CACHE_LOCKS[docs_path]


def _initialize_kb_sync(docs_path: str, use_bm25: bool) -> Dict[str, Any]:
    """Synchronous KB initialization logic."""
    kb = load_kb_from_disk(docs_path, use_bm25=use_bm25)

    if kb is None:
        print("KB cache not found. Building from scratch...")
        kb = build_kb(docs_path, use_bm25=use_bm25)
        save_kb_to_disk(docs_path, kb, use_bm25=use_bm25)

        if kb.get("knowledge_graph") is not None:
            save_graph_to_disk(docs_path, kb["knowledge_graph"])
    else:
        print("KB loaded from disk cache.")
        graph = load_graph_from_disk(docs_path)
        if graph is None:
            print("Graph cache not found. Building graph from chunks...")
            graph = build_knowledge_graph(kb["chunks"])
            save_graph_to_disk(docs_path, graph)

        kb["knowledge_graph"] = graph

    return kb


@track_node_time
def initialize_kb(state: Dict[str, Any]) -> Dict[str, Any]:
    warnings = list(state.get("warnings", []))
    errors = list(state.get("errors", []))
    paths = state.get("guideline_paths", [])

    if not paths:
        return {
            **state,
            "errors": errors + ["Не переданы пути к гайдлайнам"],
            "warnings": warnings,
        }

    docs_path = paths[0]
    use_bm25 = True

    try:
        # Get or create lock for this specific docs_path
        lock = _get_or_create_lock(docs_path)

        # Acquire lock for this specific docs_path
        with lock:
            if docs_path not in KB_CACHE:
                # Double-check after acquiring lock
                kb = _initialize_kb_sync(docs_path, use_bm25)
                KB_CACHE[docs_path] = kb
            kb = KB_CACHE[docs_path]

        if not kb.get("chunks"):
            warnings.append("После разбиения документов не получено ни одного чанка")

        if "knowledge_graph" not in kb:
            warnings.append("Граф знаний не был построен")

        return {
            **state,
            "chunks_count": len(kb.get("chunks", [])),
            "warnings": warnings,
            "errors": errors,
        }

    except Exception as exc:
        logger.exception("Не удалось инициализировать KB для %s", docs_path)
        tb_text = traceback.format_exc()
        return {
            **state,
            "warnings": warnings + [f"Не удалось инициализировать KB: {exc}\n\nTraceback:\n{tb_text}"],
            "errors": errors,
            "chunks_count": 0,
        }
