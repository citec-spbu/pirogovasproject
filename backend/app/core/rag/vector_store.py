from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import pickle

import numpy as np

try:
    from .bm25_index import build_bm25_corpus, build_bm25_index
    from .embedder import embed_texts
except ImportError:
    from bm25_index import build_bm25_corpus, build_bm25_index
    from embedder import embed_texts


KB_DISK_CACHE_DIR = Path(".kb_cache")
KB_BM25_DISK_CACHE_DIR = Path(".kb_cache_bm25")
def _get_faiss():
    import faiss

    return faiss


def get_cache_prefix(folder_path: str, use_bm25: bool = False) -> Path:
    base_dir = KB_BM25_DISK_CACHE_DIR if use_bm25 else KB_DISK_CACHE_DIR
    safe_name = Path(folder_path).name.replace(" ", "_")
    return base_dir / safe_name


def build_faiss_index(embeddings: np.ndarray):
    faiss = _get_faiss()
    dim = embeddings.shape[1] if len(embeddings) > 0 else 0

    # Return early if no embeddings to avoid creating invalid FAISS index
    if dim == 0:
        return None, 0

    index = faiss.IndexHNSWFlat(dim, 32)
    if len(embeddings) > 0:
        index.add(embeddings)
    return index, dim


def build_cluster_indexes(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    cluster_to_chunks: Dict[str, List[Dict[str, Any]]] = {}

    for chunk in chunks:
        for cluster in chunk.get("clusters", ["general"]):
            cluster_to_chunks.setdefault(cluster, []).append(chunk)

    cluster_indexes: Dict[str, Dict[str, Any]] = {}

    for cluster_name, cluster_chunks in cluster_to_chunks.items():
        texts = [chunk["text"] for chunk in cluster_chunks]
        embeddings = embed_texts(texts) if texts else np.empty((0, 0), dtype=np.float32)

        if len(embeddings) == 0:
            continue

        faiss_index, dim = build_faiss_index(embeddings)
        corpus = build_bm25_corpus(texts)

        cluster_indexes[cluster_name] = {
            "chunks": cluster_chunks,
            "faiss_index": faiss_index,
            "dim": dim,
            "bm25_corpus": corpus,
            "bm25_index": build_bm25_index(corpus),
        }

    return cluster_indexes


def _cluster_cache_dir(prefix: Path) -> Path:
    return Path(f"{prefix}_clusters")


def save_kb_to_disk(folder_path: str, kb: Dict[str, Any], use_bm25: bool = False) -> None:
    faiss = _get_faiss()
    prefix = get_cache_prefix(folder_path, use_bm25)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    with open(f"{prefix}_chunks.json", "w", encoding="utf-8") as file:
        json.dump(kb["chunks"], file, ensure_ascii=False, indent=2, default=str)

    if kb.get("faiss_index") is not None:
        faiss.write_index(kb["faiss_index"], f"{prefix}_faiss.index")

    if use_bm25 and "bm25_corpus" in kb:
        with open(f"{prefix}_bm25_corpus.pkl", "wb") as file:
            pickle.dump(kb["bm25_corpus"], file)

    cluster_indexes = kb.get("cluster_indexes", {})
    if cluster_indexes:
        cluster_dir = _cluster_cache_dir(prefix)
        cluster_dir.mkdir(parents=True, exist_ok=True)

        for cluster_name, cluster_kb in cluster_indexes.items():
            safe_cluster_name = cluster_name.replace(" ", "_")
            cluster_prefix = cluster_dir / safe_cluster_name

            # Save original cluster name in metadata file
            with open(f"{cluster_prefix}_meta.json", "w", encoding="utf-8") as file:
                json.dump({"original_name": cluster_name}, file, ensure_ascii=False)

            with open(f"{cluster_prefix}_chunks.json", "w", encoding="utf-8") as file:
                json.dump(cluster_kb["chunks"], file, ensure_ascii=False, indent=2, default=str)

            if cluster_kb.get("faiss_index") is not None:
                faiss.write_index(cluster_kb["faiss_index"], f"{cluster_prefix}_faiss.index")

            if "bm25_corpus" in cluster_kb:
                with open(f"{cluster_prefix}_bm25_corpus.pkl", "wb") as file:
                    pickle.dump(cluster_kb["bm25_corpus"], file)


def _load_cluster_indexes(prefix: Path) -> Dict[str, Dict[str, Any]]:
    faiss = _get_faiss()
    cluster_dir = _cluster_cache_dir(prefix)
    if not cluster_dir.exists():
        return {}

    cluster_indexes: Dict[str, Dict[str, Any]] = {}
    for chunks_file in cluster_dir.glob("*_chunks.json"):
        cluster_name_safe = chunks_file.name[: -len("_chunks.json")]
        meta_file = cluster_dir / f"{cluster_name_safe}_meta.json"
        index_file = cluster_dir / f"{cluster_name_safe}_faiss.index"
        corpus_file = cluster_dir / f"{cluster_name_safe}_bm25_corpus.pkl"

        if not index_file.exists():
            continue

        # Load original cluster name from metadata file if it exists
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as file:
                meta = json.load(file)
                cluster_name = meta.get("original_name", cluster_name_safe)
        else:
            # Fallback to filename-derived name if metadata file is missing
            cluster_name = cluster_name_safe

        with open(chunks_file, "r", encoding="utf-8") as file:
            chunks = json.load(file)

        index = faiss.read_index(str(index_file))
        cluster_kb: Dict[str, Any] = {
            "chunks": chunks,
            "faiss_index": index,
            "dim": index.d,
        }

        if corpus_file.exists():
            with open(corpus_file, "rb") as file:
                corpus = pickle.load(file)
            cluster_kb["bm25_corpus"] = corpus
            cluster_kb["bm25_index"] = build_bm25_index(corpus)

        cluster_indexes[cluster_name] = cluster_kb

    return cluster_indexes


def load_kb_from_disk(folder_path: str, use_bm25: bool = False) -> Optional[Dict[str, Any]]:
    faiss = _get_faiss()
    prefix = get_cache_prefix(folder_path, use_bm25)
    chunks_file = Path(f"{prefix}_chunks.json")
    index_file = Path(f"{prefix}_faiss.index")

    if not (chunks_file.exists() and index_file.exists()):
        return None

    with open(chunks_file, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    index = faiss.read_index(str(index_file))
    kb: Dict[str, Any] = {
        "chunks": chunks,
        "faiss_index": index,
        "dim": index.d,
        "cluster_indexes": _load_cluster_indexes(prefix),
    }

    if use_bm25:
        corpus_file = Path(f"{prefix}_bm25_corpus.pkl")
        if corpus_file.exists():
            with open(corpus_file, "rb") as file:
                corpus = pickle.load(file)
            kb["bm25_corpus"] = corpus
            kb["bm25_index"] = build_bm25_index(corpus)

    return kb
