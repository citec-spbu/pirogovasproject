from pathlib import Path
import faiss
import numpy as np
from app.core.config import settings

class VectorIndexManager:
    def __init__(self, dim: int, metric: str = "cosine"):
        self.dim = dim
        self.metric = metric
        self.cache_dir = Path(settings.KB_CACHE_ROOT)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        if embeddings is None or len(embeddings) == 0:
            return faiss.IndexHNSWFlat(self.dim, 32)

        index = faiss.IndexHNSWFlat(self.dim, 32)
        index.add(embeddings)
        return index

    def search(self, index: faiss.Index, query_embedding: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if index.ntotal == 0:
            return np.array([]), np.array([])
        k_search = min(k, index.ntotal)
        scores, indices = index.search(query_embedding, k_search)
        return scores, indices

    def save_index(self, index: faiss.Index, prefix: str) -> Path:
        path = self.cache_dir / f"{prefix}_faiss.index"
        faiss.write_index(index, str(path))
        return path

    def load_index(self, prefix: str) -> faiss.Index | None:
        path = self.cache_dir / f"{prefix}_faiss.index"
        if not path.exists():
            return None
        return faiss.read_index(str(path))

    def get_cache_path(self, prefix: str) -> Path:
        return self.cache_dir / f"{prefix}_faiss.index"