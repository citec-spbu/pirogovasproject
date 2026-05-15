import os
from typing import List, Optional

import numpy as np


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
_EMBEDDER = None


def get_embedder():
    """Loads the embedding model only when embeddings are really needed."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer, models

        word_embedding_model = models.Transformer(EMBEDDING_MODEL_NAME)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        _EMBEDDER = SentenceTransformer(
            modules=[word_embedding_model, pooling_model],
            device="cpu",
        )
    return _EMBEDDER


def embed_texts(texts: List[str], is_query: bool = False) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    prefix = "query: " if is_query else "passage: "
    prepared_texts = [prefix + text for text in texts]

    embeddings = get_embedder().encode(
        prepared_texts,
        normalize_embeddings=True,
        batch_size=8,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    return embeddings.astype("float32")
