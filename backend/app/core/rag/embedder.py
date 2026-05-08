import numpy as np
from sentence_transformers import SentenceTransformer, models, CrossEncoder
from app.core.config import settings

class EmbeddingService:
    def __init__(self, model_name: str = None, device: str = "cpu"):
        model_name = model_name or settings.EMBEDDING_MODEL_NAME
        word_embedding_model = models.Transformer(model_name)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )
        self.model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model],
            device=device
        )
        self.dim = self.model.get_sentence_embedding_dimension()

        ce_name = settings.CROSS_ENCODER_MODEL_NAME
        self.cross_encoder = CrossEncoder(ce_name, device=device)

    def encode(self, texts: list[str], is_query: bool = False,
               normalize: bool = None, batch_size: int = None) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        normalize = normalize if normalize is not None else settings.EMBEDDING_NORMALIZE
        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

        prefix = "query: " if is_query else "passage: "
        prepared = [prefix + t for t in texts]

        embeddings = self.model.encode(prepared, normalize_embeddings=normalize, batch_size=batch_size,
                                       convert_to_numpy=True, show_progress_bar=False)
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings.astype("float32")

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        pairs = [[query, c] for c in candidates]
        return self.cross_encoder.predict(pairs, show_progress_bar=False).tolist()
