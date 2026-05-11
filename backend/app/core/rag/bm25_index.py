import re
import json
from pathlib import Path
from typing import List, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from app.core.config import get_settings

settings = get_settings()

class BM25Manager:
    def __init__(self):
        self.cache_root = Path(settings.KB_CACHE_BM25_ROOT)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r'[а-яА-ЯёЁa-zA-Z0-9]{2,}', text.lower())

    def build_corpus(self, texts: List[str]) -> List[List[str]]:
        if not texts:
            return []
        return [self.tokenize(t) for t in texts]

    def build_index(self, corpus: List[List[str]]) -> Optional[BM25Okapi]:
        if not corpus:
            return None
        return BM25Okapi(corpus)

    def get_scores(self, index: BM25Okapi, query: str) -> np.ndarray:
        if index is None:
            return np.array([], dtype=np.float32)
        q_tokens = self.tokenize(query)
        return np.array(index.get_scores(q_tokens), dtype=np.float32)

    def get_ranked_indices(self, index: BM25Okapi, query: str) -> np.ndarray:
        scores = self.get_scores(index, query)
        if len(scores) == 0:
            return np.array([], dtype=int)
        return np.argsort(scores)[::-1]

    def save_corpus(self, corpus: List[List[str]], prefix: str) -> None:
        corpus_file = self.cache_root / f"{prefix}_corpus.json"
        with open(corpus_file, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)

    def load_corpus(self, prefix: str) -> Optional[List[List[str]]]:
        corpus_file = self.cache_root / f"{prefix}_corpus.json"
        if not corpus_file.exists():
            return None
        with open(corpus_file, "r", encoding="utf-8") as f:
            return json.load(f)

