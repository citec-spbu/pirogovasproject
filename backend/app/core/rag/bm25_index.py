import re
from typing import List, Optional
import numpy as np
from rank_bm25 import BM25Okapi

class BM25Manager:
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

