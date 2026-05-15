from typing import List, Optional
import re


def _tokenize_bm25(text: str) -> List[str]:
    return re.findall(r"[а-яА-ЯёЁa-zA-Z0-9]{2,}", text.lower())


def build_bm25_corpus(texts: List[str]) -> List[List[str]]:
    return [_tokenize_bm25(text) for text in texts]


def build_bm25_index(corpus: List[List[str]]):
    if not corpus:
        return None

    from rank_bm25 import BM25Okapi

    return BM25Okapi(corpus)
