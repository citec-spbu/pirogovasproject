import re
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from .embedder import EmbeddingService
from .graph_builder import _chunk_node_id, KnowledgeGraphBuilder
import networkx as nx
from app.core.config import get_settings

settings = get_settings()

class HybridRetriever:
    #Гибридный поиск: FAISS + BM25 + RRF + CrossEncoder + граф знаний
    def __init__(self, embedder: EmbeddingService, top_k: int = None,
                 min_score: float = None, rrf_k: float = None, ce_weight: float = None):
        self.embedder = embedder
        self.top_k = top_k if top_k is not None else settings.TOP_K_RETRIEVAL
        self.min_score = min_score if min_score is not None else settings.MIN_RELEVANCE_SCORE
        self.rrf_k = rrf_k if rrf_k is not None else settings.RRF_K
        self.ce_weight = ce_weight if ce_weight is not None else settings.CROSS_ENCODER_WEIGHT
        self.cross_encoder = CrossEncoder(settings.CROSS_ENCODER_MODEL_NAME, device="cpu")

    def _apply_rrf(self, ranks: np.ndarray, target: np.ndarray):
        for rank, idx in enumerate(ranks, start=1):
            target[idx] += 1.0 / (self.rrf_k + rank)

    def _apply_json_heuristics(self, candidates: List[Dict[str, Any]], keywords_from_json: set) -> List[Dict[str, Any]]:
        for c in candidates:
            chunk_lower = c["text"].lower()
            chunk_normalized = KnowledgeGraphBuilder._normalize_text(c["text"])
            # NER-бонус
            matches = keywords_from_json.intersection(chunk_normalized)
            c["score"] += len(matches) * 0.05
            # Бонус за числовые нормативы
            numeric_patterns = [
                r'\d{1,2}(\.\d)?\s?мм',
                r'[><=]\s?\d{1,2}',
                r'от\s\d{1,2}\sдо\s\d{1,2}'
            ]
            for pattern in numeric_patterns:
                if re.search(pattern, chunk_lower):
                    c["score"] += 0.15
            # Штраф за картинки
            stop_patterns = ["рис.", "рисунок", "вид сбоку", "снимок", "визуализация", "иллюстрация", "график"]
            for stop_word in stop_patterns:
                if stop_word in chunk_lower:
                    c["score"] -= 0.10
            # Штраф за ссылки на литературу
            reference_matches = re.findall(r'\[[\d,\s\-]+\]', chunk_lower)
            if reference_matches:
                c["score"] -= len(reference_matches) * 0.005

        return candidates

    def hybrid_search(self, query: str, chunks: List[Dict[str, Any]], faiss_index,
                    bm25_index=None, top_k: int = None) -> List[Dict[str, Any]]:
        if not chunks or faiss_index is None or faiss_index.ntotal == 0:
            return []
        top_k = top_k or self.top_k

        # 1. FAISS cosine similarity
        q_emb = self.embedder.encode([query], is_query=True)
        k_search = min(max(len(chunks), top_k * 3), faiss_index.ntotal)
        cos_scores, cos_indices = faiss_index.search(q_emb, k_search)
        cos_scores, cos_indices = cos_scores[0], cos_indices[0]

        # Filter out invalid indices from FAISS
        valid_mask = (cos_indices >= 0) & (cos_indices < len(chunks))
        cos_indices = cos_indices[valid_mask]

        # 2. BM25
        bm25_indices = None
        if bm25_index:
            from app.core.rag.bm25_index import BM25Manager
            query_tokens = BM25Manager.tokenize(query)
            bm25_scores = bm25_index.get_scores(query_tokens)
            bm25_indices = np.argsort(bm25_scores)[::-1]

            # Filter out invalid indices from BM25
            valid_mask = (bm25_indices >= 0) & (bm25_indices < len(chunks))
            bm25_indices = bm25_indices[valid_mask]

        # 3. RRF Fusion
        rrf_scores = np.zeros(len(chunks))
        self._apply_rrf(cos_indices, rrf_scores)
        if bm25_indices is not None and bm25_indices.size > 0:
            self._apply_rrf(bm25_indices, rrf_scores)

        candidate_indices = np.argsort(rrf_scores)[::-1][:(top_k * 3)]
        results = []
        for idx in candidate_indices:
            idx = int(idx)
            results.append({
                "source": chunks[idx]["source"],
                "chunk_id": chunks[idx]["chunk_id"],
                "text": chunks[idx]["text"],
                "index": idx,
                "score": float(rrf_scores[idx]),
            })
        # 4. Cross-Encoder reranking
        if results:
            pairs = [[query, r["text"]] for r in results]
            ce_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
            for i, r in enumerate(results):
                r["score"] = (1 - self.ce_weight) * r["score"] + self.ce_weight * float(ce_scores[i])
        results.sort(key=lambda x: x["score"], reverse=True)
        for r in results:
            r.pop("index", None)

        return results[:top_k]

    def graph_expand(self, seed_results: List[Dict[str, Any]], graph, chunks: List[Dict[str, Any]],
                    max_hops: int = 2, max_graph_chunks: int = 8) -> List[Dict[str, Any]]:
        if graph is None or not seed_results:
            return []
        chunk_lookup = {
            (chunk["source"], chunk["chunk_id"]): chunk
            for chunk in chunks
        }
        graph_candidates = {}
        for result in seed_results:
            source = result["source"]
            chunk_id = result["chunk_id"]
            seed_node = _chunk_node_id(source, chunk_id)

            if seed_node not in graph:
                continue

            lengths = nx.single_source_shortest_path_length(graph, seed_node, cutoff=max_hops)

            for node, distance in lengths.items():
                if not node.startswith("chunk::"):
                    continue

                node_data = graph.nodes[node]
                candidate_key = (node_data["source"], node_data["chunk_id"])

                if candidate_key not in chunk_lookup:
                    continue

                graph_score = 1.0 / (1.0 + distance)

                if candidate_key not in graph_candidates:
                    chunk = chunk_lookup[candidate_key]
                    graph_candidates[candidate_key] = {
                        "source": chunk["source"],
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "score": graph_score,
                        "graph_score": graph_score,
                        "graph_path": f"graph_distance={distance}",
                    }
                else:
                    graph_candidates[candidate_key]["score"] += graph_score
                    graph_candidates[candidate_key]["graph_score"] += graph_score

        results = list(graph_candidates.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_graph_chunks]

    def merge_results(self, vector_results: List[Dict[str, Any]], graph_results: List[Dict[str, Any]],
                      final_top_k: int = 8) -> List[Dict[str, Any]]:
        merged = {}
        for item in vector_results:
            key = (item["source"], item["chunk_id"])
            merged[key] = {
                **item,
                "vector_score": item.get("score", 0.0),
                "graph_score": 0.0,
                "final_score": item.get("score", 0.0),
                "retrieval_type": "vector",
            }

        for item in graph_results:
            key = (item["source"], item["chunk_id"])
            if key in merged:
                merged[key]["graph_score"] = item.get("graph_score", item.get("score", 0.0))
                merged[key]["graph_path"] = item.get("graph_path", "")
                merged[key]["retrieval_type"] = "vector+graph"
                merged[key]["final_score"] = (
                        0.65 * merged[key].get("vector_score", 0.0) +
                        0.35 * merged[key].get("graph_score", 0.0)
                )
            else:
                merged[key] = {
                    **item,
                    "vector_score": 0.0,
                    "graph_score": item.get("graph_score", item.get("score", 0.0)),
                    "final_score": 0.35 * item.get("score", 0.0),
                    "retrieval_type": "graph",
                }

        results = list(merged.values())
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:final_top_k]
