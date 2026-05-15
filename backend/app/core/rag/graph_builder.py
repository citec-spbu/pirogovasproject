from pathlib import Path
from typing import Any, Dict, List, Optional
import pickle

import networkx as nx

try:
    from .chunker import extract_entities_from_text
except ImportError:
    from chunker import extract_entities_from_text


KB_GRAPH_DISK_CACHE_DIR = Path(".kb_cache_graph")
def _entity_node_id(entity: Dict[str, str]) -> str:
    return f"entity::{entity['type']}::{entity['name']}"


def _chunk_node_id(source: str, chunk_id: int) -> str:
    return f"chunk::{source}::{chunk_id}"


def get_graph_cache_path(folder_path: str) -> Path:
    safe_name = Path(folder_path).name.replace(" ", "_")
    return KB_GRAPH_DISK_CACHE_DIR / f"{safe_name}_graph.pkl"


def save_graph_to_disk(folder_path: str, graph: nx.Graph) -> None:
    path = get_graph_cache_path(folder_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(graph, file)


def load_graph_from_disk(folder_path: str) -> Optional[nx.Graph]:
    path = get_graph_cache_path(folder_path)
    if not path.exists():
        return None

    with open(path, "rb") as file:
        return pickle.load(file)


def build_knowledge_graph(chunks: List[Dict[str, Any]]) -> nx.Graph:
    graph = nx.Graph()

    for idx, chunk in enumerate(chunks):
        source = chunk["source"]
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]
        chunk_node = _chunk_node_id(source, chunk_id)

        graph.add_node(
            chunk_node,
            node_type="CHUNK",
            source=source,
            chunk_id=chunk_id,
            text=text,
            index=idx,
        )

        entities = extract_entities_from_text(text)

        for entity in entities:
            entity_node = _entity_node_id(entity)
            graph.add_node(
                entity_node,
                node_type="ENTITY",
                entity_type=entity["type"],
                name=entity["name"],
            )
            graph.add_edge(
                chunk_node,
                entity_node,
                relation="MENTIONS",
                weight=1.0,
            )

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                e1 = _entity_node_id(entities[i])
                e2 = _entity_node_id(entities[j])

                if graph.has_edge(e1, e2):
                    graph[e1][e2]["weight"] += 1.0
                else:
                    graph.add_edge(
                        e1,
                        e2,
                        relation="CO_OCCURS",
                        weight=1.0,
                    )

    return graph


def graph_expand_from_chunks(
    seed_results: List[Dict[str, Any]],
    kb: Dict[str, Any],
    max_hops: int = 2,
    max_graph_chunks: int = 4,
) -> List[Dict[str, Any]]:
    graph = kb.get("knowledge_graph")
    chunks = kb.get("chunks", [])

    if graph is None or not seed_results:
        return []

    chunk_lookup = {
        (chunk["source"], chunk["chunk_id"]): chunk
        for chunk in chunks
    }
    graph_candidates: Dict[Any, Dict[str, Any]] = {}

    for result in seed_results:
        source = result["source"]
        chunk_id = result["chunk_id"]
        seed_node = _chunk_node_id(source, chunk_id)

        if seed_node not in graph:
            continue

        lengths = nx.single_source_shortest_path_length(
            graph,
            seed_node,
            cutoff=max_hops,
        )

        for node, distance in lengths.items():
            if not str(node).startswith("chunk::"):
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
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:max_graph_chunks]
