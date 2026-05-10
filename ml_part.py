import os
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional
import re
import networkx as nx
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langsmith import Client
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, models, CrossEncoder
import json 
import pymorphy3
import faiss
import pickle
from rank_bm25 import BM25Okapi
import time
from functools import wraps

os.environ["CUDA_VISIBLE_DEVICES"] = ""
load_dotenv()
client = Client()
project_name = "clinical-rag"
runs = client.list_runs(
    project_name=project_name,
    limit=11,
)

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "token-abc123")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CROSS_ENCODER = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device="cpu")

SYMPTOM_TOP_K = 4
JSON_CANDIDATES_TOP_K = 10
JSON_RESULTS_TOP_K = 5
GRAPH_EXPAND_TOP_K = 8
FINAL_GUIDELINES_TOP_K = 4
GUIDELINES_FOR_LLM_TOP_K = 4

MIN_JSON_RELEVANCE_SCORE = 3.0
MIN_FINAL_RELEVANCE_SCORE = 3.0

KB_DISK_CACHE_DIR = Path(".kb_cache")
KB_DISK_CACHE_DIR.mkdir(exist_ok=True)
KB_GRAPH_DISK_CACHE_DIR = Path(".kb_cache_graph")
KB_GRAPH_DISK_CACHE_DIR.mkdir(exist_ok=True)
KB_BM25_DISK_CACHE_DIR = Path(".kb_cache_bm25")
KB_BM25_DISK_CACHE_DIR.mkdir(exist_ok=True)

word_embedding_model = models.Transformer(EMBEDDING_MODEL_NAME)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False, pooling_mode_max_tokens=False)
EMBEDDER = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cpu")

KB_CACHE: Dict[str, Dict[str, Any]] = {}

with open(f"content/0.json", "r", encoding="utf-8") as f:
    patient_data = json.load(f)

class MedGraphState(TypedDict, total=False):
    query: str
    patient_history: str
    guideline_paths: List[str]
    persist_dir: str
    patient_data: Dict[str, Any]

    warnings: List[str]
    errors: List[str]

    retrieved_guidelines: List[Dict[str, Any]]
    fused_context: str
    final_prompt: str
    raw_llm_output: str
    chunks_count: int

MEDICAL_ENTITY_PATTERNS = {
    "ANATOMY": [
        "восходящая аорта",
        "нисходящая аорта",
        "дуга аорты",
        "перешеек аорты",
        "грудная аорта",
        "брюшная аорта",
        "аортальный клапан",
        "левая подключичная артерия",
        "плечеголовной ствол",
    ],
    "DISEASE": [
        "аневризма",
        "расслоение",
        "диссекция",
        "стеноз",
        "разрыв",
        "расширение",
        "тромбоз",
    ],
    "CLINICAL": [
        "боль в груди",
        "боль в спине",
        "боль между лопатками",
        "одышка",
        "кашель",
        "хрипы",
        "затруднённое дыхание",
        "ком в горле",
    ],
    "TACTIC": [
        "наблюдение",
        "кт-контроль",
        "хирургическое лечение",
        "эндоваскулярное лечение",
        "протезирование",
        "стентирование",
        "консультация кардиохирурга",
    ],
    "GUIDELINE": [
        "рекомендуется",
        "показания",
        "противопоказания",
        "порог",
        "норма",
        "нормальный диаметр",
        "риск расслоения",
        "риск разрыва",
    ],
}

CLUSTER_PATTERNS = {
    "thoracic_aorta": [
        "грудная аорта",
        "восходящая аорта",
        "нисходящая аорта",
        "дуга аорты",
        "перешеек аорты",
    ],
    "abdominal_aorta": [
        "брюшная аорта",
        "инфраренальный отдел",
        "супраренальный отдел",
    ],
    "ascending_aorta": [
        "восходящая аорта",
        "аортальный клапан",
        "синусы вальсальвы",
        "синотубулярное соединение",
    ],
    "aortic_arch": [
        "дуга аорты",
        "плечеголовной ствол",
        "левая подключичная артерия",
        "левая общая сонная артерия",
    ],
    "descending_aorta": [
        "нисходящая аорта",
        "грудной отдел аорты",
    ],
    "aneurysm": [
        "аневризма",
        "расширение",
        "дилатация",
    ],
    "dissection": [
        "расслоение",
        "диссекция",
        "ложный просвет",
        "истинный просвет",
    ],
    "stenosis_thrombosis": [
        "стеноз",
        "тромбоз",
        "окклюзия",
    ],
    "surgery_indications": [
        "показания к операции",
        "хирургическое лечение",
        "эндоваскулярное лечение",
        "протезирование",
        "стентирование",
        "порог вмешательства",
    ],
    "follow_up": [
        "наблюдение",
        "кт-контроль",
        "динамическое наблюдение",
        "контроль через",
    ],
}

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
)

def _entity_node_id(entity: Dict[str, str]) -> str:
    return f"entity::{entity['type']}::{entity['name']}"

def assign_clusters_to_text(text: str) -> List[str]:
    text_lower = text.lower()
    clusters = []

    for cluster_name, patterns in CLUSTER_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in text_lower:
                clusters.append(cluster_name)
                break

    if not clusters:
        clusters.append("general")

    return clusters

def track_node_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[{func.__name__}] Выполнено за {elapsed:.3f} сек.")
        return result
    return wrapper

def _chunk_node_id(source: str, chunk_id: int) -> str:
    return f"chunk::{source}::{chunk_id}"

def get_graph_cache_path(folder_path: str) -> Path:
    safe_name = Path(folder_path).name.replace(" ", "_")
    return KB_GRAPH_DISK_CACHE_DIR / f"{safe_name}_graph.pkl"

def save_graph_to_disk(folder_path: str, graph: nx.Graph) -> None:
    path = get_graph_cache_path(folder_path)
    with open(path, "wb") as f:
        pickle.dump(graph, f)

def load_graph_from_disk(folder_path: str) -> Optional[nx.Graph]:
    path = get_graph_cache_path(folder_path)
    if not path.exists():
        return None

    with open(path, "rb") as f:
        return pickle.load(f)

def extract_entities_from_text(text: str) -> List[Dict[str, str]]:
    text_lower = text.lower()
    entities = []

    for entity_type, terms in MEDICAL_ENTITY_PATTERNS.items():
        for term in terms:
            if term in text_lower:
                entities.append({
                    "name": term,
                    "type": entity_type,
                })

    numeric_patterns = re.findall(
        r'(?:>=|<=|≥|≤|>|<)?\s?\d+[.,]?\d*\s?(?:мм|см)',
        text_lower
    )

    for value in numeric_patterns:
        entities.append({
            "name": value.strip(),
            "type": "MEASUREMENT",
        })

    unique = {}
    for e in entities:
        key = (e["name"], e["type"])
        unique[key] = e

    return list(unique.values())

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

        # связываем сущности внутри одного чанка
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

def _tokenize_bm25(text: str) -> List[str]:
    return re.findall(r'[а-яА-ЯёЁa-zA-Z0-9]{2,}', text.lower())

def get_cache_prefix(folder_path: str, use_bm25: bool = False) -> Path:
    base_dir = KB_BM25_DISK_CACHE_DIR if use_bm25 else KB_DISK_CACHE_DIR
    safe_name = Path(folder_path).name.replace(" ", "_")
    return base_dir / safe_name

def save_kb_to_disk(folder_path: str, kb: Dict[str, Any], use_bm25: bool = False) -> None:
    prefix = get_cache_prefix(folder_path, use_bm25)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{prefix}_chunks.json", "w", encoding="utf-8") as f:
        json.dump(kb["chunks"], f, ensure_ascii=False, indent=2)

    if kb.get("faiss_index"):
        faiss.write_index(kb["faiss_index"], f"{prefix}_faiss.index")

    if use_bm25 and "bm25_corpus" in kb:
        with open(f"{prefix}_bm25_corpus.pkl", "wb") as f:
            pickle.dump(kb["bm25_corpus"], f)

def load_kb_from_disk(folder_path: str, use_bm25: bool = False) -> Optional[Dict[str, Any]]:
    prefix = get_cache_prefix(folder_path, use_bm25)
    chunks_file = Path(f"{prefix}_chunks.json")
    index_file = Path(f"{prefix}_faiss.index")

    if not (chunks_file.exists() and index_file.exists()):
        return None

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    index = faiss.read_index(str(index_file))

    kb = {"chunks": chunks, "faiss_index": index, "dim": index.d}
    if use_bm25:
        corpus_file = Path(f"{prefix}_bm25_corpus.pkl")
        if corpus_file.exists():
            with open(corpus_file, "rb") as f:
                corpus = pickle.load(f)
            kb["bm25_corpus"] = corpus
            kb["bm25_index"] = BM25Okapi(corpus)
    return kb

def build_chunks(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    
    split_docs = text_splitter.split_documents(docs)
    source_counters: Dict[str, int] = {}

    for doc in split_docs:
        source_path = doc.metadata.get("source", "unknown")
        source = Path(source_path).name

        chunk_id = source_counters.get(source, 0)
        source_counters[source] = chunk_id + 1

        result.append({
            "source": source,
            "chunk_id": chunk_id,
            "text": doc.page_content,
            "metadata": doc.metadata,
            "clusters": assign_clusters_to_text(doc.page_content),
            "entities": extract_entities_from_text(doc.page_content)
        }) 
    
    return result

def route_query_to_clusters(
    query: str,
    patient_history: str,
    patient_data: Dict[str, Any],
) -> List[str]:
    query_parts = [query, patient_history]

    if isinstance(patient_data, dict):
        query_parts.append(_build_json_query(patient_data))

    full_query = "\n".join(query_parts).lower()

    clusters = []

    for cluster_name, patterns in CLUSTER_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in full_query:
                clusters.append(cluster_name)
                break

    if not clusters:
        clusters.append("general")

    return clusters

def embed_texts(texts: List[str], is_query: bool = False) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    prefix = "query: " if is_query else "passage: "
    prepared_texts = [prefix + t for t in texts]

    embeddings = EMBEDDER.encode(prepared_texts, normalize_embeddings=True, batch_size=8, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings.astype("float32")

def build_kb(folder_path: str, use_bm25: bool = True) -> Dict[str, Any]:
    folder = Path(folder_path)
    PDFloader = DirectoryLoader(folder_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = PDFloader.load()
    chunks = build_chunks(docs)
    texts = [c["text"] for c in chunks]
    cluster_indexes = build_cluster_indexes(chunks)

    embeddings = embed_texts(texts) if texts else np.empty((0, 0), dtype=np.float32)
    d = embeddings.shape[1] if len(embeddings) > 0 else 0

    index = faiss.IndexHNSWFlat(d, 32)
    if len(embeddings) > 0:
        index.add(embeddings)

    graph = build_knowledge_graph(chunks)

    kb = {
        "docs": docs,
        "chunks": chunks,
        "faiss_index": index,
        "dim": d,
        "knowledge_graph": graph,
        "cluster_indexes": cluster_indexes,
    }

    if use_bm25 and texts:
        corpus = [_tokenize_bm25(t) for t in texts]
        kb["bm25_corpus"] = corpus
        kb["bm25_index"] = BM25Okapi(corpus)

    return kb

def hybrid_retrieve_from_clusters(
    query: str,
    kb: Dict[str, Any],
    target_clusters: List[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    cluster_indexes = kb.get("cluster_indexes", {})

    if not target_clusters or "general" in target_clusters:
        return hybrid_retrieve(query, kb, top_k)

    all_results = []

    for cluster in target_clusters:
        cluster_kb = cluster_indexes.get(cluster)

        if not cluster_kb:
            continue

        cluster_results = hybrid_retrieve(
            query=query,
            kb=cluster_kb,
            top_k=top_k,
        )

        for result in cluster_results:
            result["matched_cluster"] = cluster

        all_results.extend(cluster_results)

    if not all_results:
        return hybrid_retrieve(query, kb, top_k)

    seen = {}
    for item in all_results:
        key = (item["source"], item["chunk_id"])
        if key not in seen or item["score"] > seen[key]["score"]:
            seen[key] = item

    results = list(seen.values())
    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]

morph = pymorphy3.MorphAnalyzer()

def build_cluster_indexes(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    cluster_to_chunks: Dict[str, List[Dict[str, Any]]] = {}

    for chunk in chunks:
        for cluster in chunk.get("clusters", ["general"]):
            cluster_to_chunks.setdefault(cluster, []).append(chunk)

    cluster_indexes = {}

    for cluster_name, cluster_chunks in cluster_to_chunks.items():
        texts = [c["text"] for c in cluster_chunks]

        embeddings = embed_texts(texts) if texts else np.empty((0, 0), dtype=np.float32)

        if len(embeddings) == 0:
            continue

        d = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(d, 32)
        index.add(embeddings)

        corpus = [_tokenize_bm25(t) for t in texts]

        cluster_indexes[cluster_name] = {
            "chunks": cluster_chunks,
            "faiss_index": index,
            "dim": d,
            "bm25_corpus": corpus,
            "bm25_index": BM25Okapi(corpus),
        }

    return cluster_indexes

def _normalize_text(text: str) -> set:
    words = re.findall(r'\b[а-яА-ЯёЁa-zA-Z]{3,}\b', text.lower())
    return {morph.parse(w)[0].normal_form for w in words}

def hybrid_retrieve(query: str, kb: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    chunks = kb.get("chunks", [])
    if "faiss_index" not in kb or not kb["chunks"]:
        return []
            
    q_emb = embed_texts([query], is_query=True)
    k_search = max(len(kb["chunks"]), top_k * 3)
    D, I = kb["faiss_index"].search(q_emb, k_search)
    cos_scores = D[0]  
    cos_ranks = I[0]  
        
    bm25_ranks = None
    if kb.get("bm25_index"):
        q_tokens = _tokenize_bm25(query)
        bm25_scores = kb["bm25_index"].get_scores(q_tokens)
        bm25_ranks = np.argsort(bm25_scores)[::-1]

    rrf_scores = np.zeros(len(chunks))
    K_RRF = 60.0
    def apply_rrf(ranks, target):
        for rank, idx in enumerate(ranks, start=1):
            target[idx] += 1.0 / (K_RRF + rank)
    apply_rrf(cos_ranks, rrf_scores)
    if bm25_ranks is not None:
        apply_rrf(bm25_ranks, rrf_scores)

    candidate_indices = np.argsort(rrf_scores)[::-1][:(top_k * 3)]
    results = []
    for idx in candidate_indices:
        idx = int(idx)
        results.append({"source": chunks[idx]["source"], "chunk_id": chunks[idx]["chunk_id"], "text": chunks[idx]["text"],
            "index": idx, "score": float(rrf_scores[idx])})

    if results:
        pairs = [[query, r["text"]] for r in results]
        ce_scores = CROSS_ENCODER.predict(pairs, show_progress_bar=False)
        for i, r in enumerate(results):
            r["score"] = 0.3 * r["score"] + 0.7 * float(ce_scores[i])

    results.sort(key=lambda x: x["score"], reverse=True)
    for r in results:
        r.pop("index", None)
    return results[:top_k]

def _apply_json_heuristics(candidates: List[Dict[str, Any]], keywords_from_json: set) -> List[Dict[str, Any]]:
    for c in candidates:
        chunk_lower = c["text"].lower()
        chunk_normalized = _normalize_text(c["text"])

        matches = keywords_from_json.intersection(chunk_normalized)
        c["score"] += len(matches) * 0.05

        numeric_patterns = [
            r'\d{1,2}(\.\d)?\s?мм',
            r'[><=]\s?\d{1,2}',
            r'от\s\d{1,2}\sдо\s\d{1,2}'
        ]
        for pattern in numeric_patterns:
            if re.search(pattern, chunk_lower):
                c["score"] += 0.15

        stop_patterns = ["Рис.", "рисунок", "вид сбоку", "снимок", "визуализация", "иллюстрация", "график"]
        for stop_word in stop_patterns:
            if stop_word in chunk_lower:
                c["score"] -= 0.10

        reference_matches = re.findall(r'\[[\d,\s\-]+\]', chunk_lower)
        if reference_matches:
            c["score"] -= len(reference_matches) * 0.005
    return candidates

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

    graph_candidates = {}

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
            if not node.startswith("chunk::"):
                continue

            node_data = graph.nodes[node]
            candidate_key = (
                node_data["source"],
                node_data["chunk_id"],
            )

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

def merge_retrieval_results(
    vector_results: List[Dict[str, Any]],
    graph_results: List[Dict[str, Any]],
    final_top_k: int = FINAL_GUIDELINES_TOP_K,
) -> List[Dict[str, Any]]:
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
                0.65 * merged[key].get("vector_score", 0.0)
                + 0.35 * merged[key].get("graph_score", 0.0)
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

@track_node_time
def retrieve_graph_context(state: MedGraphState) -> MedGraphState:
    warnings = list(state.get("warnings", []))
    paths = state.get("guideline_paths", [])

    if not paths:
        return {
            **state,
            "retrieved_guidelines": [],
            "warnings": warnings + ["Не переданы пути к гайдлайнам"],
        }

    docs_path = paths[0]
    kb = KB_CACHE.get(docs_path)

    if kb is None:
        return {
            **state,
            "retrieved_guidelines": [],
            "warnings": warnings + ["KB не инициализирована"],
        }

    patient_data = state.get("patient_data", {})

    symptom_query = (
        f"{state.get('query', '')}\n"
        f"{state.get('patient_history', '')}"
    ).strip()

    target_clusters = route_query_to_clusters(
        query=state.get("query", ""),
        patient_history=state.get("patient_history", ""),
        patient_data=patient_data,
    )

    symptom_results = hybrid_retrieve_from_clusters(
        query=symptom_query,
        kb=kb,
        target_clusters=target_clusters,
        top_k=SYMPTOM_TOP_K,
    )

    json_query = ""
    keywords_from_json = set()

    if patient_data and isinstance(patient_data, dict):
        json_query = _build_json_query(patient_data)
        keywords_from_json = _collect_json_keywords(patient_data)

    json_candidates = (
        hybrid_retrieve_from_clusters(
            query=json_query,
            kb=kb,
            target_clusters=target_clusters,
            top_k=JSON_CANDIDATES_TOP_K,
        )
        if json_query
        else []
    )
    json_candidates = _apply_json_heuristics(json_candidates, keywords_from_json)

    for c in json_candidates:
        chunk_lower = c["text"].lower()

        if any(term in chunk_lower for term in [
            "норма", "нормаль", "аневризм", "порог", "рекомендуется",
            "диаметр", "расслоени", "разрыв", "вмешательств", "таблица"
        ]):
            c["score"] += 0.20

        if re.search(r'(>=|<=|≤|≥|>|<)\s?\d+[.,]?\d*\s?(мм|см)', chunk_lower):
            c["score"] += 0.25

        if re.search(r'\d+[.,]?\d*\s?(мм|см)', chunk_lower):
            c["score"] += 0.10

    json_candidates = [c for c in json_candidates if c["score"] >= MIN_JSON_RELEVANCE_SCORE]
    json_candidates.sort(key=lambda x: x["score"], reverse=True)
    json_results = json_candidates[:JSON_RESULTS_TOP_K]

    vector_results = []

    seen = set()
    for item in symptom_results + json_results:
        key = (item["source"], item["chunk_id"])
        if key not in seen:
            seen.add(key)
            vector_results.append(item)

    graph_results = graph_expand_from_chunks(
        seed_results=vector_results,
        kb=kb,
        max_hops=2,
        max_graph_chunks=GRAPH_EXPAND_TOP_K,
    )

    final_guidelines = merge_retrieval_results(
        vector_results=vector_results,
        graph_results=graph_results,
        final_top_k=FINAL_GUIDELINES_TOP_K,
    )

    if not final_guidelines:
        warnings.append("Ретривер не нашёл релевантных фрагментов")

    return {
        **state,
        "retrieved_guidelines": final_guidelines,
        "warnings": warnings,
    }

@track_node_time
def ingest_request(state: MedGraphState) -> MedGraphState:
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

@track_node_time
def initialize_kb(state: MedGraphState) -> MedGraphState:
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
    USE_BM25 = True

    try:
        if docs_path not in KB_CACHE:
            kb = load_kb_from_disk(docs_path, use_bm25=USE_BM25)

            if kb is None:
                print("KB cache not found. Building from scratch...")
                kb = build_kb(docs_path, use_bm25=USE_BM25)
                save_kb_to_disk(docs_path, kb, use_bm25=USE_BM25)

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

            KB_CACHE[docs_path] = kb
        else:
            kb = KB_CACHE[docs_path]

        if not kb["chunks"]:
            warnings.append("После разбиения документов не получено ни одного чанка")

        if "knowledge_graph" not in kb:
            warnings.append("Граф знаний не был построен")

        return {
            **state,
            "chunks_count": len(kb["chunks"]),
            "warnings": warnings,
            "errors": errors,
        }

    except Exception as e:
        return {
            **state,
            "warnings": warnings + [f"Не удалось инициализировать KB: {e}"],
            "errors": errors,
            "chunks_count": 0,
        }

TRANSLATION_MAP = {
    "Descending Aorta": "Нисходящая аорта",
    "Isthmus": "Перешеек аорты",
    "Arch after LSA": "Дуга аорты после отхождения левой подвключичной артерии",
    "Arch after TBC": "Дуга аорты после отхождения плечеголовного ствола",
    "Ascending Aorta befor TBC": "Восходящая аорта перед плечеголовным стволом",
    "Ascending Aorta": "Восходящая аорта",
    "max_diam_1": "Максимальные диаметры",
    "max_diam_2": "Максимальные диаметры",
    "min_diam": "Минимальный диаметр",
    "perimetr": "Периметр сосуда",
    "area": "Площадь поперечного сечения"
}

@track_node_time
def _flatten_patient_measurements(patient_data: Dict[str, Any]) -> List[str]:
    """
    Превращает JSON пациента в плоский список фраз для retrieval.
    Пример:
    'Восходящая аорта максимальный диаметр 39.44 мм'
    """
    facts: List[str] = []

    if not isinstance(patient_data, dict):
        return facts

    for zone_key, zone_value in patient_data.items():
        zone_name = TRANSLATION_MAP.get(zone_key, zone_key)

        if isinstance(zone_value, dict):
            for metric_key, metric_value in zone_value.items():
                metric_name = TRANSLATION_MAP.get(metric_key, metric_key)

                if isinstance(metric_value, (int, float)):
                    facts.append(f"{zone_name} {metric_name} {metric_value} мм")
                elif isinstance(metric_value, list):
                    numeric_values = [str(v) for v in metric_value if isinstance(v, (int, float))]
                    if numeric_values:
                        facts.append(f"{zone_name} {metric_name} {' '.join(numeric_values)} мм")
                elif isinstance(metric_value, dict):
                    for sub_key, sub_val in metric_value.items():
                        sub_name = TRANSLATION_MAP.get(sub_key, sub_key)
                        if isinstance(sub_val, (int, float)):
                            facts.append(f"{zone_name} {metric_name} {sub_name} {sub_val} мм")
                        elif isinstance(sub_val, list):
                            numeric_values = [str(v) for v in sub_val if isinstance(v, (int, float))]
                            if numeric_values:
                                facts.append(f"{zone_name} {metric_name} {sub_name} {' '.join(numeric_values)} мм")

        elif isinstance(zone_value, (int, float)):
            facts.append(f"{zone_name} {zone_value} мм")

    return facts

def _collect_json_keywords(patient_data: Dict[str, Any]) -> set:
    """
    Собирает лемматизированные ключевые слова из названий зон и числовых меток.
    """
    keywords = set()

    if not isinstance(patient_data, dict):
        return keywords

    for zone_key, zone_value in patient_data.items():
        zone_name = TRANSLATION_MAP.get(zone_key, zone_key)
        keywords.update(_normalize_text(zone_name))

        if isinstance(zone_value, dict):
            for metric_key in zone_value.keys():
                metric_name = TRANSLATION_MAP.get(metric_key, metric_key)
                keywords.update(_normalize_text(metric_name))

    return keywords

def _build_json_query(patient_data: Dict[str, Any]) -> str:
    """
    Строит retrieval-запрос не только из названий сегментов,
    но и из конкретных численных значений пациента.
    """
    facts = _flatten_patient_measurements(patient_data)

    base_terms = (
        "норма аорты нормальный диаметр аневризма расширение "
        "порог вмешательства показания к операции риск расслоения "
        "восходящая аорта нисходящая аорта дуга аорты перешеек "
        "максимальный диаметр минимальный диаметр периметр площадь поперечного сечения "
        "мм см таблица рекомендации"
    )

    return f"{base_terms} {' '.join(facts)}".strip()

MAX_GUIDELINE_CHARS = 1200

def clean_context_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_GUIDELINE_CHARS:
        text = text[:MAX_GUIDELINE_CHARS] + "..."
    return text

@track_node_time
def fuse_context(state: MedGraphState) -> MedGraphState:
    blocks = []

    guidelines = state.get("retrieved_guidelines", [])
    guidelines = sorted(
        guidelines,
        key=lambda x: x.get("final_score", x.get("score", 0.0)),
        reverse=True
    )[:GUIDELINES_FOR_LLM_TOP_K]

    for i, item in enumerate(guidelines, start=1):
        score = item.get("score", item.get("final_score", 0.0))
        text = clean_context_text(item["text"])

        blocks.append(
            f"[GUIDELINE {i}]\n"
            f"Источник: {item['source']}\n"
            f"Chunk: {item['chunk_id']}\n"
            f"Score: {score:.4f}\n"
            f"Текст: {text}"
        )

    fused_context = (
        f"Жалобы:\n{state.get('query', '')}\n\n"
        f"Анамнез:\n{state.get('patient_history', '')}\n\n"
        f"Контекст из гайдлайнов:\n"
        + ("\n\n".join(blocks) if blocks else "Ничего не найдено")
    )

    return {
        **state,
        "fused_context": fused_context,
    }

def format_patient_data(patient_data: Dict[str, Any]) -> str:
    if not patient_data:
        return "Данные пациента не переданы."
    return json.dumps(patient_data, ensure_ascii=False, indent=2)
    
@track_node_time
def build_prompt(state: MedGraphState) -> MedGraphState:
    patient_data_text = format_patient_data(state.get("patient_data", {}))
    prompt = f"""
Вы — врач-кардиохирург, эксперт в области кардиологии и сосудистой хирургии с 50-летним опытом. Ваша специализация: анализ КТ-ангиографии аорты, оценка аневризм, диссекций и стенозирующих поражений.
Сформируйте заключение врача-кардиохирурга, основываясь на данных измерений КТ аорты
Справочные данные и клинические рекомендации, a так же жалобы и анамнез:
{state.get("fused_context", "")}

Результаты КТ (структурированные данные):
{patient_data_text}

<rules>
1. Анализируйте данные ТОЛЬКО на основе предоставленных материалов. Не используйте внешние медицинские базы и не домысливайте значения.
2. Если в данных или контексте отсутствует информация для вывода по конкретному параметру, явно укажите: "Недостаточно данных для оценки [параметр]". Не делайте предположений.
3. Сопоставляйте измерения из JSON с референсными значениями из контекста. Указывайте единицы измерения и нормативные диапазоны.
4. Используйте строгую медицинскую терминологию. Избегайте разговорных формулировок.
5. Заключение носит информационно-аналитический характер и требует очной верификации лечащим врачом.
6. Не пересказывайте служебную информацию из клинических рекомендаций: разработчиков, организации, авторов, оглавление, список литературы, сокращения. Используйте только медицинские нормы, пороги, показания, противопоказания и тактику. Каждый раздел должен содержать 2–4 содержательных пункта. Если данных недостаточно, всё равно сохраните раздел и явно напишите, чего не хватает.
</rules>
<output_format>
Верните ответ строго в следующей структуре. Не добавляйте вводные/заключительные фразы вне схемы:
[Контекст] Краткая выжимка релевантных норм/рекомендаций из подгруженного контекста.
[Анализ] Сопоставление измерений пациента с нормами. Выявленные отклонения (с конкретными цифрами).
[Интерпретация] Клиническая значимость изменений. Оценка рисков (стабильность, прогрессирование, угроза разрыва и т.д.).
[Заключение] Предварительный диагноз/статус. Рекомендации по тактике (наблюдение, КТ-контроль через Х мес., консультация, хирургическое/эндоваскулярное лечение).
</output_format>
""".strip()

    return {
        **state,
        "final_prompt": prompt,
    }

@track_node_time
def call_local_llm(state: MedGraphState) -> MedGraphState:
    retrieved = state.get("retrieved_guidelines", [])
    if not retrieved:
        return {
            **state,
            "raw_llm_output": "Релевантные клинические рекомендации не найдены. Недостаточно данных для анализа.",
        }

    llm = ChatOpenAI(model=VLLM_MODEL,
                     base_url=VLLM_BASE_URL,
                     api_key=VLLM_API_KEY,
                     temperature=0.5,
                     max_tokens=5000,
                     )

    response = llm.invoke(state.get("final_prompt", ""))
    answer = response.content if hasattr(response, "content") else str(response)

    return {
        **state,
        "raw_llm_output": answer,
    }

def build_graph():
    builder = StateGraph(MedGraphState)

    builder.add_node("ingest_request", ingest_request)
    builder.add_node("initialize_kb", initialize_kb)
    builder.add_node("retrieve_graph_context", retrieve_graph_context)
    builder.add_node("fuse_context", fuse_context)
    builder.add_node("build_prompt", build_prompt)
    builder.add_node("call_local_llm", call_local_llm)

    builder.add_edge(START, "ingest_request")
    builder.add_edge("ingest_request", "initialize_kb")
    builder.add_edge("initialize_kb", "retrieve_graph_context")
    builder.add_edge("retrieve_graph_context", "fuse_context")
    builder.add_edge("fuse_context", "build_prompt")
    builder.add_edge("build_prompt", "call_local_llm")
    builder.add_edge("call_local_llm", END)

    return builder.compile(checkpointer=InMemorySaver())

if __name__ == "__main__":
    graph = build_graph()

    initial_state: MedGraphState = {
        "query": "боль в груди, между лопатками, в верхней части спины, шее, затруднённое дыхание, одышка, хрипы, кашель,ощущение комка в горле. Врожденных патологий нет",
        "patient_history": "Мужчина, 54 лет, длительный стаж курения",
        "guideline_paths": ["docs"],
        "warnings": [],
        "errors": [],
        "patient_data": patient_data,
    }

    config = {
        "configurable": {
            "thread_id": "local-vllm-demo-001"
        }
    }

    result = graph.invoke(initial_state, config=config)

    print("\n=== WARNINGS ===")
    print(result.get("warnings", []))

    print("\n=== ERRORS ===")
    print(result.get("errors", []))

    print("\n=== CHUNKS ===")
    print(result.get("chunks_count", 0))

    print("\n=== RETRIEVED ===")
    for item in result.get("retrieved_guidelines", []):
        print(item)

    print("\n=== FUSED CONTEXT ===")
    print(result.get("fused_context", ""))

    print("\n=== ANSWER ===")
    print(result.get("raw_llm_output", ""))
    with open("langsmith_runs.jsonl", "w", encoding="utf-8") as f:
        for run in runs:
            data = {
                "id": str(run.id),
                "name": run.name,
                "start_time": str(run.start_time),
                "end_time": str(run.end_time),
                "inputs": run.inputs,
                "outputs": run.outputs,
            }

            f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
     


