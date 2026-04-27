import os
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional
import re

import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, models, CrossEncoder
import json 
import pymorphy3
import faiss

import pickle
from rank_bm25 import BM25Okapi

import time
from functools import wraps

def track_node_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[{func.__name__}] Выполнено за {elapsed:.3f} сек.")
        return result
    return wrapper

os.environ["CUDA_VISIBLE_DEVICES"] = ""
load_dotenv()

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "token-abc123")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

EMBEDDING_MODEL_NAME = "DmitryPogrebnoy/MedRuBertTiny2" #"sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CROSS_ENCODER_MODEL_NAME = "DmitryPogrebnoy/MedRuBertTiny2" #"cross-encoder/ms-marco-MiniLM-L12-v2"
CROSS_ENCODER = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device="cpu")

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 4
MIN_RELEVANCE_SCORE = 0.0 #0.20

KB_DISK_CACHE_DIR = Path(".kb_cache")
KB_DISK_CACHE_DIR.mkdir(exist_ok=True)

KB_BM25_DISK_CACHE_DIR = Path(".kb_cache_bm25")
KB_BM25_DISK_CACHE_DIR.mkdir(exist_ok=True)

word_embedding_model = models.Transformer(EMBEDDING_MODEL_NAME)

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False, pooling_mode_max_tokens=False)

EMBEDDER = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cpu")
KB_CACHE: Dict[str, Dict[str, Any]] = {}

with open(f"dataset/0.json", "r", encoding="utf-8") as f:
    patient_data = json.load(f)


class MedGraphState(TypedDict, total=False):
    query: str
    patient_history: str
    guideline_paths: List[str]
    persist_dir: str
    patient_data: str

    warnings: List[str]
    errors: List[str]

    guideline_docs: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    retrieved_guidelines: List[Dict[str, Any]]
    fused_context: str
    final_prompt: str
    raw_llm_output: str

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

#for vector base
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


def read_documents(folder: Path) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    if not folder.exists() or not folder.is_dir():
        return docs

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        text = ""

        try:
            if suffix == ".txt":
                text = file_path.read_text(encoding="utf-8").strip()
            elif suffix == ".pdf":
                reader = PdfReader(str(file_path))
                pages = []
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        pages.append(page_text)
                text = "\n".join(pages).strip()
        except Exception as e:
            print(f"Ошибка чтения файла {file_path}: {e}")
            continue

        if text:
            docs.append({
                "source": file_path.name,
                "text": text,
            })


    return docs

def recursive_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    raw_chunks = []

    def _split_recursive(current_text: str, sep_idx: int):
        if len(current_text) <= chunk_size:
            if current_text.strip():
                raw_chunks.append(current_text.strip())
            return

        sep = separators[sep_idx] if sep_idx < len(separators) else ""
        parts = current_text.split(sep) if sep else list(current_text)
        merged = ""
        for part in parts:
            if len(merged) + len(sep) + len(part) > chunk_size and merged:
                _split_recursive(merged, sep_idx + 1)
                merged = part
            else:
                merged = merged + sep + part if merged else part
        if merged:
            _split_recursive(merged, sep_idx + 1)

    _split_recursive(text.strip(), 0)

    # Применяем overlap только если предыдущий чанк достаточно длинный
    final_chunks = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and len(raw_chunks[i-1]) > overlap:
            overlap_prefix = raw_chunks[i-1][-overlap:]
            chunk = (overlap_prefix + chunk) if overlap_prefix[-1].isalnum() else (overlap_prefix + " " + chunk)
        final_chunks.append(chunk)
    return final_chunks

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    # Очистка текста из PDF перед чанкингом
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Заменяем одиночные переносы строк на пробелы
    text = re.sub(r'\s{2,}', ' ', text)  # Убираем множественные пробелы
    text = re.sub(r'\n{3,}', '\n\n', text)  # Оставляем не более 2 пустых строк

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start += step

    return chunks


def build_chunks(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for doc in docs:
        pieces = recursive_chunk_text(doc["text"])
        for i, piece in enumerate(pieces):
            result.append({
                "source": doc["source"],
                "chunk_id": i,
                "text": piece,
            })
    return result


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
    docs = read_documents(folder)
    chunks = build_chunks(docs)
    texts = [c["text"] for c in chunks]

    embeddings = embed_texts(texts) if texts else np.empty((0, 0), dtype=np.float32)
    d = embeddings.shape[1] if len(embeddings) > 0 else 0

    # FAISS IndexFlatIP = точный поиск по скалярному произведению == Cosine Similarity
    index = faiss.IndexHNSWFlat(d, 32)
    index.add(embeddings)

    kb = {"docs": docs, "chunks": chunks, "faiss_index": index, "dim": d}
    if use_bm25 and texts:
        corpus = [_tokenize_bm25(t) for t in texts]
        kb["bm25_corpus"] = corpus
        kb["bm25_index"] = BM25Okapi(corpus)
    return kb

morph = pymorphy3.MorphAnalyzer()

def _normalize_text(text: str) -> set:
    words = re.findall(r'\b[а-яА-ЯёЁa-zA-Z]{3,}\b', text.lower())
    return {morph.parse(w)[0].normal_form for w in words}

def hybrid_retrieve(query: str, kb: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    chunks = kb.get("chunks", [])
    if "faiss_index" not in kb or not kb["chunks"]:
        return []
            
    # #Cosine Similarity for numpy
    # q_emb = embed_texts([query], is_query=True)
    # cos_scores = np.dot(embeddings, q_emb[0])
    # cos_ranks = np.argsort(cos_scores)[::-1]

    # Cosine Similarity через FAISS
    q_emb = embed_texts([query], is_query=True)
    k_search = max(len(kb["chunks"]), top_k * 3)
    D, I = kb["faiss_index"].search(q_emb, k_search)
    cos_scores = D[0]  # Scores (shape: k_search,)
    cos_ranks = I[0]  # Indices (shape: k_search,)
        
    #BM25
    bm25_ranks = None
    if kb.get("bm25_index"):
        q_tokens = _tokenize_bm25(query)
        bm25_scores = kb["bm25_index"].get_scores(q_tokens)
        bm25_ranks = np.argsort(bm25_scores)[::-1]

    #RRF Fusion
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

    #Cross-Encoder Reranking
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

        # А) Базовый NER-бонус (совпадение терминов)
        matches = keywords_from_json.intersection(chunk_normalized)
        c["score"] += len(matches) * 0.05

        # Б) Бонус за числовые нормативы
        numeric_patterns = [
            r'\d{1,2}(\.\d)?\s?мм',
            r'[><=]\s?\d{1,2}',
            r'от\s\d{1,2}\sдо\s\d{1,2}'
        ]
        for pattern in numeric_patterns:
            if re.search(pattern, chunk_lower):
                c["score"] += 0.15

        # В) Штраф за картинки
        stop_patterns = ["Рис.", "рисунок", "вид сбоку", "снимок", "визуализация", "иллюстрация", "график"]
        for stop_word in stop_patterns:
            if stop_word in chunk_lower:
                c["score"] -= 0.10

        # Г) Штраф за ссылки на литературу [1]
        reference_matches = re.findall(r'\[[\d,\s\-]+\]', chunk_lower)
        if reference_matches:
            c["score"] -= len(reference_matches) * 0.005
    return candidates


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
        return { **state, "errors": errors + ["Не переданы пути к гайдлайнам"], "warnings": warnings}
            
    docs_path = paths[0]
    USE_BM25 = True 
    try:
        if docs_path not in KB_CACHE:
            kb = load_kb_from_disk(docs_path, use_bm25=USE_BM25)
            if kb is None:
                print("KB cache not found. Building from scratch...")
                kb = build_kb(docs_path, use_bm25=USE_BM25)
                save_kb_to_disk(docs_path, kb, use_bm25=USE_BM25)
            else:
                print("KB loaded from disk cache.")
            KB_CACHE[docs_path] = kb
        else: 
            kb = KB_CACHE[docs_path]
        if not kb["chunks"]:
            warnings.append("После разбиения документов не получено ни одного чанка")
        return { **state,
                "chunks": kb["chunks"],
                "warnings": warnings,
                "errors": errors,
                "guideline_docs": kb.get("docs", []),
                }

    except Exception as e:
        return { **state,
                "warnings": warnings + [f"Не удалось инициализировать KB: {e}"],
                "errors": errors,
                "guideline_docs": kb.get("docs", []),
                "chunks": [],
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
def retrieve_text_context(state: MedGraphState) -> MedGraphState:
    warnings = list(state.get("warnings", []))
    paths = state.get("guideline_paths", [])
    if not paths:
        return {**state, "retrieved_guidelines": [], "warnings": warnings + ["Не переданы пути к гайдлайнам"]}

    docs_path = paths[0]
    kb = KB_CACHE.get(docs_path)
    if kb is None:
        return {**state, "retrieved_guidelines": [], "warnings": warnings + ["KB не инициализирована"]}

    patient_data = state.get("patient_data", {})
    
    symptom_query = (
        f"{state.get('query', '')}\n"
        f"{state.get('patient_history', '')}\n"
        "клинические рекомендации тактика ведения показания противопоказания диагностика лечение классификация"
    ).strip()
    symptom_results = hybrid_retrieve(symptom_query, kb, top_k=3)

    json_query = ""
    keywords_from_json = set()
    if patient_data and isinstance(patient_data, dict):
        translated_list = []
        for key in patient_data.keys():
            clean_key = key.strip()
            russian_term = TRANSLATION_MAP.get(clean_key) if 'TRANSLATION_MAP' in globals() else None
            if russian_term:
                translated_list.append(russian_term)
                keywords_from_json.update(_normalize_text(russian_term))
        zones_str = " ".join(translated_list)
        json_query += f" {zones_str} Максимальные диаметры, Минимальный диаметр, Периметр сосуда, Площадь поперечного сечения, норма диаметр классификация показатели мм превышать"

    json_candidates = hybrid_retrieve(json_query, kb, top_k=10)
    json_candidates = _apply_json_heuristics(json_candidates, keywords_from_json)
    json_candidates = [c for c in json_candidates if c["score"] >= MIN_RELEVANCE_SCORE]
    json_candidates.sort(key=lambda x: x["score"], reverse=True)
    json_results = json_candidates[:2]

    seen = set()
    merged_results = []
    for item in symptom_results + json_results:
        uid = (item["source"], item["chunk_id"])
        if uid not in seen:
            seen.add(uid)
            merged_results.append(item)

    merged_results.sort(key=lambda x: x["score"], reverse=True)
    final_guidelines = merged_results[:5]  # Гарантируем максимум 3+2 = 5

    if not final_guidelines:
        warnings.append("Ретривер не нашёл релевантных фрагментов")

    return {**state, "retrieved_guidelines": final_guidelines, "warnings": warnings}


@track_node_time
def fuse_context(state: MedGraphState) -> MedGraphState:
    blocks = []

    for i, item in enumerate(state.get("retrieved_guidelines", []), start=1):
        score = item.get("score", item.get("final_score", 0.0))
        blocks.append(
            f"[GUIDELINE {i}]\n"
            f"Источник: {item['source']}\n"
            f"Chunk: {item['chunk_id']}\n"
            f"Score: {score:.4f}\n"
            f"Текст: {item['text']}")

    fused_context = (
            f"Жалобы:\n{state.get('query', '')}\n\n"
            f"Анамнез:\n{state.get('patient_history', '')}\n\n"
            f"Контекст из гайдлайнов:\n"
            + ("\n\n".join(blocks) if blocks else "Ничего не найдено"))

    return {
        **state,
        "fused_context": fused_context,
    }


def format_patient_data(patient_data: Dict[str, Any]) -> str:
    if not patient_data:
        return "Данные пациента не переданы."
    return json.dumps(patient_data, ensure_ascii=False, indent=2)
    
patient_data_text = format_patient_data(patient_data)

@track_node_time
def build_prompt(state: MedGraphState) -> MedGraphState:
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

    llm = ChatOpenAI(model=VLLM_MODEL, base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY, temperature=0.0)

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
    builder.add_node("retrieve_text_context", retrieve_text_context)
    builder.add_node("fuse_context", fuse_context)
    builder.add_node("build_prompt", build_prompt)
    builder.add_node("call_local_llm", call_local_llm)

    builder.add_edge(START, "ingest_request")
    builder.add_edge("ingest_request", "initialize_kb")
    builder.add_edge("initialize_kb", "retrieve_text_context")
    builder.add_edge("retrieve_text_context", "fuse_context")
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
    print(len(result.get("chunks", [])))

    print("\n=== RETRIEVED ===")
    for item in result.get("retrieved_guidelines", []):
        print(item)

    print("\n=== FUSED CONTEXT ===")
    print(result.get("fused_context", ""))

    print("\n=== ANSWER ===")
    print(result.get("raw_llm_output", ""))
