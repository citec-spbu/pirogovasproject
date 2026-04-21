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
from sentence_transformers import SentenceTransformer, models
import json 
import pymorphy3

os.environ["CUDA_VISIBLE_DEVICES"] = ""
load_dotenv()

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "token-abc123")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 4
MIN_RELEVANCE_SCORE = 0.20

KB_DISK_CACHE_DIR = Path(".kb_cache")
KB_DISK_CACHE_DIR.mkdir(exist_ok=True)

word_embedding_model = models.Transformer(EMBEDDING_MODEL_NAME)

pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
)

EMBEDDER = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cpu")
KB_CACHE: Dict[str, Dict[str, Any]] = {}


with open(f"content/0.json", "r", encoding="utf-8") as f:
    patient_data = json.load(f)


class MedGraphState(TypedDict, total=False):
    query: str
    patient_history: str
    guideline_paths: List[str]
    persist_dir: str
    patient_data = str

    warnings: List[str]
    errors: List[str]

    guideline_docs: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    retrieved_guidelines: List[Dict[str, Any]]
    fused_context: str
    final_prompt: str
    raw_llm_output: str

# Вспомогательные функции
def get_cache_prefix(folder_path: str) -> Path:
    safe_name = Path(folder_path).name.replace(" ", "_")
    return KB_DISK_CACHE_DIR / safe_name

def save_kb_to_disk(folder_path: str, kb: Dict[str, Any]) -> None:
    prefix = get_cache_prefix(folder_path)

    with open(f"{prefix}_chunks.json", "w", encoding="utf-8") as f:
        json.dump(kb["chunks"], f, ensure_ascii=False, indent=2)

    np.save(f"{prefix}_embeddings.npy", kb["embeddings"])

def load_kb_from_disk(folder_path: str) -> Optional[Dict[str, Any]]:
    prefix = get_cache_prefix(folder_path)

    chunks_file = Path(f"{prefix}_chunks.json")
    emb_file = Path(f"{prefix}_embeddings.npy")

    if not (chunks_file.exists() and emb_file.exists()):
        return None

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(emb_file)

    return {
            "chunks": chunks,
            "embeddings": embeddings,
        }

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


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
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
        pieces = chunk_text(doc["text"])
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

    embeddings = EMBEDDER.encode(
        prepared_texts,
        normalize_embeddings=True,
        batch_size=8,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return embeddings.astype("float32")


def build_kb(folder_path: str) -> Dict[str, Any]:
    folder = Path(folder_path)
    docs = read_documents(folder)
    chunks = build_chunks(docs)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts) if texts else np.empty((0, 0), dtype=np.float32)

    return {
        "docs": docs,
        "chunks": chunks,
        "embeddings": embeddings,
    }

morph = pymorphy3.MorphAnalyzer()

def _normalize_text(text: str) -> set:
    words = re.findall(r'\b[а-яА-ЯёЁa-zA-Z]{3,}\b', text.lower())
    return {morph.parse(w)[0].normal_form for w in words}

def retrieve_top_k(query: str, kb: Dict[str, Any], json_data: Optional[Dict] = None, top_k: int = 5, min_score: float = 0.5) -> Dict[str, Any]:
    enriched_query = str(query)
    keywords_from_json = set()
    if json_data:
        translated_list = []
        for key in json_data.keys():
            clean_key = key.strip()
            russian_term = TRANSLATION_MAP.get(clean_key)
            if russian_term:
                translated_list.append(russian_term)
                keywords_from_json.update(_normalize_text(russian_term))
        
        zones_str = " ".join(translated_list)
        enriched_query += f" {zones_str} Максимальные диаметры, Минимальный диаметр, Периметр сосуда, Площадь поперечного сечения, норма диаметр классификация показатели мм превышать"
        
        raw_keywords = _extract_keywords_from_json(json_data)
        for kw in raw_keywords:
            keywords_from_json.update(_normalize_text(kw))
    chunks = kb.get("chunks", [])
    embeddings = kb.get("embeddings")
    if not chunks or embeddings is None or len(chunks) == 0:
        return {"chunks": [], "sources": []}
    query_embedding = embed_texts([enriched_query], is_query=True)
    if query_embedding.size == 0:
        return {"chunks": [], "sources": []}
    scores = np.dot(embeddings, query_embedding[0])
    candidate_indices = np.argsort(scores)[::-1][:(top_k * 3)]
    preliminary_results = []
    for idx in candidate_indices:
        chunk = chunks[int(idx)]
        preliminary_results.append({
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "score": float(scores[idx])})
    for item in preliminary_results:
        text_lower = item["text"].lower()
        text_normalized = _normalize_text(item["text"])

        # А) Бонус за совпадение терминов из JSON
        matches = keywords_from_json.intersection(text_normalized)
        item["score"] += len(matches) * 0.05

        # Б) Бонус за числовые нормативы
        numeric_patterns = [
            r'\d{1,2}(\.\d)?\s?мм', 
            r'[><=]\s?\d{1,2}', 
            r'от\s\d{1,2}\sдо\s\d{1,2}'
        ]
        if any(re.search(p, text_lower) for p in numeric_patterns):
            item["score"] += 0.2

        # В) Штраф за описание изображений
        stop_patterns = ["рис.", "рисунок", "вид сбоку", "снимок", "визуализация", "иллюстрация", "график"]
        if any(stop in text_lower for stop in stop_patterns):
            item["score"] -= 0.15

        # Г) Штраф за ссылки на литературу
        #reference_matches = re.findall(r'\[[\d,\s\-]+\]', text_lower)
        #item["score"] -= len(reference_matches) * 0.005

    final_chunks = [c for c in preliminary_results if c["score"] >= min_score]
    final_chunks.sort(key=lambda x: x["score"], reverse=True)
    selected_chunks = final_chunks[:top_k]

    return {
        "chunks": selected_chunks,
        "sources": list(set(c["source"] for c in selected_chunks))
    }

# Узлы графа

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

    try:
        if docs_path not in KB_CACHE:
            kb = load_kb_from_disk(docs_path)

            if kb is None:
                print("KB cache not found. Building from scratch...")
                kb = build_kb(docs_path)
                save_kb_to_disk(docs_path, kb)
            else:
                print("KB loaded from disk cache.")

            KB_CACHE[docs_path] = kb

        kb = KB_CACHE[docs_path]

        if not kb["chunks"]:
            warnings.append("После разбиения документов не получено ни одного чанка")

        return {
            **state,
            "chunks": kb["chunks"],
            "warnings": warnings,
            "errors": errors,
        }

    except Exception as e:
        return {
            **state,
            "warnings": warnings + [f"Не удалось инициализировать KB: {e}"],
            "errors": errors,
            "guideline_docs": [],
            "chunks": [],
        }

def retrieve_text_context(state: MedGraphState) -> MedGraphState:
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

    query = f"{state.get('query', '')}\n{state.get('patient_history', '')}".strip()
    retrieved = retrieve_top_k(query, kb=kb, top_k=TOP_K, min_score=MIN_RELEVANCE_SCORE)

    if not retrieved:
        warnings.append("Ретривер не нашёл релевантных фрагментов")

    return {
        **state,
        "retrieved_guidelines": retrieved["chunks"],
        "warnings": warnings,
    }



def fuse_context(state: MedGraphState) -> MedGraphState:
    blocks = []

    for i, item in enumerate(state.get("retrieved_guidelines", []), start=1):
        blocks.append(
            f"[GUIDELINE {i}]\n"
            f"Источник: {item['source']}\n"
            f"Chunk: {item['chunk_id']}\n"
            f"Score: {item['score']:.4f}\n"
            f"Текст: {item['text']}"
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
    
patient_data_text = format_patient_data(patient_data)

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



def call_local_llm(state: MedGraphState) -> MedGraphState:
    retrieved = state.get("retrieved_guidelines", [])
    if not retrieved:
        return {
            **state,
            "raw_llm_output": "Релевантные клинические рекомендации не найдены. Недостаточно данных для анализа.",
        }

    llm = ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        temperature=0.0,
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

    print("\n=== DOCS ===")
    print(len(result.get("guideline_docs", [])))

    print("\n=== CHUNKS ===")
    print(len(result.get("chunks", [])))

    print("\n=== RETRIEVED ===")
    for item in result.get("retrieved_guidelines", []):
        print(item)

    print("\n=== FUSED CONTEXT ===")
    print(result.get("fused_context", ""))

    print("\n=== ANSWER ===")
    print(result.get("raw_llm_output", ""))

