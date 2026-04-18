import os
from pathlib import Path
from typing import TypedDict, List, Dict, Any

import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import json 

os.environ["CUDA_VISIBLE_DEVICES"] = ""
load_dotenv()

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "token-abc123")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

EMBEDDING_MODEL_NAME = "DmitryPogrebnoy/MedRuBertTiny2"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 4
MIN_RELEVANCE_SCORE = 0.20

EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
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


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    embeddings = EMBEDDER.encode(
        texts,
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


def retrieve_top_k(query: str, kb: Dict[str, Any], top_k: int = TOP_K, min_score: float = MIN_RELEVANCE_SCORE) -> List[Dict[str, Any]]:
    chunks = kb.get("chunks", [])
    embeddings = kb.get("embeddings")

    if not chunks or embeddings is None or len(chunks) == 0:
        return []

    query_embedding = embed_texts([query])
    if query_embedding.size == 0:
        return []

    scores = np.dot(embeddings, query_embedding[0])
    ranked_indices = np.argsort(scores)[::-1]

    results: List[Dict[str, Any]] = []
    for idx in ranked_indices[: max(top_k * 3, top_k)]:
        score = float(scores[idx])
        if score < min_score:
            continue

        chunk = chunks[int(idx)]
        results.append({
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "score": score,
        })

        if len(results) >= top_k:
            break

    return results


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
            KB_CACHE[docs_path] = build_kb(docs_path)

        kb = KB_CACHE[docs_path]
        if not kb["docs"]:
            warnings.append(f"В папке {docs_path!r} не найдено ни одного читаемого .txt/.pdf документа")
        if not kb["chunks"]:
            warnings.append("После разбиения документов не получено ни одного чанка")

        return {
            **state,
            "guideline_docs": kb["docs"],
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
        "retrieved_guidelines": retrieved,
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
Ты — ассистент врача, работающий строго по клиническим рекомендациям.

Правила:
1. Используй ТОЛЬКО данные из:
   - жалоб,
   - анамнеза,
   - приведённых ниже фрагментов гайдлайнов.
2. Запрещено использовать внешние знания.
3. Если информации в гайдлайнах недостаточно — явно напиши об этом.
4. Не ставь окончательный диагноз.
5. Каждое утверждение о гипотезе или рекомендации обязательно сопровождай ссылкой вида [GUIDELINE i].
6. Если в контексте нет ни одного фрагмента [GUIDELINE i], ответ должен быть ровно таким:
   Релевантные клинические рекомендации не найдены. Недостаточно данных для анализа.

Формат ответа:
1. Вероятные гипотезы (максимум - 5 гипотез)
2. Обоснование
3. Что нужно уточнить
4. Краткие рекомендации
5. Ограничения

Данные пациента:
{patient_data_text}

Контекст:
{state.get("fused_context", "")}
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
        "query": "Кашель, температура, слабость, одышка",
        "patient_history": "Пациент 54 лет, длительный стаж курения",
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
