import os # Обращаемся к системе
from pathlib import Path # Путь
from typing import TypedDict, List, Dict, Any # Аннотация типов 
from pypdf import PdfReader 

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from dotenv import load_dotenv # Загрузка переменных из файла .env
from langgraph.graph import StateGraph, START, END 
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI # Интерфейс к API модели чата OpenAI

from piragi import Ragi # Библиотека с RAG

load_dotenv()

# Для vllm параметры
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "token-abc123")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

PIRAGI_CONFIG = {
    "llm": {
        "model": VLLM_MODEL,
        "base_url": VLLM_BASE_URL,
        "api_key": VLLM_API_KEY,
        "max_tokens": 2048,
        "temperature": 0.1,
    },
    "embedding": {
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "batch_size": 8,
        "device": "cpu",
    },
    "chunk": {
        "strategy": "semantic",
        "size": 600,
        "overlap": 100,
    },
    "retrieval": {
        "use_hybrid_search": True,
        "use_cross_encoder": False,
        "min_relevance_score": 0.3,
        "max_chunks_per_doc": 3,
    },
}

class MedGraphState(TypedDict, total=False): # Тут описаны все необходимые параметры
    query: str # Запрос
    patient_history: str
    guideline_paths: List[str]
    persist_dir = str

    warnings: List[str]
    errors: List[str]

    guideline_docs: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    retrieved_guidelines: List[Dict[str, Any]]

    fused_context: str
    final_prompt: str
    raw_llm_output: str

# Вспомогательные функции 
def read_txt_documents(folder: Path) -> List[Dict[str, Any]]: # ожидаемый тип у folder - Path (типо путь к папке), сама функция просто считывает текст из всех файлов в папке
    docs = []
    if not folder.exists():
        return docs

    for file_path in folder.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8").strip()
        if text:
            docs.append({
                "source": file_path.name,
                "text": text
            })

    for file_path in folder.glob("*.pdf"):
        try:
            reader = PdfReader(file_path)
            text = ""
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            text = text.strip()

            if text:
                docs.append({
                    "source": file_path.name,
                    "text": text
            })
        except Exctption as e:
            print(f'Ошибка чтения PDF {file_path}: {e}')

    return docs


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> List[str]: # С помощью этой функции рабиваем большой текст на маленькие куски
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def build_chunks(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]: # Здесь берутся документы и режутся соответственно на куски
    result = []
    for doc in docs:
        pieces = chunk_text(doc["text"])
        for i, piece in enumerate(pieces):
            result.append({
                "source": doc["source"],
                "chunk_id": i,
                "text": piece
            })
    return result


def retrieve_top_k(query: str, chunks: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
    if not chunks:
        return []

    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True, batch_size=8, convert_to_numpy=True, show_progress_bar=True)
    query_embedding = embedder.encode([query], normalize_embeddings=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))

    scores, indices = index.search(query_embedding.astype("float32"), top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append({
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "score": float(score)
        })
    return results

# Узлы графа
def ingest_request(state: MedGraphState) -> MedGraphState: # Смотрим, какие есть проблемы
    errors = state.get("errors", [])
    warnings = state.get("warnings", [])

    if not state.get("query"):
        errors.append("Не передан query")

    return {
        **state,
        "errors": errors,
        "warnings": warnings,
    }

def initialize_kb(state: MedGraphState) -> MedGraphState:
    warnings = state.get("warnings", [])
    paths = state.get("guideline_paths", [])

    if not paths:
        warnings.append("Не переданы пути к гайдлайнам")
        return {**state, "warnings": wanrnings}

    docs_path = paths[0]
    persist_dir = ".piragi_index"

    try:
        Ragi(docs_path, persist_dir=persist_dir, config=PIRAGI_CONFIG)
    except Exception as e:
        warnings.append(f"Не удалось инициализировать RAG-индекс: {e}")
        return {**state, "warnings": warnings}

    return {
        **state,
        "persist_dir": persist_dir,
        "warnings": warnings,
    }
"""
def load_guidelines(state: MedGraphState) -> MedGraphState:
    docs = []
    for path_str in state.get("guideline_paths", []):
        docs.extend(read_txt_documents(Path(path_str)))

    warnings = state.get("warnings", [])
    if not docs:
        warnings.append("Не удалось загрузить гайдлайны")


    return {
        **state,
        "guideline_docs": docs,
        "warnings": warnings,
    }


def make_chunks(state: MedGraphState) -> MedGraphState:
    chunks = build_chunks(state.get("guideline_docs", []))
    return {
        **state,
        "chunks": chunks,
    }
"""
"""
def retrieve_text_context(state: MedGraphState) -> MedGraphState:
    query = f"{state.get('query', '')}\n{state.get('patient_history', '')}".strip()
    retrieved = retrieve_top_k(query, state.get("chunks", []), top_k=4)

    return {
        **state,
        "retrieved_guidelines": retrieved,
    }
"""

def retrieve_text_context(state: MedGraphState) -> MedGraphState:
    warnings = state.get("warnings", [])
    paths = state.get("guideline_paths", [])
    persist_dir = state.get("persist_dir", ".piragi_index")

    if not paths:
        return {
            **state,
            "retrieved_guidelines": [],
            "warnings": warnings + ["Не переданы пути к гайдлайнам"],
        }

    try:
        kb = Ragi(paths[0], persist_dir=persist_dir, config=PIRAGI_CONFIG)

        query = f"{state.get('query', '')}\n{state.get('patient_history', '')}".strip()
        citations = kb.retrieve(query, top_k=6)

        retrieved = []
        for i, c in enumerate(citations):
            score = float(getattr(c, "score", 0.0))
            if score < 0.3:
                continue

            retrieved.append({
                "source": getattr(c, "source", "unknown"),
                "chunk_id": i,
                "text": getattr(c, "chunk", ""),
                "score": score,
            })

        return {
            **state,
            "retrieved_guidelines": retrieved[:4],
        }

    except Exception as e:
        return {
            **state,
            "retrieved_guidelines": [],
            "warnings": warnings + [f"Ошибка retrieval: {e}"],
        }

def fuse_context(state: MedGraphState) -> MedGraphState: # Финальный текст-контекст для LLM
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

Формат ответа:

1. Вероятные гипотезы
   - только если они поддерживаются гайдлайнами
   - укажи, на какой фрагмент опираешься ([GUIDELINE i])

2. Обоснование
   - ссылайся на конкретные фрагменты
   - не придумывай факты

3. Что нужно уточнить
   - только если это следует из гайдлайнов

4. Краткие рекомендации
   - только из гайдлайнов

5. Ограничения
   - если данных недостаточно → напиши это явно

Если релевантные фрагменты отсутствуют:
напиши: "Релевантные клинические рекомендации не найдены. Недостаточно данных для анализа."

Контекст:
{state.get("fused_context", "")}
""".strip()
    return {
        **state,
        "final_prompt": prompt,
    }


def call_local_llm(state: MedGraphState) -> MedGraphState:
    llm = ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        temperature=0.1,
    )

    response = llm.invoke(state.get("final_prompt", ""))

    return {
        **state,
        "raw_llm_output": response.content,
    }

# Строим граф
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

    initial_state = {
        "query": "Кашель, температура, слабость, одышка",
        "patient_history": "Пациент 54 лет, длительный стаж курения",
        "guideline_paths": ["docs"],
        "warnings": [],
        "errors": [],
    }

    config = {
        "configurable": {
            "thread_id": "local-vllm-demo-001"
        }
    }

    result = graph.invoke(initial_state, config=config)

    print("\n=== WARNINGS ===")
    print(result.get("warnings", []))

    print("\n=== RETRIEVED ===")
    for item in result.get("retrieved_guidelines", []):
        print(item)

    print("\n=== ANSWER ===")
    print(result.get("raw_llm_output", ""))
