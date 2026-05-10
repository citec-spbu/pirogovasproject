import os
import re
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import TypedDict, List, Dict, Any
import pymorphy3

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from functools import wraps
import time

from app.core.config import get_settings
from app.core.rag.kb_manager import KBOrchestrator
from app.core.rag.embedder import EmbeddingService
from app.core.rag.retriever import HybridRetriever

settings = get_settings()
logger = logging.getLogger(__name__)

# Module-level MorphAnalyzer for reuse
morph = pymorphy3.MorphAnalyzer()

# Lazy singletons
_kb_manager = None
_embedder = None
_retriever = None
_graph = None

def get_kb_orchestrator() -> KBOrchestrator:
    global _kb_manager
    if _kb_manager is None:
        _kb_manager = KBOrchestrator()
    return _kb_manager

def get_embedder() -> EmbeddingService:
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingService()
    return _embedder

def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        embedder = get_embedder()
        _retriever = HybridRetriever(embedder=embedder)
    return _retriever

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

VLLM_BASE_URL = settings.VLLM_BASE_URL
VLLM_API_KEY = settings.VLLM_API_KEY
VLLM_MODEL = settings.VLLM_MODEL

TRACE_FILE = Path("llm_traces.jsonl")


def save_llm_trace_jsonb(prompt: str, response: str, model: str, metadata: dict) -> str:
    trace = {
        "trace_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "input_prompt": prompt,
        "output_response": response,
        "metadata": metadata,
    }
    try:
        import fcntl
        with TRACE_FILE.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        logger.warning(f"Failed to write trace to file: {e}")
    return json.dumps(trace, ensure_ascii=False)


class MedGraphState(TypedDict, total=False):
    query: str
    patient_history: str
    guideline_paths: List[str]
    patient_data: Dict[str, Any]
    thread_id: str
    warnings: List[str]
    errors: List[str]
    retrieved_guidelines: List[Dict[str, Any]]
    fused_context: str
    final_prompt: str
    raw_llm_output: str

def track_node_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"[{func.__name__}] Выполнено за {elapsed:.3f} сек.")
        return result

    return wrapper

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

def _normalize_text(text: str) -> set:
    words = re.findall(r'\b[а-яА-ЯёЁa-zA-Z]{3,}\b', text.lower())
    return {morph.parse(w)[0].normal_form for w in words}


def _flatten_patient_measurements(patient_data: Dict[str, Any]) -> List[str]:
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
        elif isinstance(zone_value, (int, float)):
            facts.append(f"{zone_name} {zone_value} мм")
    return facts


def _collect_json_keywords(patient_data: Dict[str, Any]) -> set:
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
    facts = _flatten_patient_measurements(patient_data)
    base_terms = (
        "норма аорты нормальный диаметр аневризма расширение "
        "порог вмешательства показания к операции риск расслоения "
        "восходящая аорта нисходящая аорта дуга аорты перешеек "
        "максимальный диаметр минимальный диаметр периметр площадь поперечного сечения "
        "мм см таблица рекомендации"
    )
    return f"{base_terms} {' '.join(facts)}".strip()

@track_node_time
def ingest_request(state: MedGraphState) -> MedGraphState:
    errors = list(state.get("errors", []))
    warnings = list(state.get("warnings", []))
    if not state.get("query", "").strip():
        errors.append("Не передан query")
    if not state.get("guideline_paths"):
        errors.append("Не переданы пути к гайдлайнам")
    return {**state, "errors": errors, "warnings": warnings}


@track_node_time
def initialize_kb(state: MedGraphState) -> MedGraphState:
    warnings = list(state.get("warnings", []))
    errors = list(state.get("errors", []))
    paths = state.get("guideline_paths", [])

    if not paths:
        return {**state, "errors": errors + ["Не переданы пути к гайдлайнам"], "warnings": warnings}

    docs_path = paths[0]
    try:
        kb = get_kb_orchestrator().get_kb(docs_path, use_bm25=True)
        if not kb.get("chunks"):
            warnings.append("После разбиения документов не получено ни одного чанка")
        return {
            **state,
            "chunks": kb.get("chunks", []),
            "warnings": warnings,
            "errors": errors,
        }
    except Exception as e:
        return {
            **state,
            "warnings": warnings + [f"Не удалось инициализировать KB: {e}"],
            "errors": errors,
            "chunks": [],
        }


@track_node_time
def retrieve_graph_context(state: MedGraphState) -> MedGraphState:
    warnings = list(state.get("warnings", []))
    paths = state.get("guideline_paths", [])
    if not paths:
        return {**state, "retrieved_guidelines": [], "warnings": warnings + ["Не переданы пути к гайдлайнам"]}

    docs_path = paths[0]
    kb = get_kb_orchestrator().load_kb(docs_path, use_bm25=True)
    if not kb:
        return {**state, "retrieved_guidelines": [], "warnings": warnings + ["KB не инициализирована"]}

    patient_data = state.get("patient_data", {})

    symptom_query = f"{state.get('query', '')}\n{state.get('patient_history', '')}".strip()
    retriever = get_retriever()
    symptom_results = retriever.hybrid_search(
        query=symptom_query,
        chunks=kb["chunks"],
        faiss_index=kb.get("faiss_index"),
        bm25_index=kb.get("bm25_index"),
        top_k=4,
    )

    json_results = []
    if patient_data and isinstance(patient_data, dict):
        json_query = _build_json_query(patient_data)
        keywords = _collect_json_keywords(patient_data)
        json_candidates = get_retriever().hybrid_search(
            query=json_query,
            chunks=kb["chunks"],
            faiss_index=kb.get("faiss_index"),
            bm25_index=kb.get("bm25_index"),
            top_k=20,
        )
        json_candidates = retriever._apply_json_heuristics(json_candidates, keywords)
        # Буст за релевантные термины
        for c in json_candidates:
            chunk_lower = c["text"].lower()
            if any(term in chunk_lower for term in ["норма", "аневризм", "порог", "рекомендуется", "диаметр"]):
                c["score"] += 0.20
            if re.search(r'(>=|<=|≥|≤|>|<)\s?\d+[.,]?\d*\s?(мм|см)', chunk_lower):
                c["score"] += 0.25
        json_candidates = [c for c in json_candidates if c["score"] >= settings.MIN_RELEVANCE_SCORE]
        json_results = sorted(json_candidates, key=lambda x: x["score"], reverse=True)[:5]

    vector_results = []
    seen = set()
    for item in symptom_results + json_results:
        key = (item["source"], item["chunk_id"])
        if key not in seen:
            seen.add(key)
            vector_results.append(item)

    retriever_inst = get_retriever()
    graph_results = retriever_inst.graph_expand(seed_results=vector_results, graph=kb.get("knowledge_graph"), chunks=kb["chunks"],
                                           max_hops=2, max_graph_chunks=10)

    final_guidelines = retriever_inst.merge_results(vector_results=vector_results, graph_results=graph_results, final_top_k=8)
    if not final_guidelines:
        warnings.append("Ретривер не нашел релевантных фрагментов")

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
            f"Текст: {item['text']}"
        )

    query = state.get("query", "").strip()
    history = state.get("patient_history", "").strip()

    clinical_data = []
    if query:
        clinical_data.append(f"Жалобы/Симптомы:{query}")
    if history:
        clinical_data.append(f"История болезни (Анамнез):{history}")
    if not clinical_data:
        clinical_data.append("Жалобы и анамнез не предоставлены.")

    fused_context = (
            f"Данные пациента:\n" + "\n\n".join(clinical_data) +
            f"Контекст из гайдлайнов:\n" + ("\n\n".join(blocks) if blocks else "Ничего не найдено")
    )
    return {**state, "fused_context": fused_context}


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
</rules>
<output_format>
Верните ответ строго в следующей структуре. Не добавляйте вводные/заключительные фразы вне схемы:
[Контекст] Краткая выжимка релевантных норм/рекомендаций из подгруженного контекста.
[Анализ] Сопоставление измерений пациента с нормами. Выявленные отклонения (с конкретными цифрами).
[Интерпретация] Клиническая значимость изменений. Оценка рисков (стабильность, прогрессирование, угроза разрыва и т.д.).
[Заключение] Предварительный диагноз/статус. Рекомендации по тактике (наблюдение, КТ-контроль через Х мес., консультация, хирургическое/эндоваскулярное лечение).
</output_format>  """.strip()

    return {**state, "final_prompt": prompt}

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

    metadata = {
        "thread_id": state.get("thread_id", f"unknown-{uuid.uuid4()}"),
        "patient_data_keys": list(state.get("patient_data", {}).keys()),
        "retrieved_chunks_count": len(retrieved),
        "status": "success",
    }
    save_llm_trace_jsonb(prompt=state.get("final_prompt", ""), response=answer, model=VLLM_MODEL, metadata=metadata)
    return {**state, "raw_llm_output": answer}


def build_graph():
    builder = StateGraph(MedGraphState)
    builder.add_node("ingest_request", ingest_request)
    builder.add_node("initialize_kb", initialize_kb)
    builder.add_node("retrieve_graph_context", retrieve_graph_context)
    builder.add_node("fuse_context", fuse_context)
    builder.add_node("build_prompt", build_prompt)
    builder.add_node("call_local_llm", call_local_llm)

    # Conditional routing based on errors
    def should_continue_after_ingest(state: MedGraphState) -> str:
        return END if state.get("errors") else "initialize_kb"

    def should_continue_after_init(state: MedGraphState) -> str:
        return END if state.get("errors") else "retrieve_graph_context"

    builder.add_edge(START, "ingest_request")
    builder.add_conditional_edges("ingest_request", should_continue_after_ingest, {END: END, "initialize_kb": "initialize_kb"})
    builder.add_conditional_edges("initialize_kb", should_continue_after_init, {END: END, "retrieve_graph_context": "retrieve_graph_context"})
    builder.add_edge("retrieve_graph_context", "fuse_context")
    builder.add_edge("fuse_context", "build_prompt")
    builder.add_edge("build_prompt", "call_local_llm")
    builder.add_edge("call_local_llm", END)

    return builder.compile(checkpointer=InMemorySaver())


def generate_medical_report(query: str, patient_history: str, patient_data: dict, guideline_paths: list[str]) -> Dict[str, Any]:
    initial_state: MedGraphState = {
        "query": query,
        "patient_history": patient_history,
        "guideline_paths": guideline_paths,
        "patient_data": patient_data,
        "thread_id": f"api-{uuid.uuid4()}",
        "warnings": [],
        "errors": [],
    }

    thread_id = initial_state["thread_id"]
    config = {"configurable": {"thread_id": thread_id}}
    graph = get_graph()
    result = graph.invoke(initial_state, config=config)

    return {
        "report": result.get("raw_llm_output"),
        "warnings": result.get("warnings", []),
        "errors": result.get("errors", []),
    }
