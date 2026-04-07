from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class MedGraphState(TypedDict, total=False):
    query: str 
    patient_history: str
    guideline_paths: List[str]
    image_paths: List[str]

    warnings: List[str]
    errors: List[str]

    retrieved_guidelines: List[Dict[str, Any]]
    image_findings: List[Dict[str, Any]]

    fused_context: str
    final_promt: str

    diagnosis_json: Dict[str, Any]
    raw_llm_output: str

    requires_human_review: bool
    approved_by_human: bool

    trace_meta: Dict[str, Any]

def ingest_request(state: MedGraphState) -> MedGraphState: 
    warnings = state.get("warnings", [])
    errors = state.get("errors", [])

    if not state.get("query"):
        errors.append("Не передан запрос")

    return {
            **state,
            "warnings": warnings,
            "errors": errors,
    }

def validate_inputs(state: MedGraphState) -> MedGraphState:
    warnings = state.get("warnings", [])
    errors = state.get("errors", [])

    if not state.get("image_paths"):
        warnings.append("Изображения пока не переданы")

    if not state.get("guideline_paths"):
        warnings.append("Гайдлайны пока не переданы")

    return {
        **state,
        "warnings": warnings,
        "errors": errors,
    }

def retrieve_text_context(state: MedGraphState) -> MedGraphState: 
    query = state.get("query", "").lower()

    mocked_guidelines = [
            {
                "text": "При кашле и температуре следует оценить наличие инфекции нижних дыхательных путей.",
            "source": "clinical_guideline_1.pdf",
            "section": "Раздел 2.1",
            "score": 0.91,
        },
        {
            "text": "При наличии одышки, лихорадки и изменений на КТ необходимо исключить пневмонию.",
            "source": "clinical_guideline_2.pdf",
            "section": "Раздел 4.3",
            "score": 0.88,
        },
        {
            "text": "Курение в анамнезе повышает вероятность хронической патологии лёгких и осложняет интерпретацию симптомов.",
            "source": "clinical_guideline_3.pdf",
            "section": "Раздел 1.5",
            "score": 0.84,
        },
    ]

    selected = []

    if "каш" in query or "темпера" in query:
        selected = mocked_quidelines[:2]
    else:
        selected = mocked_quidelines[:1]

    return {
            **state,
            "retrieved_guidelines": selected,
    }

def fuse_context(state: MedGraphState) -> MedGraphState:
    query = state.get("query", "")
    patient_history = state.get("patient_history", "")
    retrieved_guidelines = state.get("retrieved_guidelines", [])

    guideline_blocks = []
    for i, item in enumerate(retrieved_guidelines, start=1):
        block = (
            f"[GUIDELINE {i}]\n"
            f"Источник: {item.get('source', 'unknown')}\n"
            f"Раздел: {item.get('section', 'unknown')}\n"
            f"Score: {item.get('score', 'n/a')}\n"
            f"Текст: {item.get('text', '')}"
        )
        guideline_blocks.append(block)

     fused_context = (
        f"Жалобы/запрос:\n{query}\n\n"
        f"Анамнез:\n{patient_history}\n\n"
        f"Релевантные фрагменты гайдлайнов:\n"
        + "\n\n".join(guideline_blocks)
    )

    return {
        **state,
        "fused_context": fused_context,
    }

def build_prompt(state: MedGraphState) -> MedGraphState:
    """
    Формируем финальный промпт для будущей LLM.
    Пока LLM не вызываем.
    """
    fused_context = state.get("fused_context", "")

    prompt = f"""
Ты — медицинский AI-ассистент поддержки врача.
Используй только предоставленный контекст.
Не придумывай факты вне контекста.
Сформируй:
1. список вероятных диагнозов,
2. краткое обоснование,
3. рекомендуемые следующие шаги,
4. уровень неопределённости,
5. предупреждение, что итог требует проверки врачом.

Контекст:
{fused_context}
""".strip()

    return {
        **state,
        "final_prompt": prompt,
    }

def build_graph():
    builder = StateGraph(MedGraphState)

    builder.add_node("ingest_request", ingest_request)
    builder.add_node("validate_inputs", validate_inputs)
    builder.add_node("retrieve_text_context", retrieve_text_context)
    builder.add_node("fuse_context", fuse_context)
    builder.add_node("build_prompt", build_prompt)

    builder.add_edge(START, "ingest_request")
    builder.add_edge("ingest_request", "validate_inputs")
    builder.add_edge("validate_inputs", "retrieve_text_context")
    builder.add_edge("retrieve_text_context", "fuse_context")
    builder.add_edge("fuse_context", "build_prompt")
    builder.add_edge("build_prompt", END)

    graph = builder.compile(checkpointer=InMemorySaver())
    return graph

if __name__ == "__main__":
    graph = build_graph()

    initial_state = {
        "query": "Кашель, слабость, температура",
        "patient_history": "Пациент 54 лет, курение 20 лет",
        "guideline_paths": ["data/guidelines"],
        "image_paths": ["data/images/ct_001.png"],
        "warnings": [],
        "errors": [],
        "trace_meta": {"case_id": "demo-001"},
    }

    config = {
        "configurable": {
            "thread_id": "demo-thread-002"
        }
    }

    result = graph.invoke(initial_state, config=config)
    print("Результат выполнения графа:")
    print(result)















