from functools import wraps
from typing import Any, Dict
import json
import os
import re
import time

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency guard
    def load_dotenv(*args, **kwargs):
        return None


load_dotenv()

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "token-abc123")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

GUIDELINES_FOR_LLM_TOP_K = 4
MAX_GUIDELINE_CHARS = 1200


def track_node_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[{func.__name__}] Выполнено за {elapsed:.3f} сек.")
        return result

    return wrapper


def clean_context_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_GUIDELINE_CHARS:
        text = text[:MAX_GUIDELINE_CHARS] + "..."
    return text


@track_node_time
def fuse_context(state: Dict[str, Any]) -> Dict[str, Any]:
    blocks = []
    guidelines = state.get("retrieved_guidelines", [])
    guidelines = sorted(
        guidelines,
        key=lambda item: item.get("final_score", item.get("score", 0.0)),
        reverse=True,
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
def build_prompt(state: Dict[str, Any]) -> Dict[str, Any]:
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
def call_local_llm(state: Dict[str, Any]) -> Dict[str, Any]:
    retrieved = state.get("retrieved_guidelines", [])
    if not retrieved:
        return {
            **state,
            "raw_llm_output": "Релевантные клинические рекомендации не найдены. Недостаточно данных для анализа.",
        }

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=VLLM_MODEL,
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
