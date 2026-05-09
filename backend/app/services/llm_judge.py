import json
from pathlib import Path
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from app.core.config import settings

VLLM_BASE_URL = settings.VLLM_BASE_URL
VLLM_API_KEY = settings.VLLM_API_KEY
VLLM_MODEL = settings.VLLM_MODEL

TRACE_FILE = Path("llm_traces.jsonl")
OUTPUT_FILE = Path("judge_scores.json")

class LLMJudge:
    def __init__(self):
        self.llm = ChatOpenAI(model=VLLM_MODEL, base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY, temperature=0.0)

    def load_latest_trace(self) -> Dict[str, Any]:
        if not TRACE_FILE.exists():
            raise FileNotFoundError("Trace file not found")

        with TRACE_FILE.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            raise ValueError("Trace file is empty")
        return json.loads(lines[-1])

    def build_prompt(self, criterion: str, trace: Dict[str, Any]) -> str:
        definitions = {
            "usefulness": """
USEFULNESS (Полезность) - Оцените, насколько итоговое заключение полезно для кардиохирурга при принятии клинического решения по КТ-аорте.
Содержит ли конкретные клинические выводы о диагнозе?
Помогает ли определить дальнейшую тактику?
Есть ли клинические рекомендации?
Пример оценивания предоставленных примеров:
Хорошо:
"В предоставленных измерениях диаметр восходящей аорты 56 мм, что свидетельствует о значительном
расширении этого отдела аорты и дилатации аорты. Показанием к хирургической коррекции восходящего отдела аорты
считается аневризма диаметром 55 мм и более при отсутствии генетических дегенеративных заболеваний и двустворчатого аортального клапана."
Плохо:
"Есть изменения. Требуется наблюдение и консультация врача-кардиохирурга."

Шкала:
1–3: почти бесполезно
4–6: частично полезно
7–8: клинически полезно
9–10: может использоваться врачом
""",

            "groundedness": """
GROUNDEDNESS (Обоснованность) - Оцените, насколько ответ строго основан на входных измерениях JSON, анамнезе, предоставленном контексте.
Есть ли галлюцинации или утверждения не подкрепленные данными?
Правильно ли интерпретированы размеры аорты?
Ссылается ли логика заключения на предоставленный контекст?
Есть ли ложные диагнозы, вымышленные пороги вмешательства или рекомендации без доказательств?

Пример оценивания предоставленных примеров:
Хорошо:
"Максимальный диаметр аорты 56 мм. Симптомов и жалоб у пациента нет. Согласно источнику 1, рекомендуется выполнять протезирование аорты
бессимптомным пациентам с аневризмами корня и/или ВА при максимальном диаметре аорты более 5,5 см.
Пациенту необходимо провести протезирование аорты"
Плохо:
"Максимальный диаметр аорты 45 мм. Пациенту необходимо провести протезирование аорты"

Шкала:
1–3: сильная галлюцинация
4–6: частичная опора на факты
7–8: в целом обосновано
9–10: строго обосновано
""",

            "efficiency": """
Эффективность (EFFICIENCY) - Оцените оптимальность использованных инструментов.
Есть ли лишние шаги в траектории?
Рационально ли используется поиск информации в базе знаний?
Нет ли избыточных повторов операций или инструментов?
Минимально ли время выполнения операций (latency) при неизменной клинической достоверности?

Пример оценивания предоставленных примеров:
Хорошо:
retrieval -> fusion -> prompt -> inference
Плохо:
несколько избыточных поисковых циклов или чрезмерный контекст

Шкала:
1–3: крайне неэффективно
4–6: средне
7–8: эффективно
9–10: оптимально
"""
        }

        return f"""
Вы — объективный клинический аудитор качества медицинских заключений по КТ-измерениям аорты.
Ваша задача — оценить качество медицинского заключения, сгенерированного LLM, по критерию: {definitions[criterion]}.

Перед финальным ответом ОБЯЗАТЕЛЬНО выполни:

1. REASONING: Сначала подробно рассуждай. Проанализируй ответ, приведи цитаты из текста, укажи на совпадения или расхождения с исходными данными.
Если данных для оценки критерия недостаточно (например, ответ пустой, RAG не нашёл контекста, или критерий не применим), ты можешь ответить "Неизвестно/Данные не предоставлены" и обязательно обоснуй почему.

2. FINAL SCORE: Верни ТОЛЬКО число 1–10 или строку "Неизвестно/Данные не предоставлены". Если данных недостаточно, обоснуй это.

Данные для оценки (трассировка запроса к LLM):
{json.dumps(trace, ensure_ascii=False, indent=2)}

Формат ответа:

REASONING:
...

FINAL SCORE:
...
"""

    def extract_score(self, response: str):
        marker = "FINAL SCORE:"
        if marker not in response:
            return "Неизвестно/Данные не предоставлены"

        score_part = response.split(marker)[-1].strip()

        if "Неизвестно" in score_part:
            return "Неизвестно/Данные не предоставлены"

        try:
            score = int(score_part.split()[0])
            if 1 <= score <= 10:
                return score
        except:
            pass
        return "Неизвестно/Данные не предоставлены"

    def evaluate_one(self, criterion: str, trace: Dict[str, Any]):
        prompt = self.build_prompt(criterion, trace)
        response = self.llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        return self.extract_score(text)

    def evaluate(self):
        trace = self.load_latest_trace()
        results = {
            "usefulness": self.evaluate_one("usefulness", trace),
            "groundedness": self.evaluate_one("groundedness", trace),
            "efficiency": self.evaluate_one("efficiency", trace)
        }
        with OUTPUT_FILE.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return results
