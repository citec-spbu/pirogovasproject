from piragi import Ragi
import os
import time
import json
from datetime import datetime
#from langfuse import Langfuse
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling
from transformers import AutoTokenizer, AutoModel
from typing import Optional, List, Dict, Any
import hashlib
import gc
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = ""
CONFIG = {
    "llm": {
        "model": os.getenv("VLLM_MODEL", "qwen3.5-35b-32k"),
        "base_url": os.getenv("VLLM_BASE_URL", "https://llm.ai-expert-opinion.ru/v1"),
        "api_key": os.getenv("LLM_API_KEY", "sk-bc9a3a95e190fc9cf5525a4726b16658")
        # "model": "qwen/qwen3.6-plus-preview:free",
        # "base_url": "https://openrouter.ai/api/v1",
        # "token": "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"
    },
    "embedding": {
        "model": 'DmitryPogrebnoy/MedRuBertTiny2', # Model for vectorization
        "batch_size": 8,
    },
    "chunk": {
        "strategy": "semantic", # Strategy for splitting documents into chunks for vectorization
        "size": 300,  # Chunk size
        "overlap": 50,  # Overlap between chunks
    },
    "retrieval": {
        "use_hybrid_search": True, # Use hybrid search (BM25 + embeddings)
        "use_cross_encoder": False, # Not using cross-encoder for re-ranking
        "priority_documents": ["docs"],
        "min_relevance_score": 0.00,  # Minimum relevance score for retrieved chunks
        "max_chunks_per_doc": 3 # Max chunks per document in retrieval results
    }
}

def clear_memory():
    gc.collect()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    print("Память очищена")

def is_index_exist(persist_dir: str) -> bool:
    """Checks if the Ragi index directory exists and is not empty."""
    if not os.path.exists(persist_dir):
        return False
    if not os.listdir(persist_dir):
        return False
    return True

def initialize_kb(docs_path: str = "/content/docs",
                  persist_dir: str = "/content/.piragi_index") -> Ragi:
    """
    Initializes or loads the Ragi knowledge base (vector database).
    This function performs the 'vectorization of the database' by creating or loading the index.
    """
    clear_memory()
    if is_index_exist(persist_dir):
        print("Найден индекс, загружаем его")
    else:
        print("Индекс не найден, создаём новый")
    kb = Ragi(docs_path, persist_dir=persist_dir, config=CONFIG)
    return kb

#setting embedding model
embedding_model_name = CONFIG["embedding"]["model"]
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
word_embedding_model = AutoModel.from_pretrained(embedding_model_name)

# Define a pooling layer
pooling_model = Pooling(word_embedding_model.config.hidden_size,
                        pooling_mode_mean_tokens=True,
                        pooling_mode_cls_token=False,
                        pooling_mode_max_tokens=False)

embedding_model_instance = SentenceTransformer(modules=[word_embedding_model, pooling_model])
print(f"Используемая эмбеддинг модель: {embedding_model_name}")

start_time = time.time()
kb = initialize_kb()

with open(f"/content/0.json", "r", encoding="utf-8") as f:
    patient_data=json.load(f)

def debug_rag_pipeline(kb, query: str, test_sentence_from_pdf: str):
    print("ЗАПУСК ДИАГНОСТИКИ RAG")
    # ЭТАП 1: Проверка наличия данных в индексе
    try:
        sources = kb.count()
        print(f"Этап 1: Документы в базе: {sources}")
        if sources == 0:
            print("[!!!] ОШИБКА: База пуста. Проверь путь к индексу.")
    except Exception as e:
        print(f"[!] Не удалось получить список источников: {e}")

    # ЭТАП 2: Проверка векторизации (Эмбеддингов)
    try:
        results = kb.retrieve(test_sentence_from_pdf, top_k=1)
        if results:
          sim = results[0].score
          print(f"Этап 2: Сходство найденного фрагмента: {sim:.4f}")
          if sim < 0.3:
            print("[!] ПРЕДУПРЕЖДЕНИЕ: Низкое сходство даже для точного совпадения. Проверь модель.")
          else:
            print("[!] Текст из PDF вообще не найден в индексной базе.")
    except Exception as e:
        print(f"[!] Ошибка при проверке векторизации: {e}")

test_txt = "Параренальная аневризма брюшной аорты"
debug_rag_pipeline(kb, "расширение аорты", test_txt)

print("\nПроверка NER-подобного переранжирования")
ner_test_query = "Параренальная аневризма брюшной аорты"
ner_retrieval_results = retrieve_with_ner_re_ranking(kb, ner_test_query, json_data=patient_data, top_k=5)

if ner_retrieval_results["chunks"]:
    print(f"Результаты для запроса с NER-переранжированием: '{ner_test_query}' ")
    for i, chunk in enumerate(ner_retrieval_results["chunks"]):
        print(f"  Chunk {i+1} (Score: {chunk.score:.4f}, Source: {chunk.source}): {chunk.chunk[:50]}...")
else:
    print(f"Для запроса '{ner_test_query}' с NER-переранжированием не найдено релевантных чанков.")

import pymorphy3
import re

CONFIG = {
    "llm": {
        "model": "qwen3.5-35b-32k",
        "base_url": "https://llm.ai-expert-opinion.ru/v1",
        "api_key": os.getenv("LLM_API_KEY", "sk-bc9a3a95e190fc9cf5525a4726b16658")
        # "model": "qwen/qwen3.6-plus-preview:free",
        # "base_url": "https://openrouter.ai/api/v1",
        # "token": "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"
    },
    "embedding": {
        "model": 'DmitryPogrebnoy/MedRuBertTiny2', # Model for vectorization
        "batch_size": 16,
    },
    "chunk": {
        "strategy": "semantic", # Strategy for splitting documents into chunks for vectorization
        "size": 300,  # Chunk size
        "overlap": 50,  # Overlap between chunks
    },
    "retrieval": {
        "use_hybrid_search": True, # Use hybrid search (BM25 + embeddings)
        "use_cross_encoder": False, # Not using cross-encoder for re-ranking
        "priority_documents": ["./content/new"],
        "min_relevance_score": 0.00,  # Minimum relevance score for retrieved chunks
        "max_chunks_per_doc": 3 # Max chunks per document in retrieval results
    }
}

TRANSLATION_MAP = {
    "Descending Aorta": "Нисходящая аорта",
    "Isthmus": "Перешеек аорты",
    "Arch after LSA": "Дуга аорты после отхождения левой подвключичной артерии", # Исправлен ключ под ваш JSON
    "Arch after TBC": "Дуга аорты после отхождения плечеголовного ствола",   # Исправлен ключ под ваш JSON
    "Ascending Aorta befor TBC": "Восходящая аорта перед плечеголовным стволом", # Исправлен ключ
    "Ascending Aorta": "Восходящая аорта",
    "max_diam_1": "Максимальные диаметры",
    "max_diam_2": "Максимальные диаметры",
    "min_diam": "Минимальный диаметр",
    "perimetr": "Периметр сосуда",
    "area": "Площадь поперечного сечения"
}
EXCLUDE_KEYS = {"ct_img", "mask_img"}
def _extract_keywords_from_json(json_data: Dict) -> List[str]:
    keywords = set()
    def extract(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in EXCLUDE_KEYS: continue
                # Добавляем ключ и его перевод
                if isinstance(k, str) and len(k) > 2:
                    keywords.add(k.lower())
                    if k in TRANSLATION_MAP:
                        keywords.add(TRANSLATION_MAP[k].lower())
                # Добавляем значение, только если это не путь к картинке
                if isinstance(v, str) and len(v) > 2 and not v.endswith('.png'):
                    keywords.add(v.lower())
                # Рекурсия только для структур данных
                if isinstance(v, (dict, list)):
                    extract(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item)

    extract(json_data)
    return list(keywords)

morph = pymorphy3.MorphAnalyzer()

def _normalize_text(text: str) -> set:
    """Очистка текста и приведение слов к начальной форме."""
    words = re.findall(r'\b[а-яА-ЯёЁa-zA-Z]{3,}\b', text.lower())
    return {morph.parse(w)[0].normal_form for w in words}

def retrieve_with_ner_re_ranking(kb: Ragi, query: str, json_data: Optional[Dict] = None, top_k: int = 3, min_score: float = 0.00) -> Dict[str, Any]:
    search_query = query
    if json_data:
        # Добавляем названия отделов и "триггеры" нормативов к поиску
        zones = " ".join(json_data.keys())
        search_query += f" {zones} норма диаметр классификация показатели мм превышать"

    citations = kb.retrieve(search_query, top_k=30)
    if not citations:
        return {"chunks": [], "sources": []}

    keywords_from_json = set()
    if json_data:
        raw_keywords = _extract_keywords_from_json(json_data)
        for kw in raw_keywords:
            keywords_from_json.update(_normalize_text(kw))

    # --- ПЕРЕРАНЖИРОВАНИЕ ---
    for c in citations:
        chunk_lower = c.chunk.lower()
        chunk_normalized = _normalize_text(c.chunk)

        # А) Базовый NER-бонус (совпадение терминов)
        matches = keywords_from_json.intersection(chunk_normalized)
        c.score += len(matches) * 0.05

        # Б) БОНУС ЗА ЧИСЛОВЫЕ НОРМАТИВЫ
        # Ищем паттерны: цифра+мм, знаки сравнения, диапазоны
        numeric_patterns = [
            r'\d{1,2}(\.\d)?\s?мм',   # 30 мм, 32.5 мм
            r'[><=]\s?\d{1,2}',       # > 40, <= 30
            r'от\s\d{1,2}\sдо\s\d{1,2}' # от 20 до 30
        ]
        for pattern in numeric_patterns:
            if re.search(pattern, chunk_lower):
                c.score += 0.15

        # В) ШТРАФ ЗА КАРТИНКИ
        stop_patterns = ["Рис.", "рисунок", "вид сбоку", "снимок", "визуализация", "иллюстрация", "график"]
        for stop_word in stop_patterns:
            if stop_word in chunk_lower:
                c.score -= 0.15

        # Г) ШТРАФ ЗА ССЫЛКИ НА ЛИТЕРАТУРУ [1]
        # Находим все вхождения цифр в квадратных скобках
        reference_matches = re.findall(r'\[[\d,\s\-]+\]', chunk_lower)
        if reference_matches:
            c.score -= len(reference_matches) * 0.005

    chunks = [c for c in citations if c.score >= min_score]
    chunks.sort(key=lambda c: c.score, reverse=True)

    return {
        "chunks": chunks[:top_k],
        "sources": list(set(c.source for c in chunks[:top_k]))
    }

# def retrieve_with_ner_re_ranking(kb: Ragi, query: str, json_data: Optional[Dict] = None, top_k: int = 3, min_score: float = 0.00) -> Dict[str, Any]:
#     citations = kb.retrieve(query, top_k=30)
#     if not citations:
#         return {"chunks": [], "sources": []}

#     if json_data:
#         raw_keywords = _extract_keywords_from_json(json_data)
#         normalized_keywords = set()
#         for kw in raw_keywords:
#             normalized_keywords.update(_normalize_text(kw))

#         for c in citations:
#             chunk_words = _normalize_text(c.chunk)
#             matches = normalized_keywords.intersection(chunk_words)
#             keyword_bonus = len(matches) * 0.1
#             c.score += keyword_bonus

#     filtered_chunks = [c for c in citations if c.score >= min_score]
#     filtered_chunks.sort(key=lambda c: c.score, reverse=True)

#     final_selection = filtered_chunks[:top_k]
#     return {
#         "chunks": final_selection,
#         "sources": list(set(c.source for c in final_selection))
#     }

#prompt engineering
def format_prompt(json_data: Dict[str, Any], retrieved_context: str, instruction: str = "Сформируйте заключение врача-кардиохирурга, основываясь на данных измерений КТ аорты") -> str:
    json_formatted = json.dumps(json_data, ensure_ascii=False, indent=2) #Dict->json
    return f"""<system>
Вы — врач-кардиохирург, эксперт в области кардиологии и сосудистой хирургии с пятидесятилетним опытом работы.
Отвечай ТОЛЬКО на основе предоставленного контекста. Если информации недостаточно, выведи ошибку
Ваша задача: сформировать профессиональное медицинское заключение на основе предоставленных данных.

Роль:
-Врач-кардиохирург
-Эксперт по компьютерно-томографическим снимкам
-Профессионал в области кардиологии и сосудистой хирургии

Инструкции:
1. Проанализируйте данные КТ снимков пациента из файла json: <patient_data> {json_formatted} </patient_data>
2. Выводы делайте только на основе предоставленного контекста: <clinical_context> {retrieved_context} </clinical_context>
3. Отвечай предельно качественно на сформулированный вопрос: <task> {instruction}
2. Указывайте только подтверждённые измерениями факты, если данных нет, указывайте "данные не предоставлены"
3. Используйте необходимую медицинскую терминологию из подтаскиваемого контекста
4. Структура ответа: [Предоставленный контекст] → [Анализ данных] → [Интерпретация] → [Диагноз+Рекомендации]
</system>

Рассуждайте по шагам:
1. Какие ключевые параметры аорты представлены в данных?
2. Есть ли отклонения от предоставленных из контекста значений?
3. Какой клинический вывод следует из сопоставления измерений и контекста?
4. Сформулируйте итоговое заключение: диагноз и рекомендации врача.
</task>"""

def ask_with_rag(query: str, json_data: Dict[str, Any], kb: Ragi, top_k: int = 5) -> Dict[str, Any]:
    #сразу проверяем есть ли этот запрос + json в кеше, если есть повторно не используем llm
    #cache_key = _get_cache_key(query, json_data)
    #cached = _load_from_cache(cache_key)
    #if cached:
    #    print("Ответ загружен из кэша")
    #    return cached

    #trace = langfuse.trace(name="rag_medical_query", user_id="mashaa",
    #    session_id=f"session_{int(time.time())}", metadata={"model": CONFIG["llm"]["model"], "top_k": top_k})
    try:
        retrieval_result = retrieve_with_ner_re_ranking(kb=kb, query=query, json_data=json_data,
                                                  top_k=top_k, min_score=CONFIG["retrieval"]["min_relevance_score"])
        if not retrieval_result["chunks"]:
            print("Контекст не найден ни в одном из документов")
            context_text = "Дополнительный клинический контекст не предоставлен."
        else:
            context_text = "\n\n".join([
                f"[{i + 1}] {c.chunk} (источник: {c.source}, релевантность: {c.score:.2%})"
                for i, c in enumerate(retrieval_result["chunks"])
            ])
            avg_score = sum(c.score for c in retrieval_result["chunks"]) / len(retrieval_result["chunks"])
            print(f"Средняя релевантность: {avg_score:.2%}")

        prompt = format_prompt(json_data=json_data, retrieved_context=context_text) #, instruction=query)

        #DO NOT WORK WITH LLM CORRECT IIIIIT
        #answer = kb.ask(prompt, top_k=1)
        result = {
            #"query": query,
            #"answer": answer.text,
            "timestamp": datetime.now().isoformat(),
            "citations": [
                {"source": c.source, "score": c.score, "chunk": c.chunk}
                for c in retrieval_result["chunks"]
            ],
            "sources_used": retrieval_result["sources"]
        }

        filename = f"rag_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        #_save_to_cache(cache_key, result)
        print(f"Ответ сохранён в: {filename}")
        return result

    except Exception as e:
        raise

with open(f"/content/0.json", "r", encoding="utf-8") as f:
    patient_data=json.load(f)

result = ask_with_rag(query="Нисходящая аорта, Перешеек аорты, Дуга аорты после отхождения левой подвключичной артерии, Дуга аорты после отхождения плечеголовного ствола, Восходящая аорта перед плечеголовным стволом, Восходящая аорта, Максимальные диаметры. Минимальный диаметр. Периметр сосуда. Площадь поперечного сечения.", json_data=patient_data, kb=kb, top_k=7)
print(f"\nЗаключение:\n{result['answer']}")
print(f"\nВремя выполнения: {time.time() - start_time:.2f} сек")

