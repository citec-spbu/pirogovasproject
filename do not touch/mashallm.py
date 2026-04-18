from piragi import Ragi
import os
import time
import json
from datetime import datetime
from langfuse import Langfuse
import requests

langfuse = Langfuse()
start_time = time.time()
config = {
    "llm": {
        "model": "qwen3.5-35b-32k",
        "base_url": "https://llm.ai-expert-opinion.ru/v1",
        "api_key": os.getenv("LLM_API_KEY", "sk-bc9a3a95e190fc9cf5525a4726b16658"),
        "max_tokens": 4096,
    },
    "embedding": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 32,
    },
    "chunk": {
        "strategy": "semantic",  # семантическое разбиение (умнее фиксированного)
        "size": 512, #размер чанка в токенах
    },
    "retrieval": {
        "use_hybrid_search": True,  # BM25 + векторный поиск
        "use_cross_encoder": False,  # можно включить для точности
    }
}

kb = Ragi("./docs", # "https://docs.example.com/**",  # веб-краулинг
    persist_dir=".piragi_index", config=config)

#При инициализации загружаются документы -> чанкинг -> эмбеддинги -> индекс сохраняется в .piragi_index"

# kb.add("./new_docs") # Добавление документов (если нужно обновить)

def ask_with_rag(query: str, top_k: int = 5):
    #Создаём трейс для этого запроса
    trace = langfuse.trace(name="rag_query", user_id="user_1", session_id=f"session_{int(time.time())}",
        metadata={"model": config["llm"]["model"], "top_k": top_k})

    #Логгируем входные данные
    trace.generation(name="rag_input", input={"query": query, "top_k": top_k},
                                        model=config["embedding"]["model"])

    answer = kb.ask(query, top_k=top_k)
    #query -> эмбеддинг через sentence-transformers
    #Векторный поиск в database: top_k ближайших чанков
    #(Опционально) BM25 + re-ranking если use_hybrid_search=True
    #Формирование промпта.
    #POST-запрос к /v1/chat/completions с промптом
    #Парсинг ответа + извлечение метаданных чанков
    print(f"Ответ:\n{answer.text}\n")
    print("Источники:")
    for i, cite in enumerate(answer.citations, 1):
        print(f"{i}. {cite.source} (score: {cite.score:.2%})")

    result = {"query": query, "answer": answer.text, "timestamp": datetime.now().isoformat(),
        "citations": [ {"source": c.source, "score": c.score, "chunk": c.chunk} for c in answer.citations]}

    filename = f"rag_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Ответ сохранён в: {filename}")

    trace.generation(name="rag_output", output={"answer": answer.text, "citations_count": len(answer.citations)},
        usage_details={"total_citations": len(answer.citations)})

    trace.update(output={"status": "success"}) #Завершаем трейс (отправляет данные на сервер)

    return answer

def evaluate_with_llm_judge(query: str, answer: str):
    judge_trace = langfuse.trace(name="llm_judge_evaluation") # Создаём отдельный трейс для оценки
    judge_prompt = f"""
    Ты — эксперт-оценщик. Оцени ответ на вопрос по шкале от 1 до 10 по критериям: полезность, точность, полнота.
    
    Вопрос: {query}
    Ответ: {answer}

    Верни ответ в формате JSON:
    {{
        "score": <число 1-10>,
        "reasoning": "<краткое обоснование>",
        "improvements": ["<совет 1>", "<совет 2>"]
    }}
    """

    response = requests.post(f"https://llm.ai-expert-opinion.ru/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {config['llm']['api_key']}",
            "Content-Type": "application/json"},
        json={"model": "qwen3.5-35b-32k",
              "messages": [{"role": "user", "content": judge_prompt}],
              "max_tokens": 500})

    evaluation = response.json()["choices"][0]["message"]["content"]

    judge_trace.generation(name="judge_evaluation", input={"prompt": judge_prompt},
                           output={"evaluation": evaluation})

    judge_trace.update(output={"status": "evaluated"})

    return {"raw_evaluation": evaluation}

result = ask_with_rag(query="Как работает механизм внимания в трансформерах?")

end_time = time.time()
execution_time = end_time - start_time
print(f"Время выполнения: {execution_time} секунд")

eval_result = evaluate_with_llm_judge(query="Как работает механизм внимания в трансформерах?", answer=result.text)
print(f"Оценка судьи: {eval_result}")

langfuse.flush() #Финальная синхронизация (гарантирует отправку всех данных)
