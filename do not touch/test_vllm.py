import os
import json
from dotenv import load_dotenv
from langfuse import observe, Langfuse
from langfuse.openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("VLLM_BASE_URL"),
    api_key=os.getenv("VLLM_API_KEY"),
)

langfuse = Langfuse()


@observe()
def ask_llm(question: str) -> str:
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        messages=[
            {"role": "system", "content": "Отвечай кратко и по делу."},
            {"role": "user", "content": question}
        ],
        temperature=0.3,
        max_tokens=100,
    )
    return response.choices[0].message.content


@observe()
def judge_answer(question: str, answer: str) -> dict:
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты оцениваешь ответ другой системы. "
                    "Оцени по 3 критериям: correctness, relevance, clarity. "
                    "Каждый критерий от 0 до 1. "
                    "Верни только JSON."
                )
            },
            {
                "role": "user",
                "content": f"""
Вопрос:
{question}

Ответ:
{answer}

Верни JSON в формате:
{{
  "correctness": 0.0,
  "relevance": 0.0,
  "clarity": 0.0,
  "comment": "короткий комментарий"
}}
"""
            }
        ],
        temperature=0.0,
        max_tokens=200,
    )

    text = response.choices[0].message.content
    return json.loads(text)


question = "Почему vLLM быстрее transformers?"
answer = ask_llm(question)
evaluation = judge_answer(question, answer)

print("ANSWER:")
print(answer)
print("\nEVALUATION:")
print(evaluation)

