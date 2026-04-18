import os
from dotenv import load_dotenv
from langfuse import observe
from langfuse.openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("VLLM_BASE_URL"),
    api_key=os.getenv("VLLM_API_KEY"),
)

@observe()
def ask_llm(question):
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        messages=[
            {"role": "system", "content": "Отвечай кратко."},
            {"role": "user", "content": question}
        ],
        temperature=0.7,
    )

    answer = response.choices[0].message.content
    return answer

questions = [
    "Что такое vLLM?",
    "Что такое KV cache?",
    "Почему vLLM быстрее transformers?",
    "Что такое инференс LLM?"
]

for q in questions:
    answer = ask_llm(q)
    print("QUESTION:", q)
    print("ANSWER:", answer)
    print()












