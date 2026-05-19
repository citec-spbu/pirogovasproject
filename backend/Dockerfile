FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Hugging Face cache paths
ENV HF_HOME=/opt/huggingface
ENV HF_HUB_CACHE=/opt/huggingface/hub
ENV SENTENCE_TRANSFORMERS_HOME=/opt/huggingface/sentence-transformers
ENV TRANSFORMERS_CACHE=/opt/huggingface/hub

# Build-time variables: откуда скачиваем модели
ARG HF_TOKEN
ARG EMBEDDING_MODEL_REPO=DmitryPogrebnoy/MedRuBertTiny2
ARG CROSS_ENCODER_MODEL_REPO=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

# Runtime variables: откуда приложение использует модели
ENV EMBEDDING_MODEL_NAME=/opt/models/embedding
ENV CROSS_ENCODER_MODEL_NAME=/opt/models/cross_encoder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pandoc \
    wkhtmltopdf \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Скачиваем модели с Hugging Face в фиксированные локальные папки внутри image
RUN python - <<'PY'
import os
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN") or None

embedding_repo = os.environ.get(
    "EMBEDDING_MODEL_REPO",
    "DmitryPogrebnoy/MedRuBertTiny2",
)

cross_encoder_repo = os.environ.get(
    "CROSS_ENCODER_MODEL_REPO",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
)

models = [
    (embedding_repo, "/opt/models/embedding"),
    (cross_encoder_repo, "/opt/models/cross_encoder"),
]

for repo_id, local_dir in models:
    print(f"Downloading model: {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=token,
    )
    print(f"Downloaded: {repo_id}")

print("All Hugging Face models downloaded.")
PY

# Проверяем, что embedding-модель грузится из локального пути
RUN python - <<'PY'
from sentence_transformers import models

print("Checking embedding model loading from local path...")
model = models.Transformer("/opt/models/embedding")
print("Embedding model loaded successfully.")
PY

# Проверяем, что CrossEncoder грузится из локального пути
RUN python - <<'PY'
from sentence_transformers import CrossEncoder

print("Checking CrossEncoder loading from local path...")
model = CrossEncoder("/opt/models/cross_encoder", device="cpu")
score = model.predict([["тестовый запрос", "тестовый документ"]], show_progress_bar=False)
print("CrossEncoder check score:", score)
PY

# После скачивания и проверки запрещаем runtime ходить в Hugging Face
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
