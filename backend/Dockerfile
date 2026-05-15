FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update

RUN apt-get install -y pandoc

RUN apt-get install -y wkhtmltopdf

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]