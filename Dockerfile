FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download pl_core_news_lg

COPY . .

RUN useradd -m appuser
USER appuser

EXPOSE 7860


CMD ["python", "app.py"]