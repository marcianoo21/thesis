FROM python:3.10-slim

# Bezpieczny, niewielki obraz bazowy i katalog roboczy
WORKDIR /app

# Instalacja minimalnego zestawu zależności systemowych
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Kopiujemy wymagania i instalujemy je bez cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalujemy model spaCy pl_core_news_lg w czasie budowy obrazu,
# żeby nie dociągać go przy każdym starcie kontenera
RUN python -m spacy download pl_core_news_lg

# Kopiujemy resztę kodu (bez pliku .env – jest w .gitignore)
COPY . .

# Tworzymy nieuprzywilejowanego użytkownika do uruchamiania aplikacji
RUN useradd -m appuser
USER appuser

# Hugging Face Spaces i większość PaaS używa zmiennej PORT; domyślnie 7860
EXPOSE 7860

# Domyślna komenda startowa
# Aplikacja Flask w app.py musi odczytywać zmienną PORT i nasłuchiwać na 0.0.0.0
CMD ["python", "app.py"]