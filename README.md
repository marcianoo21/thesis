# 🤖 Konwersacyjny System Rekomendacji Restauracji

System do rekomendacji restauracji w Łodzi wykorzystujący:
- **PLLuM-12B** - polski model językowy do naturalnej konwersacji
- **RoBERTa embeddings** - semantyczne wyszukiwanie
- **FAISS** - szybkie wyszukiwanie wektorowe
- **RAG** - Retrieval Augmented Generation

## 🚀 Instalacja

### 1. Sklonuj repozytorium
```bash
git clone <your-repo>
cd restaurant-recommender
```

### 2. Zainstaluj zależności
```bash
pip install -r requirements.txt
```

### 3. Konfiguracja API Token

Potrzebujesz tokenu Hugging Face:
1. Zarejestruj się na https://huggingface.co/
2. Przejdź do https://huggingface.co/settings/tokens
3. Utwórz nowy token (Read access wystarczy)

Skopiuj `.env.example` do `.env` i wklej swój token:
```bash
cp .env.example .env
nano .env  # lub edytor tekstu
```

Wpisz:
```
HF_TOKEN=hf_twój_token_tutaj
```

### 4. Przygotuj embeddingi

Jeśli jeszcze nie masz pliku z embeddingami:
```bash
python create_embeddings_mean.py
# lub
python create_embeddings_cls.py
```

## 📁 Struktura projektu

```
.
├── conversational_rag.py          # Główny system RAG
├── chat_interface.py              # Interfejs czatu
├── test_system.py                 # Skrypty testowe
├── create_embeddings_mean.py      # Tworzenie embeddingów (mean pooling)
├── create_embeddings_cls.py       # Tworzenie embeddingów (CLS pooling)
├── embedding_model.py             # Model embeddingów
├── search_restaurants.py          # Podstawowe wyszukiwanie
├── requirements.txt               # Zależności
├── .env                          # Konfiguracja (nie commitować!)
└── output_files/
    ├── lodz_restaurants_cafes_emb_input.jsonl
    └── lodz_restaurants_cafes_embeddings_mean.jsonl
```

## 🎮 Użycie

### Podstawowy interfejs czatu
```bash
python chat_interface.py
```

Przykładowa konwersacja:
```
👤 Ty: Cześć!
🤖 Asystent: Witaj! Jak mogę Ci pomóc w znalezieniu restauracji w Łodzi?

👤 Ty: Szukam dobrej pizzerii
🤖 Asystent: Świetnie! Znalazłem kilka doskonałych pizzerii:
1. Pizzeria Napoletana - ul. Piotrkowska 50
   ⭐ 4.8/5.0 (1234 opinii)
   ...
```

### Komendy specjalne
- `exit`, `quit`, `q` - zakończ program
- `clear`, `reset` - wyczyść historię konwersacji
- `save` - zapisz konwersację do JSON

### Testy systemowe
```bash
python test_system.py
```

## 🔧 Konfiguracja

### conversational_rag.py

Główne parametry do dostosowania:

```python
rag = ConversationalRAG(
    llm_client=llm,
    search_function=search,
    max_history=10,  # Liczba par w historii
)
```

W `generate_response()`:
```python
response = rag.generate_response(
    user_message,
    k=5  # Liczba wyników wyszukiwania
)
```

### Prompt systemowy

Możesz dostosować prompt w `conversational_rag.py`:
```python
custom_prompt = """
Twój własny prompt systemowy...
"""

rag = ConversationalRAG(
    llm_client=llm,
    search_function=search,
    system_prompt=custom_prompt
)
```

## 🧠 Jak to działa?

### 1. Ekstakcja zapytania
```
Użytkownik: "Szukam dobrej pizzerii"
    ↓
LLM ekstrahuje: "pizzeria"
```

### 2. Wyszukiwanie semantyczne
```
"pizzeria" → embedding (1024 wymiarów)
    ↓
FAISS wyszukuje podobne wektory
    ↓
Top 5 najbardziej podobnych restauracji
```

### 3. Generowanie odpowiedzi
```
Historia + Zapytanie + Wyniki → PLLuM
    ↓
Naturalna odpowiedź w języku polskim
```

## 📊 Model embeddingów

Używamy **sdadas/mmlw-retrieval-roberta-large**:
- Polski model RoBERTa
- 1024 wymiarów
- Mean lub CLS pooling
- Znormalizowane embeddingi

## 🔍 Wyszukiwanie

FAISS IndexFlatIP:
- Inner Product similarity
- Dokładne wyniki (nie przybliżone)
- Szybkie dla małych zbiorów (<100k)

## 🎯 Przykłady użycia

### Prosty kod
```python
from conversational_rag import create_rag_system

# Inicjalizacja
rag, search = create_rag_system()

# Konwersacja
response = rag.generate_response("Szukam sushi")
print(response)

# Kolejne pytanie (pamięta kontekst)
response = rag.generate_response("Ale w centrum miasta")
print(response)
```

### Z własną funkcją wyszukiwania
```python
from conversational_rag import PLLuMLLM, ConversationalRAG

llm = PLLuMLLM()

def my_search(query, k=5):
    # Twoja własna logika
    return results

rag = ConversationalRAG(llm, my_search)
```

## 🐛 Troubleshooting

### Błąd: "Brak HF_TOKEN"
- Sprawdź czy plik `.env` istnieje
- Sprawdź czy `HF_TOKEN=...` jest poprawny
- Spróbuj: `export HF_TOKEN='twój_token'`

### Błąd: "FileNotFoundError: embeddings"
- Uruchom najpierw: `python create_embeddings_mean.py`
- Sprawdź czy plik istnieje w `output_files/`

### Wolne odpowiedzi
- Model PLLuM-12B jest duży (12B parametrów)
- Pierwsza odpowiedź jest najwolniejsza (ładowanie modelu)
- Rozważ użycie mniejszego modelu lub GPU

### Błąd FAISS
- Dla CPU: `pip install faiss-cpu`
- Dla GPU: `pip install faiss-gpu`

## 📝 TODO / Rozszerzenia

- [ ] Wsparcie dla filtrów (cena, kuchnia, ocena)
- [ ] Geolokalizacja użytkownika
- [ ] Wykresy i wizualizacje
- [ ] Web interface (Streamlit/Gradio)
- [ ] Wielojęzyczność
- [ ] Integracja z API restauracji
- [ ] Rezerwacje stolików
- [ ] Historia preferencji użytkownika (persistent)

## 📄 Licencja

MIT

## 🤝 Kontakt

Pytania? Sugestie? Otwórz issue na GitHubie!

---

**Stworzone z ❤️ wykorzystując PLLuM, FAISS i RoBERTa**