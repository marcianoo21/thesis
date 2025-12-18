# 🤖 Konwersacyjny System Rekomendacji Restauracji — Łódź

Pełnoprawny system RAG (Retrieval-Augmented Generation) do rekomendacji restauracji i kawiarni w Łodzi z możliwością prowadzenia naturalnej konwersacji.

## 🎯 Komponenty

### 1. **`conversational_rag.py`** — Silnik RAG

- `ConversationHistory` — zarządza historią czatu
- `ConversationalRAG` — główna klasa integrująca LLM + RAG
- **Adaptery LLM:**
  - `OpenAILLM` — OpenAI API (GPT-3.5/GPT-4)
  - `OllamaLLM` — lokalny Ollama (offline)
  - `SimpleLLM` — tryb demo (bez API)

### 2. **`chat_interface.py`** — Interaktywny interfejs

- Główna pętla czatu
- Integracja z embeddingami FAISS
- Export konwersacji do JSON

### 3. **`example_rag_usage.py`** — Przykłady użycia

- Demo z SimpleLLM
- Demo z OpenAI API
- Niestandardowe system prompty

---

## 🚀 Szybki Start

### Opcja 1: SimpleLLM (brak API, tryb demo)

```bash
python chat_interface.py
```

**Wyjście:**

```
🤖 KONWERSACYJNY SYSTEM REKOMENDACJI RESTAURACJI — ŁÓDŹ

👤 Ty: Szukam dobrej kawiarni blisko centrum
🤖 Asystent: Na podstawie Twojego zapytania znalazłem kilka świetnych opcji:

Znalezione restauracje:
1. **The Brick Coffee Factory**
   Typ: kawa
   Adres: ul. Piotrkowska 123
   Ocena: 4.6⭐ (314 opinii)
   Dopasowanie: 0.856

...
```

### Opcja 2: OpenAI API (wymaga API key)

**1. Ustaw zmienne środowiskowe w `.env`:**

```env
OPENAI_API_KEY=sk-proj-xxxxx
```

**2. Uruchom czat:**

```bash
python chat_interface.py
```

**3. (Opcjonalnie) Testuj przykłady:**

```bash
python example_rag_usage.py
```

---

## 💬 Jak to działa?

### Flow Konwersacji

```
Użytkownik: "Szukam sushi dla 4 osób"
    ↓
[chat_interface.py] → wczytuje input
    ↓
[ConversationalRAG.generate_response()]
    ├─ 1. Dodaj do historii
    ├─ 2. Wyszukaj RAG: search_restaurants("szukam sushi...")
    ├─ 3. Pobierz kontekst: restauracje + oceny + adresy
    ├─ 4. Konstruuj prompt:
    │    System: "Jesteś asystentem rekomendacji..."
    │    Kontekst: "Znalezione: Sphinx (4.7⭐), Hana Sushi (4.1⭐)..."
    │    Historia: "Poprzednie pytania..."
    └─ 5. Wyślij do LLM → Otrzymaj odpowiedź
    ↓
Asystent: "Świetnie! Mam idealne opcje dla Was:
1. Sphinx - 4.7⭐ (10616 opinii) - najwyżej oceniany
2. Hana Sushi - 4.1⭐ (1035 opinii) - bardziej kameralne
Czy któraś z nich Wam pasuje?"
    ↓
[Dodaj odpowiedź do historii] → Gotowe na następne pytanie
```

---

## ⚙️ Konfiguracja

### System Prompt

Domyślny system prompt znajduje się w `ConversationalRAG._default_system_prompt()`.

Aby zmienić, użyj:

```python
custom_prompt = "Jesteś asystentem specjalizującym się w gastronomii..."

rag = ConversationalRAG(
    llm_client=llm,
    search_function=search,
    system_prompt=custom_prompt,
)
```

### Model LLM

**OpenAI:**

```python
from conversational_rag import OpenAILLM

llm = OpenAILLM(
    api_key="sk-...",
    model="gpt-4"  # lub "gpt-3.5-turbo"
)
```

**Ollama (offline):**

```python
from conversational_rag import OllamaLLM

llm = OllamaLLM(
    model="mistral",  # lub "llama2", "neural-chat"
    base_url="http://localhost:11434"
)
```

**SimpleLLM (demo):**

```python
from conversational_rag import SimpleLLM

llm = SimpleLLM()  # Zwraca template, nie wymaga API
```

---

## 📊 Struktura Danych

### Historia Konwersacji

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Szukam sushi",
      "timestamp": "2025-11-30T10:30:45.123456"
    },
    {
      "role": "assistant",
      "content": "Znalazłem kilka opcji...",
      "timestamp": "2025-11-30T10:30:46.234567"
    }
  ]
}
```

### Wyniki RAG

```json
{
  "name": "Sphinx",
  "type": "restauracja",
  "address": "ul. Piotrkowska 100",
  "rating": 4.7,
  "reviews": 10616,
  "relevance_score": 0.856
}
```

---

## 🔧 Zaawansowane Użycie

### Programmatyczne Wyzwolenie

```python
from conversational_rag import ConversationalRAG, SimpleLLM

# Załaduj komponenty
search = load_search_engine()
llm = SimpleLLM()

# Stwórz RAG
rag = ConversationalRAG(
    llm_client=llm,
    search_function=search,
    max_history=10,  # Ostatnie 10 wiadomości
)

# Prowadź konwersację
response = rag.generate_response("Chcę pizzę")
print(response)

# Eksportuj historię
rag.export_conversation("my_chat.json")
```

### Dostęp do Historii

```python
# Pobierz wszystkie wiadomości
history = rag.get_history()

# Uzyskaj kontekst dla innego modelu
context = rag.history.get_context()

# Pobierz w formacie OpenAI API
messages = rag.history.get_messages_for_api()
```

---

## 📋 Wymagania

```
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0  # lub faiss-gpu
numpy>=1.21.0
openai>=1.0.0  # Tylko jeśli chcesz OpenAI API
geopy>=2.2.0
python-dotenv>=0.19.0
```

**Instalacja:**

```bash
pip install sentence-transformers faiss-cpu numpy openai geopy python-dotenv
```

---

## 🎓 Przykłady Promptów do Testowania

Spróbuj tych pytań:

1. **Wyszukiwanie typu kuchni:**

   - "Gdzie mogę zjeść sushi w Łodzi?"
   - "Szukam dobrej włoskiej restauracji"

2. **Wyszukiwanie z kontekstem:**

   - "Restauracja na romantyczną kolację dla dwojga"
   - "Gdzie mogę pić piwo z przyjaciółmi?"

3. **Wyszukiwanie z ograniczeniami:**

   - "Dobra pizza poniżej 40 zł za osobę"
   - "Kawiarnia do pracy ze spokojną atmosferą"

4. **Follow-up pytania:**
   - "Czy to blisko centrum?"
   - "Ile to kosztuje?"
   - "Jak mogę tam dojechać?"

---

## 🐛 Troubleshooting

### Błąd: "ModuleNotFoundError: No module named 'openai'"

```bash
pip install openai
```

### Błąd: "FAISS index is empty"

Upewnij się, że `lodz_restaurants_cafes_embeddings.jsonl` istnieje i zawiera dane.

### Błąd: "API rate limit exceeded"

OpenAI ma limity. Poczekaj chwilę lub użyj `SimpleLLM`.

### Odpowiedzi są generyczne

Sprawdź, czy RAG znajduje restauracje:

```python
results = rag.search_restaurants_rag("twoje pytanie", k=5)
print(results)
```

---

## 📝 Licencja

MIT

---

## 🤝 Kontakt

Pytania? Zgłaszaj na GitHub! 🚀
