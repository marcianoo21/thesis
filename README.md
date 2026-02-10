## 🎓 Konwersacyjny System Rekomendacji Restauracji w Łodzi — README (praca dyplomowa)

Ten projekt jest kompletnym systemem **RAG (Retrieval-Augmented Generation)** do rekomendacji restauracji, kawiarni i innych lokali gastronomicznych w **Łodzi**.  
System łączy:
- **PLLuM-12B** (model językowy, Hugging Face Inference API) – naturalna konwersacja po polsku,
- **własne embeddingi** (RoBERTa / STELLA) przechowywane w plikach `*.jsonl`,
- **FAISS** – szybkie wektorowe wyszukiwanie,
- **warstwę lokalizacji** (spaCy + Nominatim) – rozumienie potocznych nazw miejsc i dzielnic w Łodzi,
- **logikę rankingową** – łączenie podobieństwa semantycznego z ocenami, popularnością i odległością,
- **interfejs tekstowy oraz prosty frontend webowy (Flask + HTML)**.

Celem README jest umożliwienie **osobie z zewnątrz**:
- zainstalowania środowiska,
- zainicjowania danych / embeddingów,
- uruchomienia systemu w trybie linii komend i przez przeglądarkę,
- uruchomienia skryptów testowych i ewaluacyjnych.

---

## 📦 Struktura projektu (wysoki poziom)

Najważniejsze katalogi i pliki:

```text
.
├── app.py                     # Backend Flask (API do interfejsu webowego)
├── chat_ui.html               # Frontend webowy (czat w przeglądarce)
├── src/
│   ├── __init__.py            # Ujawnia API pakietu src (create_rag_system, LocationService, ModelMeanPooling, itp.)
│   ├── conversational_rag.py  # Główny silnik RAG (PLLuM + FAISS + ranking)
│   ├── config.py              # Profile konfiguracyjne (style odpowiedzi, top_k, itp.)
│   ├── embedding_model.py     # Wrapper na SentenceTransformers (ModelMeanPooling)
│   └── location_service.py    # Warstwa lokalizacji (spaCy + Nominatim)
│
├── scripts/
│   ├── run_pipeline.py        # Konsolowy pipeline rekomendacji (1 zapytanie → lista miejsc)
│   ├── chat_interface.py      # Interaktywny czat w terminalu (konwersacyjny asystent)
│   ├── data_gathering.py      # (opcjonalnie) pobieranie / przygotowanie danych źródłowych
│   ├── extract_keywords.py    # ekstrakcja słów kluczowych
│   ├── key_words_and_context_creation.py
│   ├── context_creation_only_words.py
│   ├── chunk_divide.py        # dzielenie dłuższych opisów na fragmenty
│   ├── search_restaurants.py  # narzędzia wyszukiwawcze
│   └── ...                    # inne skrypty analityczne i pomocnicze
│
├── embedding_creation/
│   ├── create_embeddings_mean.py
│   ├── create_embeddings_cls.py
│   ├── create_embeddings_mean_words.py
│   ├── create_embeddings_cls_words.py
│   ├── create_embeddings_mean_stella.py
│   ├── create_embeddings_cls_stella.py
│   ├── create_embeddings_mean_words_stella.py
│   ├── create_embeddings_cls_words_stella.py
│   ├── create_embeddings_cls_words_v2.py
│   └── __init__.py
│
├── output_files/              # Dane wejściowe / pośrednie / embeddingi
│   ├── lodz_restaurants_cafes.csv
│   ├── lodz_restaurants_cafes_with_ratings*.jsonl
│   ├── lodz_restaurants_cafes_with_key_words*.jsonl
│   ├── lodz_restaurants_cafes_chunks.jsonl
│   ├── lodz_restaurants_cafes_emb_input*.jsonl
│   ├── context_from_filtered_keywords.jsonl
│   ├── lodz_restaurants_cafes_embeddings_*.jsonl   # Różne warianty embeddingów
│   └── ...
│
├── tests/
│   ├── test_location_layer.py               # test warstwy lokalizacji
│   ├── run_embedding_tests.py               # porównanie różnych wariantów embeddingów
│   ├── run_embedding_tests_v2_comparison.py # porównanie modelu v1 vs v2
│   ├── run_embedding_tests_stella.py        # testy dla modeli STELLA
│   ├── run_query_expansion_tests.py         # test HyDE / query expansion
│   ├── run_location_tests.py                # test normalizacji lokalizacji (LLM + spaCy)
│   └── evaluate_full_pipeline.py            # ewaluacja pełnego pipeline'u (MRR, hit rate)
│
├── requirements.txt
├── Dockerfile
└── README.md                # (ten plik)
```

---

## ✅ Wymagania wstępne

- **Python**: rekomendowana wersja 3.10+  
- **System**: Linux / macOS / Windows (testowane na Windows 10/11)
- **Konto Hugging Face** z ważnym tokenem API (używany przez `PLLuMLLM`):
  - model: `CYFRAGOVPL/PLLuM-12B-nc-chat`
- Dostęp do Internetu (PLLuM działa przez inference API + Nominatim dla geokodowania).
- Wymagane pakiety Python (instalowane przez `requirements.txt`), m.in.:
  - `huggingface_hub`
  - `sentence-transformers`
  - `faiss-cpu`
  - `spacy`
  - `geopy`
  - `python-dotenv`
  - `flask`, `flask-cors`
  - `scipy`, `numpy`, `pandas`, itp.

Dodatkowo:
- Model spaCy: **`pl_core_news_lg`** (do wykrywania nazw lokalizacji).

---

## 🔧 Instalacja krok po kroku

### 1. Sklonuj repozytorium

W terminalu / PowerShell:

```bash
git clone <URL_twojego_repozytorium>
cd Inżynierka    # lub inna nazwa katalogu z projektem
```

### 2. Utwórz i aktywuj wirtualne środowisko (zalecane)

Przykład (Windows, `venv`):

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Zainstaluj zależności

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Jeśli pojawią się problemy z FAISS, możesz jawnie doinstalować:

```bash
pip install faiss-cpu
```

### 4. Zainstaluj model spaCy `pl_core_news_lg`

```bash
python -m spacy download pl_core_news_lg
```

### 5. Konfiguracja tokenu Hugging Face (`HF_TOKEN`)

1. Załóż konto na `https://huggingface.co/` (jeśli jeszcze nie masz).
2. Wejdź w ustawienia tokenów: `https://huggingface.co/settings/tokens`.
3. Utwórz token z uprawnieniami **read**.
4. W katalogu projektu utwórz plik `.env` (jeśli go nie ma) i wpisz:

```env
HF_TOKEN=hf_...tutaj_twój_token...
```

Plik `.env` **nie powinien być commitowany** do repozytorium (jest w `.gitignore`).

---

## 🧱 Dane i embeddingi – jak to jest zorganizowane

W folderze `output_files/` znajdują się kolejne etapy przetwarzania danych:

- **Dane źródłowe**:  
  - `lodz_restaurants_cafes.csv` – surowa tabela restauracji/kawiarni.

- **Wzbogacanie danych** (oceny, słowa kluczowe, opisy):
  - `lodz_restaurants_cafes_with_ratings*.jsonl`
  - `lodz_restaurants_cafes_with_key_words*.jsonl`
  - `lodz_restaurants_cafes_chunks.jsonl` – podział dłuższych opisów na fragmenty.

- **Przygotowanie tekstów do embeddingów**:
  - `lodz_restaurants_cafes_emb_input*.jsonl` – teksty kontekstowe (pełne opisy) dla embeddingów,
  - `context_from_filtered_keywords.jsonl` – teksty zbudowane tylko z wybranych słów kluczowych.

- **Gotowe embeddingi** (różne warianty modelu i poolingu):
  - `lodz_restaurants_cafes_embeddings_mean.jsonl`
  - `lodz_restaurants_cafes_embeddings_cls.jsonl`
  - `lodz_restaurants_cafes_embeddings_mean_words.jsonl`
  - `lodz_restaurants_cafes_embeddings_cls_words.jsonl`
  - `lodz_restaurants_cafes_embeddings_mean_stella.jsonl`
  - `lodz_restaurants_cafes_embeddings_cls_stella.jsonl`
  - `lodz_restaurants_cafes_embeddings_mean_words_stella.jsonl`
  - `lodz_restaurants_cafes_embeddings_cls_words_stella.jsonl`
  - `lodz_restaurants_cafes_embeddings_cls_words_v2.jsonl`
  - itp.

W praktyce **do działania systemu nie musisz generować embeddingów od zera** – w repozytorium są już gotowe pliki.  
Domyślnie backend (`app.py`) korzysta z:

- `output_files/lodz_restaurants_cafes_embeddings_cls_words.jsonl`

Jeśli chcesz samodzielnie odtworzyć pipeline (np. do celów naukowych), zobacz następny punkt.

---

## 🔬 (Opcjonalnie) Pełny pipeline tworzenia embeddingów

Ten krok jest **opcjonalny** – potrzebny tylko wtedy, gdy chcesz:
- zbudować embeddingi z nowego źródła danych,
- przetestować inne warianty (np. inny model, inny pooling).

### 1. Przygotowanie danych wejściowych

Źródłowym plikiem jest zwykle `lodz_restaurants_cafes.csv` (kolumny z nazwą, opisem, adresem, itp.).  
Następnie skrypty w `scripts/` wykonują m.in.:

- ekstrakcję słów kluczowych (`extract_keywords.py`, `key_words_and_context_creation.py`),
- budowę kontekstów (`context_creation_only_words.py`),
- dzielenie tekstów na fragmenty (`chunk_divide.py`).

W rezultacie powstają pliki `*_with_key_words*.jsonl` oraz `context_from_filtered_keywords.jsonl` / `lodz_restaurants_cafes_emb_input*.jsonl`.

### 2. Generowanie embeddingów – przykłady

W katalogu `embedding_creation/` masz zestaw skryptów. Uruchamiasz je **z katalogu głównego projektu**, np.:

```bash
python -m embedding_creation.create_embeddings_mean
python -m embedding_creation.create_embeddings_cls
python -m embedding_creation.create_embeddings_mean_words
python -m embedding_creation.create_embeddings_cls_words
```

Odpowiadają one za:
- **`create_embeddings_mean.py`** – pełne konteksty, pooling *mean*,
- **`create_embeddings_cls.py`** – pełne konteksty, pooling *CLS*,
- **`create_embeddings_mean_words.py`** – tylko słowa kluczowe, pooling *mean*,
- **`create_embeddings_cls_words.py`** – tylko słowa kluczowe, pooling *CLS*.

Analogicznie, warianty `*_stella.py` oraz `*_v2.py` korzystają z modelu **STELLA** lub nowszej wersji modelu embeddingowego.

Każdy skrypt:
- wczytuje metadane z `lodz_restaurants_cafes_with_key_words.jsonl`,
- wczytuje tekst do zakodowania (`context`),
- wywołuje `ModelMeanPooling` z `src/embedding_model.py`,
- zapisuje wynik do `output_files/lodz_restaurants_cafes_embeddings_*.jsonl`.

---

## 🚀 Uruchomienie systemu – tryb webowy (Flask + przeglądarka)

To jest **najważniejszy scenariusz użytkowy** – uruchomienie asystenta rekomendacji w przeglądarce.

### 1. Upewnij się, że masz:

- aktywne środowisko wirtualne,
- zainstalowane zależności (`pip install -r requirements.txt`),
- zainstalowany model spaCy: `pl_core_news_lg`,
- plik `.env` z poprawnym `HF_TOKEN`,
- istnieje plik embeddingów wskazywany w `app.py` (domyślnie):

```python
embedding_file = "output_files/lodz_restaurants_cafes_embeddings_cls_words.jsonl"
```

### 2. Uruchom backend Flask

Z katalogu głównego projektu:

```bash
python app.py
```

Co się dzieje w `app.py`:
- wczytywany jest `.env`,
- tworzony jest `LocationService` (spaCy + Nominatim),
- wywoływana jest funkcja `create_rag_system(...)` z `src/conversational_rag.py`,
- budowany jest globalny obiekt `rag_chain` (silnik RAG),
- startuje serwer Flask (domyślnie na porcie `5000`).

Jeśli wszystko jest poprawnie skonfigurowane, w konsoli zobaczysz komunikaty typu:

```text
--- Inicjalizacja serwera backendu ---
Ładowanie modelu i embeddingów...
...
--- System gotowy do pracy ---
 * Running on http://127.0.0.1:5000
```

### 3. Otwórz interfejs webowy

Plik `chat_ui.html` jest statycznym frontendem, który łączy się z API `app.py`.  
Możesz go otworzyć na dwa sposoby:

- **Bezpośrednio z dysku** (double-click / „Otwórz w przeglądarce”):
  - strona będzie wysyłać żądania `POST` na `http://localhost:5000/chat`.

- Lub hostując go przez serwer (np. inny prosty backend) – ale w tym projekcie wystarcza zwykłe otwarcie pliku .html.

Po otwarciu zobaczysz pole czatu, suwak ceny, itp.  
Backend endpointy w `app.py`:
- `GET /` – serwuje `chat_ui.html` (jeśli otwierasz przez Flask),
- `POST /chat` – przyjmuje JSON `{ message: "...", price_level: 0..3 }` i zwraca HTML z listą rekomendacji.

---

## 💬 Uruchomienie systemu – tryb czat w terminalu

Jeśli wolisz interfejs konsolowy, użyj `scripts/chat_interface.py`.

### 1. Uruchom

Z katalogu głównego:

```bash
python scripts/chat_interface.py --profile default \
  --embedding-file output_files/lodz_restaurants_cafes_embeddings_mean.jsonl
```

Parametry:
- `--profile` – wybór profilu z `src/config.py` (`default`, `fast`, `detailed`, `friendly`, `professional`, `local`, `budget`, `foodie`),
- `--embedding-file` – ścieżka do pliku z embeddingami.

### 2. Komendy specjalne w czacie

W trakcie rozmowy możesz wpisać:
- `exit`, `quit`, `q` – zakończenie programu,
- `clear`, `reset` – wyczyszczenie historii konwersacji,
- `save` / `zapisz` – zapisanie historii konwersacji do pliku JSON,
- `profile` / `profil` – wyświetlenie aktualnego profilu konfiguracyjnego.

Przykład dialogu:

```text
Ty: Dobra pizza w centrum
Asystent: (lista kilku pizzerii z adresami, oceną, odległością, itp.)

Ty: A coś romantycznego na randkę?
Asystent: (kolejna lista, pamiętająca wcześniejsze preferencje)
```

---

## 🔎 Jednorazowy pipeline rekomendacji (skrypt `run_pipeline.py`)

Jeśli chcesz jednorazowo przetestować pipeline (bez konwersacji, ale z wejściami z klawiatury), użyj:

```bash
python scripts/run_pipeline.py \
  --embedding-file output_files/lodz_restaurants_cafes_embeddings_cls_words.jsonl \
  -k 5
```

Skrypt:
- poprosi Cię o tekst zapytania (np. „tania pizza na widzewie”),
- spróbuje wywnioskować lokalizację (LLM + spaCy),
- ewentualnie dopyta o lokalizację i przedział cenowy,
- wypisze listę rekomendacji w terminalu (z ocenami, odległością, kontekstem).

---

## 🧪 Testy i ewaluacja

W katalogu `tests/` znajdują się skrypty analizujące różne aspekty systemu.

### 1. Test warstwy lokalizacji

```bash
python tests/test_location_layer.py
```

Skrypt:
- uruchamia LLM (`PLLuMLLM`) i `LocationService`,
- przechodzi przez listę testowych zapytań z łódzkim slangiem (np. „kawa koło manu”, „kebab na górniaku”),
- sprawdza, czy system poprawnie normalizuje lokalizacje i znajduje współrzędne,
- zapisuje wyniki do `location_test_results.csv`.

### 2. Ewaluacja embeddingów (różne warianty)

```bash
python tests/run_embedding_tests.py
```

Skrypt:
- testuje kilka plików embeddingów (`mean`, `cls`, `*_words`, itp.),
- dla predefiniowanego zestawu zapytań (`GROUND_TRUTH`) oblicza:
  - Hit Rate@5,
  - MRR,
  - Precision@5,
- zapisuje wyniki do:
  - `embedding_test_results_all_retrieval.txt`,
  - `embedding_metrics_summary_retrieval.csv`.

Analogiczne skrypty:
- `tests/run_embedding_tests_stella.py` – warianty z modelami STELLA,
- `tests/run_embedding_tests_v2_comparison.py` – porównanie roberta v1 vs v2.

### 3. Test query expansion (HyDE)

```bash
python tests/run_query_expansion_tests.py
```

Skrypt porównuje wyniki wyszukiwania dla:
- surowego zapytania użytkownika,
- zapytania rozszerzonego przez LLM (HyDE) na opisy „idealnych” restauracji.

Wyniki zapisuje do `query_expansion_comparison_results.txt`.

### 4. Test detekcji lokalizacji (LLM + spaCy)

```bash
python tests/run_location_tests.py
```

Testowane jest działanie metody `normalize_location` w `ConversationalRAG` oraz `LocationService`.

### 5. Ewaluacja pełnego pipeline'u

```bash
python tests/evaluate_full_pipeline.py
```

Skrypt:
- pobiera kandydatów z FAISS,
- stosuje reranker,
- oblicza metryki dla kolejnych etapów (`bi-encoder`, `reranker`, końcowy algorytm wagowy),
- wypisuje tabelę z wynikami (Hit Rate@5, MRR, Precision@5).

---

## ⚙️ Główne komponenty kodu (skrót)

- **`src/conversational_rag.py`**
  - `PLLuMLLM` – klient Hugging Face Inference API dla PLLuM-12B,
  - `ConversationalRAG` – główna klasa systemu:
    - `analyze_user_intent` – jedna rozmowa z LLM, która wyciąga intencję, lokalizację, typ kuchni, cenę,
    - `extract_search_query` – HyDE / query expansion,
    - `normalize_location`, `extract_cuisine_type`, `normalize_price` – pomocnicze ekstraktory,
    - `generate_response` – łączy wyszukiwanie wektorowe, reranking i generowanie odpowiedzi,
    - `_format_search_results` – zamienia listę wyników na HTML (używane w `app.py`),
    - `_is_open_now` – heurystyczne sprawdzanie, czy lokal jest aktualnie otwarty.
  - `create_rag_system(...)` – główna fabryka systemu:
    - ładuje embeddingi z pliku,
    - tworzy indeks FAISS,
    - ładuje model `ModelMeanPooling`,
    - ładuje reranker (`sdadas/polish-reranker-roberta-v2`),
    - składa to w gotowy `ConversationalRAG` + funkcję `search(...)`.

- **`src/config.py`**
  - zawiera klasy i profile konfiguracyjne (`RAGConfig` + `PROFILES`),
  - każdy profil ma zdefiniowany system prompt (z mocnym ograniczeniem: tylko gastronomia w Łodzi).

- **`src/location_service.py`**
  - `LocationService` – ekstrakcja nazw lokalizacji (spaCy) + geokodowanie (Nominatim),
  - potrafi radzić sobie z fałszywymi pozytywami (np. „włoska” jako kuchnia, a nie lokalizacja).

- **`src/embedding_model.py`**
  - `ModelMeanPooling` – wrapper na SentenceTransformers z wyborem strategii: `mean` lub `cls`.

---

## 🐛 Typowe problemy i rozwiązania

- **Błąd: „Brak HF_TOKEN” lub „HF_TOKEN not set”**
  - Sprawdź plik `.env` w katalogu projektu,
  - Upewnij się, że jest tam linia `HF_TOKEN=...`,
  - Możesz też ustawić zmienną środowiskową systemowo.

- **Błąd: „OSError: [E050] Can't find model 'pl_core_news_lg'”**
  - Uruchom:
    ```bash
    python -m spacy download pl_core_news_lg
    ```

- **Błąd: „FileNotFoundError: ...embeddings_*.jsonl”**
  - Sprawdź, czy wskazany w `app.py` / `scripts/*` plik istnieje w `output_files/`,
  - Jeśli nie istnieje, wygeneruj go wybranym skryptem z `embedding_creation/`.

- **Backend Flask się uruchamia, ale frontend nie widzi odpowiedzi**
  - Upewnij się, że:
    - backend działa na `http://localhost:5000`,
    - przeglądarka nie blokuje żądań CORS (w `app.py` jest `flask_cors.CORS`).

- **Odpowiedzi nie są związane z gastronomią lub mówią o podróżach / innych sprawach**
  - W aktualnej wersji prompt systemowy jest ograniczony **wyłącznie** do gastronomii w Łodzi,
  - jeśli model mimo to „ucieka” w inne tematy, to jest to właściwość modelu PLLuM; w pracy dyplomowej opisuj to jako ograniczenie LLM.

---

## 📄 Licencja

Projekt jest udostępniany na licencji **MIT** (patrz plik `LICENSE`).  

---

## 📚 Kontekst pracy dyplomowej

Ten projekt stanowi część pracy inżynierskiej poświęconej:
- budowie konwersacyjnego systemu rekomendacji gastronomicznych w mieście (Łódź),
- porównaniu różnych strategii embeddingów i modeli,
- analizie wpływu dodatkowych czynników (oceny, liczba opinii, odległość) na ranking,
- testowaniu warstwy lokalizacji i query expansion.
