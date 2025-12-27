"""
conversational_rag.py

Konwersacyjny system RAG dla rekomendacji restauracji.
Używa modelu PLLuM do naturalnej konwersacji.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from dotenv import load_dotenv
from math import radians, sin, cos, sqrt, atan2, log1p
import os

load_dotenv()

def distance_km(lat1, lon1, lat2, lon2):
    """Oblicza odległość w km między dwoma punktami GPS."""
    if None in [lat1, lon1, lat2, lon2]:
        return float('inf')  # Zwróć nieskończoność, jeśli brakuje koordynatów

    R = 6371  # Promień Ziemi w km
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c



class PLLuMLLM:
    """Klient LLM używający modelu PLLuM przez Hugging Face Inference API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicjalizacja klienta PLLuM.
        
        Args:
            api_key: Klucz API Hugging Face (jeśli None, pobiera z env)
        """
        from huggingface_hub import InferenceClient
        
        self.api_key = api_key or os.getenv('HF_TOKEN')
        if not self.api_key:
            raise ValueError("Brak HF_TOKEN! Ustaw zmienną środowiskową lub przekaż jako argument.")
        
        self.client = InferenceClient(api_key=self.api_key)
        self.model = "CYFRAGOVPL/PLLuM-12B-nc-chat"
        
        print(f"Zainicjalizowano PLLuM model: {self.model}")
    
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """
        Generuje odpowiedź na podstawie historii konwersacji.
        
        Args:
            messages: Lista wiadomości w formacie [{"role": "user/assistant", "content": "..."}]
            max_tokens: Maksymalna długość odpowiedzi
            temperature: Temperatura generowania (0.0-1.0)
        
        Returns:
            Wygenerowana odpowiedź
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Błąd podczas generowania odpowiedzi: {e}")
            return "Przepraszam, wystąpił problem z wygenerowaniem odpowiedzi."


class ConversationalRAG:
    """
    System konwersacyjny RAG dla rekomendacji restauracji.
    Łączy wyszukiwanie semantyczne z naturalną konwersacją.
    """
    
    def __init__(
        self,
        llm_client: PLLuMLLM,
        search_function: Callable,
        max_history: int = 10,
        system_prompt: Optional[str] = None
    ):
        """
        Inicjalizacja systemu RAG.
        
        Args:
            llm_client: Klient LLM (PLLuM)
            search_function: Funkcja wyszukiwania restauracji
            max_history: Maksymalna liczba par w historii
            system_prompt: Opcjonalny własny prompt systemowy
        """
        self.llm = llm_client
        self.search = search_function
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
        self.user_preferences: Dict[str, Any] = {}
        self.user_location: Optional[tuple] = None
        
        # Domyślny prompt systemowy
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """Zwraca domyślny prompt systemowy."""
        return """Jesteś asystentem rekomendacji restauracji i kawiarni w Łodzi. Twoim zadaniem jest:

Twoim zadaniem jest WYŁĄCZNIE:
- interpretować potrzeby użytkownika
- korzystać z dostarczonych wyników wyszukiwania
- prezentować rekomendacje w ściśle określonym formacie
- Analizować wyniki wyszukiwania i przedstawiać je w atrakcyjny sposób
- Wykorzystywać informacje z pola 'Kontekst' (menu, atmosfera, opinie), aby uzasadnić rekomendację
- Pamiętać wcześniejsze preferencje użytkownika w ramach konwersacji

WAŻNE:
- Używaj naturalnego, ciepłego języka polskiego
- Dostosuj rekomendacje do preferencji użytkownika
- Jeśli wyniki wyszukiwania są dostępne, wykorzystaj je w odpowiedzi
- Zawsze przedstawiaj rekomendacje jako listę, podając nazwę miejsca i wynik dopasowania (score). Na przykład: "Oto co znalazłem: 1. Nazwa Miejsca (Dopasowanie: 0.75)". Nie dodawaj opisów, tylko listę.
- Nie wymyślaj informacji - bazuj tylko na danych z wyszukiwania
- Jeśli nie ma wyników, zapytaj o inne preferencje"""
    
    def extract_search_query(self, user_message: str) -> Optional[str]:
        """
        Inteligentne rozszerzanie zapytania (Query Expansion).
        Zamienia intencję użytkownika na optymalne zapytanie dla bazy wektorowej.
        Generuje opis idealnej restauracji na podstawie intencji użytkownika (HyDE).
        """
        extraction_prompt = [
            {
                "role": "system",
                "content": """Twoim zadaniem jest stworzenie hipotetycznego opisu idealnej restauracji pasującej do zapytania użytkownika.
NIE wymyślaj nazwy restauracji. NIE pisz wstępów typu "Oto opis" czy "Jeśli szukasz...".
Skup się wyłącznie na typie kuchni, atmosferze i ofercie.

Wymagany format:
"To miejsce typu: [typ kuchni/lokalu]. Atmosfera jest opisywana jako: [klimat]. W ofercie znajduje się: [ogólne kategorie dań]."

Unikaj wymieniania konkretnych nazw potraw (np. "rosół", "schabowy"), chyba że użytkownik wyraźnie o nie pyta. Używaj ogólnych kategorii (np. "zupy", "dania mięsne", "kuchnia polska").
W ofercie wymień MAKSYMALNIE 3 kategorie i tylko te, które są kluczowe dla intencji użytkownika.

KOLEJNOŚĆ JEST OBOWIĄZKOWA:
1. Typ miejsca
2. Atmosfera
3. Oferta

Przykłady:
User: "Szukam taniej pizzy na randkę w centrum"
Assistant: "To miejsce typu: Pizzeria, Kuchnia włoska. Atmosfera jest opisywana jako: Romantyczna atmosfera, Kameralne. W ofercie znajduje się: Pizza."

User: "Gdzie na szybką kawę i ciastko?"
Assistant: "To miejsce typu: Kawiarnia, Cukiernia. W ofercie znajduje się: Kawa, Dobre desery, Szybka przekąska. Atmosfera jest opisywana jako: Niezobowiązująca atmosfera."

User: "klimatyczne miejsce z jedzeniem azjatyckim w okolicach galerii łódzkiej"
Assistant: "To miejsce typu: Kuchnia azjatycka, Restauracja orientalna. Atmosfera jest opisywana jako: Klimatyczne, Niezobowiązująca atmosfera. W ofercie znajduje się: Dania kuchni azjatyckiej."

Jeśli użytkownik nie szuka jedzenia/lokalu (np. "Cześć"), zwróć "BRAK"."""
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
        
        query = self.llm.generate(extraction_prompt, max_tokens=100, temperature=0.4)
        query = query.strip().replace('"', '')
        
        if query.upper() == "BRAK" or len(query) < 2:
            return None
            
        print(f"DEBUG: Wygenerowany hipotetyczny kontekst: '{query}'")
        return query

    def normalize_location(self, user_message: str) -> Optional[str]:
        """
        Tłumaczy potoczne określenia lokalizacji na nazwy geograficzne zrozumiałe dla geokodera.
        """
        prompt = [
            {
                "role": "system",
                "content": """Jesteś ekstraktorem danych. Twoim zadaniem jest wyciągnięcie nazwy lokalizacji (ulica, dzielnica, punkt orientacyjny, centrum handlowe) z tekstu użytkownika.

Jesteś ekstraktorem lokalizacji w MIEŚCIE ŁÓDŹ.

Zwracaj TYLKO:
- dzielnice Łodzi
- ulice w Łodzi
- znane punkty orientacyjne w Łodzi
- centra handlowe w Łodzi

ZASADY:
1. Zamieniaj potoczne nazwy na oficjalne
2. Zwróć wyłącznie nazwę własną
3. Jeśli lokalizacja jest niejednoznaczna → zwróć BRAK
4. Jeśli lokalizacja NIE JEST w Łodzi → zwróć BRAK

Zasady:
1. Zamieniaj potoczne nazwy na oficjalne (np. "koło manu" -> "Manufaktura", "przy polibudzie" -> "Politechnika Łódzka", "pietryna" -> "Ulica Piotrkowska").
2. Zwróć TYLKO nazwę lokalizacji. Nie pisz pełnych zdań. Nie dodawaj "Lokalizacja to...".
3. Jeśli w wypowiedzi nie ma wskazania lokalizacji, zwróć słowo "BRAK".

Przykłady:
User: "blisko manufaktury" -> Assistant: "Manufaktura"
User: "okolice dworca fabrycznego" -> Assistant: "Dworzec Łódź Fabryczna"
User: "na teofilowie" -> Assistant: "Teofilów"
User: "klimatyczne miejsce z jedzeniem azjatyckim w okolicach galerii łódzkiej" -> Assistant: "Galeria Łódzka"
User: "blisko galerii" -> Assistant: "Galeria Łódzka"
User: "centrum" -> Assistant: "Śródmieście"
User: "zjem pizzę" -> Assistant: "BRAK" """
            },
            {"role": "user", "content": user_message}
        ]
        response = self.llm.generate(prompt, max_tokens=20, temperature=0.1).strip().replace('"', '').replace("Assistant: ", "")
        if response.endswith('.'): response = response[:-1]
        return None if "BRAK" in response.upper() or len(response) < 2 or len(response) > 50 else response

    def extract_cuisine_type(self, user_message: str) -> Optional[str]:
        """
        Ekstrahuje typ kuchni z zapytania użytkownika w celu ścisłego filtrowania.
        """
        prompt = [
            {
                "role": "system",
                "content": """Jesteś narzędziem do ekstrakcji danych. Twoim zadaniem jest zidentyfikować JEDEN, główny typ kuchni lub dania z zapytania użytkownika.

ZASADY:
1. Zwróć DOKŁADNIE JEDNO słowo lub frazę w mianowniku (np. "pizza", "kuchnia azjatycka", "sushi", "burger").
2. Jeśli użytkownik poda kilka typów, wybierz ten, który wydaje się najważniejszy.
3. Jeśli w zapytaniu nie ma mowy o jedzeniu, zwróć DOKŁADNIE słowo "BRAK".
4. NIE odpowiadaj zdaniami. NIE dodawaj wyjaśnień. Zwróć tylko nazwę kuchni lub "BRAK".

Przykłady:
User: "Szukam dobrej pizzy" -> Assistant: "pizza"
User: "Gdzie na dobrego burgera w centrum?" -> Assistant: "burger"
User: "kuchnia amerykańska, jakies burgery" -> Assistant: "amerykańska"
User: "Chcę zjeść coś azjatyckiego, może ramen?" -> Assistant: "azjatycka"
User: "Cześć, jak się masz?" -> Assistant: "BRAK"
User: "Klimatyczne miejsce na kolację" -> Assistant: "BRAK" """
            },
            {"role": "user", "content": user_message}
        ]
        response = self.llm.generate(prompt, max_tokens=10, temperature=0.0).strip().replace('"', '').replace("Assistant: ", "").replace(".", "")
        return None if "BRAK" in response.upper() or len(response) < 2 else response

    def normalize_price(self, user_message: str) -> Optional[str]:
        """
        Tłumaczy intencje cenowe użytkownika na konkretne przedziały liczbowe.
        """
        prompt = [
            {
                "role": "system",
                "content": """Jesteś narzędziem do ekstrakcji danych. Zinterpretuj preferencje cenowe użytkownika i zwróć TYLKO jeden z trzech przedziałów: '0-40', '40-80', '80-1000' lub słowo "BRAK".

ZASADY:
1. Tanie/studenckie/niedrogie/budżetowe/street food/$ -> zwróć '0-40'
2. Średnie/umiarkowane/$$ -> zwróć '40-80'
3. Drogie/ekskluzywne/luksusowe/fine dining/$$$ -> zwróć '80-1000'
4. Jeśli nie ma preferencji cenowej, zwróć DOKŁADNIE słowo "BRAK".
5. NIE odpowiadaj zdaniami. NIE dodawaj wyjaśnień.

Przykłady:
User: "szukam czegoś taniego dla studenta" -> Assistant: "0-40"
User: "dobre burgery w centrum" -> Assistant: "BRAK"
User: "ekskluzywna restauracja na rocznicę" -> Assistant: "80-1000"
User: "kuchnia amerykańska, jakies burgery w okolicach manufaktury" -> Assistant: "BRAK"
User: "coś w średniej cenie" -> Assistant: "40-80" """

            },
            {"role": "user", "content": user_message}
        ]
        response = self.llm.generate(prompt, max_tokens=10, temperature=0.0).strip().replace('"', '').replace("Assistant: ", "").replace(".", "")
        return None if "BRAK" in response.upper() else response
    
    def _format_search_results(self, results: List[Dict]) -> str:
        """Formatuje wyniki wyszukiwania do kontekstu dla LLM."""
        if not results:
            return "Brak wyników wyszukiwania."
        
        formatted = "WYNIKI WYSZUKIWANIA:\n\n"
        for i, r in enumerate(results[:5], 1):
            formatted += f"{i}. {r['name']}\n"
            formatted += f"   Typ: {r['type']}\n"
            formatted += f"   Adres: {r['address']}\n"
            
            if r.get('google_rating'):
                formatted += f"   Ocena: {r['google_rating']}/5.0 ({r.get('google_reviews_total', 0)} opinii)\n"
            
            if r.get('coords'):
                formatted += f"   Współrzędne: {r['coords']}\n"
            
            if r.get('context'):
                formatted += f"   Kontekst/Opis: {r['context'][:400]}...\n"
            
            score = r.get('final_score', r.get('semantic_score', 0.0))
            formatted += f"   Dopasowanie: {score:.2f}\n\n"
        return formatted
    
    def _prepare_messages(self, user_message: str, k: int = 5):
        """
        Przygotowuje listę wiadomości dla LLM.
        
        Args:
            user_message: Wiadomość od użytkownika
            k: Liczba wyników do wyszukania
        
        Returns:
            Lista wiadomości gotowa dla LLM
        """
        # 1. Sprawdź czy potrzebne jest wyszukiwanie
        search_query = self.extract_search_query(user_message)
        
        search_results = None
        if search_query:
            location_info = f"(Lokalizacja: {self.user_location})" if self.user_location else ""
            print(f"Wyszukuję: '{search_query}' {location_info}...")
            try:
                search_results = self.search(
                    search_query, k=k, user_location=self.user_location
                )
            except Exception as e:
                print(f"Błąd wyszukiwania: {e}")
        
        # 2. Przygotuj kontekst dla LLM
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Dodaj historię (ograniczoną)
        history_start = max(0, len(self.conversation_history) - self.max_history * 2)
        messages.extend(self.conversation_history[history_start:])
        
        # 3. Dodaj aktualną wiadomość użytkownika
        user_content = user_message
        
        # Jeśli są wyniki wyszukiwania, dodaj je do kontekstu
        if search_results:
            results_context = self._format_search_results(search_results)
            user_content = f"{user_message}\n\n{results_context}"
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    def generate_response(self, user_message: str, k: int = 5) -> str:
        """
        Generuje odpowiedź na wiadomość użytkownika.
        
        Args:
            user_message: Wiadomość od użytkownika
            k: Liczba wyników do wyszukania
        
        Returns:
            Odpowiedź asystenta
        """
        # Przygotuj wiadomości
        messages = self._prepare_messages(user_message, k)
        
        # Wygeneruj odpowiedź
        response = self.llm.generate(messages, max_tokens=500, temperature=0.7)
        
        # Zaktualizuj historię
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def clear_history(self):
        """Czyści historię konwersacji."""
        self.conversation_history = []
        self.user_preferences = {}
        print("Historia konwersacji wyczyszczona.")
    
    def reset_history(self):
        """Alias for clear_history."""
        self.clear_history()
    
    def export_conversation(self, filepath: str):
        """Eksportuje konwersację do pliku JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "history": self.conversation_history,
            "preferences": self.user_preferences
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Konwersacja zapisana do: {filepath}")

    def set_user_location(self, location: Optional[tuple]):
        """Ustawia lub resetuje lokalizację użytkownika."""
        self.user_location = location
        print(f"INFO: Ustawiono lokalizację użytkownika na: {location}")
    
    def load_conversation(self, filepath: str):
        """Wczytuje konwersację z pliku JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.conversation_history = data.get("history", [])
        self.user_preferences = data.get("preferences", {})
        
        print(f"Konwersacja wczytana z: {filepath}")

    def summarize_recommendations(self, results: List[Dict], user_query: str) -> str:
        """
        Generuje podsumowanie rekomendacji przy użyciu LLM, wyjaśniając dlaczego pasują.
        """
        if not results:
            return "Nie znaleziono pasujących miejsc."

        context_text = self._format_search_results(results)
        
        prompt = [
            {"role": "system", "content": "Jesteś ekspertem kulinarnym. Twoim zadaniem jest krótko podsumować znalezione restauracje i doradzić użytkownikowi, którą wybrać, bazując na ich opisach (polu Kontekst). Odnoś się do konkretnych cech miejsc."},
            {"role": "user", "content": f"Użytkownik szukał: '{user_query}'.\n\nOto znalezione miejsca:\n{context_text}\n\nNapisz krótkie podsumowanie i rekomendację dla użytkownika."}
        ]
        
        print("Generowanie podsumowania przez PLLuM...")
        response = self.llm.generate(prompt, max_tokens=400, temperature=0.7)
        return response

def _is_open_now(opening_hours: Optional[Dict[str, str]]) -> bool:
    """Sprawdza, czy miejsce jest otwarte w danym momencie."""
    if not opening_hours or not isinstance(opening_hours, dict):
        return True  # Zakładamy, że otwarte, jeśli brak danych, by nie odrzucać

    now = datetime.now()
    # Python's weekday(): Monday is 0 and Sunday is 6
    day_map = {
        0: "poniedziałek", 1: "wtorek", 2: "środa", 3: "czwartek",
        4: "piątek", 5: "sobota", 6: "niedziela"
    }
    day_name_pl = day_map[now.weekday()]

    # Znajdź godziny dla dzisiejszego dnia
    hours_today_str = opening_hours.get(day_name_pl) or opening_hours.get(day_name_pl.capitalize())

    if not hours_today_str or hours_today_str.lower() in ['zamknięte', 'closed']:
        return False

    if 'całą dobę' in hours_today_str.lower() or '24 hours' in hours_today_str.lower():
        return True

    # Przetwarzanie przedziałów czasowych, np. "12:00–22:00"
    time_ranges = re.findall(r'(\d{1,2}:\d{2})[–-](\d{1,2}:\d{2})', hours_today_str)

    if not time_ranges:
        return True  # Zakładamy, że otwarte, jeśli nie uda się sparsować

    current_time = now.time()

    for open_str, close_str in time_ranges:
        try:
            open_time = datetime.strptime(open_str, "%H:%M").time()
            close_time = datetime.strptime(close_str, "%H:%M").time()

            if (open_time <= close_time and open_time <= current_time <= close_time) or \
               (open_time > close_time and (current_time >= open_time or current_time <= close_time)):
                return True
        except ValueError:
            continue  # Ignoruj błędnie sformatowane przedziały
    return False

def create_rag_system(
    embeddings_file: str = "output_files/lodz_restaurants_cafes_embeddings_mean.jsonl",
    pooling_type: str | None = None,
    embedding_model_name: str = "sdadas/mmlw-retrieval-roberta-large" # sdadas/stella-pl-retrieval
):
    """
    Tworzy kompletny system RAG z wczytanymi embeddingami.

    Args:
        embeddings_file: Ścieżka do pliku z embeddingami
        pooling_type: "cls" lub "mean", jeśli None – wykrywa automatycznie
        embedding_model_name: Nazwa modelu embeddingów z Hugging Face.

    Returns:
        Tuple (ConversationalRAG, search_function)
    """
    import numpy as np
    import faiss
    import json
    from embedding_model import ModelMeanPooling

    print("Ładowanie modelu i embeddingów...")

    # 1. Wybór strategii poolingu
    if pooling_type is None:
        pooling_strategy = "cls" if "cls" in embeddings_file else "mean"
    else:
        pooling_strategy = pooling_type

    print(f"INFO: Inicjalizuję model wykonawczy | pooling='{pooling_strategy}'")

    # 2. Wczytanie embeddingów z pliku
    records = []
    embeddings = []

    with open(embeddings_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)
            embeddings.append(np.array(rec["embedding"], dtype="float32"))

    embeddings = np.vstack(embeddings)
    n_samples, embedding_dim = embeddings.shape

    print(f"Załadowano {n_samples} embeddingów o wymiarze {embedding_dim}")

    # 3. Inicjalizacja modelu do enkodowania zapytań
    model_wrapper = ModelMeanPooling(
        embedding_model_name,
        word_embedding_dimension=embedding_dim,
        pooling_strategy=pooling_strategy
    )
    print("Pooling strategy:", model_wrapper.pooling_strategy)
    model_dim = model_wrapper.model.get_sentence_embedding_dimension()

    if model_dim != embedding_dim:
        print(f"UWAGA: Wymiar modelu ({model_dim}) nie zgadza się z wymiarem embeddingów ({embedding_dim})")

    # 4. Normalizacja embeddingów i indeks FAISS
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    print(f"Indeks gotowy! Liczba restauracji: {index.ntotal}")

    query_prefix = "zapytanie: "

    # 5. Obiekt wektorowy z metodą wyszukiwania
    class SimpleVectorStore:
        def similarity_search_with_score(self, query: str, k: int = 5):
            full_query = query_prefix + query
            q_emb = model_wrapper.encode(full_query, normalize=True)
            q_emb = np.array(q_emb, dtype="float32").reshape(1, -1)

            # Bezpieczny check wymiaru
            assert q_emb.shape[1] == embedding_dim, (
                f"Embedding mismatch: query={q_emb.shape[1]} index={embedding_dim}"
            )

            scores, idxs = index.search(q_emb, k)

            class Document:
                def __init__(self, page_content, metadata):
                    self.page_content = page_content
                    self.metadata = metadata

            results = []
            for score, idx in zip(scores[0], idxs[0]):
                rec = records[idx]
                doc = Document(page_content=rec.get("name"), metadata=rec)
                results.append((doc, float(score)))
            return results

    vector_store = SimpleVectorStore()

    # 6. Funkcja wyszukiwania i re-rankingu dla agenta
    def search(
        query: str,
        k: int = 5,
        user_location: Optional[tuple] = None,
        price_preference: Optional[str] = None,
        cuisine_filter: Optional[str] = None
    ):
        """
        Wyszukuje, deduplikuje i re-rankuje restauracje, łącząc podobieństwo
        łącząc podobieństwo semantyczne z ocenami, popularnością i odległością.

        Args:
            query (str): Zapytanie użytkownika.
            k (int): Liczba wyników do zwrócenia.
            user_location (tuple, optional): Krotka (lat, lon) lokalizacji użytkownika.
            price_preference (str, optional): Preferowany przedział cenowy do filtrowania.
            cuisine_filter (str, optional): Typ kuchni do filtrowania (np. "azjatycka").
        """
        # Krok 1: Wyszukaj semantycznie większą pulę kandydatów
        # Jeśli mamy filtr kuchni, pobieramy znacznie więcej kandydatów, aby mieć co filtrować
        # (np. 100 zamiast 25 dla k=5), bo azjatyckie mogą być dalej na liście semantycznej
        initial_k = k * 20 if cuisine_filter else k * 5
        docs_with_scores = vector_store.similarity_search_with_score(query, k=initial_k)

        # Krok 2: Odrzuć duplikaty i przygotuj listę do dalszego przetwarzania
        unique_results = {}
        for doc, score in docs_with_scores:
            rec = doc.metadata
            name = rec.get("name")
            if name not in unique_results:
                key_words_data = rec.get("key_words", {})
                
                # Pobieranie danych z korzenia rekordu (priorytet) lub z key_words (fallback)
                rating = rec.get("google_rating") or key_words_data.get("google_rating")
                reviews = rec.get("google_reviews_total") or key_words_data.get("google_reviews_total")
                price_range = rec.get("google_price_range") or key_words_data.get("google_price_range")
                opening_hours = rec.get("opening_hours") or key_words_data.get("opening_hours")

                unique_results[name] = {
                    "semantic_score": score,
                    "name": name,
                    "type": key_words_data.get("types") or rec.get("types") or [],
                    "address": key_words_data.get("address"),
                    "coords": rec.get("Współrzędne"),
                    "google_rating": rating,
                    "google_reviews_total": reviews,
                    "google_price_range": price_range,
                    "opening_hours": opening_hours,
                    "context": rec.get("context") or key_words_data.get("context"),
                    "distance_km": float('inf'),
                    "final_score": 0.0
                }

        processed_results = list(unique_results.values())

        # Krok 2a: Filtrowanie po typie kuchni (jeśli podano) - PRIORYTETOWE
        if cuisine_filter:
            filtered_by_cuisine = []
            filter_lower = cuisine_filter.lower()
            # Uproszczenie filtra (np. "kuchnia azjatycka" -> "azjatycka")
            keywords = filter_lower.replace("kuchnia", "").replace("jedzenie", "").strip().split()
            
            for r in processed_results:
                types_list = r.get("type") or []
                types = [t.lower() for t in types_list]
                context = r.get("context", "").lower()
                name = r.get("name", "").lower()
                
                # Sprawdź czy którykolwiek keyword pasuje do typów, kontekstu LUB NAZWY
                match = False
                for kw in keywords:
                    if len(kw) > 2 and (any(kw in t for t in types) or kw in context or kw in name):
                        match = True
                        break
                if match:
                    filtered_by_cuisine.append(r)
            
            if filtered_by_cuisine:
                print(f"INFO: Przefiltrowano wyniki po kuchni '{cuisine_filter}'. Pozostało: {len(filtered_by_cuisine)}")
                processed_results = filtered_by_cuisine
            else:
                print(f"INFO: Nie znaleziono wyników dla kuchni '{cuisine_filter}'. Pokazuję wyniki semantyczne.")

        # Krok 2b: Filtrowanie po cenie (jeśli podano)
        if price_preference:
            filtered_results = []
            
            def parse_price_range(text):
                if not text: return None
                text_str = str(text).lower()
                
                # 1. Obsługa słów kluczowych
                if any(w in text_str for w in ['tanie', 'tanio', 'tani', 'niedrogie', 'budżetowe', 'ekonomiczne']):
                    return (0, 40)
                if any(w in text_str for w in ['średnie', 'średnio', 'umiarkowane', 'przystępne']):
                    return (40, 80)
                if any(w in text_str for w in ['drogie', 'drogo', 'drogi', 'ekskluzywne', 'luksusowe']):
                    return (80, 1000)
                
                # 2. Obsługa symboli $
                if '$' in text_str:
                    count = text_str.count('$')
                    if count == 1: return (0, 40)
                    if count == 2: return (40, 80)
                    if count >= 3: return (80, 1000)

                # 3. Obsługa liczb (np. "20-40")
                nums = [int(n) for n in re.findall(r'\d+', text_str)]
                if not nums: return None
                if len(nums) == 1: return (nums[0], nums[0])
                return (min(nums), max(nums))

            user_price_range = parse_price_range(price_preference)
            if user_price_range:
                print(f"DEBUG: Zinterpretowano preferencję ceny '{price_preference}' jako zakres: {user_price_range}")

            for r in processed_results:
                price = r.get("google_price_range")
                match = False
                
                # 1. Try numeric range matching
                if user_price_range:
                    place_price_range = parse_price_range(price)
                    if place_price_range:
                        # Check overlap: max(start1, start2) <= min(end1, end2)
                        if max(user_price_range[0], place_price_range[0]) <= min(user_price_range[1], place_price_range[1]):
                            match = True
                
                # 2. Fallback to string matching
                if not match and price and price_preference.lower() in str(price).lower():
                    match = True
                
                if match:
                    filtered_results.append(r)
            
            if filtered_results:
                processed_results = filtered_results
                print(f"INFO: Przefiltrowano wyniki po cenie '{price_preference}'. Pozostało: {len(processed_results)}")
            else:
                print(f"INFO: Brak wyników dla ceny '{price_preference}'. Ignoruję filtr ceny.")

        # Krok 3: Wzbogać dane i przygotuj do re-rankingu
        max_dist = 0.0
        # Użyj 1.0 jako minimum, aby uniknąć dzielenia przez zero
        max_reviews_log = 1.0
        
        # Oblicz odległości i znajdź wartości do normalizacji
        for result in processed_results:
            if user_location and result.get("coords"):
                try:
                    lat, lon = map(float, result["coords"].split(","))
                    dist = distance_km(user_location[0], user_location[1], lat, lon)
                    result["distance_km"] = dist
                    if dist != float('inf'): max_dist = max(max_dist, dist)
                except (ValueError, TypeError):
                    pass
            
            reviews = result.get("google_reviews_total") or 0
            max_reviews_log = max(max_reviews_log, log1p(reviews))

        # Krok 4: Re-Ranking - obliczanie złożonego wyniku
        # Zmieniono wagi: priorytet dla semantyki (0.85), mniejszy wpływ popularności (reviews 0.02)
        # Zapobiega to sytuacji, gdzie popularna pierogarnia wyskakuje na hasło "sushi".
        weights = {"semantic": 0.86, "rating": 0.07, "reviews": 0.02, "distance": 0.05}

        for result in processed_results:
            # Normalizacja oceny (1-5 -> 0-1)
            rating = result.get("google_rating")
            score_rating = (rating - 1) / 4.0

            # Normalizacja logarytmiczna liczby opinii
            reviews = result.get("google_reviews_total")
            score_reviews = log1p(reviews) / max_reviews_log if max_reviews_log > 0 else 0

            # Normalizacja odległości (odwrócona)
            score_dist = 0.0
            if user_location and max_dist > 0 and result["distance_km"] != float('inf'):
                score_dist = 1.0 - (result["distance_km"] / max_dist)

            score_semantic = result.get("semantic_score", 0.0)

            # Obliczenie wyniku końcowego
            result["final_score"] = (weights["semantic"] * score_semantic +
                                     weights["rating"] * score_rating +
                                     weights["reviews"] * score_reviews +
                                     weights["distance"] * score_dist)

        # Krok 5: Sortowanie i zwrócenie najlepszych wyników
        processed_results.sort(key=lambda x: x["final_score"], reverse=True)
        return processed_results[:k]

    # 7. Funkcja do filtrowania otwartych miejsc
    def filter_open_places(results: List[Dict]) -> List[Dict]:
        """
        Filtruje listę wyników, zwracając tylko te miejsca, które są aktualnie otwarte.
        
        Args:
            results: Lista słowników z wynikami (zwrócona przez funkcję `search`).
        
        Returns:
            Przefiltrowana lista wyników.
        """
        return [r for r in results if _is_open_now(r.get("opening_hours"))]

    # 8. Inicjalizacja LLM
    llm = PLLuMLLM()

    # 9. Stworzenie systemu RAG
    rag = ConversationalRAG(
        llm_client=llm,
        search_function=search,
        max_history=10
    )

    # Dołączenie vector_store, aby był dostępny w testach
    rag.vectorstore = vector_store

    return rag, search, filter_open_places
