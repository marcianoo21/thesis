"""
conversational_rag.py

Konwersacyjny system RAG dla rekomendacji restauracji.
Używa modelu PLLuM do naturalnej konwersacji.
"""

import json
import os
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()



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
        
        # Domyślny prompt systemowy
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """Zwraca domyślny prompt systemowy."""
        return """Jesteś asystentem rekomendacji restauracji w Łodzi. Twoim zadaniem jest:

1. Prowadzić naturalną, przyjazną konwersację z użytkownikiem
2. Zadawać pytania, aby lepiej zrozumieć preferencje (np. typ kuchni, budżet, lokalizacja)
3. Analizować wyniki wyszukiwania i przedstawiać je w atrakcyjny sposób
4. Pamiętać wcześniejsze preferencje użytkownika w ramach konwersacji
5. Sugerować pytania uzupełniające, jeśli informacje są niepełne

WAŻNE:
- Używaj naturalnego, ciepłego języka polskiego
- Dostosuj rekomendacje do preferencji użytkownika
- Jeśli wyniki wyszukiwania są dostępne, wykorzystaj je w odpowiedzi
- Zawsze przedstawiaj rekomendacje jako listę, podając nazwę miejsca i wynik dopasowania (score). Na przykład: "Oto co znalazłem: 1. Nazwa Miejsca (Dopasowanie: 0.75)". Nie dodawaj opisów, tylko listę.
- Nie wymyślaj informacji - bazuj tylko na danych z wyszukiwania
- Jeśli nie ma wyników, zapytaj o inne preferencje"""
    
    def _extract_search_query(self, user_message: str) -> Optional[str]:
        """
        Ekstrakcja zapytania wyszukiwania z wiadomości użytkownika.
        Używa LLM do inteligentnej ekstrakcji.
        """
        extraction_prompt = [
            {
                "role": "system",
                "content": """Twoim zadaniem jest wyekstrahowanie kluczowych słów do wyszukiwania restauracji.
                
Przykłady:
- "Szukam dobrej pizzerii" -> "pizzeria"
- "Chciałbym coś włoskiego niedaleko centrum" -> "włoska kuchnia centrum"
- "Gdzie mogę zjeść śniadanie?" -> "śniadanie"
- "Cześć" -> BRAK
- "Dziękuję" -> BRAK

Odpowiedz TYLKO kluczowymi słowami lub słowem "BRAK" jeśli nie ma intencji wyszukiwania."""
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
        
        query = self.llm.generate(extraction_prompt, max_tokens=50, temperature=0.3)
        query = query.strip()
        
        if query.upper() == "BRAK" or len(query) < 2:
            return None
        return query
    
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
            
            formatted += f"   Dopasowanie: {r['score']:.2f}\n\n"
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
        search_query = self._extract_search_query(user_message)
        
        search_results = None
        if search_query:
            print(f"Wyszukuję: '{search_query}'...")
            try:
                search_results = self.search(search_query, k=k)
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
    
    def load_conversation(self, filepath: str):
        """Wczytuje konwersację z pliku JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.conversation_history = data.get("history", [])
        self.user_preferences = data.get("preferences", {})
        
        print(f"Konwersacja wczytana z: {filepath}")

def create_rag_system(
    embeddings_file: str = "output_files/lodz_restaurants_cafes_embeddings_mean.jsonl",
    pooling_type: str | None = None
):
    """
    Tworzy kompletny system RAG z wczytanymi embeddingami.

    Args:
        embeddings_file: Ścieżka do pliku z embeddingami
        pooling_type: "cls" lub "mean", jeśli None – wykrywa automatycznie

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
        "sdadas/mmlw-retrieval-roberta-large",
        pooling_strategy=pooling_strategy
    )
    model = model_wrapper.model
    model_dim = model.get_sentence_embedding_dimension()

    if model_dim != embedding_dim:
        print(f"UWAGA: Wymiar modelu ({model_dim}) nie zgadza się z wymiarem embeddingów ({embedding_dim})")
        # Można wybrać dopasowanie modelu do embeddingów lub odrzucić
        # W tym przypadku użyjemy wymiaru z embeddingów
        embedding_dim = embedding_dim

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
            q_emb = model.encode(full_query, normalize_embeddings=True)
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

    # 6. Funkcja wyszukiwania dla agenta konwersacyjnego
    def search(query, k=5):
        docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
        results = []
        for doc, score in docs_with_scores:
            rec = doc.metadata
            results.append({
                "score": score,
                "name": rec.get("name"),
                "type": rec.get("type"),
                "address": rec.get("Adres", "brak"),
                "coords": rec.get("Współrzędne"),
                "google_rating": rec.get("google_rating"),
                "google_reviews_total": rec.get("google_reviews_total"),
            })
        results.sort(
            key=lambda x: (-(x["google_rating"] or 0), -(x["google_reviews_total"] or 0))
        )
        return results

    # 7. Inicjalizacja LLM
    llm = PLLuMLLM()

    # 8. Stworzenie systemu RAG
    rag = ConversationalRAG(
        llm_client=llm,
        search_function=search,
        max_history=10
    )

    # Dołączenie vector_store, aby był dostępny w testach
    rag.vectorstore = vector_store

    return rag, search
