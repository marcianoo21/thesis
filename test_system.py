"""
test_system.py

Skrypt testowy do weryfikacji systemu RAG.
"""

import os
from dotenv import load_dotenv
from conversational_rag import create_rag_system

def test_search_only():
    """Test samego wyszukiwania bez LLM."""
    print("=" * 60)
    print("TEST 1: Wyszukiwanie semantyczne")
    print("=" * 60)
    print()
    
    try:
        from conversational_rag import create_rag_system
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer
        import json
        
        print("Ładowanie modelu i embeddingów...")
        
        model = SentenceTransformer("sdadas/mmlw-retrieval-roberta-large")
        query_prefix = "zapytanie: "
        
        records = []
        embeddings = []
        
        with open("output_files/lodz_restaurants_cafes_embeddings_mean.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                records.append(rec)
                embeddings.append(rec["embedding"])
        
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        print(f"Załadowano {index.ntotal} restauracji\n")
        
        # Test query
        test_query = "pizzeria"
        print(f"Testuję zapytanie: '{test_query}'\n")
        
        full_query = query_prefix + test_query
        q_emb = model.encode(full_query, normalize_embeddings=True)
        q_emb = q_emb.astype("float32").reshape(1, -1)
        
        scores, idxs = index.search(q_emb, 3)
        
        print("Top 3 wyniki:")
        for i, (score, idx) in enumerate(zip(scores[0], idxs[0]), 1):
            rec = records[idx]
            print(f"{i}. {rec.get('name')} - Score: {score:.3f}")
        
        print("\nTest wyszukiwania zakończony sukcesem!\n")
        
    except Exception as e:
        print(f"Błąd: {e}")
        import traceback
        traceback.print_exc()


def test_llm_only():
    """Test samego LLM bez wyszukiwania."""
    print("=" * 60)
    print("TEST 2: Model PLLuM")
    print("=" * 60)
    print()
    
    load_dotenv()
    
    if not os.getenv("HF_TOKEN"):
        print("Brak HF_TOKEN! Ustaw token w .env")
        return
    
    try:
        from conversational_rag import PLLuMLLM
        
        print("Inicjalizacja PLLuM...")
        llm = PLLuMLLM()
        
        print("Model załadowany!\n")
        
        # Test prosty
        messages = [
            {"role": "user", "content": "Cześć! Co potrafisz?"}
        ]
        
        print("Test generowania odpowiedzi...\n")
        response = llm.generate(messages, max_tokens=100)
        
        print(f"Odpowiedź: {response}\n")
        print("Test LLM zakończony sukcesem!\n")
        
    except Exception as e:
        print(f"Błąd: {e}")
        import traceback
        traceback.print_exc()


def test_full_system():
    """Test pełnego systemu RAG."""
    print("=" * 60)
    print("TEST 3: Pełny system RAG")
    print("=" * 60)
    print()
    
    load_dotenv()
    
    if not os.getenv("HF_TOKEN"):
        print("Brak HF_TOKEN! Ustaw token w .env")
        return
    
    try:
        print("Inicjalizacja systemu RAG...")
        rag, search = create_rag_system()
        
        print("\nSystem gotowy!\n")
        
        # Test konwersacji
        test_messages = [
            # 1. Proste
            "Gdzie zjem dobrego burgera?",
            
            # 2. Złożone
            "Szukam restauracji z kuchnią włoską, ale nie pizzerii, która będzie odpowiednia na spotkanie biznesowe w centrum.",
            
            # 3. Opisowe
            "Chciałbym znaleźć jakieś przytulne, ciche miejsce z dobrą kawą i ciastem, idealne do poczytania książki po południu. Coś z miłą atmosferą, może trochę na uboczu."
        ]
        
        # Resetuj historię przed każdym testem, aby były niezależne
        rag.reset_history()
        print("Historia konwersacji zresetowana.\n")
        
        for msg in test_messages:
            print(f"Ty: {msg}")
            response = rag.generate_response(msg)
            print(f"Asystent: {response}\n")
        
        print("Test pełnego systemu zakończony sukcesem!\n")
        
    except Exception as e:
        print(f"Błąd: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Uruchom wszystkie testy."""
    print("\nURUCHAMIANIE TESTOW SYSTEMU\n")

    # Test 1: Tylko wyszukiwanie
    print("\n--- Uruchamiam Test 1: Tylko wyszukiwanie ---\n")
    test_search_only()
    
    # Test 2: Tylko LLM
    print("\n--- Uruchamiam Test 2: Tylko LLM ---\n")
    test_llm_only()
    
    # Test 3: Pełny system
    print("\n--- Uruchamiam Test 3: Pelny system RAG ---\n")
    test_full_system()
    
    print("\nWszystkie testy zakonczone!")


if __name__ == "__main__":
    main()
