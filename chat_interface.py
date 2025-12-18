"""
chat_interface.py

Interaktywny interfejs do konwersacyjnego systemu rekomendacji.
Używa PLLuM do naturalnej konwersacji po polsku.
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from conversational_rag import create_rag_system, PLLuMLLM, ConversationalRAG
from config import get_config, list_profiles


def parse_arguments():
    """Parsuj argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description="Konwersacyjny System Rekomendacji Restauracji - Łódź"
    )
    
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        help="Profil konfiguracyjny (default, fast, detailed, friendly, professional, local, budget, foodie)"
    )
    
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Wyświetl dostępne profile i zakończ"
    )
    
    parser.add_argument(
        "--embedding-file",
        type=str,
        default="output_files/lodz_restaurants_cafes_embeddings_mean.jsonl",
        help="Ścieżka do pliku z embeddingami"
    )
    
    return parser.parse_args()


def print_welcome(profile_name: str):
    """Wyświetla powitalny banner."""
    print("\n" + "=" * 70)
    print("KONWERSACYJNY SYSTEM REKOMENDACJI RESTAURACJI – ŁÓDŹ")
    print("=" * 70)
    print(f"\nProfil: {profile_name.upper()}")
    print("Powered by PLLuM-12B + FAISS + RoBERTa embeddings")
    print()


def print_instructions():
    """Wyświetla instrukcje użytkowania."""
    print("Komendy specjalne:")
    print("   • 'exit', 'quit', 'q' - zakończ konwersację")
    print("   • 'clear', 'reset' - wyczyść historię konwersacji")
    print("   • 'save' - zapisz konwersację do pliku")
    print("   • 'profile' - pokaż aktualny profil")
    print()
    print("-" * 70)
    print()


def main():
    """Główna pętla czatu."""
    # Parsuj argumenty
    args = parse_arguments()
    
    # Jeśli --list-profiles, wyświetl i zakończ
    if args.list_profiles:
        list_profiles()
        return
    
    # Załaduj zmienne środowiskowe
    load_dotenv()
    
    # Sprawdź czy jest token HF
    if not os.getenv("HF_TOKEN"):
        print("BŁĄD: Brak HF_TOKEN w zmiennych środowiskowych!")
        print("   Ustaw token: export HF_TOKEN='twój_token'")
        print("   Lub dodaj do pliku .env: HF_TOKEN=twój_token")
        return
    
    # Załaduj konfigurację
    config = get_config(args.profile)
    
    print_welcome(args.profile)
    
    try:
        # Inicjalizuj system RAG
        print("Inicjalizacja systemu...")
        print(f"   Profil: {args.profile}")
        print(f"   Top-K: {config.top_k}")
        print(f"   Max tokens: {config.max_tokens}")
        print(f"   Temperature: {config.temperature}")
        print()
        
        # Załaduj wyszukiwarkę
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer
        import json
        
        model = SentenceTransformer(config.embedding_model)
        query_prefix = "zapytanie: "
        
        records = []
        embeddings = []
        
        with open(config.embedding_file, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                records.append(rec)
                embeddings.append(rec["embedding"])
        
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        def search(query, k=None):
            if k is None:
                k = config.top_k
            full_query = query_prefix + query
            q_emb = model.encode(full_query, normalize_embeddings=True)
            q_emb = q_emb.astype("float32").reshape(1, -1)
            
            scores, idxs = index.search(q_emb, k)
            
            results = []
            for score, idx in zip(scores[0], idxs[0]):
                rec = records[idx]
                results.append({
                    "score": float(score),
                    "name": rec.get("name"),
                    "type": rec.get("type"),
                    "address": rec.get("Adres", "brak"),
                    "coords": rec.get("Współrzędne"),
                    "google_rating": rec.get("google_rating"),
                    "google_reviews_total": rec.get("google_reviews_total"),
                })
            
            results.sort(
                key=lambda x: (
                    -(x["google_rating"] or 0),
                    -(x["google_reviews_total"] or 0)
                )
            )
            
            return results
        
        # Stwórz system RAG
        llm = PLLuMLLM()
        rag = ConversationalRAG(
            llm_client=llm,
            search_function=search,
            max_history=config.max_history,
            system_prompt=config.system_prompt
        )
        
        print(f"Indeks wczytany! Liczba restauracji: {index.ntotal}\n")
        
        print("Witaj! Jestem Twoim asystentem do rekomendacji restauracji.")
        print("Mogę Ci pomóc w wyborze miejsca do jedzenia w Łodzi.\n")
        
        print_instructions()
        
        # Główna pętla konwersacji
        while True:
            try:
                # Pobierz wiadomość od użytkownika
                user_input = input("Ty: ").strip()
                
                if not user_input:
                    continue
                
                # Obsługa komend specjalnych
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("\nDziękuję za rozmowę! Do widzenia! 👋\n")
                    
                    # Zapytaj czy zapisać konwersację
                    save = input("Czy zapisać konwersację? (t/n): ").strip().lower()
                    if save in ["t", "y", "tak", "yes"]:
                        filename = f"conversation_{args.profile}_{len(rag.conversation_history)//2}_messages.json"
                        rag.export_conversation(filename)
                    
                    break
                
                elif user_input.lower() in ["clear", "reset", "wyczyść"]:
                    rag.clear_history()
                    print("Historia konwersacji wyczyszczona!\n")
                    continue
                
                elif user_input.lower() in ["save", "zapisz"]:
                    filename = input("Nazwa pliku (Enter = domyślna): ").strip()
                    if not filename:
                        filename = f"conversation_{args.profile}_{len(rag.conversation_history)//2}_messages.json"
                    rag.export_conversation(filename)
                    print()
                    continue
                
                elif user_input.lower() in ["profile", "profil"]:
                    print(f"\nAktualny profil: {args.profile.upper()}")
                    print(f"   Top-K: {config.top_k}")
                    print(f"   Max history: {config.max_history}")
                    print(f"   Max tokens: {config.max_tokens}")
                    print(f"   Temperature: {config.temperature}")
                    print()
                    continue
                
                # Wygeneruj odpowiedź
                print("\nAsystent: ", end="", flush=True)
                
                # Użyj konfiguracji dla generowania
                response = rag.llm.generate(
                    rag._prepare_messages(user_input),
                    max_tokens=config.max_tokens,
                    temperature=config.temperature
                )
                
                # Zaktualizuj historię
                rag.conversation_history.append({"role": "user", "content": user_input})
                rag.conversation_history.append({"role": "assistant", "content": response})
                
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\nPrzerwano przez użytkownika. Do widzenia!\n")
                break
                
            except Exception as e:
                print(f"\nBłąd: {e}\n")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"\nBłąd inicjalizacji: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()