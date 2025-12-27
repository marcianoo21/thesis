"""
chat_interface.py

Interaktywny interfejs do konwersacyjnego systemu rekomendacji.
Używa PLLuM do naturalnej konwersacji po polsku.
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from config import get_config, list_profiles
from conversational_rag import create_rag_system
from location_service import LocationService


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
        print("Inicjalizacja systemu RAG i serwisu lokalizacji...")
        location_service = LocationService()
        rag_chain, search_and_rank, filter_open = create_rag_system(
            embeddings_file=args.embedding_file,
            # Możesz tu nadpisać inne parametry, np. model embeddingu
        )
        # Ustawienie promptu z profilu
        rag_chain.system_prompt = config.system_prompt
        
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
                    
                    # Zapytaj, czy zapisać konwersację
                    save = input("Czy zapisać konwersację? (t/n): ").strip().lower()
                    if save in ["t", "y", "tak", "yes"]:
                        filename = f"conversation_{args.profile}_{len(rag_chain.conversation_history)//2}_messages.json"
                        rag_chain.export_conversation(filename)
                    
                    break
                
                elif user_input.lower() in ["clear", "reset", "wyczyść"]:
                    rag_chain.clear_history()
                    print("Historia konwersacji wyczyszczona!\n")
                    continue
                
                elif user_input.lower() in ["save", "zapisz"]:
                    filename = input("Nazwa pliku (Enter = domyślna): ").strip()
                    if not filename:
                        filename = f"conversation_{args.profile}_{len(rag_chain.conversation_history)//2}_messages.json"
                    rag_chain.export_conversation(filename)
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

                # --- NOWA LOGIKA PIPELINE'U ---
                # 1. Wykryj i ustaw lokalizację na podstawie zapytania
                coords = location_service.get_location_from_query(user_input)
                if coords:
                    rag_chain.set_user_location(coords)
                    print(f"(INFO: Wykryto lokalizację, wyniki będą sortowane względem {coords})")

                # 2. Wygeneruj odpowiedź za pomocą pełnego systemu RAG
                print("\nAsystent: ", end="", flush=True)
                response = rag_chain.generate_response(
                    user_input, k=config.top_k
                )
                # --- KONIEC NOWEJ LOGIKI ---

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