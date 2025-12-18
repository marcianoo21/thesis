"""
examples.py

Przykłady użycia systemu RAG w różnych scenariuszach.
"""

import os
from dotenv import load_dotenv
from conversational_rag import create_rag_system, PLLuMLLM, ConversationalRAG


def example_1_basic_conversation():
    """Przykład 1: Podstawowa konwersacja."""
    print("\n" + "="*60)
    print("PRZYKŁAD 1: Podstawowa konwersacja")
    print("="*60 + "\n")
    
    load_dotenv()
    rag, _ = create_rag_system()
    
    messages = [
        "Cześć!",
        "Szukam miejsca na romantyczną kolację",
        "Coś z widokiem byłoby idealne",
    ]
    
    for msg in messages:
        print(f"👤 Ty: {msg}")
        response = rag.generate_response(msg)
        print(f"🤖 Asystent: {response}\n")


def example_2_specific_search():
    """Przykład 2: Konkretne wyszukiwanie."""
    print("\n" + "="*60)
    print("PRZYKŁAD 2: Konkretne wyszukiwanie")
    print("="*60 + "\n")
    
    load_dotenv()
    rag, _ = create_rag_system()
    
    query = "najlepsza pizzeria w Łodzi z dobrymi opiniami"
    print(f"👤 Ty: {query}")
    response = rag.generate_response(query)
    print(f"🤖 Asystent: {response}\n")


def example_3_context_aware():
    """Przykład 3: Kontekst w konwersacji."""
    print("\n" + "="*60)
    print("PRZYKŁAD 3: Pamiętanie kontekstu")
    print("="*60 + "\n")
    
    load_dotenv()
    rag, _ = create_rag_system()
    
    messages = [
        "Lubię włoską kuchnię",
        "Ale nie jestem fanem pizzy",
        "Co możesz mi polecić?",
        "A gdzie jest to miejsce?",
    ]
    
    for msg in messages:
        print(f"👤 Ty: {msg}")
        response = rag.generate_response(msg)
        print(f"🤖 Asystent: {response}\n")


def example_4_export_import():
    """Przykład 4: Zapisywanie i wczytywanie konwersacji."""
    print("\n" + "="*60)
    print("PRZYKŁAD 4: Zapis i odczyt konwersacji")
    print("="*60 + "\n")
    
    load_dotenv()
    rag, _ = create_rag_system()
    
    # Prowadź krótką konwersację
    rag.generate_response("Szukam kawiarni")
    rag.generate_response("Z dobrą kawą")
    
    # Zapisz
    filename = "test_conversation.json"
    rag.export_conversation(filename)
    print(f"✅ Konwersacja zapisana do {filename}\n")
    
    # Wyczyść historię
    rag.clear_history()
    print(f"📊 Historia po wyczyszczeniu: {len(rag.conversation_history)} wiadomości\n")
    
    # Wczytaj
    rag.load_conversation(filename)
    print(f"📊 Historia po wczytaniu: {len(rag.conversation_history)} wiadomości\n")
    
    # Kontynuuj konwersację
    response = rag.generate_response("A w centrum?")
    print(f"🤖 Kontynuacja: {response}\n")


def example_5_custom_prompt():
    """Przykład 5: Własny prompt systemowy."""
    print("\n" + "="*60)
    print("PRZYKŁAD 5: Własny prompt systemowy")
    print("="*60 + "\n")
    
    load_dotenv()
    
    # Własny prompt - bardziej zwięzły styl
    custom_prompt = """Jesteś asystentem rekomendacji restauracji w Łodzi.

Zasady:
- Odpowiadaj zwięźle i konkretnie
- Podawaj maksymalnie TOP 3 miejsca
- Zawsze wspominaj oceny Google
- Używaj emoji 🍕🍔🍜 dla typów kuchni

Bądź profesjonalny ale przyjazny."""
    
    # Stwórz system z własnym promptem
    _, search = create_rag_system()
    llm = PLLuMLLM()
    
    rag = ConversationalRAG(
        llm_client=llm,
        search_function=search,
        system_prompt=custom_prompt
    )
    
    query = "polecisz pizzerię?"
    print(f"👤 Ty: {query}")
    response = rag.generate_response(query)
    print(f"🤖 Asystent: {response}\n")


def example_6_direct_llm():
    """Przykład 6: Bezpośrednie użycie LLM bez RAG."""
    print("\n" + "="*60)
    print("PRZYKŁAD 6: Bezpośrednie użycie modelu PLLuM")
    print("="*60 + "\n")
    
    load_dotenv()
    
    llm = PLLuMLLM()
    
    messages = [
        {"role": "system", "content": "Jesteś pomocnym asystentem."},
        {"role": "user", "content": "Jakie są główne atrakcje Łodzi?"}
    ]
    
    response = llm.generate(
        messages,
        max_tokens=200,
        temperature=0.7
    )
    
    print(f"🤖 Odpowiedź: {response}\n")


def example_7_multi_turn_refinement():
    """Przykład 7: Wieloetapowe doprecyzowanie."""
    print("\n" + "="*60)
    print("PRZYKŁAD 7: Doprecyzowanie wymagań")
    print("="*60 + "\n")
    
    load_dotenv()
    rag, _ = create_rag_system()
    
    messages = [
        "Szukam restauracji",
        "Azjatyckiej",
        "Ale bez sushi",
        "W budżecie do 100zł na osobę",
        "I żeby była w centrum",
    ]
    
    for msg in messages:
        print(f"👤 Ty: {msg}")
        response = rag.generate_response(msg)
        print(f"🤖 Asystent: {response}\n")


def example_8_error_handling():
    """Przykład 8: Obsługa błędów."""
    print("\n" + "="*60)
    print("PRZYKŁAD 8: Obsługa błędów")
    print("="*60 + "\n")
    
    # Test bez tokenu
    if "HF_TOKEN" in os.environ:
        backup_token = os.environ["HF_TOKEN"]
        del os.environ["HF_TOKEN"]
    else:
        backup_token = None
    
    try:
        llm = PLLuMLLM()
    except ValueError as e:
        print(f"✅ Poprawnie złapano błąd: {e}\n")
    
    # Przywróć token
    if backup_token:
        os.environ["HF_TOKEN"] = backup_token
    
    # Test z pustym zapytaniem
    load_dotenv()
    rag, _ = create_rag_system()
    
    empty_queries = ["", "   ", "asdfghjkl"]
    
    for query in empty_queries:
        print(f"👤 Ty: '{query}'")
        try:
            response = rag.generate_response(query)
            print(f"🤖 Asystent: {response}\n")
        except Exception as e:
            print(f"❌ Błąd: {e}\n")


def main():
    """Uruchom wszystkie przykłady."""
    examples = [
        ("Podstawowa konwersacja", example_1_basic_conversation),
        ("Konkretne wyszukiwanie", example_2_specific_search),
        ("Pamiętanie kontekstu", example_3_context_aware),
        ("Zapis/odczyt konwersacji", example_4_export_import),
        ("Własny prompt", example_5_custom_prompt),
        ("Bezpośrednie użycie LLM", example_6_direct_llm),
        ("Doprecyzowanie wymagań", example_7_multi_turn_refinement),
        ("Obsługa błędów", example_8_error_handling),
    ]
    
    print("\n🎯 PRZYKŁADY UŻYCIA SYSTEMU RAG\n")
    print("Dostępne przykłady:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    choice = input("\nWybierz przykład (1-8) lub 'all' aby uruchomić wszystkie: ").strip()
    
    if choice.lower() == "all":
        for name, func in examples:
            func()
            input("\nNaciśnij Enter aby kontynuować...")
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                examples[idx][1]()
            else:
                print("❌ Nieprawidłowy wybór!")
        except ValueError:
            print("❌ Nieprawidłowy wybór!")


if __name__ == "__main__":
    main()