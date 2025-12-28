import argparse
import sys
import re
from dotenv import load_dotenv

from conversational_rag import create_rag_system
from location_service import LocationService


def main():
    """
    Główny skrypt do uruchamiania pełnego pipeline'u rekomendacji.
    1. Analizuje zapytanie w poszukiwaniu lokalizacji.
    2. Wyszukuje semantycznie pasujące miejsca.
    3. Stosuje re-ranking uwzględniający ocenę, popularność i odległość.
    4. Wyświetla ostateczne, posortowane wyniki.
    """
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Uruchom pipeline rekomendacji restauracji.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--embedding-file",
        type=str,
        default="output_files/lodz_restaurants_cafes_embeddings_cls_words.jsonl",
        help="Ścieżka do pliku z embeddingami, który zawiera pełne metadane."
    )
    parser.add_argument("-k", type=int, default=5, help="Liczba wyników do zwrócenia.")
    args = parser.parse_args()

    print("Inicjalizacja systemu...")
    try:
        location_service = LocationService()
        rag_chain, search_and_rank, _filter_open = create_rag_system(
            embeddings_file=args.embedding_file
        )
    except Exception as e:
        print(f"\nBłąd inicjalizacji: {e}")
        print("Upewnij się, że masz zainstalowane wszystkie zależności (pip install -r requirements.txt) i poprawnie skonfigurowane pliki.")
        sys.exit(1)

    while True:
        print("\n--- Preferencje Użytkownika (wpisz 'exit' aby zakończyć) ---")
        
        # 1. Zapytanie główne (Input 1)
        user_input = input("Co chciałbyś zjeść? (np. 'dobra pizza'): ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Do widzenia!")
            break
            
        if not user_input:
            print("Musisz podać zapytanie!")
            continue

        # --- NOWA LOGIKA: JEDNO ZAPYTANIE ANALITYCZNE ---
        print("Analizuję zapytanie (LLM)...")
        analysis = rag_chain.analyze_user_intent(user_input)
        
        if not analysis:
            print("⚠️  UWAGA: Nie udało się uzyskać analizy od LLM (np. limit API lub błąd sieci).")
            print("   System przechodzi w tryb awaryjny: Wyszukiwanie lokalne (RoBERTa) na podstawie surowego tekstu.")

        # 1. Lokalizacja - Wielostopniowa detekcja (zgodnie z prośbą o debugowanie)
        user_location = None

        # Krok A: LLM Normalizacja
        detected_location_llm = analysis.get("location")
        
        if detected_location_llm:
            print(f"INFO: LLM wykrył lokalizację: '{detected_location_llm}'")
            # Próba geokodowania wyniku LLM
            user_location = location_service.geocode(detected_location_llm)

            # Jeśli geokodowanie wyniku LLM się nie uda, spróbuj przepuścić przez spaCy
            if not user_location:
                print(f"INFO: Geokodowanie LLM nieudane. Przekazuję wynik LLM do spaCy...")
                spacy_from_llm = location_service.extract_location_name(detected_location_llm)
                if spacy_from_llm:
                    print(f"INFO: spaCy (z wyniku LLM) wykryło: '{spacy_from_llm}'")
                    user_location = location_service.geocode(spacy_from_llm)

        # Krok B: Fallback - spaCy na oryginalnym zapytaniu (jeśli ścieżka LLM zawiodła)
        if not user_location:
            print("INFO: Próba bezpośredniej ekstrakcji lokalizacji z zapytania (spaCy)...")
            spacy_direct = location_service.extract_location_name(user_input)
            if spacy_direct:
                print(f"INFO: spaCy (z zapytania) wykryło: '{spacy_direct}'")
                user_location = location_service.geocode(spacy_direct)

        if user_location:
            print(f"INFO: Sukces! Ustalono współrzędne: {user_location}")

        # Krok C: Jeśli nadal brak lokalizacji - dopytujemy
        if not user_location:
            location_input = input("Gdzie szukać? (Lokalizacja, np. 'Manufaktura' lub Enter by pominąć): ").strip()
            if location_input:
                # Normalizujemy również input od użytkownika, na wszelki wypadek
                normalized_input = rag_chain.normalize_location(location_input) or location_input
                if normalized_input:
                    user_location = location_service.geocode(normalized_input)

        # 2. HyDE (Otoczka zapytania)
        expanded_query = analysis.get("search_query")
        if expanded_query:
            print(f"\n=== [HyDE] Wygenerowany kontekst (Hipotetyczny Dokument) ===")
            print(f"{expanded_query}")
            print(f"==========================================================\n")
        else:
            expanded_query = user_input

        # 3. Typ kuchni i Cena (Ekstrakcja)
        cuisine_type = analysis.get("cuisine")
        if cuisine_type:
            print(f"INFO: Wykryto preferencję kuchni: '{cuisine_type}'")

        price_preference = analysis.get("price")
        
        if price_preference:
            print(f"INFO: Model wykrył preferencję cenową: '{price_preference}'")
        else:
            price_input = input("Jaki przedział cenowy? (np. '20-40', '$$', 'tanio' lub Enter by pominąć): ").strip()
            if price_input:
                price_preference = rag_chain.normalize_price(price_input) or price_input

        # 2. Wyszukaj i re-rankuj wyniki
        print(f"\nWyszukiwanie i ranking dla k={args.k}...")
        try:
            ranked_results = search_and_rank(
                expanded_query, # Używamy rozszerzonego zapytania!
                k=args.k, 
                user_location=user_location,
                price_preference=price_preference,
                cuisine_filter=cuisine_type
            )
        except Exception as e:
            print(f"\nBłąd podczas wyszukiwania: {e}")
            print("\nWAŻNE: Upewnij się, że plik z embeddingami jest aktualny i zawiera wszystkie potrzebne metadane (oceny, współrzędne, etc.).")
            continue
        # 3. Wyświetl wyniki
        if not ranked_results:
            print("\nNie znaleziono żadnych pasujących miejsc.")
            continue

        print("\n--- Najlepsze rekomendacje ---")
        for i, r in enumerate(ranked_results, 1):
            rating_info = f"Ocena: {r.get('google_rating', 'Brak')}/5 ({r.get('google_reviews_total', 0)} opinii)" if r.get('google_rating') else "Brak oceny"
            price_info = f"Cena na osobę: {r.get('google_price_range', 'Brak danych')}zł"

            print(f"\n{i}. {r['name']} (Wynik: {r['final_score']:.3f})")
            
            # Wyświetlanie typu kuchni
            types = r.get('type', [])
            if types:
                print(f"   Typ: {', '.join(types) if isinstance(types, list) else str(types)}")
            
            print(f"   Adres: {r.get('address', 'Brak adresu')}")
            print(f"   {rating_info} | {price_info}")
            if user_location and r.get('distance_km') is not None and r['distance_km'] != float('inf'):
                print(f"   Odległość: {r['distance_km']:.2f} km")
            
            # Wyświetlanie kontekstu
            context_preview = r.get('context') or 'Brak kontekstu'
            print(f"   Kontekst (fragment): {context_preview[:200]}...")
        
        # --- NOWOŚĆ: Podsumowanie od PLLuM ---
        # print("\n--- Komentarz Asystenta (PLLuM) ---")
        # summary = rag_chain.summarize_recommendations(ranked_results[:3], user_input)
        # print(summary)
        # print("-" * 40)

if __name__ == "__main__":
    main()