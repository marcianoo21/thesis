import os
import sys
import csv
import time
from dotenv import load_dotenv

# Dodajemy katalog główny projektu do ścieżki
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import PLLuMLLM, ConversationalRAG, LocationService

# --- LISTA 50 TESTOWYCH ZAPYTAŃ (ŁÓDŹ - SLANG I NAZWY POTOCZNE) ---
TEST_QUERIES = [
    "Gdzie zjem dobrą pizzę na pietrynie?",
    "Szukam kawy koło manu",
    "Jakiś tani bar przy polibudzie",
    "Restauracja na offie",
    "Coś do jedzenia blisko kaliskiego",
    "Obiad w okolicach fabrycznego",
    "Gdzie na piwo na lumumbowie?",
    "Szukam jedzenia przy stajni jednorożców",
    "Kebab na górniaku",
    "Lody na księżym młynie",
    "Kolacja w monopolis",
    "Coś wegańskiego blisko galerii",
    "Burger na retkini",
    "Pizza na widzewie",
    "Jedzenie na teofilowie",
    "Sushi na bałutach",
    "Gdzie zjeść na dąbrowie?",
    "Kawiarnia przy placu wolności",
    "Restauracja blisko pasażu róży",
    "Coś dobrego koło atlas areny",
    "Obiad po spacerze na zdrowiu",
    "Jedzenie blisko fali",
    "Restauracja przy ec1",
    "Gdzie na randkę na rudzie?",
    "Coś przy stawach jana",
    "Obiad w porcie",
    "Jedzenie na radogoszczu",
    "Pizza na chojnach",
    "Bar mleczny na śródmieściu",
    "Coś ekskluzywnego w grand hotelu",
    "Jedzenie koło beczek grohmana",
    "Kawa przy białej fabryce",
    "Lunch blisko sądu na placu dąbrowskiego",
    "Coś przy Piotrkowskiej 217",
    "Jedzenie na julianowie",
    "Restauracja na złotnie",
    "Gdzie zjeść na stokach?",
    "Coś blisko dworca północnego",
    "Jedzenie przy rondzie solidarności",
    "Pizza na olechowie",
    "Obiad przy szpitalu kopernika",
    "Coś przy kampusie uł",
    "Jedzenie na nowym centrum łodzi",
    "Restauracja blisko parku śledzia",
    "Kawa na woonerfie",
    "Coś przy katedrze",
    "Jedzenie na rynku bałuckim",
    "Gdzie na zapiekankę na polesiu?",
    "Restauracja blisko teatru wielkiego",
    "Coś przy muzeum włókiennictwa"
]

def main():
    """
    Automatyczny tester warstwy lokalizacji.
    Przetwarza listę zapytań i zapisuje wyniki do CSV.
    """
    load_dotenv()
    
    if not os.getenv("HF_TOKEN"):
        print("BŁĄD: Brak HF_TOKEN w pliku .env.")
        return

    print("--- Inicjalizacja Systemu ---")
    try:
        llm = PLLuMLLM()
        location_service = LocationService()
        # Dummy search function, bo testujemy tylko normalizację
        rag = ConversationalRAG(llm_client=llm, search_function=lambda x: [])
    except Exception as e:
        print(f"Błąd inicjalizacji: {e}")
        return

    output_file = "location_test_results.csv"
    results = []

    print(f"\nRozpoczynam testowanie {len(TEST_QUERIES)} zapytań...")
    print(f"Wyniki będą zapisywane do: {output_file}\n")

    start_time = time.time()

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"[{i}/{len(TEST_QUERIES)}] Przetwarzanie: '{query}'")
        
        detected_name = None
        method = "None"
        
        # 1. Próba LLM (Normalizacja)
        try:
            # Używamy analyze_user_intent (tak jak w run_pipeline.py) dla spójności z główną aplikacją
            rag.clear_history()
            analysis = rag.analyze_user_intent(query)
            detected_name = analysis.get("location")
            
            if detected_name:
                method = "LLM (Intent Analysis)"
        except Exception as e:
            print(f"  ! Błąd LLM: {e}")

        # 2. Fallback spaCy (NER)
        if not detected_name:
            try:
                detected_name = location_service.extract_location_name(query)
                if detected_name:
                    method = "spaCy (NER)"
            except Exception as e:
                print(f"  ! Błąd spaCy: {e}")

        # 3. Geokodowanie (weryfikacja współrzędnych)
        coords = None
        if detected_name:
            coords = location_service.geocode(detected_name)

        # Logowanie wyniku w konsoli
        status_icon = "✅" if detected_name else "❌"
        coords_str = str(coords) if coords else "BRAK"
        print(f"   {status_icon} Wynik: {detected_name} (Metoda: {method}) | Coords: {coords_str}")

        # Dodanie do listy wyników
        results.append({
            "Original_Query": query,
            "Detected_Location_Formal": detected_name if detected_name else "BRAK",
            "Detection_Method": method,
            "Coordinates": coords_str
        })

    # Zapis do CSV
    try:
        with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["Original_Query", "Detected_Location_Formal", "Detection_Method", "Coordinates"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in results:
                writer.writerow(row)
        
        duration = time.time() - start_time
        print(f"\n--- Zakończono testy w {duration:.2f}s ---")
        print(f"Plik CSV gotowy: {os.path.abspath(output_file)}")
        
    except Exception as e:
        print(f"\nBłąd podczas zapisu CSV: {e}")

if __name__ == "__main__":
    main()
