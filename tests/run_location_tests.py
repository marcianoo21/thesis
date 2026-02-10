import os
import sys
import time
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import create_rag_system, LocationService

def run_location_tests():
    """
    Testuje skuteczność wykrywania lokalizacji w zapytaniach.
    Symuluje logikę z run_pipeline.py (LLM -> Geocode oraz Fallback spaCy).
    """
    load_dotenv()
    
    print("Inicjalizacja usług...")
    # Inicjalizujemy LocationService
    location_service = LocationService()
    
    # Inicjalizujemy RAG tylko dla funkcji normalize_location (LLM)
    # Plik embeddingów nie ma znaczenia dla tego testu, ale musi istnieć, by funkcja zadziałała
    rag_chain, _, _ = create_rag_system(
        embeddings_file="output_files/lodz_restaurants_cafes_embeddings_cls.jsonl"
    )
    
    # Lista zapytań testowych (skupiona wyłącznie na lokalizacjach)
    queries = [
        # --- Centrum / Śródmieście ---
        "Najlepsza kawiarnia w centrum z miejscem do pracy.",
        "Smash burger w centrum",
        "Jedzenie na dowóz centrum",
        
        # --- Piotrkowska / Off ---
        "Tanie jedzenie dla studenta, najlepiej blisko Piotrkowskiej",
        "Restauracja na Off Piotrkowska",
        "Kawa na Pietrynie",
        
        # --- Manufaktura / Centra Handlowe ---
        "Gdzie zjeść w Manufakturze?",
        "Coś dobrego koło Manu",
        "Restauracja blisko Galerii Łódzkiej",
        "Obiad w Porcie Łódź",

        # --- Dworce ---
        "Restauracja blisko dworca Łódź Fabryczna",
        "Gdzie zjem blisko dworca Kaliskiego?",
        "Coś do jedzenia przy Fabrycznym",

        # --- Dzielnice / Osiedla ---
        "Klimatyczne miejsce na Teofilowie",
        "Jedzenie na Widzewie",
        "Pizza na Retkini",
        "Szukam lokalu na Bałutach",
        "Coś na Górnej",

        # --- Punkty Orientacyjne / Uczelnie ---
        "Obiad przy Polibudzie",
        "Lunch w okolicach Placu Wolności",
        "Restauracja w Monopolis",
        "Coś na Księżym Młynie",
        "Blisko Lumumby",
        

    ]
    
    print(f"\n=== TESTOWANIE DETEKCJI LOKALIZACJI ({len(queries)} zapytań) ===\n")
    
    found_count = 0
    
    for i, query in enumerate(queries, 1):
        # 1. LLM Normalizacja
        llm_loc = rag_chain.normalize_location(query)
        
        # 2. Próba geokodowania wyniku LLM
        coords = None
        if llm_loc:
            coords = location_service.geocode(llm_loc)
        
        if coords:
            found_count += 1
            print(f"[{i}] '{query}'\n    ✅ ZNALEZIONO: '{llm_loc}' -> {coords}")
            
    print(f"\nPodsumowanie: Wykryto lokalizację w {found_count}/{len(queries)} zapytań.")

if __name__ == "__main__":
    run_location_tests()
