"""
extract_google_ratings.py

Wyciąga TYLKO średnie oceny (rating) i zakres cenowy z Google Maps dla restauracji w Łodzi.
Aktualizuje plik JSONL o nowe pole 'google_rating'.
"""
# W TYM API JEST TEZ PRZEDZIAŁ CENOWY!!!!!!!!!!
from serpapi.google_search import GoogleSearch
import json
import time
from dotenv import load_dotenv
import os

# =============================================================
# KONFIGURACJA
# =============================================================

load_dotenv()


SERPAPI_KEY = os.getenv('SERP_API_KEY')
INPUT_FILE = "output_files/lodz_restaurants_cafes_ready_for_embd.jsonl"
OUTPUT_FILE = "output_files/lodz_restaurants_cafes_with_ratings.jsonl"

# Opóźnienie między requestami (SerpAPI ma limity)
DELAY_SECONDS = 1


# =============================================================
# FUNKCJA: Pobierz data_id z Google Maps
# =============================================================
def get_place_data_id(name, coords):
    """
    Wyszukuje miejsce w Google Maps i zwraca data_id.
    """
    if not coords:
        return None
    
    try:
        lat, lon = map(float, coords.replace(" ", "").split(","))
    except:
        return None
    
    params = {
        "engine": "google_maps",
        "q": name,
        "ll": f"@{lat},{lon},14z",  # Koordynaty + zoom
        "type": "search",
        "hl": "pl",              
        "gl": "pl",
        "api_key": SERPAPI_KEY
    }
    
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if results["local_results"][0]['title'] != name:
            print(f"    Nie znaleziono miejsca: {name}")
            return None
        else:
        # Pobierz pierwszy wynik (najbardziej pasujący)
            if results.get("local_results") and len(results["local_results"]) > 0:
                place = results["local_results"][0]
                data_id = place.get("data_id")
                rating = place.get("rating")  # Średnia ocena
                reviews = place.get("reviews")  # Liczba opinii
                prices = place.get("price") # przedział cenowy
                stripped_price = prices.split()[0].strip() # bez waluty dla czystszego zapisu (wszystko i tak w PLN)
                print("PLACE", place)
                print(f"    Znaleziono: {place.get('title')} - Rating: {rating} ({reviews} opinii) - Przedział cenowy: {prices}")
                
                return {
                    "data_id": data_id,
                    "google_rating": rating,
                    "google_reviews_total": reviews,
                    "google_price_range": stripped_price
                }
            else:
                print(f"    Nie znaleziono miejsca: {name}")
                return None
            
    except Exception as e:
        print(f"    Błąd dla {name}: {e}")
        return None


# =============================================================
# FUNKCJA: Pobierz tylko średnią ocenę (bez szczegółowych reviews)
# =============================================================
def get_google_rating(name, coords):
    """
    Prostsza wersja - pobiera tylko data_id i zwraca średnią ocenę.
    """
    place_data = get_place_data_id(name, coords)
    
    if place_data:
        return {
            "google_rating": place_data.get("google_rating"),
            "google_reviews_total": place_data.get("google_reviews_total"),
            "google_price_range": place_data.get("google_price_range")
        }
    
    return {
        "google_rating": None,
        "google_reviews_total": None,
        "google_price_range": None
    }


# =============================================================
# GŁÓWNA FUNKCJA: Przetwórz wszystkie restauracje
# =============================================================
def main():
    print("Pobieranie ocen Google Maps dla restauracji w Łodzi\n")
    print("="*60)
    
    records_processed = 0
    records_with_rating = 0
    
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            try:
                rec = json.loads(line)
            except:
                continue
            
            osm_id = rec.get("oms_id")
            name = rec.get("name", "Nieznane")
            coords = rec.get("Współrzędne")
            
            print(f"\n[{records_processed + 1}] {name}")
            print(f" Koordynaty: {coords}")
            
            # Pobierz rating
            rating_data = get_google_rating(name, coords)
            
            # Dodaj do rekordu
            # rec["google_rating"] = rating_data["google_rating"]
            # rec["google_reviews_total"] = rating_data["google_reviews_total"]
            
            if rating_data["google_rating"]:
                records_with_rating += 1
                
            # dane do zapisu
            output_rec = {
                "oms_id": osm_id,
                "name": name,
                "Współrzędne": coords,
                "google_rating": rating_data["google_rating"],
                "google_reviews_total": rating_data["google_reviews_total"],
                "google_price_range": rating_data["google_price_range"]
            }
            # Zapis
            fout.write(json.dumps(output_rec, ensure_ascii=False) + "\n")
            
            records_processed += 1
            
            # Rate limiting
            time.sleep(DELAY_SECONDS)
    
    print("\n" + "="*60)
    print(" GOTOWE!")
    print(f" Przetworzono: {records_processed} restauracji")
    print(f" Z ratingiem: {records_with_rating} restauracji")
    print(f" Zapisano do: {OUTPUT_FILE}")


# =============================================================
# URUCHOMIENIE
# =============================================================
if __name__ == "__main__":
    if SERPAPI_KEY == "YOUR_API_KEY":
        print(" BŁĄD: Ustaw swój klucz SerpAPI w zmiennej SERPAPI_KEY")
        print("Pobierz klucz z: https://serpapi.com/")
    else:
        main()
        
        
        
