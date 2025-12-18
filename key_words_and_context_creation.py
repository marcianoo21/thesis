# W TYM API JEST TEZ PRZEDZIAŁ CENOWY!!!!!!!!!!
from serpapi.google_search import GoogleSearch
import json
import time
from dotenv import load_dotenv
import os


load_dotenv()

SERPAPI_KEY = os.getenv('SERP_API_KEY_4')
INPUT_FILE = "output_files/lodz_restaurants_cafes_ready_for_embd.jsonl"
OUTPUT_FILE = "output_files/lodz_restaurants_cafes_with_key_words2.jsonl"
EMBED_FILE = "output_files/lodz_restaurants_cafes_emb_input2.jsonl"
RATINGS_FILE = "output_files/lodz_restaurants_cafes_with_ratings2.jsonl"

# Opóźnienie między requestami (SerpAPI ma limity)
DELAY_SECONDS = 1
 
# PRIORYTETY INFORMACJI (im mniejsza liczba, tym ważniejsze)
PRIORITY = {
    "types": 1,
    "atmosphere": 4,
    "popular_for": 8, 
    "offerings": 2,
    "amenities": 6, 
    "service_options": 9,
    "specials": 3,
    "crowd": 7,
    "children": 11,
    "accessibility": 5,
    "parking": 10,
}

# maksymalna długość pola context (znaki)
MAX_CONTEXT_LEN = 800

def parse_coordinates(coords):
    if not coords:
        return None, None
    try:
        lat, lon = map(float, coords.replace(" ", "").split(","))
        return lat, lon
    except Exception:
        return None, None


def flatten_extensions(ext):
    if not isinstance(ext, list):
        return {}
    return {k: v for d in ext if isinstance(d, dict) for k, v in d.items()}


def build_place_description(name, place_data):
    def join_list(values):
        shortened_values = values[:8]
        return ", ".join(values) if isinstance(values, list) and values else None

    types_txt = join_list(place_data.get("types"))
    # service_options_txt = join_list(place_data.get("service_options"))
    specials_txt = join_list(place_data.get("specials"))
    popular_for_txt = join_list(place_data.get("popular_for"))
    accessibility_txt = join_list(place_data.get("accessibility"))
    offerings_txt = join_list(place_data.get("offerings"))
    amenities_txt = join_list(place_data.get("amenities"))
    atmosphere_txt = join_list(place_data.get("atmosphere"))
    crowd_txt = join_list(place_data.get("crowd"))
    # children_txt = join_list(place_data.get("children"))
    # parking_txt = join_list(place_data.get("parking"))


    description_parts = []

    base = f"Obiekt o nazwie {name}"
    if types_txt:
        base += f" to miejsce typu: {types_txt}"
    description_parts.append(base + ".")

    

    if atmosphere_txt:
        description_parts.append(f"Atmosfera jest opisywana jako: {atmosphere_txt}.")

    if popular_for_txt:
        description_parts.append(f"Jest popularne szczególnie na: {popular_for_txt}.")

    if amenities_txt:
        description_parts.append(f"Dostępne udogodnienia to: {amenities_txt}.")
        
    # if service_options_txt:
    #     description_parts.append(f"Opcje usług obejmują: {service_options_txt}.")

    if offerings_txt:
        description_parts.append(f"W ofercie znajduje się: {offerings_txt}.")

    if specials_txt:
        description_parts.append(f"Specjalne cechy miejsca: {specials_txt}.")

    if crowd_txt:
        description_parts.append(f"Typowa grupa odwiedzających: {crowd_txt}.")

    if accessibility_txt:
        description_parts.append(f"Miejsce oferuje udogodnienia dostępności: {accessibility_txt}.")
        
    # if parking_txt:
    #     description_parts.append(f"Parking: {parking_txt}.")

    # # Dla dzieci
    # if children_txt:
    #     description_parts.append(f"Udogodnienia dla dzieci: {children_txt}.")

    

    return " ".join(description_parts)


def build_place_description_with_priority(name, place_data):
    """Builds a short context string ordering fragments by PRIORITY and truncating to MAX_CONTEXT_LEN."""
    def join_list(values):
        shortened_values = values[:8]
        return ", ".join(shortened_values) if isinstance(shortened_values, list) and shortened_values else None
    
    types_txt = join_list(place_data.get("types"))
    specials_txt = join_list(place_data.get("specials"))
    popular_for_txt = join_list(place_data.get("popular_for"))
    accessibility_txt = join_list(place_data.get("accessibility"))
    offerings_txt = join_list(place_data.get("offerings"))
    amenities_txt = join_list(place_data.get("amenities"))
    atmosphere_txt = join_list(place_data.get("atmosphere"))
    crowd_txt = join_list(place_data.get("crowd"))

    components = {}
    base = f"Miejsce o nazwie {name}"
    if types_txt:
        base += f" to miejsce typu: {types_txt}"
    components['types'] = base + "."

    if atmosphere_txt:
        components['atmosphere'] = f"Atmosfera jest opisywana jako: {atmosphere_txt}."
    if popular_for_txt:
        components['popular_for'] = f"Jest popularne szczególnie na: {popular_for_txt}."
    if offerings_txt:
        components['offerings'] = f"W ofercie znajduje się: {offerings_txt}."
    if amenities_txt:
        components['amenities'] = f"Dostępne udogodnienia to: {amenities_txt}."
    if place_data.get('service_options'):
        so = join_list(place_data.get('service_options') or [])
        if so:
            components['service_options'] = f"Opcje serwisu: {so}."
    if specials_txt:
        components['specials'] = f"Specjalne cechy miejsca: {specials_txt}."
    if crowd_txt:
        components['crowd'] = f"Typowa grupa odwiedzających: {crowd_txt}."
    if place_data.get('children'):
        ch = join_list(place_data.get('children') or [])
        if ch:
            components['children'] = f"Dla dzieci: {ch}."
    if accessibility_txt:
        components['accessibility'] = f"Miejsce oferuje udogodnienia dostępności: {accessibility_txt}."
    if place_data.get('parking'):
        pk = join_list(place_data.get('parking') or [])
        if pk:
            components['parking'] = f"Parking: {pk}."

    ordered_keys = sorted(components.keys(), key=lambda k: PRIORITY.get(k, 999))
    ordered_parts = [components[k] for k in ordered_keys if components.get(k)]
    text = ' '.join(ordered_parts)
    if MAX_CONTEXT_LEN and len(text) > MAX_CONTEXT_LEN:
        text = text[:MAX_CONTEXT_LEN-3].rstrip() + '...'
    return text



def names_match(search_name: str, result_title: str) -> bool:
    """Elastyczne porównanie nazw — sprawdza czy są wystarczająco podobne."""
    s = search_name.lower().strip()
    r = result_title.lower().strip()
    # dokładne dopasowanie
    if s == r:
        return True
    # jedno jest zawarte w drugim (bez znaków punkuacji)
    s_clean = ''.join(c for c in s if c.isalnum())
    r_clean = ''.join(c for c in r if c.isalnum())
    if s_clean in r_clean or r_clean in s_clean:
        return True
    # pierwsza część się zgadza (np. "Pod Kapliczkami" vs "Pod Kapliczkami - Cafe")
    if s.split()[0] == r.split()[0] and len(s.split()[0]) > 2:
        return True
    return False


def get_ratings_data(name, coords, place_result=None):
    """Pobiera oceny, liczbę opinii i przedział cenowy z Google Maps."""
    lat, lon = parse_coordinates(coords)
    if lat is None or lon is None:
        return {
            "google_rating": None,
            "google_reviews_total": None,
            "google_price_range": None
        }

    # Jeśli miejsce już znaleźliśmy wcześniej w get_key_words, użyj tego
    if place_result:
        place = place_result
    else:
        params = {
            "engine": "google_maps",
            "q": name,
            "ll": f"@{lat},{lon},14z",
            "type": "search",
            "hl": "pl",
            "gl": "pl",
            "api_key": SERPAPI_KEY
        }
        
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            local = results.get("local_results", [])
            if not local:
                return {
                    "google_rating": None,
                    "google_reviews_total": None,
                    "google_price_range": None
                }
            place = local[0]
        except Exception:
            return {
                "google_rating": None,
                "google_reviews_total": None,
                "google_price_range": None
            }
    
    rating = place.get("rating")
    reviews = place.get("reviews")
    price = place.get("price", "")
    
    # Wyciągnij tylko pierwszą część (walutę/przedział)
    stripped_price = None
    if price:
        try:
            stripped_price = price.split()[0].strip()
        except:
            stripped_price = price
    
    return {
        "google_rating": rating,
        "google_reviews_total": reviews,
        "google_price_range": stripped_price
    }


def get_key_words(name, coords):
    lat, lon = parse_coordinates(coords)
    if lat is None or lon is None:
        return None

    params = {
        "engine": "google_maps",
        "q": name,
        "ll": f"@{lat},{lon},14z",
        "type": "search",
        "hl": "pl",
        "gl": "pl",
        "api_key": SERPAPI_KEY
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()

        local = results.get("local_results", [])
        if not local:
            print(f"Nie znaleziono miejsca: {name}")
            return None

        place = local[0]

        # elastyczne dopasowanie nazwy
        if not names_match(name, place.get("title", "")):
            print(f"Nie znaleziono dokładnego dopasowania: {name} (znaleziono: {place.get('title')})")
            return None

        types_of_goods = place.get("types", [])
        address = place.get("address", "")
        phone = place.get("phone", "")
        operating_hours = place.get("operating_hours", {})

        # zmiana listy słowników na jeden słownik - łatwiejsze w obsłudze
        flat_ext = flatten_extensions(place.get("extensions", []))

        place_dict = {
        "types": types_of_goods,
        # "service_options": flat_ext.get("service_options", []),
        "specials": flat_ext.get("highlights", []),
        "popular_for": flat_ext.get("popular_for", []),
        "accessibility": flat_ext.get("accessibility", []),
        "offerings": flat_ext.get("offerings", []),
        "amenities": flat_ext.get("amenities", []),
        "atmosphere": flat_ext.get("atmosphere", []),
        "crowd": flat_ext.get("crowd", []),
        # "children": flat_ext.get("children", []),
        # "parking": flat_ext.get("parking", []),
    }

    
        # użyj wersji z priorytetami i ograniczeniem długości
        text_description = build_place_description_with_priority(name, place_dict)

        # Pobierz oceny (używając już znalezionego place)
        ratings_data = get_ratings_data(name, coords, place)

        return {
            "types": types_of_goods,
            "address": address,
            "phone": phone,
            "opening_hours": operating_hours,
            "service_options": flat_ext.get("service_options", []),
            "specials": flat_ext.get("highlights", []),
            "popular_for": flat_ext.get("popular_for", []),
            "accessibility": flat_ext.get("accessibility", []),
            "offerings": flat_ext.get("offerings", []),
            "amenities": flat_ext.get("amenities", []),
            "atmosphere": flat_ext.get("atmosphere", []),
            "crowd": flat_ext.get("crowd", []),
            "children": flat_ext.get("children", []),
            "parking": flat_ext.get("parking", []),
            "context": text_description,
            "google_rating": ratings_data.get("google_rating"),
            "google_reviews_total": ratings_data.get("google_reviews_total"),
            "google_price_range": ratings_data.get("google_price_range")
        }

    except Exception as e:
        print(f"Błąd dla {name}: {e}")
        return None



def main():
    print("Pobieranie danych Google Maps (słowa kluczowe, oceny, ceny)...\n")
    print("="*60)

    records_processed = 0
    records_ok = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout, \
         open(RATINGS_FILE, "w", encoding="utf-8") as frat, \
         open(EMBED_FILE, "w", encoding="utf-8") as fembed:


        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except:
                continue

            name = rec.get("name", "Nieznane")
            coords = rec.get("Współrzędne")
            oms_id = rec.get("oms_id")

            key_words = get_key_words(name, coords)
            
            print(f"\n[{records_processed + 1}] {name}")
            if key_words:
                print(f" Zapisano {len(key_words.items())} kluczowych słów.")
                if key_words.get("google_rating"):
                    print(f" Ocena: {key_words.get('google_rating')} ({key_words.get('google_reviews_total')} opinii) | Cena: {key_words.get('google_price_range')}")
            else:
                print(f" Nie udało się pobrać danych (None).")
                records_processed += 1
                continue

            # Zapis do pliku z keywords
            output_rec = {
                "oms_id": oms_id,
                "name": name,
                "Współrzędne": coords,
                "key_words": key_words or {},
            }
            fout.write(json.dumps(output_rec, ensure_ascii=False) + "\n")

            # Zapis do pliku z ratings (uproszczony format)
            ratings_rec = {
                "oms_id": oms_id,
                "name": name,
                "Współrzędne": coords,
                "google_rating": key_words.get("google_rating") if key_words else None,
                "google_reviews_total": key_words.get("google_reviews_total") if key_words else None,
                "google_price_range": key_words.get("google_price_range") if key_words else None
            }
            frat.write(json.dumps(ratings_rec, ensure_ascii=False) + "\n")
            
            # Zapis do pliku dla embeddingów - z keywords
            if key_words:
                context = key_words["context"]
                embed_obj = {
                    "oms_id": oms_id,
                    "name": name,
                    "coords": coords,
                    "context": context
                }
                fembed.write(json.dumps(embed_obj, ensure_ascii=False) + "\n")

            if key_words:
                records_ok += 1

            records_processed += 1
            time.sleep(DELAY_SECONDS)

    print("\n" + "="*60)
    print(" GOTOWE!")
    print(f" Przetworzono: {records_processed} miejsc")
    print(f" Udało się pobrać dane dla: {records_ok}")
    print(f" Wynik zapisano do:")
    print(f"  - {OUTPUT_FILE}")
    print(f"  - {RATINGS_FILE}")
    print(f"  - {EMBED_FILE}")



if __name__ == "__main__":
    if SERPAPI_KEY == "YOUR_API_KEY":
        print(" BŁĄD: Ustaw swój klucz SerpAPI w zmiennej SERPAPI_KEY")
        print("Pobierz klucz z: https://serpapi.com/")
    else:
        main()
        
        
        
