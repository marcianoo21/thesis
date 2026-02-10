import requests
import time
import csv
import json
from math import ceil
import pandas as pd

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# jeśli masz problemy z limitem, spróbuj innego publicznego endpointu lub dziel zapytanie na kafelki

# podstawowe zapytanie - pobiera nodes/ways/relations z tagiem amenity=restaurant|cafe dla obszaru "Łódź"
OVERPASS_QUERY = """
[out:json][timeout:180];
area["name"="Łódź"]["boundary"="administrative"]->.searchArea;
(
  node["amenity"~"restaurant|cafe"](area.searchArea);
  way["amenity"~"restaurant|cafe"](area.searchArea);
  relation["amenity"~"restaurant|cafe"](area.searchArea);
);
out center tags;
"""

# lista pól, które wyciągniemy z tagów (rozszerzaj według potrzeby)
WANTED_TAGS = [
    # podstawowe informacje
    "name", "amenity", "cuisine", "opening_hours", "website", "phone",
    "contact:phone", "contact:website", "contact:email",
    
    # adres
    "addr:street", "addr:housenumber", "addr:postcode", "addr:city",
    
    # operator i opis
    "operator", "description", 
    
    # internet i media
    "internet_access", "wifi",
    
    # dostępność i udogodnienia
    "wheelchair", "outdoor_seating", "indoor_seating", "smoking", "smoking:outside",
    "toilets", "toilets:wheelchair", "changing_table", "highchair",
    "air_conditioning", "dog", "reservations", "max_seats",
    
    # usługi
    "takeaway", "delivery", "drive_through", "self_service",
    
    # płatności
    "payment:cash", "payment:cards", "payment:debit_cards", "payment:credit_cards",
    "payment:contactless", "payment:mastercard", "payment:visa",
    "payment:american_express", "payment:google_pay", "payment:apple_pay",
    "payment:blik", "payment:mobile_phone",
    
    # diety i alergeny
    "diet:vegetarian", "vegetarian",
    "diet:vegan", "vegan",
    "diet:gluten_free", "gluten_free",
    
    # social media i kontakt dodatkowy
    "contact:facebook", "contact:instagram", "contact:twitter",
    "contact:youtube", "contact:tripadvisor", "contact:booking"
]

def fetch_overpass(query, max_retries=3, backoff=5):
    for attempt in range(max_retries):
        try:
            r = requests.post(OVERPASS_URL, data={"data": query}, timeout=300)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"Overpass request failed (attempt {attempt+1}): {e}")
            time.sleep(backoff * (attempt+1))
    raise RuntimeError("Nie udało się pobrać danych z Overpass po kilku próbach.")

def extract_pois(osm_json):
    elements = osm_json.get("elements", [])
    rows = []
    for el in elements:
        tags = el.get("tags", {}) or {}
        # lat/lon: nodes mają lat/lon bezpośrednio, ways/relations - center
        if el["type"] == "node":
            lat = el.get("lat")
            lon = el.get("lon")
        else:
            center = el.get("center", {})
            lat = center.get("lat")
            lon = center.get("lon")

        row = {
            "osm_type": el.get("type"),
            "osm_id": el.get("id"),
            "lat": lat,
            "lon": lon,
        }
        # wypakuj zadane tagi (wybiórczo) oraz dodaj pełne tags jako JSON
        for tag in WANTED_TAGS:
            # preferuj contact:* jeśli existuje - ale zapisujemy obie wersje
            row[tag] = tags.get(tag) or tags.get(tag.replace("contact:", "")) or None
        row["all_tags"] = json.dumps(tags, ensure_ascii=False)
        rows.append(row)
    return rows

def save_to_csv(rows, filename="lodz_restaurants_cafes.csv"):
    df = pd.DataFrame(rows)
    # dodatkowe kolumny porządkujące
    cols_order = ["osm_type", "osm_id", "name", "amenity", "cuisine",
                  "addr:street", "addr:housenumber", "addr:postcode", "addr:city",
                  "phone", "website", "opening_hours", "operator", "lat", "lon", "all_tags"]
    # zachowaj wszystkie kolumny, ale jeśli brak niektórych w df, dołącz je na końcu
    remaining = [c for c in cols_order if c in df.columns]
    other = [c for c in df.columns if c not in remaining]
    df = df[remaining + other]
    # wygeneruj szczegółowy opis kontekstowy (text_chunk) dla każdego rekordu
    try:
        # jeśli kolumna już istnieje, nie nadpisujemy jeśli pusta
        if 'text_chunk' not in df.columns:
            df['text_chunk'] = df.apply(build_description, axis=1)
        else:
            # nadpisz tylko jeśli wartości są puste/NaN
            mask = df['text_chunk'].isna() | (df['text_chunk'].astype(str).str.strip() == '')
            if mask.any():
                df.loc[mask, 'text_chunk'] = df[mask].apply(build_description, axis=1)
    except Exception as e:
        print(f"Nie udało się wygenerować text_chunk: {e}")
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Zapisano {len(df)} rekordów do {filename}")


def build_description(row: pd.Series) -> str:
    """Zbuduj szczegółowy, wieloczęściowy opis (text_chunk) dla jednego wiersza.

    Cel: dać modelowi kontekst obejmujący wszystkie istotne pola
    (nazwa, typ, kuchnia, adres, kontakt, godziny, dostępność, płatności, diety,
    udogodnienia, współrzędne, oraz szkic tagów dodatkowych).

    Funkcja zwraca czysty tekst gotowy do indeksowania. Obsługuje brakujące dane.
    """
    # preferencje pól kontaktowych
    phone = (row.get('contact:phone') or row.get('phone') or row.get('contact_phone') or '')
    website = (row.get('contact:website') or row.get('website') or row.get('contact_website') or '')

    parts = []
    name = row.get('name') if pd.notna(row.get('name')) else 'Miejsce'
    amenity = row.get('amenity') if pd.notna(row.get('amenity')) else ''
    header = f"{name}"
    if amenity:
        header += f" — {amenity}"
    parts.append(header.strip())

    # kuchnia i cechy gastronomiczne
    cuisine = row.get('cuisine')
    if pd.notna(cuisine) and str(cuisine).strip():
        # rozbijamy wielokrotne rodzaje kuchni
        try:
            cuisines = [c.strip() for delim in [';', ','] for c in str(cuisine).split(delim)]
            # usuń puste i zduplikuj
            cuisines = [c for c in cuisines if c]
            cuisine_str = ', '.join(dict.fromkeys(cuisines))
        except Exception:
            cuisine_str = str(cuisine)
        parts.append(f"Kuchnia: {cuisine_str}.")

    # adres
    street = row.get('addr:street')
    housenr = row.get('addr:housenumber')
    postcode = row.get('addr:postcode')
    city = row.get('addr:city')
    addr_items = []
    if pd.notna(street) and str(street).strip():
        addr_items.append(str(street).strip())
    if pd.notna(housenr) and str(housenr).strip():
        addr_items.append(str(housenr).strip())
    if addr_items or pd.notna(city) or pd.notna(postcode):
        addr = ' '.join(addr_items).strip()
        if postcode and pd.notna(postcode):
            addr += (', ' + str(postcode)) if addr else str(postcode)
        if city and pd.notna(city):
            addr += (', ' + str(city)) if addr else str(city)
        parts.append(f"Adres: {addr}.")

    # godziny, operator
    if pd.notna(row.get('opening_hours')) and str(row.get('opening_hours')).strip():
        parts.append(f"Godziny otwarcia: {row['opening_hours']}.")
    if pd.notna(row.get('operator')) and str(row.get('operator')).strip():
        parts.append(f"Operator: {row['operator']}.")

    # kontakt i social media
    if phone:
        parts.append(f"Telefon: {phone}.")
    if website:
        parts.append(f"Strona: {website}.")
    
    # social media i dodatkowe kontakty
    social = []
    for fld, label in [
        ('contact:facebook', 'Facebook'),
        ('contact:instagram', 'Instagram'),
        ('contact:twitter', 'Twitter'),
        ('contact:youtube', 'YouTube'),
        ('contact:tripadvisor', 'TripAdvisor'),
        ('contact:booking', 'Booking')
    ]:
        v = row.get(fld)
        if pd.notna(v) and str(v).strip():
            social.append(f"{label}: {v}")
    if social:
        parts.append('Social media i dodatkowe profile: ' + '; '.join(social) + '.')

    # dostępność i udogodnienia
    acc = []
    
    # dostępność dla niepełnosprawnych i dzieci
    for fld, label in [
        ('wheelchair', 'Dostęp dla wózków'),
        ('toilets:wheelchair', 'Toalety dla niepełnosprawnych'),
        ('changing_table', 'Przewijak'),
        ('highchair', 'Krzesełko dla dzieci')
    ]:
        val = row.get(fld)
        if pd.notna(val) and str(val).strip():
            acc.append(f"{label}: {val}")
    
    # miejsca i palenie
    for fld, label in [
        ('indoor_seating', 'Miejsca wewnątrz'),
        ('outdoor_seating', 'Miejsca na zewnątrz'),
        ('smoking', 'Palenie'),
        ('smoking:outside', 'Palenie na zewnątrz'),
        ('max_seats', 'Liczba miejsc')
    ]:
        val = row.get(fld)
        if pd.notna(val) and str(val).strip():
            acc.append(f"{label}: {val}")
    
    # internet i udogodnienia techniczne
    for fld, label in [
        ('internet_access', 'Internet'),
        ('wifi', 'Wi-Fi'),
        ('air_conditioning', 'Klimatyzacja')
    ]:
        val = row.get(fld)
        if pd.notna(val) and str(val).strip():
            acc.append(f"{label}: {val}")
    
    # usługi
    for fld, label in [
        ('takeaway', 'Na wynos'),
        ('delivery', 'Dostawa'),
        ('drive_through', 'Drive-through'),
        ('self_service', 'Samoobsługa'),
        ('reservations', 'Rezerwacje')
    ]:
        val = row.get(fld)
        if pd.notna(val) and str(val).strip():
            acc.append(f"{label}: {val}")
    
    # inne udogodnienia
    for fld, label in [
        ('dog', 'Przyjazne zwierzętom'),
        ('toilets', 'Toalety')
    ]:
        val = row.get(fld)
        if pd.notna(val) and str(val).strip():
            acc.append(f"{label}: {val}")
            
    if acc:
        parts.append('Udogodnienia: ' + '; '.join(acc) + '.')

    # płatności
    pay = []
    payment_methods = {
        'payment:cash': 'gotówka',
        'payment:cards': 'karty płatnicze',
        'payment:debit_cards': 'karty debetowe',
        'payment:credit_cards': 'karty kredytowe',
        'payment:contactless': 'płatności zbliżeniowe',
        'payment:mastercard': 'Mastercard',
        'payment:visa': 'Visa',
        'payment:american_express': 'American Express',
        'payment:google_pay': 'Google Pay',
        'payment:apple_pay': 'Apple Pay',
        'payment:blik': 'BLIK',
        'payment:mobile_phone': 'płatność telefonem'
    }
    for fld, label in payment_methods.items():
        v = row.get(fld) or row.get(fld.replace(':', '_'))
        if pd.notna(v) and str(v).strip():
            pay.append(f"{label}: {v}")
    if pay:
        parts.append('Dostępne formy płatności: ' + '; '.join(pay) + '.')

    # diety, alergeny i specjalne potrzeby
    diets = []
    # sprawdź obie wersje tagów (diet:xxx i xxx)
    for fld, alt_fld, label in [
        ('diet:vegetarian', 'vegetarian', 'wegetariańskie'),
        ('diet:vegan', 'vegan', 'wegańskie'),
        ('diet:gluten_free', 'gluten_free', 'bezglutenowe')
    ]:
        v = row.get(fld) or row.get(alt_fld)
        if pd.notna(v) and str(v).strip():
            diets.append(f"{label}: {v}")
    if diets:
        parts.append('Diety i alergeny: ' + '; '.join(diets) + '.')

    # wielkość / miejsca
    if pd.notna(row.get('max_seats')) and str(row.get('max_seats')).strip():
        parts.append(f"Maks. miejsc: {row.get('max_seats')}.")

    # współrzędne i odnośnik OSM
    lat = row.get('lat')
    lon = row.get('lon')
    if pd.notna(lat) and pd.notna(lon):
        parts.append(f"Współrzędne: {float(lat):.6f}, {float(lon):.6f}.")
    # link do OSM (jeśli dostępne id i typ)
    if pd.notna(row.get('osm_type')) and pd.notna(row.get('osm_id')):
        t = row.get('osm_type')
        oid = int(row.get('osm_id'))
        # typ w URL: node/way/relation
        parts.append(f"OSM: https://www.openstreetmap.org/{t}/{oid}.")

    # krótka agregacja dodatkowych tagów (all_tags jeśli jest JSON)
    # Cel: dodać wszystkie pozostałe tagi, które nie mają przypisanej kategorii
    all_tags = row.get('all_tags')
    extra = []
    if pd.notna(all_tags) and str(all_tags).strip():
        try:
            tags = all_tags if isinstance(all_tags, dict) else json.loads(all_tags)
            # Klucze, które już obsługujemy w innych sekcjach (nie duplikujemy)
            used_keys = set([
                'name','amenity','cuisine','opening_hours','website','phone','contact:phone','contact:website','contact:email',
                'addr:street','addr:housenumber','addr:postcode','addr:city','operator','description','internet_access','wifi',
                'wheelchair','outdoor_seating','indoor_seating','smoking','smoking:outside','toilets','toilets:wheelchair',
                'changing_table','highchair','air_conditioning','dog','reservations','max_seats','takeaway','delivery','drive_through',
                'self_service','payment:cash','payment:cards','payment:debit_cards','payment:credit_cards','payment:contactless',
                'payment:mastercard','payment:visa','payment:american_express','payment:google_pay','payment:apple_pay','payment:blik',
                'payment:mobile_phone','diet:vegetarian','vegetarian','diet:vegan','vegan','diet:gluten_free','gluten_free',
                'contact:facebook','contact:instagram','contact:twitter','contact:youtube','contact:tripadvisor','contact:booking',
                'all_tags','osm_type','osm_id','lat','lon'
            ])

            additional = []
            for k, v in tags.items():
                if not k or k in used_keys:
                    continue
                if v is None:
                    continue
                s = str(v).strip()
                if not s:
                    continue
                # skróć długie wartości
                if len(s) > 200:
                    s = s[:197] + '...'
                additional.append(f"{k}: {s}")

            # zachowaj sensowny limit (np. 20 tagów)
            if additional:
                parts.append('Dodatkowe tagi: ' + '; '.join(additional[:30]) + '.')
        except Exception:
            # jeśli nie JSON, dodaj surowy ciąg (limit długości)
            s = str(all_tags)
            extra.append(s[:200])
    # jeśli jest jawny opis w kolumnie, dodaj go (unikaj duplikatów)
    if pd.notna(row.get('description')) and str(row.get('description')).strip():
        desc = str(row.get('description')).strip()
        parts.append('Opis (z tagu description): ' + desc)
    # jeśli zostały nieparsowane surowe ekstra, dorzuć je
    if extra:
        parts.append('Dodatkowe informacje surowe: ' + ' | '.join(extra) + '.')

    # złącz i zwróć
    text = ' '.join([p.strip() for p in parts if p and str(p).strip()])
    return text.strip()


def generate_chunks_from_csv(infile: str, out_csv: str = None, out_jsonl: str = None):
    """Wczytaj CSV (eksport z Overpass), wygeneruj kolumnę text_chunk i zapisz CSV oraz JSONL z chunkami.

    Zwraca: tuple(out_csv, out_jsonl)
    """
    if out_csv is None:
        out_csv = infile.replace('.csv', '_with_chunks.csv')
    if out_jsonl is None:
        out_jsonl = infile.replace('.csv', '_chunks.jsonl')

    df = pd.read_csv(infile, encoding='utf-8-sig')
    # upewnij się, że puste pola rozpoznawane jako NaN
    df = df.where(pd.notnull(df), None)
    # wygeneruj chunk
    df['text_chunk'] = df.apply(build_description, axis=1)
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')

    # zapisz JSONL: id + text_chunk
    with open(out_jsonl, 'w', encoding='utf-8') as fh:
        for _, r in df.iterrows():
            rec = {
                'osm_type': str(r.get('osm_type') or "null"),
                'osm_id': int(r['osm_id']) if pd.notna(r.get('osm_id')) else "null",
                'name': str(r.get('name')) if pd.notna(r.get('name')) else "null",
                'lat': str(r.get('lat')) if pd.notna(r.get('lat')) else "null",
                'lon': str(r.get('lon')) if pd.notna(r.get('lon')) else "null",
                'text_chunk': str(r.get('text_chunk')) if pd.notna(r.get('text_chunk')) else "null"
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Zapisano CSV z chunkami: {out_csv}\nZapisano JSONL: {out_jsonl}")
    return out_csv, out_jsonl

def main():
    print("Wysyłam zapytanie do Overpass...")
    data = fetch_overpass(OVERPASS_QUERY)
    print("Pobieranie zakończone, przetwarzam...")
    rows = extract_pois(data)
    save_to_csv(rows)

if __name__ == "__main__":
    main()
