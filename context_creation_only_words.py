import json

# Ścieżki plików
INPUT_FILE = "output_files/filtered_keywords.jsonl"
HELPER_FILE = "output_files/lodz_restaurants_cafes_with_key_words.jsonl"
OUTPUT_FILE = "output_files/context_from_filtered_keywords.jsonl"

# Maksymalna długość opisu dla modelu embeddingowego
MAX_CONTEXT_LEN = 400

# Lista typów do wykluczenia (czarna lista)
EXCLUDED_TYPES = {
    "sklep", "sklep zoologiczny", "sklep rowerowy", "agencja nieruchomości",
    "sala bankietowa", "planowanie wesel", "konserwatorium muzyczne",
    "instytucja edukacyjna", "atrakcja turystyczna", "teren spacerowy",
    "serwis sprzętu agd", "sklep odzieżowy", "sklep sportowy",
    "sklep wielobranżowy", "sklep agd", "dostawca ekspresów i akcesoriów do kawy",
    "sklep z herbatą", "sklep z czekoladą", "sklep spożywczy",
    "delikatesy", "sklep mięsny"
}

def create_context_from_keywords(name, keywords_data):
    """
    Tworzy krótki, faktualny opis miejsca pod embeddingi (RAG) wg ścisłych zasad CLS-OPTIMIZED.
    Format: Definicja -> Typ -> Oferta -> Charakter -> Cechy.
    """
    def get_list(key):
        val = keywords_data.get(key, [])
        if isinstance(val, list):
            # Filtrowanie pustych stringów i duplikatów (zachowując kolejność)
            seen = set()
            res = []
            for v in val:
                if v and v not in seen:
                    seen.add(v)
                    res.append(str(v))
            return res
        return []

    parts = []

    # Pobieramy ofertę wcześniej, aby użyć jej do wnioskowania typu, jeśli brakuje 'types'
    offerings = get_list('offerings')

    # 1. Definicja (OBOWIĄZKOWE)
    # Używamy pierwszego typu jako głównego określenia, ewentualnie dodajemy drugi, jeśli istnieje.
    # Usuwamy "w Łodzi", bo to redundancja dla lokalnej bazy.
    types = get_list('types')
    if types:
        # Np. "Da Grasso to pizzeria, kuchnia włoska."
        main_definition = ", ".join([t.lower() for t in types[:2]])
    else:
        # Fallback: wnioskowanie z nazwy i oferty
        name_lower = name.lower()
        inferred = []
        
        if "pizza" in name_lower: inferred.append("pizzeria")
        elif "sushi" in name_lower: inferred.append("sushi")
        elif "kebab" in name_lower: inferred.append("kebab")
        elif "burger" in name_lower: inferred.append("burgerownia")
        elif any(x in name_lower for x in ["cafe", "caffe", "kawa", "cukiernia"]): inferred.append("kawiarnia")
        elif "bar" in name_lower or "pub" in name_lower: inferred.append("bar")
        elif "kawa" in [o.lower() for o in offerings] and "alkohol" not in [o.lower() for o in offerings]: inferred.append("kawiarnia")

        main_definition = ", ".join(inferred) if inferred else "restauracja"
    
    parts.append(f"{name} to {main_definition}.")

    # 2. Typ / kuchnia (MAX 2)
    # Jeśli mamy więcej niż 2 typy, wymieniamy kolejne tutaj.
    if len(types) > 2:
        parts.append(f"Typ miejsca: {', '.join(types[2:4])}.")

    # 3. Oferta (MAX 3)
    if offerings:
        parts.append(f"Oferta: {', '.join(offerings[:3])}.")

    # 4. Charakter / zastosowanie (MAX 2)
    # Źródła: atmosphere, crowd, popular_for
    char_sources = get_list('atmosphere') + get_list('crowd') + get_list('popular_for')
    # Deduplikacja
    seen_char = set()
    char_final = [c for c in char_sources if not (c in seen_char or seen_char.add(c))]
    
    if char_final:
        parts.append(f"Charakter miejsca: {', '.join(char_final[:2])}.")

    # 5. Cechy specjalne / udogodnienia (MAX 2)
    # Źródła: specials, amenities, accessibility, service_options, children, parking
    feat_sources = (get_list('specials') + get_list('amenities') + 
                    get_list('accessibility') + get_list('service_options') + 
                    get_list('children') + get_list('parking'))
    
    seen_feat = set()
    feat_final = [f for f in feat_sources if not (f in seen_feat or seen_feat.add(f))]
            
    if feat_final:
        parts.append(f"Cechy dodatkowe: {', '.join(feat_final[:2])}.")

    # Złożenie całości i limit znaków
    text = " ".join(parts)
    if MAX_CONTEXT_LEN and len(text) > MAX_CONTEXT_LEN:
        text = text[:MAX_CONTEXT_LEN-3].rstrip() + '...'

    return text

def load_helper_data(filepath):
    data = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if 'oms_id' in rec:
                        # Pobieramy obiekt 'key_words', który zawiera listę 'types'
                        # Struktura: {"oms_id": ..., "key_words": {"types": ["Restauracja", ...], ...}}
                        data[rec['oms_id']] = rec.get('key_words', {})
    except Exception as e:
        print(f"Warning: Could not load helper file: {e}")
    return data

def main():
    """
    Główna funkcja skryptu.
    """
    print(f"Rozpoczynam tworzenie kontekstu z pliku: {INPUT_FILE}")
    helper_data = load_helper_data(HELPER_FILE)
    records_written = 0

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
             open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

            for line in fin:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                oms_id = record.get("oms_id")
                name = record.get("name")
                keywords = record.get("keywords", {})

                # Filtrowanie po typach (czarna lista)
                types_list = keywords.get("types", [])
                # Sprawdź też w helper_data, bo tam mogą być dokładniejsze typy
                if oms_id in helper_data:
                    types_list.extend(helper_data[oms_id].get("types", []))
                
                if any(t.lower() in EXCLUDED_TYPES for t in types_list):
                    # print(f"Pominięto (czarna lista): {name} - {types_list}")
                    continue

                # Uzupełnij dane z pliku pomocniczego (np. types)
                if oms_id in helper_data:
                    helper_kw = helper_data[oms_id]
                    # Pobieramy TYLKO 'types' z pliku pomocniczego, aby zbudować lepszą definicję (CLS token)
                    if 'types' in helper_kw and helper_kw['types']:
                        keywords['types'] = helper_kw['types']

                if not name or not keywords:
                    continue

                # Generuj ciąg kontekstu
                context = create_context_from_keywords(name, keywords)

                # Przygotuj rekord wyjściowy
                output_record = {
                    "oms_id": oms_id,
                    "name": name,
                    "context": context
                }

                fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                records_written += 1

        print("\nPrzetwarzanie zakończone.")
        print(f"Zapisano {records_written} rekordów do pliku: {OUTPUT_FILE}")

    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku wejściowego: '{INPUT_FILE}'")
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd: {e}")

if __name__ == "__main__":
    main()