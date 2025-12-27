import json

# Ścieżki plików
INPUT_FILE = "output_files/filtered_keywords.jsonl"
OUTPUT_FILE = "output_files/context_from_filtered_keywords.jsonl"

# Priorytety i maksymalna długość - zaadaptowane z key_words_and_context_creation.py
PRIORITY = {
    "name_intro": 0,  # Dodany priorytet dla samego przedstawienia miejsca
    "types": 1,
    "offerings": 2,
    "specials": 3,
    "atmosphere": 4,
    "accessibility": 5,
    "amenities": 6,
    "crowd": 7,
    "popular_for": 8,
    "service_options": 9,
    "parking": 10,
    "children": 11,
}
MAX_CONTEXT_LEN = 800

def create_context_from_keywords(name, keywords_data):
    """
    Tworzy ciąg tekstowy (kontekst) z podanych słów kluczowych,
    używając priorytetów do uporządkowania informacji.
    """
    def join_list(values):
        if not isinstance(values, list) or not values:
            return None
        # Ograniczamy liczbę wartości, aby uniknąć zbyt długich ciągów
        shortened_values = values[:8]
        return ", ".join(shortened_values)

    components = {}

    # Zaczynamy od nazwy miejsca
    components['name_intro'] = f"Miejsce o nazwie {name}."

    # Przetwarzamy dostępne słowa kluczowe
    keyword_map = {
        'types': "To miejsce typu: {}.",
        'offerings': "W ofercie znajduje się: {}.",
        'specials': "Specjalne cechy miejsca: {}.",
        'atmosphere': "Atmosfera jest opisywana jako: {}.",
        'amenities': "Dostępne udogodnienia to: {}.",
        'popular_for': "Jest popularne szczególnie na: {}.",
        'crowd': "Typowa grupa odwiedzających: {}.",
        'accessibility': "Miejsce oferuje udogodnienia dostępności: {}."
    }

    for key, template in keyword_map.items():
        if key in keywords_data:
            text_values = join_list(keywords_data.get(key))
            if text_values:
                components[key] = template.format(text_values)

    # Porządkujemy komponenty zgodnie z priorytetem
    ordered_keys = sorted(components.keys(), key=lambda k: PRIORITY.get(k, 999))
    ordered_parts = [components[k] for k in ordered_keys if components.get(k)]

    text = ' '.join(ordered_parts)

    # Przycinamy tekst, jeśli jest za długi
    if MAX_CONTEXT_LEN and len(text) > MAX_CONTEXT_LEN:
        text = text[:MAX_CONTEXT_LEN - 3].rstrip() + '...'

    return text

def main():
    """
    Główna funkcja skryptu.
    """
    print(f"Rozpoczynam tworzenie kontekstu z pliku: {INPUT_FILE}")
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