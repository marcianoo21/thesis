import json

# Definicja ścieżek plików
# Plik wejściowy, z którego czytamy dane (ten z przykładów)
INPUT_FILE = "output_files/lodz_restaurants_cafes_with_key_words.jsonl"
# Plik wyjściowy, do którego zapiszemy przefiltrowane dane
OUTPUT_FILE = "output_files/filtered_keywords.jsonl"

# Klucze, które chcemy wyodrębnić z obiektu "key_words"
KEYS_TO_EXTRACT = [
    "specials",
    "popular_for",
    "accessibility",
    "offerings",
    "amenities",
    "atmosphere",
    "crowd"
]

def extract_keywords():
    """
    Czyta plik wejściowy JSONL, wyodrębnia określone słowa kluczowe
    i zapisuje je do nowego pliku JSONL.
    """
    print(f"Rozpoczynam przetwarzanie pliku: {INPUT_FILE}")
    records_written = 0

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
             open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

            for line in fin:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Ostrzeżenie: Pomijam nieprawidłową linię JSON: {line}")
                    continue

                key_words = record.get("key_words", {})

                # Przygotuj nowy rekord z podstawowymi informacjami
                filtered_record = {
                    "oms_id": record.get("oms_id"),
                    "name": record.get("name"),
                    "keywords": {}
                }

                # Dodaj tylko wybrane klucze, jeśli istnieją i nie są puste
                for key in KEYS_TO_EXTRACT:
                    if key in key_words and key_words[key]:
                        filtered_record["keywords"][key] = key_words[key]

                # Zapisz przefiltrowany rekord do pliku wyjściowego
                fout.write(json.dumps(filtered_record, ensure_ascii=False) + "\n")
                records_written += 1

        print("\nPrzetwarzanie zakończone.")
        print(f"Zapisano {records_written} rekordów do pliku: {OUTPUT_FILE}")

    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku wejściowego: '{INPUT_FILE}'")
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd: {e}")


if __name__ == "__main__":
    extract_keywords()