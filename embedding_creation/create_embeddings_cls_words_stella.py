import json
import time
from typing import Optional, List, Dict
from embedding_model import ModelMeanPooling
# python -m embedding_creation.create_embeddings_cls_words_stella
INPUT_FILE = "output_files/context_from_filtered_keywords.jsonl"
OUTPUT_FILE = "output_files/lodz_restaurants_cafes_embeddings_cls_words_stella.jsonl"
METADATA_FILE = "output_files/lodz_restaurants_cafes_with_key_words.jsonl"

# model do embeddingów
EMBED_MODEL_NAME = "sdadas/stella-pl-retrieval"
    
# Konfiguracja poolingu: "mean" lub "cls"
POOLING_STRATEGY = "cls"  # zmień na "cls" dla CLS poolingu

print("Ładowanie modelu embeddingów:", EMBED_MODEL_NAME)
print(f"Pooling strategy: {POOLING_STRATEGY}")
model = ModelMeanPooling(
    EMBED_MODEL_NAME,
    pooling_strategy=POOLING_STRATEGY
)



def create_embedding(text: str) -> List[float]:
    """Tworzy embedding dla tekstu."""
    if not text or not isinstance(text, str):
        return []
    embedding = model.encode(text, normalize=True)
    return embedding.tolist()


def main():
    """
    Główna funkcja skryptu. Wczytuje konteksty, generuje dla nich embeddingi,
    a następnie łączy je z pełnymi metadanymi restauracji, tworząc
    kompletny plik wejściowy dla systemu RAG.
    """
    # 1. Wczytaj pełne metadane do słownika dla szybkiego dostępu
    print(f"Wczytuję pełne metadane z '{METADATA_FILE}'...")
    full_metadata: Dict[int, dict] = {}
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f_meta:
            for line in f_meta:
                rec = json.loads(line)
                if rec.get("oms_id"):
                    full_metadata[rec["oms_id"]] = rec
    except FileNotFoundError:
        print(f"Błąd: Plik z metadanymi '{METADATA_FILE}' nie został znaleziony.")
        return

    print(f"Znaleziono {len(full_metadata)} rekordów z metadanymi.")

    # 2. Przetwórz plik wejściowy, stwórz embeddingi i połącz z metadanymi
    cnt = 0
    print(f"Rozpoczynam tworzenie embeddingów z pliku: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                context_rec = json.loads(line)
                oms_id = context_rec.get("oms_id")
                metadata = full_metadata.get(oms_id)

                if not oms_id or not metadata:
                    print(f"Ostrzeżenie: Pomijam rekord bez ID lub metadanych: {line}")
                    continue

                embedding = create_embedding(context_rec.get("context", ""))
                enriched_record = metadata.copy()
                enriched_record["embedding"] = embedding

                fout.write(json.dumps(enriched_record, ensure_ascii=False) + "\n")
                cnt += 1
                if cnt % 50 == 0:
                    print(f"Przetworzono {cnt} rekordów...")
            except Exception as e:
                print(f"Błąd przetwarzania rekordu {cnt+1}: {e}")
    
    print("="*60)
    print("Gotowe! Zapisano", cnt, "rekordów do", OUTPUT_FILE)

if __name__ == "__main__":
    main()