import json
import time
from typing import Optional, List
from embedding_model import ModelMeanPooling

# ---------- Ustawienia ----------
INPUT_FILE = "output_files/lodz_restaurants_cafes_emb_input.jsonl"
OUTPUT_FILE = "output_files/lodz_restaurants_cafes_embeddings_mean.jsonl"

# model do embeddingów
EMBED_MODEL_NAME = "sdadas/mmlw-retrieval-roberta-large"

# Konfiguracja poolingu: "mean" lub "cls"
POOLING_STRATEGY = "mean"  # zmień na "cls" dla CLS poolingu

# Wymiar embeddingów dla modelu
WORD_EMBEDDING_DIMENSION = 1024

# ---------- Inicjalizacja modelu ----------
print("Ładowanie modelu embeddingów:", EMBED_MODEL_NAME)
print(f"Pooling strategy: {POOLING_STRATEGY}")
model = ModelMeanPooling(
    EMBED_MODEL_NAME,
    word_embedding_dimension=WORD_EMBEDDING_DIMENSION,
    pooling_strategy=POOLING_STRATEGY
)



def create_embedding(text: str) -> List[float]:
    """Tworzy embedding dla tekstu."""
    if not text or not isinstance(text, str):
        return []
    embedding = model.encode(text, normalize=True)
    return embedding.tolist()


def build_rich_record(rec: dict) -> dict:
    """Buduje rekord ze wszystkimi danymi i embeddingiem."""
    oms_id = rec.get("oms_id")
    name = rec.get("name", "")
    context = rec.get("context", "")
    
    # Tworzymy embedding z pola "context"
    embedding = create_embedding(context)
    
    return {
        "oms_id": oms_id,
        "name": name,
        "embedding": embedding
    }


# ---------- Główna pętla ----------
def main():
    cnt = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            try:
                enriched = build_rich_record(rec)
                fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                cnt += 1
                if cnt % 50 == 0:
                    print(f"Przetworzono {cnt} rekordów...")
                time.sleep(0.02)
            except Exception as e:
                print(f"Błąd przetwarzania rekordu {cnt}: {e}")
    
    print("="*60)
    print("Gotowe! Zapisano", cnt, "rekordów do", OUTPUT_FILE)


if __name__ == "__main__":
    main()