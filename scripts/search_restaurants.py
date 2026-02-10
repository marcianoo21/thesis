import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from math import radians, sin, cos, sqrt, atan2
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="hid_training")

geo_coords = "51.779247, 19.493394"

location  = geolocator.reverse(geo_coords)

print(location)
print(location.raw)


# ===============================
#  FUNKCJA: oblicz odległość
# ===============================
def distance_km(lat1, lon1, lat2, lon2):
    R = 6371  # km
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(d_lon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


# ===============================
#  LOKALIZACJA CENTRUM ŁODZI
# ===============================
LODZ_CENTER = (51.759445, 19.457216)


# ================================================
# 1) ŁADOWANIE MODELU DO EMBEDDINGÓW
# ================================================
print("⏳ Ładowanie modelu SDADAS MMLW...")
model = SentenceTransformer("sdadas/mmlw-retrieval-roberta-large")
query_prefix = "zapytanie: "

# ================================================
# 2) WCZYTANIE EMBEDDINGÓW RESTAURACJI
# ================================================
print("⏳ Wczytywanie embeddingów...")

records = []
embeddings = []

with open("output_files/lodz_restaurants_cafes_embeddings.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        records.append(rec)
        embeddings.append(rec["embedding"])

embeddings = np.array(embeddings).astype("float32")

# ================================================
# 3) BUDOWA FAISS INDEX
# ================================================
print("⏳ Budowanie FAISS index...")

faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print(f"✅ Indeks gotowy! Liczba wektorów: {index.ntotal}")


# ===============================
#  WZBOGACENIE WYNIKU
# ===============================
def pretty_print_result(r):
    name = r["name"]
    typ = r["type"]
    addr = r["address"] or "brak adresu"
    rating = r.get("google_rating")
    reviews = r.get("google_reviews_total")

    lat, lon = None, None
    if r["coords"]:
        try:
            lat, lon = map(float, r["coords"].split(","))
        except:
            pass

    # Odległość od centrum
    if lat and lon:
        dist = distance_km(lat, lon, LODZ_CENTER[0], LODZ_CENTER[1])
        dist_text = f"{dist:.2f} km od centrum"
    else:
        dist_text = "brak danych o lokalizacji"

    # Ocena i liczba opinii
    rating_text = f" {rating} ({reviews} opinii)" if rating else " brak danych"

    print(f" **{name}**")
    print(f"    Typ: {typ}")
    print(f"    Adres: {addr}")
    print(f"    Odległość: {dist_text}")
    print(f"   {rating_text}")
    print(f"    Dopasowanie: {r['score']:.3f}")
    print()


# ================================================
# 4) FUNKCJA WYSZUKIWANIA
# ================================================
def search_restaurants(query, k=5):
    """Zwraca TOP-k najbliższych miejsc do zapytania, posortowanych po ocenie i liczbie opinii."""
    full_query = query_prefix + query

    q_emb = model.encode(full_query, normalize_embeddings=True)
    q_emb = q_emb.astype("float32").reshape(1, -1)

    scores, idxs = index.search(q_emb, k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        rec = records[idx]
        results.append({
            "score": float(score),
            "name": rec.get("name"),
            "type": rec.get("type"),
            "address": rec.get("Adres", "brak"),
            "coords": rec.get("Współrzędne"),
            "google_rating": rec.get("google_rating"),
            "google_reviews_total": rec.get("google_reviews_total"),
        })

    # Sortowanie po ocenie (malejąco), a w razie remisu po liczbie opinii (malejąco)
    results.sort(
        key=lambda x: (
            -(x["google_rating"] or 0),  # Minus, żeby był porządek malejący
            -(x["google_reviews_total"] or 0)  # Minus, żeby był porządek malejący
        )
    )

    return results


# ================================================
# 5) INTERAKCJA
# ================================================
if __name__ == "__main__":
    print("\n Inteligentny Rekomender Restauracji — Łódź\n")
    
    user_query = input(" Czego szukasz? ")
    
    print("\n Szukam najlepszych opcji...\n")
    
    results = search_restaurants(user_query, k=5)
    
    for r in results:
        pretty_print_result(r)