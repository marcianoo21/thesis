import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import log1p
from scipy.special import expit
from dotenv import load_dotenv

# Dodaj katalog główny projektu do ścieżki, aby móc importować pakiet src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import create_rag_system, distance_km

# --- KONFIGURACJA TESTU ---

# Złoty standard (Ground Truth) - te same dane co w testach embeddingów
GROUND_TRUTH = {
    "Gdzie zjem dobrą pizzę na grubym cieście?": ["Gruby Benek", "Pizzeria 105", "Grande Pizza", "Fiero", "Da Grasso", "Antonio", "Biesiadowo", "Pizza Hut", "Otto", "Pełny Brzuszek", "Pizzeria Kultowa", "Fiero Pizza", "Pizzeria Verona", "Pizzeria Osiedlowa", "Do.Orso", "Speedy Romano", "Ferment", "Pizza Lovers", "Presto", "Boska Włoska", "Basilia", "Pizzeria Do Orso", "Solo Pizza"],
    "Szukam autentycznej restauracji z kuchnią włoską.": ["Boska Włoska", "Angelo", "Roma", "Ciao", "Bawełna", "Biesiadowo", "Fiero", "Pełny Brzuszek", "Fiero Pizza", "Presto", "Pasta GO!", "Pasta Go!", "Marco"],
    "Najlepsza kawiarnia w centrum z miejscem do pracy.": ["Starbucks", "Costa", "The Brick Coffee Factory", "Prosto z Mostu", "Kofeina", "Poczekalnia", "Dybalski", "Cafe Vanilia", "Layali Shisha Club & Restaurant", "Ekspres do Kawy", "Colour Cafe", "Owoce i warzywa", "MORNING", "Caffe przy Targu", "Stacja Zero kawiarnia piekarnia", "Boogie cafe", "Chude ciacho", "MORNING coffee & more", "Crazy Bubble", "Kawka", "Ice&Coffe", "Vita Cafe"],
    "Klimatyczne miejsce na romantyczną kolację z kominkiem.": ["Soplicowo", "Polka", "Gruby Benek", "Spółdzielnia", "Beijing Taste", "Winoteka", "Indian Steak", "Angelo"],
    "Gdzie można posłuchać muzyki na żywo w weekend?": ["Willa", "Ciao", "Smak"],
    "Restauracja przyjazna rodzinom z dziećmi i placem zabaw.": ["Sphinx", "A... Nóż Widelec", "A nóż widelec", "Pora Karmienia", "Pasta GO!", "Pasta Go!", "GeoBistro"],
    "Tanie jedzenie dla studenta, najlepiej blisko Piotrkowskiej": ["Antonio", "Kucak", "Obiady domowe", "Stołówka studencka", "Małgosia", "Saga", "Kapusta z grochem", "Phuong Dong", "Lawenda", "New York Hot Dog", "Bar Orientalny Phuong Thao", "Obiady u Gosi", "Teremok", "Zahir", "A-Dong", "Bar Nam-Long", "Rajskie Jadło", "Ba Mien", "Thai Wok", "Kuchnia Marché", "Wok-Art", "Obiady Domowe", "Złoty Smok", "Pod Jabłonką", "Karczma Raz na Wozie"],
    "Gdzie na wypasionego burgera i dobre piwo kraftowe?": ["Gastromachina", "Cochise", "Szpulka", "GastroMachina"],
    "Szukam restauracji z bogatą ofertą dań wegańskich.": ["Kucak", "Manekin", "Starbucks", "Otto", "Drukarnia", "Sushi Kushi", "Fiero Pizza", "Teremok", "Kimsu", "Novo Square Lounge Bar", "Kawka", "Masala Trail", "Restauracja Europa", "Vita Cafe"],
    "Lokal z ogródkiem i dostępem dla osób na wózkach.": ["Kucak", "Obiady domowe", "Beza Krówka - naturalne lody rzemieślnicze", "Restauracja Stary Rynek 2", "New York Hot Dog", "Obiady u Gosi", "Cochise", "Montag", "Pizza Hut", "Speedy Romano", "Pizza Lovers", "Nova Sushi", "Ba Mien", "Wasabi Sushi", "Złoty Smok", "Gruby Benek"],
    "Gdzie serwują najlepsze sushi w Łodzi?": ["Sushi Kushi", "Susharnia", "Hana Sushi", "Sushi Kushi & Ramen Shop", "Koku Sushi", "Sayuri Sushi", "HASHTAG SUSHI", "Bukowiecki Sushi", "Wasabi Sushi", "Sushi w dłoń", "House of Sushi", "Nova Sushi", "Sushi w Dłoń"],
    "Polećcie jakiś dobry ramen.": ["Ato Ramen", "Sushi Kushi & Ramen Shop", "Nova Sushi", "Sushi w Dłoń", "House of Sushi", "Sushi Kushi"],
    "Miejsce na szybki i niedrogi lunch w okolicach biurowców.": ["Kofeina", "Szpulka", "Caffe Przy Ulicy", "Caffe przy ulicy", "Bułkę Przez Bibułkę", "Rajskie Jadło"],
    "Szukam eleganckiej restauracji na kolację biznesową.": ["Polka", "Angelo", "Indian Steak", "Złota Kaczka"],
    "Gdzie można zjeść tradycyjne polskie pierogi?": ["Teremok", "Lepione&Pieczone", "Pierogarnia Stary Młyn"],
    "Bar z fajkami wodnymi i dobrą herbatą.": ["Casablanca", "Layali Shisha Club & Restaurant", "Crazy Bubble"],
    "Cukiernia z najlepszymi pączkami w mieście.": ["Cukiernia Sowa", "Cukiernia Braci Miś", "Drukarnia", "Montag", "Stacja Zero kawiarnia piekarnia", "Vita Cafe"],
    "Restauracja serwująca owoce morza.": ["Sushi Kushi", "Susharnia", "Hana Sushi", "Ato Ramen", "Sushi w dłoń", "House of Sushi", "Koku Sushi", "Nova Sushi", "Sushi w Dłoń", "Sushi Kushi & Ramen Shop", "Sayuri Sushi", "HASHTAG SUSHI", "Bukowiecki Sushi", "Wasabi Sushi"],
    "Gdzie na śniadanie w sobotę rano?": ["Prosto z Mostu", "Caffe Przy Ulicy", "Szpulka", "Caffe przy ulicy", "MORNING", "Caffe przy Targu", "MORNING coffee & more", "HASHTAG SUSHI"],
    "Mam ochotę na coś ostrego, może kuchnia meksykańska albo indyjska?": ["The Mexican", "Ganesh", "Indian Steak", "Third Eye", "Masala Trail"]
}

def precompute_candidates(rag, queries):
    """
    Dla każdego zapytania pobiera kandydatów i oblicza ich surowe składowe (semantic, rating, popularity, proximity).
    Dzięki temu nie musimy uruchamiać modelu przy każdej zmianie wag.
    """
    precomputed_data = []
    
    print(f"Przetwarzanie {len(queries)} zapytań (Retrieval + Reranking)...")
    
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Analiza: {query}")
        
        # 1. Wykrywanie lokalizacji (LLM)
        user_location = None
        detected_loc = rag.normalize_location(query)
        if detected_loc:
            user_location = rag.location_service.geocode(detected_loc)

        # 2. Retrieval (Vector Search)
        # Pobieramy więcej kandydatów (50), aby mieć co sortować
        docs_with_scores = rag.vectorstore.similarity_search_with_score(query, k=50)
        
        # Deduplikacja
        unique_results = {}
        for doc, score in docs_with_scores:
            rec = doc.metadata
            name = rec.get("name")
            if name and name not in unique_results:
                unique_results[name] = {
                    "name": name,
                    "context": rec.get("context") or rec.get("key_words", {}).get("context"),
                    "coords": rec.get("Współrzędne"),
                    "google_rating": rec.get("google_rating") or rec.get("key_words", {}).get("google_rating"),
                    "google_reviews_total": rec.get("google_reviews_total") or rec.get("key_words", {}).get("google_reviews_total"),
                    "distance_km": float('inf')
                }
        
        candidates = list(unique_results.values())
        
        # 3. Reranking (Cross-Encoder)
        if candidates:
            rerank_pairs = [[query, c["context"]] for c in candidates]
            rerank_scores = rag.reranker.predict(rerank_pairs)
            normalized_semantic = expit(rerank_scores) # Sigmoid
            
            for idx, c in enumerate(candidates):
                c["score_semantic"] = float(normalized_semantic[idx])
        
        # 4. Obliczanie składowych metadanych (Rating, Popularity, Proximity)
        max_dist = 0.0
        max_reviews_log = 1.0
        
        # Pre-kalkulacja maxów do normalizacji
        for c in candidates:
            # Dystans
            if user_location and c.get("coords"):
                try:
                    lat, lon = map(float, c["coords"].split(","))
                    dist = distance_km(user_location[0], user_location[1], lat, lon)
                    c["distance_km"] = dist
                    if dist != float('inf'): max_dist = max(max_dist, dist)
                except: pass
            
            # Opinie
            reviews = c.get("google_reviews_total") or 0
            max_reviews_log = max(max_reviews_log, log1p(reviews))
            
        # Obliczanie znormalizowanych składowych
        for c in candidates:
            # Rating (0-1)
            rating = c.get("google_rating")
            c["score_rating"] = (rating - 1) / 4.0 if rating else 0.5
            
            # Popularity (0-1)
            reviews = c.get("google_reviews_total") or 0
            c["score_popularity"] = log1p(reviews) / max_reviews_log if max_reviews_log > 0 else 0
            
            # Proximity (0-1)
            c["score_proximity"] = 0.0
            if user_location and max_dist > 0 and c["distance_km"] != float('inf'):
                c["score_proximity"] = 1.0 - (c["distance_km"] / max_dist)
                
        precomputed_data.append({
            "query": query,
            "candidates": candidates,
            "ground_truth": GROUND_TRUTH.get(query, [])
        })
        
    return precomputed_data

def evaluate_weights(precomputed_data, w_sem, w_rat, w_pop, w_prox):
    """
    Oblicza średnie MRR dla zadanego zestawu wag.
    """
    total_mrr = 0.0
    
    for item in precomputed_data:
        candidates = item["candidates"]
        ground_truth = item["ground_truth"]
        
        # Oblicz final_score dla każdego kandydata
        for c in candidates:
            c["final_score"] = (
                w_sem * c["score_semantic"] +
                w_rat * c["score_rating"] +
                w_pop * c["score_popularity"] +
                w_prox * c["score_proximity"]
            )
            
        # Sortuj malejąco
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Oblicz MRR (pozycja pierwszego poprawnego wyniku)
        rank = 0
        for i, c in enumerate(candidates[:5], 1): # Patrzymy na Top-5
            # Sprawdź czy nazwa jest w ground truth (częściowe dopasowanie)
            is_relevant = any(gt.lower() in c["name"].lower() for gt in ground_truth)
            if is_relevant:
                rank = i
                break
        
        if rank > 0:
            total_mrr += 1.0 / rank
            
    return total_mrr / len(precomputed_data)

def run_analysis():
    load_dotenv()
    
    print("Inicjalizacja systemu RAG...")
    rag, _, _ = create_rag_system(embeddings_file="output_files/lodz_restaurants_cafes_embeddings_cls_words.jsonl")
    
    queries = list(GROUND_TRUTH.keys())
    data = precompute_candidates(rag, queries)
    
    # --- PARAMETRY DO OPTYMALIZACJI ---
    # Badamy wpływ wagi semantycznej (w_sem) vs reszta (metadata).
    # Reszta wag (rating, popularity, proximity) zachowuje swoje proporcje względem siebie.
    # Obecne proporcje reszty: Rat=0.35, Pop=0.1, Prox=0.2 -> Suma=0.65
    
    base_rat = 0.35 / 0.65
    base_pop = 0.10 / 0.65
    base_prox = 0.20 / 0.65
    
    x_values = [] # Waga semantyczna
    y_values = [] # MRR
    
    print("\nRozpoczynam symulację wag...")
    
    # Iterujemy wagę semantyczną od 0.0 do 1.0
    for w_sem in np.linspace(0.0, 1.0, 21): # co 0.05
        w_rest = 1.0 - w_sem
        
        w_rat = w_rest * base_rat
        w_pop = w_rest * base_pop
        w_prox = w_rest * base_prox
        
        mrr = evaluate_weights(data, w_sem, w_rat, w_pop, w_prox)
        
        x_values.append(w_sem)
        y_values.append(mrr)
        
        print(f"Sem: {w_sem:.2f} | Rat: {w_rat:.2f} | Pop: {w_pop:.2f} | Prox: {w_prox:.2f} -> MRR: {mrr:.4f}")

    # --- RYSOWANIE WYKRESU ---
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    plt.plot(x_values, y_values, marker='o', linewidth=2, label='Jakość dopasowania (MRR)')
    
    # Zaznaczenie obecnego punktu (0.35)
    current_sem = 0.35
    current_mrr = evaluate_weights(data, 0.35, 0.35, 0.1, 0.2)
    
    plt.axvline(x=current_sem, color='r', linestyle='--', label=f'Obecna konfiguracja (w={current_sem})')
    plt.scatter([current_sem], [current_mrr], color='red', s=100, zorder=5)
    
    # Znalezienie optimum
    best_idx = np.argmax(y_values)
    best_sem = x_values[best_idx]
    best_mrr = y_values[best_idx]
    
    plt.scatter([best_sem], [best_mrr], color='green', s=100, zorder=5, label=f'Optimum (w={best_sem:.2f})')

    plt.title("Wpływ wagi semantycznej na jakość rekomendacji (MRR)", fontsize=14)
    plt.xlabel("Waga Semantyczna (vs Jakość/Lokalizacja)", fontsize=12)
    plt.ylabel("Mean Reciprocal Rank (MRR)", fontsize=12)
    plt.legend()
    
    # Dodanie opisu osi X
    plt.text(0.02, min(y_values), "Tylko Metadane\n(Ocena, Opinie, Dystans)", fontsize=9, verticalalignment='bottom')
    plt.text(0.98, min(y_values), "Tylko Semantyka\n(Vector Search)", fontsize=9, horizontalalignment='right', verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig("weights_optimization.png", dpi=300)
    print(f"\nWykres zapisano jako: weights_optimization.png")
    print(f"Najlepszy wynik MRR: {best_mrr:.4f} dla wagi semantycznej: {best_sem:.2f}")
    print(f"Obecny wynik MRR: {current_mrr:.4f} dla wagi semantycznej: {current_sem:.2f}")

if __name__ == "__main__":
    run_analysis()