import os
import sys
import numpy as np
from math import log1p
from scipy.special import expit
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import create_rag_system, distance_km

# --- KONFIGURACJA TESTU ---

# Złoty standard (Ground Truth)
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

def evaluate_ranking(ranked_list, ground_truth, k=5):
    """Oblicza metryki dla posortowanej listy wyników."""
    top_k = ranked_list[:k]
    
    is_hit = 0
    first_relevant_rank = 0
    relevant_count = 0
    
    for rank, doc in enumerate(top_k, 1):
        is_relevant = any(gt.lower() in doc['name'].lower() for gt in ground_truth)
        if is_relevant:
            if is_hit == 0:
                is_hit = 1
                first_relevant_rank = rank
            relevant_count += 1
            
    mrr = 1.0 / first_relevant_rank if first_relevant_rank > 0 else 0.0
    precision = relevant_count / k
    
    return is_hit, mrr, precision

def run_full_pipeline_evaluation():
    load_dotenv()
    
    print("Inicjalizacja systemu RAG...")
    rag, _, _ = create_rag_system(embeddings_file="output_files/lodz_restaurants_cafes_embeddings_cls_words.jsonl")
    
    queries = list(GROUND_TRUTH.keys())
    
    # Słowniki do przechowywania list wyników dla każdej metryki na każdym etapie
    metrics = {
        "stage1_bi_encoder": {"hits": [], "mrr": [], "precision": []},
        "stage2_reranker": {"hits": [], "mrr": [], "precision": []},
        "stage3_final_score": {"hits": [], "mrr": [], "precision": []},
    }

    print(f"\nRozpoczynam ewaluację pełnego pipeline'u dla {len(queries)} zapytań...")

    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Przetwarzanie: '{query[:50]}...'")
        
        # --- Krok 1: Pobranie kandydatów (Bi-Encoder) ---
        docs_with_scores = rag.vectorstore.similarity_search_with_score(query, k=50)
        
        unique_results = {}
        for doc, score in docs_with_scores:
            name = doc.metadata.get("name")
            if name and name not in unique_results:
                unique_results[name] = doc.metadata
                unique_results[name]['semantic_score'] = score
        
        candidates = list(unique_results.values())

        # --- Ewaluacja Etapu 1: Tylko Bi-Encoder ---
        candidates.sort(key=lambda x: x['semantic_score'], reverse=True)
        hit, mrr, prec = evaluate_ranking(candidates, GROUND_TRUTH[query])
        metrics["stage1_bi_encoder"]["hits"].append(hit)
        metrics["stage1_bi_encoder"]["mrr"].append(mrr)
        metrics["stage1_bi_encoder"]["precision"].append(prec)

        # --- Krok 2: Reranking (Cross-Encoder) ---
        rerank_pairs = [[query, c.get("context", "")] for c in candidates]
        rerank_scores = rag.reranker.predict(rerank_pairs)
        normalized_scores = expit(rerank_scores)
        
        for idx, c in enumerate(candidates):
            c["reranker_score"] = float(normalized_scores[idx])

        # --- Ewaluacja Etapu 2: Po Rerankerze ---
        candidates.sort(key=lambda x: x['reranker_score'], reverse=True)
        hit, mrr, prec = evaluate_ranking(candidates, GROUND_TRUTH[query])
        metrics["stage2_reranker"]["hits"].append(hit)
        metrics["stage2_reranker"]["mrr"].append(mrr)
        metrics["stage2_reranker"]["precision"].append(prec)

        # --- Krok 3: Końcowy algorytm wagowy ---
        # Replikacja logiki z `conversational_rag.py`
        weights = {"semantic": 0.35, "rating": 0.35, "popularity": 0.10, "proximity": 0.20}
        max_reviews_log = max([log1p(c.get("google_reviews_total") or 0) for c in candidates] + [1.0])

        for c in candidates:
            score_semantic = c["reranker_score"] # Używamy wyniku z rerankera
            score_rating = ((c.get("google_rating") or 2.5) - 1) / 4.0
            score_popularity = log1p(c.get("google_reviews_total") or 0) / max_reviews_log
            # Dla tego testu ignorujemy dystans (brak lokalizacji użytkownika)
            score_proximity = 0.0 

            c["final_score"] = (
                weights["semantic"] * score_semantic +
                weights["rating"] * score_rating +
                weights["popularity"] * score_popularity
            )

        # --- Ewaluacja Etapu 3: Po końcowym wyniku ---
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        hit, mrr, prec = evaluate_ranking(candidates, GROUND_TRUTH[query])
        metrics["stage3_final_score"]["hits"].append(hit)
        metrics["stage3_final_score"]["mrr"].append(mrr)
        metrics["stage3_final_score"]["precision"].append(prec)

    # --- Podsumowanie wyników ---
    print("\n" + "="*80)
    print("WYNIKI EWALUACJI PEŁNEGO PIPELINE'U")
    print("="*80)
    print(f"{'Etap':<25} | {'Hit Rate@5':<12} | {'MRR':<8} | {'Precision@5':<12}")
    print("-" * 65)

    for stage, data in metrics.items():
        stage_name = stage.replace('_', ' ').title()
        hr = np.mean(data["hits"]) * 100
        mrr = np.mean(data["mrr"])
        prec = np.mean(data["precision"]) * 100
        print(f"{stage_name:<25} | {hr:<11.2f}% | {mrr:<8.4f} | {prec:<11.2f}%")

    print("="*80)

if __name__ == "__main__":
    run_full_pipeline_evaluation()