import os
import sys
import time
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import create_rag_system

def run_expansion_tests():
    """
    Uruchamia testy porównawcze dla Query Expansion (HyDE).
    Porównuje wyniki wyszukiwania dla surowego zapytania vs zapytania rozszerzonego przez LLM.
    """
    load_dotenv()

    if not os.getenv("HF_TOKEN"):
        print("Brak HF_TOKEN! Ustaw token w .env")
        return

    # Konfiguracja dla CLS z pełnym kontekstem
    config = {
        "name": "CLS (Pełny kontekst)",
        "file": "output_files/lodz_restaurants_cafes_embeddings_cls.jsonl",
        "pooling": "cls"
    }

    # Lista zapytań testowych
    queries = [
        "Gdzie zjem dobrą pizzę na grubym cieście?",
        "Szukam autentycznej restauracji z kuchnią włoską.",
        "Najlepsza kawiarnia w centrum z miejscem do pracy.",
        "Klimatyczne miejsce na romantyczną kolację z kominkiem.",
        "Gdzie można posłuchać muzyki na żywo w weekend?",
        "Restauracja przyjazna rodzinom z dziećmi i placem zabaw.",
        "Tanie jedzenie dla studenta, najlepiej blisko Piotrkowskiej",
        "Gdzie na wypasionego burgera i dobre piwo kraftowe?",
        "Szukam restauracji z bogatą ofertą dań wegańskich.",
        "Lokal z ogródkiem i dostępem dla osób na wózkach.",
        "Gdzie serwują najlepsze sushi w Łodzi?",
        "Polećcie jakiś dobry ramen.",
        "Miejsce na szybki i niedrogi lunch w okolicach biurowców.",
        "Szukam eleganckiej restauracji na kolację biznesową.",
        "Gdzie można zjeść tradycyjne polskie pierogi?",
        "Bar z fajkami wodnymi i dobrą herbatą.",
        "Cukiernia z najlepszymi pączkami w mieście.",
        "Restauracja serwująca owoce morza.",
        "Gdzie na śniadanie w sobotę rano?",
        "Mam ochotę na coś ostrego, może kuchnia meksykańska albo indyjska?"
    ]

    results_log = [f"Test mode: QUERY EXPANSION COMPARISON (CLS Full Context)\n"]
    results_log.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    print("=" * 80)
    print(f"Testing with {config['name']}")
    print(f"Embedding file: {config['file']}")
    print("=" * 80)

    if not os.path.exists(config['file']):
        print(f"ERROR: File not found: {config['file']}")
        return

    try:
        # Inicjalizacja systemu RAG
        # create_rag_system zwraca (rag, search, filter_open_places)
        rag_chain, _, _ = create_rag_system(
            embeddings_file=config['file'],
            pooling_type=config['pooling']
        )
        
        print("\nSystem initialized. Starting tests...\n")

        def get_top_results(query_text, k=5):
            """Pobiera top k unikalnych wyników dla danego tekstu zapytania."""
            # Pobieramy więcej (k*3) aby mieć zapas na deduplikację
            docs_with_scores = rag_chain.vectorstore.similarity_search_with_score(query_text, k=k*3)
            
            unique_docs = {}
            for doc, score in docs_with_scores:
                name = doc.page_content
                # Zachowujemy najwyższy wynik dla danej nazwy
                if name not in unique_docs:
                    unique_docs[name] = score
            
            # Sortujemy i bierzemy top k
            sorted_docs = sorted(unique_docs.items(), key=lambda x: x[1], reverse=True)[:k]
            return sorted_docs

        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] Processing: {query}")
            results_log.append(f"QUERY {i}: {query}\n")
            
            # 1. Baseline (Bez Query Expansion)
            baseline_results = get_top_results(query)
            
            results_log.append("-" * 20 + " Baseline (Raw Query) " + "-" * 20 + "\n")
            for name, score in baseline_results:
                results_log.append(f"{name} - {score:.4f}\n")
            
            # 2. Z Query Expansion (HyDE)
            start_exp = time.time()
            expanded_query = rag_chain.extract_search_query(query)
            exp_time = time.time() - start_exp
            
            if expanded_query:
                print(f"    Expanded in {exp_time:.2f}s")
                results_log.append(f"\n" + "-" * 20 + " Expanded Query (HyDE) " + "-" * 20 + "\n")
                results_log.append(f"Generated Context: {expanded_query}\n\n")
                
                expanded_results = get_top_results(expanded_query)
                
                results_log.append("Results with Expansion:\n")
                for name, score in expanded_results:
                    results_log.append(f"{name} - {score:.4f}\n")
            else:
                print("    No expansion generated.")
                results_log.append("\n" + "-" * 20 + " Expanded Query " + "-" * 20 + "\n")
                results_log.append("No expansion generated (returned None or empty).\n")

            results_log.append("\n" + "=" * 60 + "\n\n")

    except Exception as e:
        error_msg = f"An error occurred: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        results_log.append(f"\nERROR: {error_msg}\n")

    output_filename = "query_expansion_comparison_results.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.writelines(results_log)
    
    print(f"\nAll tests completed. Results saved to '{output_filename}'")

if __name__ == "__main__":
    run_expansion_tests()