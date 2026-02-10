import os
import sys
import time
import csv
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import create_rag_system


def run_tests(test_mode="rag"):
    """
    Runs tests on different embedding models and queries, saving the results to a file.

    Args:
        test_mode (str): 'rag' to test the full RAG system (retrieval + generation),
                         'retrieval' to test only the retrieval component.
    """
    load_dotenv()

    if not os.getenv("HF_TOKEN"):
        print("Brak HF_TOKEN! Ustaw token w .env")
        return

    # Konfiguracja testów dla wszystkich 4 typów embeddingów
    embedding_files = {
        "mean_full_context": {
            "name": "MEAN (Pełny kontekst)",
            "file": "output_files/lodz_restaurants_cafes_embeddings_mean.jsonl",
            "pooling": "mean"
        },
        "cls_full_context": {
            "name": "CLS (Pełny kontekst)",
            "file": "output_files/lodz_restaurants_cafes_embeddings_cls.jsonl",
            "pooling": "cls"
        },
        "mean_keywords_only": {
            "name": "MEAN (Tylko słowa kluczowe)",
            "file": "output_files/lodz_restaurants_cafes_embeddings_mean_words.jsonl",
            "pooling": "mean"
        },
        "cls_keywords_only": {
            "name": "CLS (Tylko słowa kluczowe)",
            "file": "output_files/lodz_restaurants_cafes_embeddings_cls_words.jsonl",
            "pooling": "cls"
        }
    }

    # Złoty standard - słowa kluczowe lub nazwy, które MUSZĄ się pojawić w wynikach dla danego zapytania
    # Kluczem jest treść zapytania (dokładnie taka jak w liście queries)
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

    # Rozszerzona lista zapytań testowych
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

    results_log = [f"Test mode: {test_mode.upper()}\n\n"]
    metrics_summary = []

    for config_key, config in embedding_files.items():
        test_name = config["name"]
        file_path = config["file"]
        pooling_type = config["pooling"]

        print("=" * 80)
        print(f"Testing with {test_name}")
        print(f"Embedding file: {file_path}")
        print("=" * 80)

        results_log.append(
            f"{'=' * 30} TEST SET: {test_name} {'=' * 30}\n"
        )

        if not os.path.exists(file_path):
            msg = f"ERROR: File not found: {file_path}"
            print(msg)
            results_log.append(msg + "\n\n")
            continue

        total_hits = 0      # Do Hit Rate (uproszczony Recall)
        total_mrr = 0       # Do MRR
        total_precision = 0 # Do Precision
        queries_with_truth = 0

        try:
            start_time = time.time()

            rag_chain, _, _ = create_rag_system(
                embeddings_file=file_path,
                pooling_type=pooling_type
            )

            init_time = time.time() - start_time
            print(f"System initialized in {init_time:.2f} seconds.\n")

            if test_mode == "rag":
                rag_chain.reset_history()

            for query in queries:
                print(f"--> Query: {query}")
                results_log.append(f"Query: {query}\n")

                if test_mode == "rag":
                    response = rag_chain.generate_response(query, k=5)
                    print(f"<-- Response: {response}\n")
                    results_log.append(f"Response:\n{response}\n")

                elif test_mode == "retrieval":
                    # Wyszukaj więcej wyników (np. 15), aby mieć z czego odfiltrować duplikaty
                    initial_k = 15
                    final_k = 5

                    docs_with_scores = (
                        rag_chain.vectorstore.similarity_search_with_score(
                            query, k=initial_k
                        )
                    )

                    # Deduplikacja wyników - zachowujemy tylko pierwszy (najlepszy) wynik dla danej nazwy
                    unique_docs = {}
                    for doc, score in docs_with_scores:
                        name = doc.page_content
                        if name not in unique_docs:
                            unique_docs[name] = (doc, score)

                    # Weź top N unikalnych wyników
                    top_unique_docs = list(unique_docs.values())[:final_k]

                    # --- EWALUACJA (Metrics) ---
                    expected_items = GROUND_TRUTH.get(query, [])
                    
                    if expected_items:
                        queries_with_truth += 1
                        found_items = []
                        first_relevant_rank = 0
                        relevant_count_in_k = 0

                        for rank, (doc, score) in enumerate(top_unique_docs, 1):
                            is_doc_relevant = False
                            for expected in expected_items:
                                if expected.lower() in doc.page_content.lower():
                                    found_items.append(doc.page_content)
                                    is_doc_relevant = True
                                    break
                            
                            if is_doc_relevant:
                                relevant_count_in_k += 1
                                if first_relevant_rank == 0:
                                    first_relevant_rank = rank
                        
                        if found_items:
                            total_hits += 1
                            total_mrr += 1.0 / first_relevant_rank
                            total_precision += relevant_count_in_k / len(top_unique_docs)
                    # ---------------------------

                    response_lines = [
                        f"{doc.page_content} - {score:.4f}" for doc, score in top_unique_docs
                    ]

                    response = "\n".join(response_lines)
                    if expected_items:
                        print(f"   [EVAL] Relevant found: {list(set(found_items))} (Expected from: {expected_items})")
                        
                    print(f"<-- Retrieved docs (name - score):\n{response}\n")
                    results_log.append(f"Retrieved docs:\n{response}\n")

                results_log.append("-" * 40 + "\n")

            if test_mode == "retrieval" and queries_with_truth > 0:
                hit_rate = (total_hits / queries_with_truth) * 100
                mrr_score = total_mrr / queries_with_truth
                avg_precision = (total_precision / queries_with_truth) * 100
                
                metrics_summary.append({
                    "Model": test_name,
                    "Hit Rate@5": f"{hit_rate:.2f}%",
                    "MRR": f"{mrr_score:.4f}",
                    "Precision@5": f"{avg_precision:.2f}%"
                })

                summary = f"\n>>> SYSTEM EVALUATION:\n    Hit Rate@5 = {hit_rate:.2f}%\n    MRR = {mrr_score:.4f}\n    Precision@5 = {avg_precision:.2f}%\n"
                print(summary)
                results_log.append(summary)

        except Exception as e:
            error_message = (
                f"An error occurred during testing with {test_name}: {e}"
            )
            print(error_message)
            import traceback
            traceback.print_exc()
            results_log.append(f"ERROR: {error_message}\n\n")

        results_log.append("\n" * 2)

    if metrics_summary:
        # Generowanie czytelnej tabeli tekstowej do logów
        header = f"{'Model':<35} | {'Hit Rate@5':<12} | {'MRR':<8} | {'Precision@5':<12}"
        sep = "-" * len(header)
        table_str = f"\n\n{sep}\nFINAL SUMMARY\n{sep}\n{header}\n{sep}\n"
        for m in metrics_summary:
            table_str += f"{m['Model']:<35} | {m['Hit Rate@5']:<12} | {m['MRR']:<8} | {m['Precision@5']:<12}\n"
        table_str += f"{sep}\n"
        
        results_log.append(table_str)
        print(table_str)
        
        # Zapis do pliku CSV (Excel-friendly)
        csv_filename = f"embedding_metrics_summary_{test_mode}.csv"
        try:
            with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["Model", "Hit Rate@5", "MRR", "Precision@5"])
                writer.writeheader()
                writer.writerows(metrics_summary)
            print(f"Zapisano tabelę wyników do pliku CSV: {csv_filename}")
        except Exception as e:
            print(f"Błąd zapisu CSV: {e}")

    output_filename = f"embedding_test_results_all_{test_mode}.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.writelines(results_log)

    print(f"All tests completed. Results saved to '{output_filename}'")


if __name__ == "__main__":
    # Możesz wybrać tryb testowy: 'retrieval' (tylko wyszukiwanie) lub 'rag' (pełny system)
    print("Running comprehensive retrieval-only tests for all embedding types...")
    run_tests(test_mode="retrieval")

    # Aby uruchomić testy RAG (z generowaniem odpowiedzi przez LLM), odkomentuj poniższą linię:
    # print("\nRunning comprehensive RAG tests for all embedding types...")
    # run_tests(test_mode="rag")
