import os
import time
from dotenv import load_dotenv
from conversational_rag import create_rag_system


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

    # WAŻNA UWAGA:
    # Poniższa konfiguracja zakłada, że chcesz testować model `sdadas/stella-pl-retrieval`.
    # Model ten generuje embeddingi o wymiarze 1024.
    # Twoje obecne pliki `.jsonl` zawierają embeddingi o wymiarze 1536, co powoduje błąd.
    # ABY TESTY ZADZIAŁAŁY POPRAWNIE, MUSISZ NAJPIERW WYGENEROWAĆ PONOWNIE
    # PLIKI EMBEDDINGÓW (`.jsonl`) UŻYWAJĄC MODELU `sdadas/stella-pl-retrieval`.
    
    embedding_files = {
        "mean_full_context": {
            "name": "MEAN (Pełny kontekst)",
            "file": "output_files/lodz_restaurants_cafes_embeddings_mean_stella.jsonl",
            "embedding_model_name": "sdadas/stella-pl-retrieval",
            "pooling": "mean"
        },
        "cls_full_context": {
            "name": "CLS (Pełny kontekst)",
            "file": "output_files/lodz_restaurants_cafes_embeddings_cls_stella.jsonl",
            "embedding_model_name": "sdadas/stella-pl-retrieval",
            "pooling": "cls"
        },
        "mean_keywords_only": {
            "name": "MEAN (Tylko słowa kluczowe)",
            "file": "output_files/lodz_restaurants_cafes_embeddings_mean_words_stella.jsonl",
            "embedding_model_name": "sdadas/stella-pl-retrieval",
            "pooling": "mean"
        },
        "cls_keywords_only": {
            "name": "CLS (Tylko słowa kluczowe)",
            "file": "output_files/lodz_restaurants_cafes_embeddings_cls_words_stella.jsonl",
            "embedding_model_name": "sdadas/stella-pl-retrieval",
            "pooling": "cls"
        }
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

    for config_key, config in embedding_files.items():
        test_name = config["name"]
        file_path = config["file"]
        pooling_type = config["pooling"]
        embedding_model_name = config["embedding_model_name"]

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

        try:
            start_time = time.time()

            rag_chain, _ = create_rag_system(
                embeddings_file=file_path,
                pooling_type=pooling_type,
                embedding_model_name=embedding_model_name
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

                    response_lines = [
                        f"{doc.page_content} - {score:.4f}" for doc, score in top_unique_docs
                    ]

                    response = "\n".join(response_lines)
                    print(f"<-- Retrieved docs (name - score):\n{response}\n")
                    results_log.append(f"Retrieved docs:\n{response}\n")

                results_log.append("-" * 40 + "\n")

        except Exception as e:
            error_message = (
                f"An error occurred during testing with {test_name}: {e}"
            )
            print(error_message)
            import traceback
            traceback.print_exc()
            results_log.append(f"ERROR: {error_message}\n\n")

        results_log.append("\n" * 2)

    output_filename = f"embedding_test_results_all_stella_{test_mode}.txt"
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
