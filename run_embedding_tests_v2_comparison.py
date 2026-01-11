import os
import time
from dotenv import load_dotenv
from conversational_rag import create_rag_system

def run_tests(test_mode="retrieval"):
    """
    Porównuje wyniki wyszukiwania dla modelu v1 i v2.
    """
    load_dotenv()

    if not os.getenv("HF_TOKEN"):
        print("Brak HF_TOKEN! Ustaw token w .env")
        return

    # Definicja plików do porównania
    embedding_files = {
        "cls_words_v1": {
            "name": "CLS Words (v1: roberta-large)",
            "file": "output_files/lodz_restaurants_cafes_embeddings_cls_words.jsonl",
            "embedding_model_name": "sdadas/mmlw-retrieval-roberta-large",
            "pooling": "cls"
        },
        "cls_words_v2": {
            "name": "CLS Words (v2: roberta-large-v2)",
            "file": "output_files/lodz_restaurants_cafes_embeddings_cls_words_v2.jsonl",
            "embedding_model_name": "sdadas/mmlw-retrieval-roberta-large-v2",
            "pooling": "cls"
        }
    }

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

    results_log = [f"Test mode: {test_mode.upper()} - COMPARISON V1 vs V2\n\n"]

    for config_key, config in embedding_files.items():
        test_name = config["name"]
        file_path = config["file"]
        pooling_type = config["pooling"]
        embedding_model_name = config["embedding_model_name"]

        print("=" * 80)
        print(f"Testing with {test_name}")
        print(f"Embedding file: {file_path}")
        print(f"Model: {embedding_model_name}")
        print("=" * 80)

        results_log.append(
            f"{'=' * 30} TEST SET: {test_name} {'=' * 30}\n"
        )

        if not os.path.exists(file_path):
            msg = f"ERROR: File not found: {file_path}. Run the creation script first!"
            print(msg)
            results_log.append(msg + "\n\n")
            continue

        try:
            start_time = time.time()

            # Tworzymy system RAG z odpowiednim modelem embeddingów dla danego pliku
            rag_chain, _, _ = create_rag_system(
                embeddings_file=file_path,
                pooling_type=pooling_type,
                embedding_model_name=embedding_model_name
            )

            init_time = time.time() - start_time
            print(f"System initialized in {init_time:.2f} seconds.\n")

            for query in queries:
                print(f"--> Query: {query}")
                results_log.append(f"Query: {query}\n")

                if test_mode == "retrieval":
                    # Wyszukaj więcej wyników (np. 15), aby mieć z czego odfiltrować duplikaty
                    initial_k = 15
                    final_k = 5

                    docs_with_scores = (
                        rag_chain.vectorstore.similarity_search_with_score(
                            query, k=initial_k
                        )
                    )

                    # Deduplikacja wyników
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
                    # print(f"<-- Retrieved docs:\n{response}\n")
                    results_log.append(f"Retrieved docs:\n{response}\n")

                results_log.append("-" * 40 + "\n")

        except Exception as e:
            print(f"ERROR: {e}")
            results_log.append(f"ERROR: {e}\n\n")

    output_filename = "embedding_test_results_v1_vs_v2.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.writelines(results_log)

    print(f"All tests completed. Results saved to '{output_filename}'")

if __name__ == "__main__":
    run_tests(test_mode="retrieval")