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

    # Jawnie mapujemy pooling -> plik
    embedding_files = {
        "mean": "output_files/lodz_restaurants_cafes_embeddings_mean.jsonl",
        "cls": "output_files/lodz_restaurants_cafes_embeddings_cls.jsonl"
    }

    queries = [
        "Gdzie zjem dobrą pizzę?",
        "Szukam restauracji z kuchnią azjatycką",
        "Jaka jest najlepsza kawiarnia w centrum?"
    ]

    results_log = [f"Test mode: {test_mode.upper()}\n\n"]

    for pooling_type, file_path in embedding_files.items():
        print("=" * 80)
        print(f"Testing with {pooling_type.upper()} pooling")
        print(f"Embedding file: {file_path}")
        print("=" * 80)

        results_log.append(
            f"{'=' * 30} TEST SET: {pooling_type.upper()} {'=' * 30}\n"
        )

        if not os.path.exists(file_path):
            msg = f"ERROR: File not found: {file_path}"
            print(msg)
            results_log.append(msg + "\n\n")
            continue

        try:
            start_time = time.time()

            # 🔑 KLUCZOWA ZMIANA – przekazujemy pooling_type
            rag_chain, _ = create_rag_system(
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
                    response = rag_chain.generate_response(query, k=3)
                    print(f"<-- Response: {response}\n")
                    results_log.append(f"Response:\n{response}\n")

                elif test_mode == "retrieval":
                    docs_with_scores = (
                        rag_chain.vectorstore.similarity_search_with_score(
                            query, k=3
                        )
                    )

                    response_lines = [
                        f"{doc.page_content} - {score:.4f}"
                        for doc, score in docs_with_scores
                    ]

                    response = "\n".join(response_lines)
                    print(f"<-- Retrieved docs (name - score):\n{response}\n")
                    results_log.append(f"Retrieved docs:\n{response}\n")

                results_log.append("-" * 40 + "\n")

        except Exception as e:
            error_message = (
                f"An error occurred during testing with {pooling_type}: {e}"
            )
            print(error_message)
            import traceback
            traceback.print_exc()
            results_log.append(f"ERROR: {error_message}\n\n")

        results_log.append("\n" * 2)

    output_filename = f"embedding_test_results_{test_mode}.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.writelines(results_log)

    print(f"All tests completed. Results saved to '{output_filename}'")


if __name__ == "__main__":
    print("Running retrieval-only tests for pooling evaluation...")
    run_tests(test_mode="retrieval")
