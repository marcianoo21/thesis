import re
from collections import defaultdict

def parse_results(input_file):
    """Analizuje plik z wynikami testów wyszukiwania."""
    results = defaultdict(dict)
    current_test_set = None
    current_query = None
    retrieved_docs = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("="):
                match = re.search(r"TEST SET: (.*) =", line)
                if match:
                    current_test_set = match.group(1).strip()
                    current_query = None
                    retrieved_docs = []
                continue

            if line.startswith("Query:"):
                if current_query and retrieved_docs:
                    results[current_query][current_test_set] = retrieved_docs
                current_query = line.replace("Query:", "").strip()
                retrieved_docs = []
                continue

            if line.startswith("Retrieved docs:"):
                continue
            
            if line.startswith("---"):
                if current_query and retrieved_docs:
                    results[current_query][current_test_set] = retrieved_docs
                current_query = None
                retrieved_docs = []
                continue

            if current_query and current_test_set:
                parts = line.split(' - ')
                if len(parts) == 2:
                    name = parts[0].strip()
                    try:
                        score = float(parts[1])
                        retrieved_docs.append((name, score))
                    except ValueError:
                        pass

    if current_query and retrieved_docs:
        results[current_query][current_test_set] = retrieved_docs

    return results

def generate_summary(results, output_file):
    """Generuje podsumowanie w formacie Markdown, porównujące wyniki dla każdego zapytania."""
    test_sets_order = [
        "CLS (Pełny kontekst)", "MEAN (Pełny kontekst)",
        "CLS (Tylko słowa kluczowe)", "MEAN (Tylko słowa kluczowe)"
    ]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Porównanie Wyników Wyszukiwania dla Poszczególnych Zapytań\n\n")
        queries = list(results.keys())

        for i, query in enumerate(queries, 1):
            f.write(f"## {i}. Zapytanie: \"{query}\"\n\n")
            f.write("| " + " | ".join(test_sets_order) + " |\n")
            f.write("|" + "---|" * len(test_sets_order) + "\n")

            max_rows = max((len(results[query].get(ts, [])) for ts in test_sets_order), default=0)

            for row_idx in range(max_rows):
                row_data = [f"{doc} ({score:.4f})" if row_idx < len(results[query].get(ts, [])) else "" for ts in test_sets_order for doc, score in [results[query].get(ts, [])[row_idx]]]
                f.write(f"| {' | '.join(row_data)} |\n")
            f.write("\n---\n\n")

    print(f"Podsumowanie zostało zapisane do pliku: {output_file}")

if __name__ == "__main__":
    parsed_data = parse_results("embedding_test_results_all_retrieval.txt")
    generate_summary(parsed_data, "query_by_query_summary.md")