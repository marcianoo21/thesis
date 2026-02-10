import re
from collections import defaultdict

def parse_results(input_file):
    """
    Analizuje plik z wynikami testów wyszukiwania.
    Ta wersja zawiera poprawki błędów, aby poprawnie przetwarzać wszystkie wpisy.
    """
    results = defaultdict(lambda: defaultdict(list))
    current_test_set = None
    current_query = None

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("="):
            match = re.search(r"TEST SET: (.*) =", line)
            if match:
                current_test_set = match.group(1).strip()
                current_query = None  # Reset query when new test set starts
            continue

        if line.startswith("Query:"):
            current_query = line.replace("Query:", "").strip()
            # Ensure a list is ready for this query and test set
            if current_test_set not in results[current_query]:
                 results[current_query][current_test_set] = []
            continue

        if line.startswith("Retrieved docs:") or line.startswith("---"):
            continue

        if current_query and current_test_set:
            parts = line.split(' - ')
            if len(parts) == 2:
                name = parts[0].strip()
                try:
                    score = float(parts[1])
                    results[current_query][current_test_set].append((name, score))
                except (ValueError, IndexError):
                    # Ignore lines that don't parse correctly
                    pass
    return results

def generate_summary(results, output_file):
    """Generuje podsumowanie w formacie Markdown, porównujące wyniki dla każdego zapytania."""
    test_sets_order = [
        "CLS (Pełny kontekst)", "MEAN (Pełny kontekst)",
        "CLS (Tylko słowa kluczowe)", "MEAN (Tylko słowa kluczowe)"
    ]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Porównanie Wyników Wyszukiwania dla Poszczególnych Zapytań (Model Stella)\n\n")
        queries = list(results.keys())

        for i, query in enumerate(queries, 1):
            f.write(f"## {i}. Zapytanie: \"{query}\"\n\n")
            f.write("| " + " | ".join(test_sets_order) + " |\n")
            f.write("|" + "---|" * len(test_sets_order) + "\n")

            max_rows = max((len(results[query].get(ts, [])) for ts in test_sets_order), default=0)

            for row_idx in range(max_rows):
                row_data = []
                for ts in test_sets_order:
                    test_set_results = results[query].get(ts, [])
                    if row_idx < len(test_set_results):
                        doc, score = test_set_results[row_idx]
                        row_data.append(f"{doc} ({score:.4f})")
                    else:
                        row_data.append("")
                f.write(f"| {' | '.join(row_data)} |\n")
            f.write("\n---\n\n")

    print(f"Podsumowanie zostało zapisane do pliku: {output_file}")

if __name__ == "__main__":
    parsed_data = parse_results("tests/results/embedding_test_results_all_stella_retrieval.txt")
    generate_summary(parsed_data, "tests/results/query_by_query_summary_stella.md")