import re
import os

def parse_results(filename):
    """
    Parsuje plik z wynikami porównania HyDE i oblicza statystyki.
    """
    if not os.path.exists(filename):
        print(f"Błąd: Nie znaleziono pliku {filename}")
        return []

    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Dzielimy plik na sekcje zapytań
    queries = content.split('============================================================')
    
    stats = []
    
    for q_section in queries:
        if "QUERY" not in q_section:
            continue
            
        # Wyciągamy ID i treść zapytania
        query_match = re.search(r'QUERY (\d+): (.*)', q_section)
        if not query_match:
            continue
        q_id = int(query_match.group(1))
        q_text = query_match.group(2).strip()

        # Wyciągamy wyniki dla Baseline (Raw Query)
        # Szukamy sekcji między nagłówkiem Baseline a Expanded Query
        # Używamy bardziej elastycznego regexa
        baseline_match = re.search(r'Baseline \(Raw Query\).*?\n(.*?)\n\s*-+ Expanded Query', q_section, re.DOTALL)
        
        baseline_scores = []
        if baseline_match:
            baseline_lines = baseline_match.group(1).strip().split('\n')
            for line in baseline_lines:
                match = re.search(r'(\d+\.\d+)', line)
                if match:
                    baseline_scores.append(float(match.group(1)))
                
        # Wyciągamy wyniki dla HyDE (Expanded Query)
        # Szukamy sekcji po nagłówku Results with Expansion
        hyde_match = re.search(r'Results with Expansion:.*?\n(.*?)(?:\n={5,}|$)', q_section, re.DOTALL)
        
        hyde_scores = []
        if hyde_match:
            hyde_lines = hyde_match.group(1).strip().split('\n')
            for line in hyde_lines:
                match = re.search(r'(\d+\.\d+)', line)
                if match:
                    hyde_scores.append(float(match.group(1)))
        
        # Obliczamy średnie dla danego zapytania
        if baseline_scores and hyde_scores:
            avg_base = sum(baseline_scores) / len(baseline_scores)
            avg_hyde = sum(hyde_scores) / len(hyde_scores)
            diff = avg_hyde - avg_base
            
            stats.append({
                'id': q_id,
                'query_text': q_text,
                'avg_base': avg_base,
                'avg_hyde': avg_hyde,
                'diff': diff
            })
            
    return stats

def generate_latex_table(stats):
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Porównanie średnich wartości podobieństwa cosinusowego dla zapytań testowych: Baseline vs. HyDE.}")
    print("\\label{tab:hyde_comparison}")
    print("\\resizebox{\\textwidth}{!}{%")
    print("\\begin{tabular}{|c|l|c|c|c|}")
    print("\\hline")
    print("\\textbf{ID} & \\textbf{Treść zapytania (skrócona)} & \\textbf{\\begin{tabular}[c]{@{}c@{}}Baseline\\\\ (Avg Score)\\end{tabular}} & \\textbf{\\begin{tabular}[c]{@{}c@{}}HyDE\\\\ (Avg Score)\\end{tabular}} & \\textbf{\\begin{tabular}[c]{@{}c@{}}Różnica\\\\ (Gain)\\end{tabular}} \\\\ \\hline")

    total_base = 0
    total_hyde = 0
    total_diff = 0

    for s in stats:
        q_text = s['query_text']
        if len(q_text) > 50:
            q_text = q_text[:47] + "..."
        
        diff_sign = "+" if s['diff'] > 0 else ""
        print(f"{s['id']} & {q_text} & {s['avg_base']:.4f} & {s['avg_hyde']:.4f} & {diff_sign}{s['diff']:.4f} \\\\ \\hline")
        
        total_base += s['avg_base']
        total_hyde += s['avg_hyde']
        total_diff += s['diff']

    if stats:
        avg_base_all = total_base / len(stats)
        avg_hyde_all = total_hyde / len(stats)
        avg_diff_all = total_diff / len(stats)
        diff_sign_all = "+" if avg_diff_all > 0 else ""

        print("\\hline")
        print(f"\\multicolumn{{2}}{{|r|}}{{\\textbf{{ŚREDNIA DLA WSZYSTKICH ZAPYTAŃ:}}}} & \\textbf{{{avg_base_all:.4f}}} & \\textbf{{{avg_hyde_all:.4f}}} & \\textbf{{{diff_sign_all}{avg_diff_all:.4f}}} \\\\ \\hline")
    
    print("\\end{tabular}%")
    print("}")
    print("\\end{table}")

def main():
    input_file = 'tests/results/query_expansion_comparison_results.txt'
    stats = parse_results(input_file)
    
    if not stats:
        print("Nie znaleziono danych do przetworzenia.")
        return

    generate_latex_table(stats)

if __name__ == "__main__":
    main()
