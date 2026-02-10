import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Nazwa pliku z wynikami (możesz tu wpisać swoją długą nazwę pliku)
INPUT_FILE = "tests/results/embedding_test_results_all_retrieval.txt"

def parse_results_file(filepath):
    data = []
    current_model = None
    current_query_idx = 0
    
    # Wzorce regex
    model_pattern = re.compile(r"={10,} TEST SET: (.*?) ={10,}")
    query_pattern = re.compile(r"Query: (.*)")
    score_pattern = re.compile(r" - (\d+\.\d+)")

    if not os.path.exists(filepath):
        print(f"Błąd: Nie znaleziono pliku {filepath}")
        return pd.DataFrame()

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Wykrywanie modelu
            m_match = model_pattern.search(line)
            if m_match:
                current_model = m_match.group(1)
                current_query_idx = 0
                continue
            
            # Wykrywanie zapytania (inkrementacja indeksu)
            if query_pattern.match(line):
                current_query_idx += 1
                continue

            # Wykrywanie wyniku (score)
            s_match = score_pattern.search(line)
            if s_match and current_model:
                score = float(s_match.group(1))
                data.append({
                    "Model": current_model,
                    "Query_ID": current_query_idx,
                    "Score": score,
                    "Rank": "Top-5" # Uproszczenie, bierzemy wszystkie z top 5
                })
                
    df = pd.DataFrame(data)
    
    # Tłumaczenie nazw modeli na angielski (dla wykresów w pracy dyplomowej)
    translation_map = {
        "MEAN (Pełny kontekst)": "MEAN & Full Context",
        "CLS (Pełny kontekst)": "CLS & Full Context",
        "MEAN (Tylko słowa kluczowe)": "MEAN & Keywords",
        "CLS (Tylko słowa kluczowe)": "CLS & Keywords"
    }
    if not df.empty:
        df["Model"] = df["Model"].replace(translation_map)
        
    return df

def plot_boxplot(df):
    """Generuje wykres pudełkowy rozkładu podobieństwa."""
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Rysowanie boxplota
    ax = sns.boxplot(x="Model", y="Score", data=df, palette="Set2", showfliers=False)
    # Dodanie punktów (strip plot) dla lepszego wglądu w gęstość
    sns.stripplot(x="Model", y="Score", data=df, color=".25", size=2, alpha=0.5)

    plt.title("Distribution of Cosine Similarity Scores for Different Strategies", fontsize=22)
    plt.ylabel("Cosine Similarity Score", fontsize=20)
    plt.xlabel("Strategy", fontsize=20)
    plt.xticks(rotation=10, fontsize=18)
    plt.yticks(fontsize=18)
    
    plt.tight_layout()
    plt.savefig("similarity_boxplot.png", dpi=300)
    print("Wygenerowano: similarity_boxplot.png")
    plt.close()

def plot_heatmap(df):
    """Generuje mapę ciepła dla najlepszego wyniku (Top-1) dla każdego zapytania."""
    # Bierzemy tylko najlepszy wynik dla każdego zapytania (maksymalny score)
    # Zakładamy, że pierwszy wynik w pliku dla danego zapytania jest najlepszy, 
    # ale dla pewności weźmiemy max.
    top1_df = df.groupby(['Model', 'Query_ID'])['Score'].max().reset_index()
    
    # Pivot table: Wiersze=Zapytania, Kolumny=Modele
    pivot_df = top1_df.pivot(index="Query_ID", columns="Model", values="Score")
    
    plt.figure(figsize=(10, 8))
    
    # Rysowanie heatmapy
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Top-1 Similarity Score'})
    
    plt.title("Heatmap: Strategy Confidence for Individual Queries", fontsize=14)
    plt.ylabel("Query Number (1-20)", fontsize=12)
    plt.xlabel("Strategy", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("similarity_heatmap.png", dpi=300)
    print("Wygenerowano: similarity_heatmap.png")
    plt.close()

def main():
    print(f"Analiza pliku: {INPUT_FILE}")
    df = parse_results_file(INPUT_FILE)
    
    if df.empty:
        print("Nie udało się wczytać danych. Sprawdź nazwę pliku.")
        return

    print(f"Wczytano {len(df)} wyników.")
    
    # 1. Wykres pudełkowy (Porównanie ogólne)
    plot_boxplot(df)
    
    # 2. Mapa ciepła (Analiza szczegółowa)
    plot_heatmap(df)
    
    print("\nGotowe! Wykresy zostały zapisane.")

if __name__ == "__main__":
    main()