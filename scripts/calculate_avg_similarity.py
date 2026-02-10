import re
import os
import numpy as np

def calculate_metrics(file_path, target_section_header):
    """
    Parses a result file and calculates the average similarity score 
    and standard deviation for a specific test section.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return 0.0, 0.0

    scores = []
    in_section = False
    
    # Regex to capture "Name - 0.1234"
    score_pattern = re.compile(r" - (\d+\.\d+)")
    # Regex to detect section headers
    # Supports both: "========== TEST SET: Name ==========" and "-------------------- Name --------------------"
    section_pattern = re.compile(r"[=\-]{10,}\s*(?:TEST SET:\s*)?(.*?)\s*[=\-]{10,}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Check for section change
            section_match = section_pattern.search(line)
            if section_match:
                current_header = section_match.group(1).strip()
                # Check if we are entering the target section
                if target_section_header in current_header:
                    in_section = True
                else:
                    in_section = False
                continue
            
            if in_section:
                # Extract score
                match = score_pattern.search(line)
                if match:
                    scores.append(float(match.group(1)))

    if not scores:
        return 0.0, 0.0
    
    return np.mean(scores), np.std(scores)

def main():
    # Define paths based on your project structure
    file_v1_all = "tests/results/embedding_test_results_all_retrieval.txt"
    file_v1_v2 = "tests/results/embedding_test_results_v1_vs_v2.txt"
    file_stella = "tests/results/embedding_test_results_all_stella_retrieval.txt"
    file_hyde = "tests/results/query_expansion_comparison_results.txt"
    
    # Define configurations to analyze
    configs = [
        # --- RoBERTa v1 (Wszystkie kombinacje) ---
        {
            "model": "RoBERTa-large (v1)",
            "strategy": "MEAN (Pełny kontekst)",
            "file": file_v1_all,
            "section": "MEAN (Pełny kontekst)"
        },
        {
            "model": "RoBERTa-large (v1)",
            "strategy": "CLS (Pełny kontekst)",
            "file": file_v1_all,
            "section": "CLS (Pełny kontekst)"
        },
        {
            "model": "RoBERTa-large (v1)",
            "strategy": "MEAN (Słowa kluczowe)",
            "file": file_v1_all,
            "section": "MEAN (Tylko słowa kluczowe)"
        },
        {
            "model": "RoBERTa-large (v1)",
            "strategy": "CLS (Słowa kluczowe)",
            "file": file_v1_all,
            "section": "CLS (Tylko słowa kluczowe)"
        },
        # --- Stella (Wszystkie kombinacje) ---
        {"model": "Stella", "strategy": "MEAN (Pełny kontekst)", "file": file_stella, "section": "MEAN (Pełny kontekst)"},
        {"model": "Stella", "strategy": "CLS (Pełny kontekst)", "file": file_stella, "section": "CLS (Pełny kontekst)"},
        {"model": "Stella", "strategy": "MEAN (Słowa kluczowe)", "file": file_stella, "section": "MEAN (Tylko słowa kluczowe)"},
        {"model": "Stella", "strategy": "CLS (Słowa kluczowe)", "file": file_stella, "section": "CLS (Tylko słowa kluczowe)"},
        
        # --- RoBERTa v2 (Tylko porównanie keywords) ---
        {"model": "RoBERTa-large-v2", "strategy": "CLS (Słowa kluczowe)", "file": file_v1_v2, "section": "CLS Words (v2: roberta-large-v2)"},

        # --- HyDE Comparison ---
        {"model": "HyDE Baseline", "strategy": "Raw Query", "file": file_hyde, "section": "Baseline (Raw Query)"},
        {"model": "HyDE Expanded", "strategy": "HyDE", "file": file_hyde, "section": "Expanded Query (HyDE)"},
    ]
    
    model_scores = {}

    print(f"{'Model':<20} | {'Strategy':<25} | {'Avg Similarity':<15} | {'Std Dev':<15}")
    print("-" * 85)
    for config in configs:
        avg_score, std_dev = calculate_metrics(config["file"], config["section"])
        strategy_name = config.get("strategy", "N/A")
        print(f"{config['model']:<20} | {strategy_name:<25} | {avg_score:.4f}          | {std_dev:.4f}")
        
        if config['model'] not in model_scores:
            model_scores[config['model']] = []
        model_scores[config['model']].append(avg_score) # Note: This only tracks avg_score for the summary

    print("-" * 85)
    print("ŚREDNIE ZBIORCZE DLA MODELI:")
    for model, scores in model_scores.items():
        print(f"{model:<20} | {np.mean(scores):.4f} (z {len(scores)} konfiguracji)")

if __name__ == "__main__":
    main()