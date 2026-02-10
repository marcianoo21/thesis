import re
import os
import numpy as np

# Złoty standard (Ground Truth)
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

def calculate_metrics(file_path, target_section_header, debug=False):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    # Regex
    section_pattern = re.compile(r"[=\-]{10,}\s*(?:TEST SET:\s*)?(.*?)\s*[=\-]{10,}")
    query_pattern = re.compile(r"Query: (.*)")
    doc_pattern = re.compile(r"(.*?) - \d+\.\d+")

    current_query = None
    retrieved_docs = []
    in_target_section = False
    
    # Metrics accumulators
    hits_list = []
    mrr_list = []
    precision_list = []
    query_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Section detection
            section_match = section_pattern.search(line)
            if section_match:
                header = section_match.group(1).strip()
                
                # Jeśli kończymy sekcję docelową, musimy przetworzyć ostatnie wiszące zapytanie
                if in_target_section and current_query and retrieved_docs:
                    hits, mrr, precision = evaluate_query(current_query, retrieved_docs, debug)
                    hits_list.append(hits)
                    mrr_list.append(mrr)
                    precision_list.append(precision)
                    query_count += 1
                    current_query = None
                    retrieved_docs = []

                if target_section_header in header:
                    in_target_section = True
                else:
                    in_target_section = False
                continue
            
            if not in_target_section:
                continue

            # Query detection
            query_match = query_pattern.match(line)
            if query_match:
                # Process previous query if exists
                if current_query and retrieved_docs:
                    hits, mrr, precision = evaluate_query(current_query, retrieved_docs, debug)
                    hits_list.append(hits)
                    mrr_list.append(mrr)
                    precision_list.append(precision)
                    query_count += 1
                
                current_query = query_match.group(1).strip()
                retrieved_docs = []
                continue
            
            # Doc detection
            doc_match = doc_pattern.match(line)
            if doc_match:
                doc_name = doc_match.group(1).strip()
                retrieved_docs.append(doc_name)

        # Process last query (if EOF reached while inside target section)
        if in_target_section and current_query and retrieved_docs:
            hits, mrr, precision = evaluate_query(current_query, retrieved_docs, debug)
            hits_list.append(hits)
            mrr_list.append(mrr)
            precision_list.append(precision)
            query_count += 1

    if query_count == 0:
        return None

    return {
        "Hit Rate@5": (np.mean(hits_list) * 100, np.std(hits_list) * 100),
        "MRR": (np.mean(mrr_list), np.std(mrr_list)),
        "Precision@5": (np.mean(precision_list) * 100, np.std(precision_list) * 100)
    }

def evaluate_query(query, retrieved_docs, debug=False, k=5):
    ground_truth = GROUND_TRUTH.get(query, [])
    if not ground_truth:
        if debug:
            print(f"[WARN] Brak Ground Truth dla zapytania: '{query}'")
        return 0, 0.0, 0.0

    top_k = retrieved_docs[:k]
    
    # Hit Rate (at least one relevant)
    is_hit = 0
    first_relevant_rank = 0
    relevant_count = 0
    
    for rank, doc_name in enumerate(top_k, 1):
        is_relevant = any(gt.lower() in doc_name.lower() for gt in ground_truth)
        if is_relevant:
            if is_hit == 0:
                is_hit = 1
                first_relevant_rank = rank
            relevant_count += 1
            
    mrr = 1.0 / first_relevant_rank if first_relevant_rank > 0 else 0.0
    precision = relevant_count / k
    
    if debug:
        print(f"Query: {query[:50]}...")
        print(f"  -> Hits: {relevant_count}/{k} | MRR: {mrr:.2f}")
    
    return is_hit, mrr, precision

def main():
    configs = [
        {
            "model": "RoBERTa-large-v2",
            "file": "tests/results/embedding_test_results_v1_vs_v2.txt",
            "section": "CLS Words (v2: roberta-large-v2)"
        },
        {
            "model": "Stella",
            "file": "tests/results/embedding_test_results_all_stella_retrieval.txt",
            "section": "CLS (Tylko słowa kluczowe)"
        },
        # Dodajmy też v1 (nasz główny model) dla porównania
        {
            "model": "RoBERTa-large (v1)",
            "file": "tests/results/embedding_test_results_all_retrieval.txt",
            "section": "CLS (Tylko słowa kluczowe)"
        }
    ]

    print(f"{'Model':<20} | {'Hit Rate@5':<18} | {'MRR':<18} | {'Precision@5':<20}")
    print("-" * 85)
    
    for config in configs:
        # Włączamy debugowanie, aby widzieć co się dzieje
        # print(f"\n--- Analiza dla: {config['model']} ---")
        metrics = calculate_metrics(config["file"], config["section"], debug=False)
        # print("-----------------------------------")
        
        if metrics:
            hr_mean, hr_std = metrics['Hit Rate@5']
            mrr_mean, mrr_std = metrics['MRR']
            prec_mean, prec_std = metrics['Precision@5']
            
            print(f"{config['model']:<20} | {hr_mean:.2f}% (±{hr_std:.2f})    | {mrr_mean:.4f} (±{mrr_std:.4f}) | {prec_mean:.2f}% (±{prec_std:.2f})")
        else:
            print(f"{config['model']:<20} | N/A          | N/A      | N/A")

if __name__ == "__main__":
    main()