from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import sys
import os

# Importujemy Twoją logikę z istniejących plików
from conversational_rag import create_rag_system
from location_service import LocationService

app = Flask(__name__)
CORS(app)  # Pozwala na połączenie z plikiem HTML otwieranym lokalnie

# Zmienne globalne na system
rag_chain = None
search_and_rank = None
location_service = None

def init_system():
    """Inicjalizacja systemu przy starcie serwera"""
    global rag_chain, search_and_rank, location_service
    load_dotenv()
    print("--- Inicjalizacja serwera backendu ---")
    try:
        # Ścieżka domyślna z Twojego run_pipeline.py
        embedding_file = "output_files/lodz_restaurants_cafes_embeddings_cls_words.jsonl"
        
        location_service = LocationService()
        rag_chain, search_and_rank, _ = create_rag_system(
            embeddings_file=embedding_file
        )
        print("--- System gotowy do pracy ---")
    except Exception as e:
        print(f"Błąd inicjalizacji: {e}")

@app.route('/')
def index():
    """Serwowanie pliku HTML przy wejściu na stronę główną"""
    return send_file('chat_ui.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint odbierający wiadomości z frontend'u"""
    data = request.json
    user_input = data.get('message', '')

    if not user_input:
        return jsonify({"response": "Proszę wpisać wiadomość."})

    # --- LOGIKA Z RUN_PIPELINE.PY ---
    
    # 1. Analiza intencji (LLM)
    try:
        analysis = rag_chain.analyze_user_intent(user_input)
    except Exception as e:
        print(f"⚠️ Wyjątek podczas analizy LLM: {e}")
        analysis = None

    if not analysis:
        # Fallback: Jeśli LLM nie działa (np. błąd 503), kontynuujemy bez analizy
        print("⚠️ Błąd analizy LLM - używam trybu awaryjnego (surowe zapytanie + spaCy)")
        analysis = {}

    # 2. Wykrywanie lokalizacji (uproszczone bez input() fallback)
    user_location = None
    detected_location_llm = analysis.get("location")
    
    if detected_location_llm:
        user_location = location_service.geocode(detected_location_llm)
        if not user_location:
            spacy_from_llm = location_service.extract_location_name(detected_location_llm)
            if spacy_from_llm:
                user_location = location_service.geocode(spacy_from_llm)

    if not user_location:
        spacy_direct = location_service.extract_location_name(user_input)
        if spacy_direct:
            user_location = location_service.geocode(spacy_direct)

    # 3. Parametry wyszukiwania
    expanded_query = analysis.get("search_query") or user_input
    cuisine_type = analysis.get("cuisine")
    price_preference = analysis.get("price")

    # 4. Ustalenie zapytania dla wyszukiwarki (HyDE)
    # Jeśli analiza zwróciła search_query, używamy go. Jeśli zwróciła null (None),
    # przekazujemy pusty string, aby ConversationalRAG wiedział, że ma NIE szukać (tryb chat).
    # Jeśli analysis było puste (błąd), przekazujemy None, aby ConversationalRAG sam spróbował wygenerować.
    search_query_override = None
    if analysis:
        search_query_override = analysis.get("search_query")
        if search_query_override is None:
            search_query_override = "" # Jawny brak wyszukiwania (np. dla "cześć")

    # 5. Generowanie odpowiedzi przez Asystenta (RAG)
    try:
        response_text = rag_chain.generate_response(
            user_input,
            k=5,
            price_preference=price_preference,
            cuisine_filter=cuisine_type,
            search_query_override=search_query_override
        )
    except Exception as e:
        print(f"Błąd generowania odpowiedzi: {e}")
        return jsonify({"response": "Przepraszam, wystąpił błąd systemu."})

    # Formatowanie markdown/text na HTML (prosta zamiana nowych linii)
    response_html = response_text.replace("\n", "<br>")
    return jsonify({"response": response_html})

if __name__ == '__main__':
    init_system()
    app.run(port=5000, debug=True)