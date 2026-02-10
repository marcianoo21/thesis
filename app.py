from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import sys
import os

from src import create_rag_system, LocationService

app = Flask(__name__)
CORS(app)  # Pozwala na połączenie z plikiem HTML otwieranym lokalnie

# Zmienne globalne na system
rag_chain = None
search_and_rank = None
location_service = None

# Inicjalizacja systemu w przestrzeni globalnej
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
    price_level = data.get('price_level', 0)  # 0=Dowolna, 1=Tanie, 2=Średnie, 3=Drogie

    if not user_input:
        return jsonify({"response": "Proszę wpisać wiadomość."})

    # --- LOGIKA Z RUN_PIPELINE.PY ---
    
    # 1. Analiza intencji (LLM)
    try:
        analysis = rag_chain.analyze_user_intent(user_input)
    except Exception as e:
        print(f"Wyjątek podczas analizy LLM: {e}")
        analysis = None

    # NEW: Obsługa bezpośredniej odpowiedzi (chit-chat)
    if analysis and analysis.get("direct_response"):
        # Aktualizacja historii konwersacji dla spójności
        rag_chain.conversation_history.append({"role": "user", "content": user_input})
        rag_chain.conversation_history.append({"role": "assistant", "content": analysis["direct_response"]})
        return jsonify({"response": analysis["direct_response"]})

    if not analysis:
        # Fallback: Jeśli LLM nie działa (np. błąd 503), kontynuujemy bez analizy
        print("Błąd analizy LLM - używam trybu awaryjnego (surowe zapytanie + spaCy)")
        analysis = {}

    # Sprawdzenie intencji (czy to luźna rozmowa?)
    is_chitchat = analysis.get("intent") == "chitchat"

    # 2. Wykrywanie lokalizacji (uproszczone bez input() fallback)
    user_location = None
    
    # Wykrywamy lokalizację TYLKO jeśli to nie jest chitchat
    if not is_chitchat:
        detected_location_llm = analysis.get("location")
        if detected_location_llm:
            user_location = location_service.geocode(detected_location_llm)
            if not user_location:
                spacy_from_llm = location_service.extract_location_name(detected_location_llm)
                if spacy_from_llm:
                    user_location = location_service.geocode(spacy_from_llm)
        
        # Fallback: spaCy bezpośrednio na zapytaniu
        if not user_location:
            spacy_direct = location_service.extract_location_name(user_input)
            if spacy_direct:
                user_location = location_service.geocode(spacy_direct)

    # WAŻNE: Przekazanie wykrytej lokalizacji do systemu RAG
    if user_location:
        rag_chain.set_user_location(user_location)
    else:
        # Resetujemy lokalizację, aby nie używać tej z poprzedniego zapytania (bo rag_chain jest globalny)
        rag_chain.set_user_location(None)

    # 3. Parametry wyszukiwania
    expanded_query = analysis.get("search_query") or user_input
    cuisine_type = analysis.get("cuisine")
    
    # Obsługa suwaka cenowego (nadpisuje LLM)
    price_map = {1: "0-40", 2: "40-80", 3: "80-1000"}
    if price_level in price_map:
        price_preference = price_map[price_level]
        print(f"INFO: Wymuszam cenę z suwaka: {price_preference}")
    else:
        # Jeśli suwak na 0 (Dowolna), bierzemy to co wykrył LLM (lub None)
        price_preference = analysis.get("price")

    # 4. Ustalenie zapytania dla wyszukiwarki (HyDE)
    # Jeśli analiza zwróciła search_query, używamy go. Jeśli zwróciła null (None),
    # przekazujemy pusty string, aby ConversationalRAG wiedział, że ma NIE szukać (tryb chat).
    # Jeśli analysis było puste (błąd), przekazujemy None, aby ConversationalRAG sam spróbował wygenerować.
    search_query_override = None
    if is_chitchat:
        search_query_override = ""

    try:
        # Wymuszenie formatu odpowiedzi (tylko lista, bez konwersacji)
        final_input = user_input
        if search_query_override != "":
            final_input += " . Odpowiedz WYŁĄCZNIE listą znalezionych miejsc. Format: 1. **Nazwa** \n  Adres: [Adres] ([Odległość]) \n  Typ kuchni: [Max 3 typy po przecinku] \n  Ocena: [Ocena] \n  [Opis]. Nie dodawaj wstępu ani zakończenia."

        response_text = rag_chain.generate_response(
            final_input,
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
    # use_reloader=False zapobiega podwójnemu ładowaniu modelu przy starcie (raz dla serwera, raz dla debuggera)
    # PORT jest odczytywany z env (np. w Hugging Face Spaces); lokalnie domyślnie 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)