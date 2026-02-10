"""
conversational_rag.py

Konwersacyjny system RAG dla rekomendacji restauracji.
Używa modelu PLLuM do naturalnej konwersacji.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from dotenv import load_dotenv
from math import radians, sin, cos, sqrt, atan2, log1p
from .location_service import LocationService
import urllib.parse

load_dotenv()

def distance_km(lat1, lon1, lat2, lon2):
    """Oblicza odległość w km między dwoma punktami GPS."""
    if None in [lat1, lon1, lat2, lon2]:
        return float('inf')  # Zwróć nieskończoność, jeśli brakuje koordynatów

    R = 6371  # Promień Ziemi w km
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c



class PLLuMLLM:
    """Klient LLM używający modelu PLLuM przez Hugging Face Inference API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicjalizacja klienta PLLuM.
        
        Args:
            api_key: Klucz API Hugging Face (jeśli None, pobiera z env)
        """
        from huggingface_hub import InferenceClient
        
        self.api_key = api_key or os.getenv('HF_TOKEN')
        if not self.api_key:
            raise ValueError("Brak HF_TOKEN! Ustaw zmienną środowiskową lub przekaż jako argument.")
        
        self.client = InferenceClient(api_key=self.api_key)
        self.model = "CYFRAGOVPL/PLLuM-12B-nc-chat"
        
        print(f"Zainicjalizowano PLLuM model: {self.model}")
    
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """
        Generuje odpowiedź na podstawie historii konwersacji.
        
        Args:
            messages: Lista wiadomości w formacie [{"role": "user/assistant", "content": "..."}]
            max_tokens: Maksymalna długość odpowiedzi
            temperature: Temperatura generowania (0.0-1.0)
        
        Returns:
            Wygenerowana odpowiedź
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["Użytkownik:", "User:", "\nUżytkownik", "\nUser"]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Błąd podczas generowania odpowiedzi: {e}")
            return "Przepraszam, wystąpił problem z wygenerowaniem odpowiedzi."


class ConversationalRAG:
    """
    System konwersacyjny RAG dla rekomendacji restauracji.
    Łączy wyszukiwanie semantyczne z naturalną konwersacją.
    """
    
    def __init__(
        self,
        llm_client: PLLuMLLM,
        search_function: Callable,
        max_history: int = 10,
        system_prompt: Optional[str] = None
    ):
        """
        Inicjalizacja systemu RAG.
        
        Args:
            llm_client: Klient LLM (PLLuM)
            search_function: Funkcja wyszukiwania restauracji
            max_history: Maksymalna liczba par w historii
            system_prompt: Opcjonalny własny prompt systemowy
        """
        self.llm = llm_client
        self.search = search_function
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
        self.user_preferences: Dict[str, Any] = {}
        self.user_location: Optional[tuple] = None
        
        # Domyślny prompt systemowy
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.location_service = LocationService()
    
    def _default_system_prompt(self) -> str:
        """Zwraca domyślny prompt systemowy."""
        return """Jesteś asystentem rekomendacji restauracji, kawiarni, barów i innych miejsc gastronomicznych w Łodzi.

Twoja rola jest ściśle ograniczona:
- zajmujesz się WYŁĄCZNIE jedzeniem i miejscami gastronomicznymi w Łodzi,
- NIE pomagasz w planowaniu podróży, życiu osobistym, pracy, finansach ani innych tematach niezwiązanych z gastronomią,
- jeśli użytkownik pyta o coś spoza jedzenia w Łodzi, uprzejmie wyjaśnij, że możesz pomagać tylko w wyborze miejsc do jedzenia w Łodzi
  i zaproś go do doprecyzowania preferencji gastronomicznych.

Twoim zadaniem jest:
- interpretować potrzeby użytkownika
- korzystać z dostarczonych wyników wyszukiwania
- prezentować rekomendacje w czytelny i atrakcyjny sposób

WAŻNE ZASADY PREZENTACJI:
1. Odpowiadaj konkretnie i na temat.
2. Przedstawiaj rekomendacje jako ponumerowaną listę.
3. Dla każdego miejsca podaj:
   - Nazwę
   - Adres
   - Ocenę Google (jeśli dostępna)
   - klimat miejsca
4. Nie wymyślaj informacji - bazuj tylko na dostarczonych danych.
5. Używaj naturalnego, pomocnego języka polskiego.

Przykład formatu:
1. **Nazwa Restauracji**
    Adres: Ulica 123
    Ocena: 4.5/5.0
    To miejsce idealne na randkę, serwujące świetną pizzę w przytulnej atmosferze.

Jeśli nie ma wyników, zapytaj o inne preferencje."""
    
    def analyze_user_intent(self, user_message: str) -> Dict[str, Any]:
        """
        Analizuje intencję użytkownika w jednym zapytaniu (oszczędność API).
        Zwraca słownik z polami: location, cuisine, price, search_query.
        """
        history_text = ""
        if self.conversation_history:
             # Bierzemy ostatnie 4 wymiany zdań dla kontekstu
             recent = self.conversation_history[-4:] 
             cleaned_recent = []
             for m in recent:
                 role = m['role']
                 content = m['content']
                 # Usuwamy HTML z historii, aby model nie próbował go naśladować zamiast zwracać JSON
                 if role == 'assistant' and ('<' in content or '&lt;' in content):
                     content = "[Asystent wyświetlił listę rekomendacji]"
                 cleaned_recent.append(f"{role}: {content}")
             history_text = "HISTORIA ROZMOWY:\n" + "\n".join(cleaned_recent) + "\n\n"

        prompt = [
            {
                "role": "system",
                "content": """Jesteś analitykiem intencji użytkownika w systemie rekomendacji restauracji w Łodzi.
Twoim zadaniem jest klasyfikacja intencji i ekstrakcja kluczowych informacji do formatu JSON.

Analizuj CAŁĄ historię rozmowy, aby zrozumieć kontekst (np. jeśli użytkownik wcześniej pytał o sushi, a teraz pisze "w centrum", to szuka SUSHI w centrum).

Zwróć obiekt JSON z polami:
1. "intent": "recommendation" (szukanie lokalu, preferencje, pytania o jedzenie) LUB "chitchat" (powitanie, podziękowanie, luźna rozmowa bez celu wyszukiwania).
2. "location": Wykryta lokalizacja w Łodzi (mianownik). Normalizuj nazwy potoczne (np. "Manu" -> "Manufaktura", "Pietryna" -> "Ulica Piotrkowska", "Polibuda" -> "Politechnika Łódzka"). Jeśli brak lub "wszędzie"/"obojętnie" -> null.
3. "cuisine": Typ kuchni/dania (np. "włoska", "sushi"). Jeśli nie podano w bieżącym zapytaniu, POBIERZ Z HISTORII.
4. "price": "0-40", "40-80", "80-1000" lub null.

ZASADY:
- "wszędzie", "obojętnie" w kontekście lokalizacji oznacza intent="recommendation", location=null.
- Ignoruj HTML w historii.
- Zwróć TYLKO czysty JSON.

PRZYKŁADY:
User: "Cześć, co tam?"
JSON:
{
  "intent": "chitchat",
  "location": null,
  "cuisine": null,
  "price": null
}

User: "Szukam pizzy na Widzewie"
JSON:
{
  "intent": "recommendation",
  "location": "Widzew",
  "cuisine": "pizza",
  "price": null
}

User: "wszędzie" (w kontekście wcześniejszego pytania o sushi)
JSON:
{
  "intent": "recommendation",
  "location": null,
  "cuisine": "sushi",
  "price": null
}
"""
            },
            {"role": "user", "content": history_text + "AKTUALNE ZAPYTANIE: " + user_message}
        ]
        
        # Zwiększamy nieco max_tokens, aby model nie uciął JSON-a
        response = self.llm.generate(prompt, max_tokens=350, temperature=0.1)
        
        try:
            # Próba znalezienia JSON-a w tekście za pomocą regex (obsługa tekstu przed/po JSON)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                clean_response = json_match.group(0)
                return json.loads(clean_response)
            
            # Fallback: proste czyszczenie markdown
            clean_response = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_response)
        except json.JSONDecodeError:
            # POPRAWKA: Sprawdzenie, czy to nie jest zwykła rozmowa (chit-chat)
            # Jeśli model zamiast JSON-a zwrócił uprzejmą odpowiedź, wykrywamy to tutaj.
            is_conversational = (
                "{" not in response and 
                len(response) < 200
            )
            if is_conversational:
                return {"direct_response": response.strip()}
                
            print(f"BŁĄD: Nie udało się sparsować JSON z LLM. Odpowiedź: {response}")
            return {}

    def extract_search_query(self, user_message: str) -> Optional[str]:
        """
        Inteligentne rozszerzanie zapytania (Query Expansion).
        Zamienia intencję użytkownika na optymalne zapytanie dla bazy wektorowej.
        Generuje opis idealnej restauracji na podstawie intencji użytkownika (HyDE).
        """
        # Budowanie kontekstu z historii dla lepszego zrozumienia intencji (np. "w centrum" -> "sushi w centrum")
        history_text = ""
        if self.conversation_history:
             recent = self.conversation_history[-2:] 
             cleaned_recent = []
             for m in recent:
                 role = "Asystent" if m['role'] == 'assistant' else "Użytkownik"
                 content = m['content']
                 if role == 'Asystent' and ('<' in content or '&lt;' in content):
                     content = "[Lista rekomendacji]"
                 cleaned_recent.append(f"{role}: {content}")
             history_text = "KONTEKST ROZMOWY:\n" + "\n".join(cleaned_recent) + "\n\n"

        prompt_content = history_text + "AKTUALNE ZAPYTANIE: " + user_message

        extraction_prompt = [
            {
                "role": "system",
                "content": """Twoim zadaniem jest stworzenie hipotetycznego opisu idealnej restauracji pasującej do zapytania użytkownika.
NIE wymyślaj nazwy restauracji, konkretnego adresu ani oceny. NIE generuj listy miejsc. NIE pisz wstępów.
Skup się wyłącznie na typie kuchni, atmosferze i ofercie.

Wymagany format:
"[Typ kuchni/lokalu]. Oferta: [dania/napoje]. Charakter miejsca: [atmosfera/zastosowanie]."

Unikaj wymieniania konkretnych nazw potraw (np. "rosół", "schabowy"), chyba że użytkownik wyraźnie o nie pyta. Używaj ogólnych kategorii (np. "zupy", "dania mięsne", "kuchnia polska").
W ofercie wymień MAKSYMALNIE 3 kategorie.

Przykłady:
User: "Szukam taniej pizzy na randkę w centrum"
Assistant: "pizzeria, kuchnia włoska. Oferta: Pizza. Charakter miejsca: Romantyczna atmosfera, Kameralne."

User: "Gdzie na szybką kawę i ciastko?"
Assistant: "kawiarnia, cukiernia. Oferta: Kawa, Dobre desery, Szybka przekąska. Charakter miejsca: Niezobowiązująca atmosfera."

User: "jak sie masz?"
Assistant: "BRAK"

User: "klimatyczne miejsce z jedzeniem azjatyckim w okolicach galerii łódzkiej"
Assistant: "kuchnia azjatycka, restauracja orientalna. Oferta: Dania kuchni azjatyckiej. Charakter miejsca: Klimatyczne, Niezobowiązująca atmosfera."

Jeśli użytkownik nie szuka jedzenia/lokalu (np. "Cześć", "Co tam?"), zwróć "BRAK"."""
            },
            {
                "role": "user",
                "content": prompt_content
            }
        ]
        
        query = self.llm.generate(extraction_prompt, max_tokens=100, temperature=0.4)
        query = query.strip().replace('"', '')
        
        if "BRAK" in query.upper() or len(query) < 2:
            return None
            
        print(f"DEBUG: Wygenerowany hipotetyczny kontekst: '{query}'")
        return query

    def normalize_location(self, user_message: str) -> Optional[str]:
        """
        Tłumaczy potoczne określenia lokalizacji na nazwy geograficzne zrozumiałe dla geokodera.
        """
        prompt = [
            {
                "role": "system",
                "content": """Jesteś ekstraktorem danych. Twoim zadaniem jest wyciągnięcie nazwy lokalizacji (ulica, dzielnica, punkt orientacyjny, centrum handlowe) z tekstu użytkownika.

Jesteś ekstraktorem lokalizacji w MIEŚCIE ŁÓDŹ.

Zwracaj TYLKO:
- dzielnice Łodzi
- ulice w Łodzi
- znane punkty orientacyjne w Łodzi
- centra handlowe w Łodzi

ZASADY:
1. Zamieniaj potoczne nazwy na oficjalne
2. Zwróć wyłącznie nazwę własną
3. Jeśli lokalizacja jest niejednoznaczna → zwróć BRAK
4. Jeśli lokalizacja NIE JEST w Łodzi → zwróć BRAK

Zasady:
1. Zamieniaj potoczne nazwy na oficjalne (np. "koło manu" -> "Manufaktura", "przy polibudzie" -> "Politechnika Łódzka", "pietryna" -> "Ulica Piotrkowska").
2. Zwróć TYLKO nazwę lokalizacji. Nie pisz pełnych zdań. Nie dodawaj "Lokalizacja to...".
3. Jeśli w wypowiedzi nie ma wskazania lokalizacji, zwróć słowo "BRAK".

Przykłady:
User: "blisko manufaktury" -> Assistant: "Manufaktura"
User: "okolice dworca fabrycznego" -> Assistant: "Dworzec Łódź Fabryczna"
User: "na teofilowie" -> Assistant: "Teofilów"
User: "klimatyczne miejsce z jedzeniem azjatyckim w okolicach galerii łódzkiej" -> Assistant: "Galeria Łódzka"
User: "blisko galerii" -> Assistant: "Galeria Łódzka"
User: "centrum" -> Assistant: "Śródmieście"
User: "zjem pizzę" -> Assistant: "BRAK" """
            },
            {"role": "user", "content": user_message}
        ]
        response = self.llm.generate(prompt, max_tokens=20, temperature=0.1).strip().replace('"', '').replace("Assistant: ", "")
        if response.endswith('.'): response = response[:-1]
        return None if "BRAK" in response.upper() or len(response) < 2 or len(response) > 50 else response

    def extract_cuisine_type(self, user_message: str) -> Optional[str]:
        """
        Ekstrahuje typ kuchni z zapytania użytkownika w celu ścisłego filtrowania.
        """
        prompt = [
            {
                "role": "system",
                "content": """Jesteś narzędziem do ekstrakcji danych. Twoim zadaniem jest zidentyfikować JEDEN, główny typ kuchni lub dania z zapytania użytkownika.

ZASADY:
1. Zwróć DOKŁADNIE JEDNO słowo lub frazę w mianowniku (np. "pizza", "kuchnia azjatycka", "sushi", "burger").
2. Jeśli użytkownik poda kilka typów, wybierz ten, który wydaje się najważniejszy.
3. Jeśli w zapytaniu nie ma mowy o jedzeniu, zwróć DOKŁADNIE słowo "BRAK".
4. NIE odpowiadaj zdaniami. NIE dodawaj wyjaśnień. Zwróć tylko nazwę kuchni lub "BRAK".

Przykłady:
User: "Szukam dobrej pizzy" -> Assistant: "pizza"
User: "Gdzie na dobrego burgera w centrum?" -> Assistant: "burger"
User: "kuchnia amerykańska, jakies burgery" -> Assistant: "amerykańska"
User: "Chcę zjeść coś azjatyckiego, może ramen?" -> Assistant: "azjatycka"
User: "Cześć, jak się masz?" -> Assistant: "BRAK"
User: "Klimatyczne miejsce na kolację" -> Assistant: "BRAK" """
            },
            {"role": "user", "content": user_message}
        ]
        response = self.llm.generate(prompt, max_tokens=10, temperature=0.0).strip().replace('"', '').replace("Assistant: ", "").replace(".", "")
        return None if "BRAK" in response.upper() or len(response) < 2 else response

    def normalize_price(self, user_message: str) -> Optional[str]:
        """
        Tłumaczy intencje cenowe użytkownika na konkretne przedziały liczbowe.
        """
        prompt = [
            {
                "role": "system",
                "content": """Jesteś narzędziem do ekstrakcji danych. Zinterpretuj preferencje cenowe użytkownika i zwróć TYLKO jeden z trzech przedziałów: '0-40', '40-80', '80-1000' lub słowo "BRAK".

ZASADY:
1. Tanie/studenckie/niedrogie/budżetowe/street food/$ -> zwróć '0-40'
2. Średnie/umiarkowane/$$ -> zwróć '40-80'
3. Drogie/ekskluzywne/luksusowe/fine dining/$$$ -> zwróć '80-1000'
4. Jeśli nie ma preferencji cenowej, zwróć DOKŁADNIE słowo "BRAK".
5. NIE odpowiadaj zdaniami. NIE dodawaj wyjaśnień.

Przykłady:
User: "szukam czegoś taniego dla studenta" -> Assistant: "0-40"
User: "dobre burgery w centrum" -> Assistant: "BRAK"
User: "ekskluzywna restauracja na rocznicę" -> Assistant: "80-1000"
User: "kuchnia amerykańska, jakies burgery w okolicach manufaktury" -> Assistant: "BRAK"
User: "coś w średniej cenie" -> Assistant: "40-80" """

            },
            {"role": "user", "content": user_message}
        ]
        response = self.llm.generate(prompt, max_tokens=10, temperature=0.0).strip().replace('"', '').replace("Assistant: ", "").replace(".", "")
        return None if "BRAK" in response.upper() else response
    
    def _format_opening_hours_html(self, opening_hours_data: Dict) -> str:
        """Formatuje godziny otwarcia do HTML <details>."""
        if not opening_hours_data or not isinstance(opening_hours_data, dict):
            return ""
        
        days_order = ["poniedziałek", "wtorek", "środa", "czwartek", "piątek", "sobota", "niedziela"]
        current_day_idx = datetime.now().weekday()
        lines = []
        for i, day in enumerate(days_order):
            # Pobieramy wartość dla danego dnia (obsługa małych/wielkich liter)
            val = opening_hours_data.get(day) or opening_hours_data.get(day.capitalize())
            if val:
                line = f"<b>{day.capitalize()}:</b> {val}"
                if i == current_day_idx:
                    line = f"<span style='color: #e67e22; font-weight: bold;'>{line}</span>"
                    line = f"<span style='color: #e67e22; font-weight: bold;'>{line} (Dzisiaj)</span>"
                lines.append(line)
        
        if not lines:
            return ""
        
        # Dodanie statusu Otwarte/Zamknięte w nagłówku
        is_open = _is_open_now(opening_hours_data)
        status_html = "<span style='color: #28a745; font-weight: bold;'> (Otwarte)</span>" if is_open else "<span style='color: #dc3545; font-weight: bold;'> (Zamknięte)</span>"

        content = "\n".join(lines)
        return f"<details style='background-color: transparent;'><summary>Godziny otwarcia{status_html}</summary><div class='hours-content'>\n{content}\n</div></details>"

    def _format_search_results(self, results: List[Dict]) -> str:
        """Formatuje wyniki wyszukiwania bezpośrednio dla użytkownika."""
        if not results:
            return ""
        
        # Logowanie szczegółowych informacji do konsoli (dla inżyniera)
        print("\n--- SZCZEGÓŁY ZNALEZIONYCH MIEJSC ---")
        for i, r in enumerate(results[:5], 1):
            print(f"{i}. {r['name']} (Score: {r.get('final_score', r.get('semantic_score', 0.0)):.2f})")
            print(f"   Adres pełny: {r.get('address')}")
            print(f"   Opis: {r.get('context', '')[:100]}...")
        print("---------------------------------------\n")

        formatted = ""
        for i, r in enumerate(results[:5], 1):
            # Używamy tagów HTML <b>, ponieważ app.py nie parsuje markdowna **
            formatted += f"{i}. <b>{r['name']}</b>\n"
            
            types = r.get('type', [])
            if types:
                types_str = ", ".join(types) if isinstance(types, list) else str(types)
                formatted += f"   <b>Kuchnia/Typ:</b> {types_str}\n"
            
            address = r.get('address')
            if address:
                # Inteligentne skracanie adresu
                parts = [p.strip() for p in address.split(',')]
                # Jeśli pierwsza część to tylko numer (np. "26, Kasprzaka..."), bierzemy też drugą
                if len(parts) > 1 and (parts[0].isdigit() or len(parts[0]) < 3):
                    simple_address = f"{parts[1]} {parts[0]}" # Format: Ulica Numer
                else:
                    simple_address = parts[0]
                
                # Link do Google Maps
                encoded_addr = urllib.parse.quote(f"{simple_address}, Łódź")
                formatted += f"   <b>Adres:</b> <a href='https://www.google.com/maps/search/?api=1&query={encoded_addr}' target='_blank' style='text-decoration: none; color: #0366d6;'>{simple_address}</a>"
                
                dist = r.get('distance_km')
                if dist is not None and dist != float('inf'):
                    formatted += f" <span style='color: #6c757d; font-size: 0.9em;'>({dist:.2f} km)</span>"
                formatted += "\n"
            
            rating = r.get('google_rating')
            reviews = r.get('google_reviews_total')
            if rating:
                color = "#28a745"
                rating_display = str(rating)
                try:
                    rating_float = float(rating)
                    if rating_float < 4.0: color = "#dc3545"
                    elif rating_float < 4.5: color = "#fd7e14"
                    rating_display = f"{rating_float:.1f}"
                except: pass
                formatted += f"   <b>Ocena:</b> <span style='color: {color}; font-weight: bold;'>{rating_display}/5.0</span> ({reviews} opinii)\n"
            
            price = r.get('google_price_range')
            if price:
                # Nie dodawaj "zł" jeśli cena jest w formacie $
                if '$' in str(price):
                    formatted += f"   <b>Cena:</b> {price}\n"
                else:
                    formatted += f"   <b>Cena:</b> {price} zł\n"
            
            # Dodanie godzin otwarcia
            opening_hours = r.get('opening_hours')
            if opening_hours:
                formatted += self._format_opening_hours_html(opening_hours) + "\n"
            
            # Usunięto opis (kontekst) i dopasowanie z widoku użytkownika
            
        return formatted
    
    def generate_response(self, user_message: str, k: int = 5, price_preference: Optional[str] = None, cuisine_filter: Optional[str] = None, search_query_override: Optional[str] = None) -> str:
        """
        Generuje odpowiedź łącząc konwersację LLM z wynikami wyszukiwania (bez ingerencji LLM w wyniki).
        
        Args:
            user_message: Wiadomość od użytkownika
            k: Liczba wyników do wyszukania
            price_preference: Preferencje cenowe
            cuisine_filter: Filtr kuchni
            search_query_override: Opcjonalne wymuszenie zapytania wyszukiwania
        
        Returns:
            Odpowiedź asystenta
        """
        # 1. Określenie zapytania wyszukiwania
        if search_query_override is not None:
             search_query = search_query_override
        else:
             search_query = self.extract_search_query(user_message)
        
        # Automatyczne wykrywanie lokalizacji i geokodowanie
        if not self.user_location:
            detected_loc = self.normalize_location(user_message)
            if detected_loc:
                coords = self.location_service.geocode(detected_loc)
                if coords:
                    self.set_user_location(coords)

        search_results = []
        relaxed_constraints = False # Flaga, czy poluzowaliśmy kryteria

        # 1a. Sprawdzenie lokalizacji (interakcja z użytkownikiem)
        if search_query and not self.user_location:
            # Sprawdzamy czy użytkownik wyraźnie zrezygnował z lokalizacji
            skip_keywords = ["wszędzie", "obojętnie", "cała łódź", "całej łodzi", "bez znaczenia", "nie ważne", "nieważne", "gdziekolwiek", "wszedzie"]
            user_msg_lower = user_message.lower().strip()
            
            # Jeśli użytkownik nie podał lokalizacji i nie napisał, że mu to obojętne -> Pytamy.
            # (Sprawdzamy też "nie" jako samodzielne słowo lub początek zdania np. "nie mam preferencji")
            is_no = user_msg_lower in ["nie", "nie."] or user_msg_lower.startswith("nie ")
            
            # Obsługa odpowiedzi twierdzącej (użytkownik chce podać lokalizację, ale jeszcze jej nie wpisał)
            is_yes = user_msg_lower in ["tak", "tak.", "poproszę", "chcę", "jasne", "pewnie"] or user_msg_lower.startswith("tak ")
            if is_yes:
                return "Gdzie dokładnie mam szukać? Podaj dzielnicę, ulicę lub punkt orientacyjny."

            if not (is_no or any(k in user_msg_lower for k in skip_keywords)):
                return "Czy szukasz lokalu w konkretnej części Łodzi, rejonie lub obok jakiegoś miejsca? Jeśli tak, napisz gdzie (np. 'blisko Manufaktury'). Jeśli nie, napisz 'wszędzie'."

        if search_query:
            try:
                search_results = self.search(
                    search_query, k=k, user_location=self.user_location,
                    price_preference=price_preference,
                    cuisine_filter=cuisine_filter
                )
                
                # SMART FALLBACK: Jeśli brak wyników, a była preferencja ceny, spróbuj bez niej
                if not search_results and price_preference:
                    print(f"INFO: Brak wyników dla ceny '{price_preference}'. Próbuję znaleźć cokolwiek...")
                    search_results = self.search(
                        search_query, k=k, user_location=self.user_location,
                        price_preference=None, # Usuwamy filtr ceny
                        cuisine_filter=cuisine_filter
                    )
                    if search_results:
                        relaxed_constraints = True

            except Exception as e:
                print(f"Błąd wyszukiwania: {e}")

        # 2. Generowanie odpowiedzi
        if search_results:
            # A. Formatowanie wyników (Python) - czysta lista bez ingerencji LLM
            results_text = self._format_search_results(search_results)
            
            # B. Zwracamy TYLKO wyniki, bez wstępu LLM (zgodnie z życzeniem)
            if relaxed_constraints:
                response = "Nie znalazłem miejsc w tej cenie, ale oto inne propozycje:\n\n" + results_text
            else:
                response = results_text
            
        else:
            # Brak wyników lub chitchat - LLM przejmuje pałeczkę
            messages = [{"role": "system", "content": self.system_prompt}]
            history_start = max(0, len(self.conversation_history) - self.max_history)
            messages.extend(self.conversation_history[history_start:])
            messages.append({"role": "user", "content": user_message})
            
            response = self.llm.generate(messages, max_tokens=300, temperature=0.7)
            
            # Czyszczenie odpowiedzi LLM
            for stop_phrase in ["Użytkownik:", "User:", "System:", "\nUżytkownik", "\nUser"]:
                if stop_phrase in response:
                    response = response.split(stop_phrase)[0].strip()
        
        # Zaktualizuj historię
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def clear_history(self):
        """Czyści historię konwersacji."""
        self.conversation_history = []
        self.user_preferences = {}
        print("Historia konwersacji wyczyszczona.")
    
    def reset_history(self):
        """Alias for clear_history."""
        self.clear_history()
    
    def export_conversation(self, filepath: str):
        """Eksportuje konwersację do pliku JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "history": self.conversation_history,
            "preferences": self.user_preferences
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Konwersacja zapisana do: {filepath}")

    def set_user_location(self, location: Optional[tuple]):
        """Ustawia lub resetuje lokalizację użytkownika."""
        self.user_location = location
        print(f"INFO: Ustawiono lokalizację użytkownika na: {location}")
    
    def load_conversation(self, filepath: str):
        """Wczytuje konwersację z pliku JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.conversation_history = data.get("history", [])
        self.user_preferences = data.get("preferences", {})
        
        print(f"Konwersacja wczytana z: {filepath}")

    def summarize_recommendations(self, results: List[Dict], user_query: str) -> str:
        """
        Generuje podsumowanie rekomendacji przy użyciu LLM, wyjaśniając dlaczego pasują.
        """
        if not results:
            return "Nie znaleziono pasujących miejsc."

        context_text = self._format_search_results(results)
        
        prompt = [
            {"role": "system", "content": "Jesteś ekspertem kulinarnym. Twoim zadaniem jest krótko podsumować znalezione restauracje i doradzić użytkownikowi, którą wybrać, bazując na ich opisach (polu Kontekst). Odnoś się do konkretnych cech miejsc."},
            {"role": "user", "content": f"Użytkownik szukał: '{user_query}'.\n\nOto znalezione miejsca:\n{context_text}\n\nNapisz krótkie podsumowanie i rekomendację dla użytkownika."}
        ]
        
        print("Generowanie podsumowania przez PLLuM...")
        response = self.llm.generate(prompt, max_tokens=400, temperature=0.7)
        return response

def _is_open_now(opening_hours: Optional[Dict[str, str]]) -> bool:
    """Sprawdza, czy miejsce jest otwarte w danym momencie."""
    if not opening_hours or not isinstance(opening_hours, dict):
        return True  # Zakładamy, że otwarte, jeśli brak danych, by nie odrzucać

    now = datetime.now()
    # Python's weekday(): Monday is 0 and Sunday is 6
    day_map = {
        0: "poniedziałek", 1: "wtorek", 2: "środa", 3: "czwartek",
        4: "piątek", 5: "sobota", 6: "niedziela"
    }
    day_name_pl = day_map[now.weekday()]

    # Znajdź godziny dla dzisiejszego dnia
    hours_today_str = opening_hours.get(day_name_pl) or opening_hours.get(day_name_pl.capitalize())

    if not hours_today_str or hours_today_str.lower() in ['zamknięte', 'closed']:
        return False

    if 'całą dobę' in hours_today_str.lower() or '24 hours' in hours_today_str.lower():
        return True

    # Przetwarzanie przedziałów czasowych, np. "12:00–22:00"
    time_ranges = re.findall(r'(\d{1,2}:\d{2})[–-](\d{1,2}:\d{2})', hours_today_str)

    if not time_ranges:
        return True  # Zakładamy, że otwarte, jeśli nie uda się sparsować

    current_time = now.time()

    for open_str, close_str in time_ranges:
        try:
            open_time = datetime.strptime(open_str, "%H:%M").time()
            close_time = datetime.strptime(close_str, "%H:%M").time()

            if (open_time <= close_time and open_time <= current_time <= close_time) or \
               (open_time > close_time and (current_time >= open_time or current_time <= close_time)):
                return True
        except ValueError:
            continue  # Ignoruj błędnie sformatowane przedziały
    return False

def create_rag_system(
    embeddings_file: str = "output_files/lodz_restaurants_cafes_embeddings_cls_words.jsonl",
    pooling_type: str | None = "cls",
    embedding_model_name: str = "sdadas/mmlw-retrieval-roberta-large" # sdadas/stella-pl-retrieval
):
    """
    Tworzy kompletny system RAG z wczytanymi embeddingami.

    Args:
        embeddings_file: Ścieżka do pliku z embeddingami
        pooling_type: "cls" lub "mean", jeśli None – wykrywa automatycznie
        embedding_model_name: Nazwa modelu embeddingów z Hugging Face.

    Returns:
        Tuple (ConversationalRAG, search_function)
    """
    import numpy as np
    import faiss
    import json
    from .embedding_model import ModelMeanPooling
    from sentence_transformers import CrossEncoder

    print("Ładowanie modelu i embeddingów...")

    # 1. Wybór strategii poolingu
    if pooling_type is None:
        pooling_strategy = "cls" if "cls" in embeddings_file else "mean"
    else:
        pooling_strategy = pooling_type

    print(f"INFO: Inicjalizuję model wykonawczy | pooling='{pooling_strategy}'")

    # 2. Wczytanie embeddingów z pliku
    records = []
    embeddings = []

    with open(embeddings_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)
            embeddings.append(np.array(rec["embedding"], dtype="float32"))

    embeddings = np.vstack(embeddings)
    n_samples, embedding_dim = embeddings.shape

    print(f"Załadowano {n_samples} embeddingów o wymiarze {embedding_dim}")

    # 3. Inicjalizacja modelu do enkodowania zapytań
    model_wrapper = ModelMeanPooling(
        embedding_model_name,
        word_embedding_dimension=embedding_dim,
        pooling_strategy=pooling_strategy
    )
    print("Pooling strategy:", model_wrapper.pooling_strategy)
    model_dim = model_wrapper.model.get_sentence_embedding_dimension()

    if model_dim != embedding_dim:
        print(f"UWAGA: Wymiar modelu ({model_dim}) nie zgadza się z wymiarem embeddingów ({embedding_dim})")

    # 4. Normalizacja embeddingów i indeks FAISS
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    print(f"Indeks gotowy! Liczba restauracji: {index.ntotal}")

    # 4a. Inicjalizacja Rerankera (Cross-Encoder)
    print("Ładowanie modelu rerankera...")
    # Używamy dedykowanego polskiego rerankera od sdadas (wersja v2)
    reranker = CrossEncoder('sdadas/polish-reranker-roberta-v2', trust_remote_code=True)

    query_prefix = "zapytanie: "

    # 5. Obiekt wektorowy z metodą wyszukiwania
    class SimpleVectorStore:
        def similarity_search_with_score(self, query: str, k: int = 5):
            full_query = query_prefix + query
            q_emb = model_wrapper.encode(full_query, normalize=True)
            q_emb = np.array(q_emb, dtype="float32").reshape(1, -1)

            # Bezpieczny check wymiaru
            assert q_emb.shape[1] == embedding_dim, (
                f"Embedding mismatch: query={q_emb.shape[1]} index={embedding_dim}"
            )

            scores, idxs = index.search(q_emb, k)

            class Document:
                def __init__(self, page_content, metadata):
                    self.page_content = page_content
                    self.metadata = metadata

            results = []
            for score, idx in zip(scores[0], idxs[0]):
                rec = records[idx]
                doc = Document(page_content=rec.get("name"), metadata=rec)
                results.append((doc, float(score)))
            return results

    vector_store = SimpleVectorStore()

    # 6. Funkcja wyszukiwania i re-rankingu dla agenta
    def search(
        query: str,
        k: int = 5,
        user_location: Optional[tuple] = None,
        price_preference: Optional[str] = None,
        cuisine_filter: Optional[str] = None
    ):
        """
        Wyszukuje, deduplikuje i re-rankuje restauracje, łącząc podobieństwo
        łącząc podobieństwo semantyczne z ocenami, popularnością i odległością.

        Args:
            query (str): Zapytanie użytkownika.
            k (int): Liczba wyników do zwrócenia.
            user_location (tuple, optional): Krotka (lat, lon) lokalizacji użytkownika.
            price_preference (str, optional): Preferowany przedział cenowy do filtrowania.
            cuisine_filter (str, optional): Typ kuchni do filtrowania (np. "azjatycka").
        """
        # Krok 1: Wyszukaj semantycznie większą pulę kandydatów
        # Jeśli mamy filtr kuchni, pobieramy znacznie więcej kandydatów, aby mieć co filtrować
        # (np. 100 zamiast 25 dla k=5), bo azjatyckie mogą być dalej na liście semantycznej
        initial_k = k * 20 if cuisine_filter else k * 10  # Zwiększamy pulę dla rerankera
        docs_with_scores = vector_store.similarity_search_with_score(query, k=initial_k)

        # Krok 2: Odrzuć duplikaty i przygotuj listę do dalszego przetwarzania
        unique_results = {}
        for doc, score in docs_with_scores:
            rec = doc.metadata
            name = rec.get("name")
            name_key = name.strip().lower() if name else "" # Normalizacja nazwy dla deduplikacji
            if name_key and name_key not in unique_results:
                key_words_data = rec.get("key_words", {})
                
                # Pobieranie danych z korzenia rekordu (priorytet) lub z key_words (fallback)
                rating = rec.get("google_rating") or key_words_data.get("google_rating")
                reviews = rec.get("google_reviews_total") or key_words_data.get("google_reviews_total")
                price_range = rec.get("google_price_range") or key_words_data.get("google_price_range")
                opening_hours = rec.get("opening_hours") or key_words_data.get("opening_hours")

                unique_results[name_key] = {
                    "semantic_score": score,
                    "name": name,
                    "type": key_words_data.get("types") or rec.get("types") or [],
                    "address": key_words_data.get("address"),
                    "coords": rec.get("Współrzędne"),
                    "google_rating": rating,
                    "google_reviews_total": reviews,
                    "google_price_range": price_range,
                    "opening_hours": opening_hours,
                    "context": rec.get("context") or key_words_data.get("context"),
                    "distance_km": float('inf'),
                    "final_score": 0.0
                }

        processed_results = list(unique_results.values())

        # Krok 2a: Filtrowanie po typie kuchni (jeśli podano) - PRIORYTETOWE
        if cuisine_filter:
            filtered_by_cuisine = []
            filter_lower = cuisine_filter.lower()
            # Uproszczenie filtra (np. "kuchnia azjatycka" -> "azjatycka")
            keywords = filter_lower.replace("kuchnia", "").replace("jedzenie", "").strip().split()
            
            for r in processed_results:
                types_list = r.get("type") or []
                types = [t.lower() for t in types_list]
                context = r.get("context", "").lower()
                name = r.get("name", "").lower()
                
                # Sprawdź czy którykolwiek keyword pasuje do typów, kontekstu LUB NAZWY
                match = False
                for kw in keywords:
                    if len(kw) > 2 and (any(kw in t for t in types) or kw in context or kw in name):
                        match = True
                        break
                if match:
                    filtered_by_cuisine.append(r)
            
            if filtered_by_cuisine:
                print(f"INFO: Przefiltrowano wyniki po kuchni '{cuisine_filter}'. Pozostało: {len(filtered_by_cuisine)}")
                processed_results = filtered_by_cuisine
            else:
                print(f"INFO: Nie znaleziono wyników dla kuchni '{cuisine_filter}'. Pokazuję wyniki semantyczne.")

        # Krok 2b: Filtrowanie po cenie (jeśli podano)
        if price_preference:
            filtered_results = []
            
            def parse_price_range(text):
                if not text: return None
                text_str = str(text).lower()
                
                # 1. Obsługa słów kluczowych
                if any(w in text_str for w in ['tanie', 'tanio', 'tani', 'niedrogie', 'budżetowe', 'ekonomiczne']):
                    return (0, 40)
                if any(w in text_str for w in ['średnie', 'średnio', 'umiarkowane', 'przystępne']):
                    return (40, 80)
                if any(w in text_str for w in ['drogie', 'drogo', 'drogi', 'ekskluzywne', 'luksusowe']):
                    return (80, 1000)
                
                # 2. Obsługa symboli $
                if '$' in text_str:
                    count = text_str.count('$')
                    if count == 1: return (0, 40)
                    if count == 2: return (40, 80)
                    if count >= 3: return (80, 1000)

                # 3. Obsługa liczb (np. "20-40")
                nums = [int(n) for n in re.findall(r'\d+', text_str)]
                if not nums: return None
                if len(nums) == 1: return (nums[0], nums[0])
                return (min(nums), max(nums))

            user_price_range = parse_price_range(price_preference)
            if user_price_range:
                print(f"DEBUG: Zinterpretowano preferencję ceny '{price_preference}' jako zakres: {user_price_range}")

            for r in processed_results:
                price = r.get("google_price_range")
                match = False
                
                # 1. Try numeric range matching
                if user_price_range:
                    place_price_range = parse_price_range(price)
                    if place_price_range:
                        # Check overlap: max(start1, start2) <= min(end1, end2)
                        if max(user_price_range[0], place_price_range[0]) <= min(user_price_range[1], place_price_range[1]):
                            match = True
                
                # 2. Fallback to string matching
                if not match and price and price_preference.lower() in str(price).lower():
                    match = True
                
                if match:
                    filtered_results.append(r)
            
            if filtered_results:
                processed_results = filtered_results
                print(f"INFO: Przefiltrowano wyniki po cenie '{price_preference}'. Pozostało: {len(processed_results)}")
            else:
                print(f"INFO: Brak wyników dla ceny '{price_preference}'. Ignoruję filtr ceny.")

        # Krok 2c: RERANKING (Cross-Encoder)
        # Oceniamy semantycznie pary (zapytanie, kontekst) dla przefiltrowanych wyników
        if processed_results:
            # Przygotuj pary do oceny
            rerank_pairs = [[query, r["context"]] for r in processed_results]
            
            # Oblicz nowe wyniki (logits)
            rerank_scores = reranker.predict(rerank_pairs)
            
            # Zaktualizuj semantic_score w wynikach
            # Normalizujemy wyniki rerankera (sigmoid), aby były w zakresie 0-1 jak cosine similarity
            from scipy.special import expit
            normalized_scores = expit(rerank_scores)
            
            print(f"\n--- RERANKING DEBUG (Query: {query}) ---")
            for i, r in enumerate(processed_results):
                old_score = r["semantic_score"]
                new_score = float(normalized_scores[i])
                r["semantic_score"] = new_score
                
                diff = new_score - old_score
                symbol = "⬆" if diff > 0 else "⬇"
                # print(f"   {r['name'][:25]:<25} | Vector: {old_score:.4f} -> Reranker: {new_score:.4f} ({symbol} {abs(diff):.4f})")
            print("----------------------------------------\n")

            # Krok 2d: FILTR BEZPIECZEŃSTWA (Threshold)
            # Jeśli Reranker (ekspert) mówi, że wynik jest słaby (< 0.15), to go wyrzucamy.
            # To chroni nas przed sytuacją, gdzie sklep zoologiczny z oceną 5.0 wygrywa ranking.
            processed_results = [r for r in processed_results if r["semantic_score"] > 0.15]
            print(f"INFO: Po filtracji semantycznej (threshold 0.15) pozostało {len(processed_results)} wyników.")

        # Krok 3: Wzbogać dane i przygotuj do re-rankingu
        max_dist = 0.0
        # Użyj 1.0 jako minimum, aby uniknąć dzielenia przez zero
        max_reviews_log = 1.0
        
        # Oblicz odległości i znajdź wartości do normalizacji
        for result in processed_results:
            if user_location and result.get("coords"):
                try:
                    lat, lon = map(float, result["coords"].split(","))
                    dist = distance_km(user_location[0], user_location[1], lat, lon)
                    result["distance_km"] = dist
                    if dist != float('inf'): max_dist = max(max_dist, dist)
                except (ValueError, TypeError):
                    pass
            
            reviews = result.get("google_reviews_total") or 0
            max_reviews_log = max(max_reviews_log, log1p(reviews))

        # Krok 4: Re-Ranking - obliczanie złożonego wyniku
        # Zmiana strategii: Reranker (semantic) działa jako filtr bezpieczeństwa (odrzuca śmieci).
        # Skoro przeszliśmy przez próg 0.15, to znaczy, że wyniki są sensowne.
        # Teraz o kolejności decyduje głównie JAKOŚĆ (Rating + Reviews).
        # Wagi: Semantic 35%, Rating 35%, Reviews 10%, Distance 20%
        weights = {"semantic": 0.35, "rating": 0.35, "popularity": 0.10, "proximity": 0.20}

        for result in processed_results:
            # Normalizacja oceny (1-5 -> 0-1)
            rating = result.get("google_rating")
            if rating is not None:
                score_rating = (rating - 1) / 4.0
            else:
                score_rating = 0.5  # Domyślna wartość, jeśli brak oceny

            # Normalizacja logarytmiczna liczby opinii (Popularność)
            reviews = result.get("google_reviews_total")
            score_popularity = log1p(reviews) / max_reviews_log if max_reviews_log > 0 else 0

            # Normalizacja odległości (odwrócona -> Bliskość)
            score_proximity = 0.0
            if user_location and max_dist > 0 and result["distance_km"] != float('inf'):
                score_proximity = 1.0 - (result["distance_km"] / max_dist)

            score_semantic = result.get("semantic_score", 0.0)

            # Obliczenie wyniku końcowego
            result["final_score"] = (weights["semantic"] * score_semantic +
                                     weights["rating"] * score_rating +
                                     weights["popularity"] * score_popularity +
                                     weights["proximity"] * score_proximity)

        # Krok 5: Sortowanie i zwrócenie najlepszych wyników
        processed_results.sort(key=lambda x: x["final_score"], reverse=True)
        return processed_results[:k]

    # 7. Funkcja do filtrowania otwartych miejsc
    def filter_open_places(results: List[Dict]) -> List[Dict]:
        """
        Filtruje listę wyników, zwracając tylko te miejsca, które są aktualnie otwarte.
        
        Args:
            results: Lista słowników z wynikami (zwrócona przez funkcję `search`).
        
        Returns:
            Przefiltrowana lista wyników.
        """
        return [r for r in results if _is_open_now(r.get("opening_hours"))]

    # 8. Inicjalizacja LLM
    llm = PLLuMLLM()

    # 9. Stworzenie systemu RAG
    rag = ConversationalRAG(
        llm_client=llm,
        search_function=search,
        max_history=10
    )

    # Dołączenie vector_store, aby był dostępny w testach
    rag.vectorstore = vector_store
    rag.reranker = reranker

    return rag, search, filter_open_places
