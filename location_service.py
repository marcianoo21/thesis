import spacy
from geopy.geocoders import Nominatim
from typing import Optional, Tuple

class LocationService:
    """
    Serwis do ekstrakcji nazw lokalizacji z tekstu i ich geokodowania.
    """
    def __init__(self):
        print("Inicjalizuję serwis lokalizacji (spaCy + Nominatim)...")
        try:
            self.nlp = spacy.load("pl_core_news_lg")
        except OSError:
            print("BŁĄD: Nie znaleziono modelu 'pl_core_news_lg'.")
            print("Aby go zainstalować, uruchom: python -m spacy download pl_core_news_lg")
            raise
        self.geolocator = Nominatim(user_agent="inzynierka_restaurant_recommender")
        print("Serwis lokalizacji gotowy.")

    # Lista słów, które spaCy często błędnie rozpoznaje jako lokalizacje (False Positives)
    IGNORED_LOCATIONS = {
        "włoska", "włoski", "polska", "polski", "chińska", "chiński",
        "azjatycka", "azjatycki", "tajska", "tajski", "indyjska", "indyjski",
        "meksykańska", "meksykański", "amerykańska", "amerykański",
        "wegańska", "wegański", "wegetariańska", "wegetariański",
        "dobra", "dobry", "tania", "tani", "droga", "drogi",
        "restauracja", "kawiarnia", "bar", "pub", "bistro", "jedzenie",
        "obiad", "kolacja", "śniadanie", "lunch", "steki", "burger", "pizza"
    }

    def extract_location_name(self, text: str) -> Optional[str]:
        """Wyciąga nazwę lokalizacji z tekstu za pomocą NER."""
        doc = self.nlp(text.title()) # Lepsze wyniki dla nazw własnych
        # Szukamy encji typu 'placeName' (miejsce) lub 'geogName' (nazwa geograficzna)
        for ent in doc.ents:
            if ent.label_ in ["placeName", "geogName", "orgName", "roadName"]: # orgName też może być lokalizacją (np. nazwa firmy), roadName dla ulic
                if ent.lemma_.lower() in self.IGNORED_LOCATIONS:
                    print(f"  Zignorowano fałszywą lokalizację (przymiotnik/typ): '{ent.text}'")
                    continue
                
                print(f"  Znaleziono encję lokalizacyjną: '{ent.text}' ({ent.label_})")
                # Używamy lematu (formy podstawowej), aby ułatwić geokodowanie (np. "Piotrkowskiej" -> "Piotrkowska")
                print(f"  Lematyzacja (forma podstawowa): '{ent.lemma_}'")
                return ent.lemma_
        return None

    def geocode(self, location_name: str) -> Optional[Tuple[float, float]]:
        """Konwertuje nazwę lokalizacji na współrzędne (lat, lon)."""
        try:
            # Ograniczamy wyszukiwanie do Łodzi dla większej precyzji
            location = self.geolocator.geocode(f"{location_name}, Łódź, Polska")
            if location:
                print(f"  Geokodowanie '{location_name}': ({location.latitude:.4f}, {location.longitude:.4f})")
                return (location.latitude, location.longitude)
            print(f"  Nie udało się znaleźć współrzędnych dla: '{location_name}'")
            return None
        except Exception as e:
            print(f"Błąd podczas geokodowania: {e}")
            return None

    def get_location_from_query(self, text: str) -> Optional[Tuple[float, float]]:
        """Pełny proces: ekstrakcja nazwy i geokodowanie."""
        location_name = self.extract_location_name(text)
        if location_name:
            return self.geocode(location_name)
        return None