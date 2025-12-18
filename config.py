"""
config.py

Profile konfiguracyjne dla różnych przypadków użycia.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RAGConfig:
    """Konfiguracja systemu RAG."""
    
    # Model settings
    model_name: str = "CYFRAGOVPL/PLLuM-12B-nc-chat:featherless-ai"
    embedding_model: str = "sdadas/mmlw-retrieval-roberta-large"
    
    # Search settings
    top_k: int = 5
    embedding_file: str = "output_files/lodz_restaurants_cafes_embeddings_mean.jsonl"
    
    # Conversation settings
    max_history: int = 10
    max_tokens: int = 500
    temperature: float = 0.7
    
    # System prompt
    system_prompt: Optional[str] = None
    
    def __repr__(self):
        return f"RAGConfig(model={self.model_name}, k={self.top_k}, history={self.max_history})"


# ============================================
# PROFIL 1: DOMYŚLNY (Zbalansowany)
# ============================================
DEFAULT = RAGConfig(
    top_k=5,
    max_history=10,
    max_tokens=500,
    temperature=0.7,
)


# ============================================
# PROFIL 2: SZYBKI (Mniej dokładny, szybsze odpowiedzi)
# ============================================
FAST = RAGConfig(
    top_k=3,
    max_history=6,
    max_tokens=300,
    temperature=0.5,
    system_prompt="""Jesteś asystentem rekomendacji restauracji w Łodzi.
    
Odpowiadaj ZWIĘŹLE i NA TEMAT:
- Maksymalnie 3 rekomendacje
- Krótkie opisy
- Tylko najważniejsze informacje"""
)


# ============================================
# PROFIL 3: DOKŁADNY (Więcej kontekstu, dłuższe odpowiedzi)
# ============================================
DETAILED = RAGConfig(
    top_k=10,
    max_history=15,
    max_tokens=800,
    temperature=0.8,
    system_prompt="""Jesteś ekspertem od gastronomii w Łodzi.

Twoje odpowiedzi powinny być:
- Szczegółowe i informacyjne
- Z kontekstem lokalnym (historia miejsca, specjalności)
- Z praktycznymi wskazówkami (parking, godziny, rezerwacje)
- Z porównaniami między opcjami

Przedstaw TOP 5-7 miejsc z pełnymi opisami."""
)


# ============================================
# PROFIL 4: PRZYJACIELSKI (Casualowy styl)
# ============================================
FRIENDLY = RAGConfig(
    top_k=5,
    max_history=12,
    max_tokens=400,
    temperature=0.9,
    system_prompt="""Cześć! Jestem Twoim kumplm od jedzenia w Łodzi! 😊

Zasady:
- Używaj emotikon 🍕🍔🍜☕
- Pisz naturalnie, jak do znajomego
- Dziel się osobistymi opiniami ("Osobiście uwielbiam...", "Musisz spróbować...")
- Bądź entuzjastyczny ale szczery

Polecaj miejsca z pasją, ale uczciwie mów o wadach!"""
)


# ============================================
# PROFIL 5: PROFESJONALNY (Biznesowy styl)
# ============================================
PROFESSIONAL = RAGConfig(
    top_k=5,
    max_history=8,
    max_tokens=500,
    temperature=0.6,
    system_prompt="""Jesteś profesjonalnym concierge specjalizującym się w gastronomii łódzkiej.

Format odpowiedzi:
- Obiektywne, oparte na faktach rekomendacje
- Konkretne dane: oceny, ceny, lokalizacje
- Struktura: 1) Rekomendacja 2) Uzasadnienie 3) Szczegóły praktyczne
- Bez emotikonów, ton formalny ale uprzejmy

Priorytet: jakość, renoma, recenzje."""
)


# ============================================
# PROFIL 6: LOKALNY PRZEWODNIK
# ============================================
LOCAL_GUIDE = RAGConfig(
    top_k=7,
    max_history=10,
    max_tokens=600,
    temperature=0.75,
    system_prompt="""Jesteś łodzianinem od pokoleń i znasz miasto jak własną kieszeń!

Twój styl:
- Dziel się lokalnymi ciekawostkami i historią miejsc
- Wspominaj o ukrytych perełkach, nie tylko o popularnych miejscach
- Polecaj trasy ("najpierw tu, potem stamtąd")
- Ostrzegaj przed pułapkami turystycznymi
- Mów o sąsiednich atrakcjach

Pokazuj Łódź oczami lokalsa, z miłością do miasta!"""
)


# ============================================
# PROFIL 7: BUDŻETOWY
# ============================================
BUDGET = RAGConfig(
    top_k=5,
    max_history=10,
    max_tokens=400,
    temperature=0.7,
    system_prompt="""Jesteś ekspertem od taniego ale dobrego jedzenia w Łodzi!

Priorytet:
- Stosunek jakości do ceny
- Promocje, happy hours, lunche biznesowe
- Porcje i ilość za cenę
- Miejsca studenckie i budżetowe

Zawsze wspominaj orientacyjne ceny i gdzie można zaoszczędzić!"""
)


# ============================================
# PROFIL 8: FOODIE (Dla smakoszy)
# ============================================
FOODIE = RAGConfig(
    top_k=6,
    max_history=12,
    max_tokens=700,
    temperature=0.8,
    system_prompt="""Jesteś koneserem kulinarnym, krytykiem gastronomicznym!

Analizuj:
- Jakość składników i świeżość
- Techniki kulinarne i prezentacja
- Autentyczność kuchni
- Kreatywność menu
- Doświadczenie sensoryczne (smak, aromat, tekstura)

Używaj terminologii kulinarnej. Porównuj do standardów światowych.
Rekomenduj miejsca dla prawdziwych miłośników jedzenia!"""
)


# ============================================
# MAPA PROFILI
# ============================================
PROFILES = {
    "default": DEFAULT,
    "fast": FAST,
    "detailed": DETAILED,
    "friendly": FRIENDLY,
    "professional": PROFESSIONAL,
    "local": LOCAL_GUIDE,
    "budget": BUDGET,
    "foodie": FOODIE,
}


def get_config(profile: str = "default") -> RAGConfig:
    """
    Pobierz konfigurację dla danego profilu.
    
    Args:
        profile: Nazwa profilu (default, fast, detailed, friendly, 
                professional, local, budget, foodie)
    
    Returns:
        Obiekt RAGConfig
    """
    if profile not in PROFILES:
        print(f"⚠️  Nieznany profil '{profile}', używam 'default'")
        profile = "default"
    
    return PROFILES[profile]


def list_profiles():
    """Wyświetl dostępne profile."""
    print("\n📋 Dostępne profile konfiguracyjne:\n")
    
    descriptions = {
        "default": "Zbalansowany - standardowa konfiguracja",
        "fast": "Szybki - krótsze odpowiedzi, mniej kontekstu",
        "detailed": "Dokładny - długie odpowiedzi, więcej szczegółów",
        "friendly": "Przyjacielski - casualowy styl, emotikony",
        "professional": "Profesjonalny - formalny ton, biznesowy",
        "local": "Lokalny przewodnik - insider tips, ciekawostki",
        "budget": "Budżetowy - focus na cenie i oszczędnościach",
        "foodie": "Smakosz - dla koneserów, język kulinarny",
    }
    
    for name, desc in descriptions.items():
        config = PROFILES[name]
        print(f"  • {name:12} - {desc}")
        print(f"                 (k={config.top_k}, tokens={config.max_tokens}, temp={config.temperature})")
    
    print()


# ============================================
# PRZYKŁAD UŻYCIA
# ============================================
if __name__ == "__main__":
    list_profiles()
    
    print("\nPrzykład użycia:")
    print("="*60)
    
    # Załaduj konfigurację
    config = get_config("friendly")
    print(f"\n{config}")
    print(f"\nSystem prompt preview:")
    print(config.system_prompt[:200] + "...")