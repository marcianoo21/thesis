# import spacy

# nlp = spacy.load("pl_core_news_lg")

# text1 = "Znajdz mi klimatyczne  miejsce przy Politechnice Łódzkiej lub przy manufakturze, ewentualnie gdzieś na teofilowie"
# text = "Politechnice Łódzkiej i Przy Manufakturze"
# text_cap = text.title()
# print(text_cap)


# doc = nlp(text_cap)

# for ent in doc.ents:
#     print(ent.text, ent.label_)

import spacy
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="hid_training")

geo_coords = "51.779247, 19.493394"


nlp = spacy.load("pl_core_news_lg")
doc = nlp("pasaż Róż".title())
for ent in doc.ents:
    print(f"Tekst: {ent.text}, Etykieta: {ent.label_}")
    location  = geolocator.geocode(ent.text)
    print(location)
    print(location.raw)