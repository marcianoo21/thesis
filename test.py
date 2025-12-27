import spacy

nlp = spacy.load("pl_core_news_lg")

text1 = "Znajdz mi klimatyczne  miejsce przy Politechnice Łódzkiej lub przy manufakturze, ewentualnie gdzieś na teofilowie"
text = "Politechnice Łódzkiej i Przy Manufakturze"
text_cap = text.title()
print(text_cap)


doc = nlp(text_cap)

for ent in doc.ents:
    print(ent.text, ent.label_)
