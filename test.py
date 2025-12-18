from sentence_transformers import SentenceTransformer
EMBED_MODEL_NAME = "sdadas/mmlw-retrieval-roberta-large" # przetestować tez sdadas/mmlw-retrieval-roberta-large-v2
model = SentenceTransformer(EMBED_MODEL_NAME)
print(model) # Pooling - wectorization technique, dokładniej CLS pooling