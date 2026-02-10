from typing import List, Optional, Union, Literal
from sentence_transformers import SentenceTransformer, models


class ModelMeanPooling:
    """Wrapper tworzący SentenceTransformer z mean-pooling lub CLS-pooling.

    Przykład użycia:
        m = ModelMeanPooling(
            "sentence-transformers/all-MiniLM-L6-v2",
            word_embedding_dimension=384,
            pooling_strategy="mean"  # lub "cls"
        )
        embs = m.encode(["tekst 1", "tekst 2"])  # numpy array / list
    """

    def __init__(
        self,
        model_name: str,
        word_embedding_dimension: Optional[int] = None,
        pooling_strategy: Literal["mean", "cls"] = "cls",
    ):
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        # Załaduj bazowy model aby pobrać warstwę słów
        base = SentenceTransformer(model_name)

        # jeśli nie podano wymiaru, spróbuj go odczytać z modelu
        if word_embedding_dimension is None:
            try:
                word_embedding_dimension = base.get_word_embedding_dimension()
            except Exception:
                # fallback: 768 — bezpieczna domyślna dla większości modeli
                word_embedding_dimension = 768

        word_model = base[0]
        
        # Ustaw odpowiednie parametry poolingu
        # Domyślnie wyłączamy wszystkie, aby mieć pewność, że tylko wybrana strategia będzie aktywna.
        pooling_kwargs = {
            "word_embedding_dimension": word_embedding_dimension,
            "pooling_mode_mean_tokens": False,
            "pooling_mode_cls_token": False,
            "pooling_mode_max_tokens": False,
        }
        
        if pooling_strategy == "mean":
            pooling_kwargs["pooling_mode_mean_tokens"] = True
        elif pooling_strategy == "cls":
            pooling_kwargs["pooling_mode_cls_token"] = True
        pooling = models.Pooling(**pooling_kwargs)

        # Złożony model z modułem token embedding + pooling
        self.model = SentenceTransformer(modules=[word_model, pooling])

    def encode(self, texts: Union[str, List[str]], normalize: bool = True):
        """Zwraca embeddingi dla pojedynczego tekstu lub listy tekstów.

        Args:
            texts: string albo lista stringów
            normalize: czy normalizować embeddingi (domyślnie True)

        Returns:
            list lub numpy array z embeddingami
        """
        return self.model.encode(texts, normalize_embeddings=normalize)
