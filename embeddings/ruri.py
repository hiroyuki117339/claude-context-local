"""Ruri v3 embedding model implementation (Japanese-optimized)."""

from typing import Optional
import numpy as np
from embeddings.sentence_transformer import SentenceTransformerModel


class RuriEmbeddingModel(SentenceTransformerModel):
    """cl-nagoya/ruri-v3-130m model - Japanese-optimized SentenceTransformer (512-dim)."""

    PROMPT_PREFIX_MAP = {
        "Retrieval-document": "検索文書: ",
        "InstructionRetrieval": "検索クエリ: ",
    }

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        device: str = "auto"
    ):
        super().__init__(
            model_name="cl-nagoya/ruri-v3-130m",
            cache_dir=cache_dir,
            device=device
        )

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        """Encode texts with Ruri v3 prefix convention.

        Ruri v3 uses prefix strings instead of prompt_name:
        - "Retrieval-document" -> prepend "検索文書: "
        - "InstructionRetrieval" -> prepend "検索クエリ: "
        - Other/None -> no prefix
        """
        prompt_name = kwargs.pop("prompt_name", None)
        prefix = self.PROMPT_PREFIX_MAP.get(prompt_name, "")
        if prefix:
            texts = [prefix + t for t in texts]
        return super().encode(texts, **kwargs)
