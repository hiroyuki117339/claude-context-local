"""MiniLM embedding model implementation (non-gated alternative)."""

from typing import Optional
import numpy as np
from embeddings.sentence_transformer import SentenceTransformerModel


class MiniLMEmbeddingModel(SentenceTransformerModel):
    """all-MiniLM-L6-v2 model - lightweight, non-gated SentenceTransformer."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        device: str = "auto"
    ):
        super().__init__(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=cache_dir,
            device=device
        )

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        """Encode texts, stripping Gemma-specific prompt_name if present."""
        kwargs.pop("prompt_name", None)
        return super().encode(texts, **kwargs)
