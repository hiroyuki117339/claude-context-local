"""Embedding models registry."""
from embeddings.gemma import GemmaEmbeddingModel
from embeddings.minilm import MiniLMEmbeddingModel
from embeddings.ruri import RuriEmbeddingModel

AVAILIABLE_MODELS = {
    "google/embeddinggemma-300m": GemmaEmbeddingModel,
    "sentence-transformers/all-MiniLM-L6-v2": MiniLMEmbeddingModel,
    "cl-nagoya/ruri-v3-130m": RuriEmbeddingModel,
}
