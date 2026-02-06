"""Cross-encoder reranker for search result refinement."""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from functools import cached_property

import torch


logger = logging.getLogger(__name__)

# Maximum document length (characters) passed to the cross-encoder.
_MAX_DOC_CHARS = 4000


class CrossEncoderReranker:
    """Reranks search results using a cross-encoder model.

    The cross-encoder scores each (query, document) pair jointly, producing
    far more accurate relevance judgements than bi-encoder dot-product alone.
    """

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-v3-reranker-310m",
        cache_dir: Optional[str] = None,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._device = self._resolve_device(device)
        self._model_loaded = False

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    @cached_property
    def model(self):
        """Load and cache the CrossEncoder model."""
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading reranker model: {self.model_name}")

        # Enable offline mode when model is already cached
        if self._is_model_cached():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            logger.info("Reranker model cache detected. Enabling offline mode.")

        # MPS does not support FlashAttention used by ModernBERT
        model_kwargs: Dict[str, Any] = {}
        if self._device == "mps":
            model_kwargs["attn_implementation"] = "eager"

        try:
            model = CrossEncoder(
                self.model_name,
                device=self._device,
                cache_folder=self.cache_dir,
                model_kwargs=model_kwargs if model_kwargs else None,
            )
            self._model_loaded = True
            logger.info(f"Reranker loaded on device: {self._device}")
            return model
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        k: int,
    ) -> List[Dict[str, Any]]:
        """Rerank documents by cross-encoder relevance to *query*.

        Args:
            query: The search query string.
            documents: List of dicts, each **must** contain ``chunk_id`` and
                should contain ``content`` (full text) or ``content_preview``.
            k: Number of top results to return.

        Returns:
            Top-*k* documents sorted by cross-encoder score (descending).
            Each dict gets an additional ``reranker_score`` key.
        """
        if not documents:
            return []

        # Build (query, doc_text) pairs
        pairs: List[tuple] = []
        for doc in documents:
            text = doc.get("content") or doc.get("content_preview") or ""
            # Truncate long documents
            if len(text) > _MAX_DOC_CHARS:
                text = text[:_MAX_DOC_CHARS]
            pairs.append((query, text))

        # Score with cross-encoder
        scores = self.model.predict(pairs)

        # Attach scores and sort
        scored_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = dict(doc)
            doc_copy["reranker_score"] = float(score)
            scored_docs.append(doc_copy)

        scored_docs.sort(key=lambda d: d["reranker_score"], reverse=True)
        return scored_docs[:k]

    def get_model_info(self) -> Dict[str, Any]:
        """Return model status information."""
        if not self._model_loaded:
            return {
                "model_name": self.model_name,
                "status": "not_loaded",
                "device": self._device,
            }
        return {
            "model_name": self.model_name,
            "status": "loaded",
            "device": self._device,
        }

    def cleanup(self):
        """Release model resources."""
        if not self._model_loaded:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Reranker cleaned up")
        except Exception as e:
            logger.warning(f"Reranker cleanup error: {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(requested: str) -> str:
        req = (requested or "auto").lower()
        if req in ("auto", ""):
            if torch.cuda.is_available():
                return "cuda"
            try:
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except Exception:
                pass
            return "cpu"
        return req

    def _is_model_cached(self) -> bool:
        """Check whether the reranker model files exist locally."""
        if not self.cache_dir:
            return False
        try:
            model_key = self.model_name.split("/")[-1].lower()
            cache_root = Path(self.cache_dir)
            if not cache_root.exists():
                return False
            for path in cache_root.rglob("config.json"):
                if model_key in str(path.parent).lower():
                    return True
        except Exception:
            pass
        return False
