"""BM25 keyword index for code search.

Uses rank-bm25 (BM25Okapi) with a custom tokenizer that handles
CJK bigrams, CamelCase, and snake_case splitting.
Persisted as pickle alongside the FAISS index.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from search.tokenizer import tokenize

logger = logging.getLogger(__name__)

# Truncate chunk text beyond this limit before tokenizing.
# BM25 gains little from very long documents; truncation speeds up build.
_MAX_TEXT_LEN = 10_000


class BM25Index:
    """BM25 keyword search index backed by rank-bm25."""

    def __init__(self, storage_dir: str):
        self._storage_dir = Path(storage_dir)
        self._pkl_path = self._storage_dir / "bm25.pkl"

        # In-memory state — populated lazily
        self._bm25: Optional[BM25Okapi] = None
        self._chunk_ids: List[str] = []
        self._dirty = False
        self._loaded = False

    # ------------------------------------------------------------------
    # Build / rebuild
    # ------------------------------------------------------------------

    def build_from_metadata_db(self, metadata_db, chunk_ids: List[str]) -> None:
        """(Re)build the BM25 index from the SQLite metadata DB.

        Args:
            metadata_db: SqliteDict with chunk metadata.
            chunk_ids: Ordered list of chunk IDs to index.
        """
        corpus: List[List[str]] = []
        valid_ids: List[str] = []

        for cid in chunk_ids:
            entry = metadata_db.get(cid)
            if entry is None:
                continue
            meta = entry["metadata"]
            text = meta.get("content") or meta.get("content_preview", "")
            if len(text) > _MAX_TEXT_LEN:
                text = text[:_MAX_TEXT_LEN]
            # Prepend name for better keyword matching
            name = meta.get("name", "")
            if name:
                text = f"{name} {text}"
            tokens = tokenize(text)
            if not tokens:
                continue
            corpus.append(tokens)
            valid_ids.append(cid)

        if corpus:
            self._bm25 = BM25Okapi(corpus)
        else:
            self._bm25 = None

        self._chunk_ids = valid_ids
        self._dirty = False
        self._loaded = True
        logger.info(f"BM25 index built with {len(valid_ids)} chunks")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        metadata_db=None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search the BM25 index.

        Args:
            query: Natural language or keyword query.
            k: Number of results to return.
            filters: Optional filters (same format as CodeIndexManager).
            metadata_db: SqliteDict for metadata lookup and filter checks.

        Returns:
            List of (chunk_id, bm25_score, metadata) tuples, sorted by score desc.
        """
        if self._bm25 is None or not self._chunk_ids:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Build (score, index) pairs and sort descending
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for idx, score in scored:
            if score <= 0:
                break
            if len(results) >= k:
                break

            cid = self._chunk_ids[idx]

            # Fetch metadata for filter check + return value
            if metadata_db is not None:
                entry = metadata_db.get(cid)
                if entry is None:
                    continue
                meta = entry["metadata"]
            else:
                meta = {}

            # Apply filters
            if filters and not self._matches_filters(meta, filters):
                continue

            results.append((cid, float(score), meta))

        return results

    # ------------------------------------------------------------------
    # Filter logic (mirrors CodeIndexManager._matches_filters)
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if key == "file_pattern":
                if not any(
                    pat in metadata.get("relative_path", "") for pat in value
                ):
                    return False
            elif key == "chunk_type":
                if metadata.get("chunk_type") != value:
                    return False
            elif key == "tags":
                chunk_tags = set(metadata.get("tags", []))
                required = set(value if isinstance(value, list) else [value])
                if not required.intersection(chunk_tags):
                    return False
            elif key == "folder_structure":
                chunk_folders = set(metadata.get("folder_structure", []))
                required = set(value if isinstance(value, list) else [value])
                if not required.intersection(chunk_folders):
                    return False
            elif key in metadata:
                if metadata[key] != value:
                    return False
        return True

    # ------------------------------------------------------------------
    # Dirty flag — callers mark dirty after add/remove
    # ------------------------------------------------------------------

    def mark_dirty(self) -> None:
        """Mark the index as needing a rebuild."""
        self._dirty = True

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist BM25 index to disk as pickle."""
        if self._bm25 is None:
            return
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "bm25": self._bm25,
            "chunk_ids": self._chunk_ids,
        }
        with open(self._pkl_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"BM25 index saved ({len(self._chunk_ids)} chunks)")

    def load(self) -> bool:
        """Load BM25 index from disk. Returns True if loaded successfully."""
        if not self._pkl_path.exists():
            return False
        try:
            with open(self._pkl_path, "rb") as f:
                data = pickle.load(f)
            self._bm25 = data["bm25"]
            self._chunk_ids = data["chunk_ids"]
            self._dirty = False
            self._loaded = True
            logger.info(f"BM25 index loaded ({len(self._chunk_ids)} chunks)")
            return True
        except Exception as e:
            logger.warning(f"Failed to load BM25 index: {e}")
            return False

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def ensure_loaded(self, metadata_db=None, chunk_ids: Optional[List[str]] = None) -> None:
        """Ensure the index is ready for search.

        1. If already loaded and not dirty → no-op.
        2. If dirty or not loaded → try loading from pickle.
        3. If pickle missing or dirty → rebuild from metadata_db.
        """
        if self._loaded and not self._dirty:
            return

        # Try loading from disk first (unless dirty — need rebuild)
        if not self._dirty and self.load():
            return

        # Rebuild from metadata
        if metadata_db is not None and chunk_ids is not None:
            self.build_from_metadata_db(metadata_db, chunk_ids)
            self.save()
        else:
            logger.warning("BM25 index not available: no pickle and no metadata_db to rebuild from")

    def clear(self) -> None:
        """Clear in-memory state and delete pickle file."""
        self._bm25 = None
        self._chunk_ids = []
        self._dirty = False
        self._loaded = False
        if self._pkl_path.exists():
            self._pkl_path.unlink()
        logger.info("BM25 index cleared")
