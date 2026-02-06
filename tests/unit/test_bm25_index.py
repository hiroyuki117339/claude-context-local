"""Unit tests for search.bm25_index."""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from search.bm25_index import BM25Index


def _make_metadata_db(entries):
    """Create a dict-like mock metadata DB.

    entries: list of (chunk_id, metadata_dict) tuples.
    """
    db = {}
    for cid, meta in entries:
        db[cid] = {"metadata": meta}
    return db


@pytest.fixture
def tmp_storage(tmp_path):
    """Temporary storage directory."""
    d = tmp_path / "bm25_test"
    d.mkdir()
    return str(d)


@pytest.fixture
def sample_entries():
    """Sample metadata entries for testing."""
    return [
        ("chunk1", {
            "content": "def calculate_total(items): return sum(items)",
            "content_preview": "def calculate_total...",
            "name": "calculate_total",
            "relative_path": "src/math.py",
            "chunk_type": "function",
            "tags": ["math"],
            "folder_structure": ["src"],
        }),
        ("chunk2", {
            "content": "class UserManager: manages user accounts and authentication",
            "content_preview": "class UserManager...",
            "name": "UserManager",
            "relative_path": "src/auth/user.py",
            "chunk_type": "class",
            "tags": ["auth"],
            "folder_structure": ["src", "auth"],
        }),
        ("chunk3", {
            "content": "def connect_database(host, port): establish database connection",
            "content_preview": "def connect_database...",
            "name": "connect_database",
            "relative_path": "src/db/connector.py",
            "chunk_type": "function",
            "tags": ["database"],
            "folder_structure": ["src", "db"],
        }),
        ("chunk4", {
            "content": "契約期間を計算する関数。開始日と終了日から日数を返す。",
            "content_preview": "契約期間を計算...",
            "name": "calculate_contract_period",
            "relative_path": "src/contract.py",
            "chunk_type": "function",
            "tags": ["contract"],
            "folder_structure": ["src"],
        }),
    ]


class TestBM25IndexBuild:
    def test_build_from_metadata_db(self, tmp_storage, sample_entries):
        idx = BM25Index(tmp_storage)
        db = _make_metadata_db(sample_entries)
        chunk_ids = [e[0] for e in sample_entries]

        idx.build_from_metadata_db(db, chunk_ids)

        assert len(idx._chunk_ids) == 4
        assert idx._bm25 is not None
        assert not idx.is_dirty

    def test_build_empty_corpus(self, tmp_storage):
        idx = BM25Index(tmp_storage)
        idx.build_from_metadata_db({}, [])

        assert idx._bm25 is None
        assert idx._chunk_ids == []

    def test_build_skips_missing_entries(self, tmp_storage, sample_entries):
        idx = BM25Index(tmp_storage)
        db = _make_metadata_db(sample_entries[:2])
        # Request all 4 but only 2 exist in db
        chunk_ids = [e[0] for e in sample_entries]

        idx.build_from_metadata_db(db, chunk_ids)
        assert len(idx._chunk_ids) == 2


class TestBM25IndexSearch:
    @pytest.fixture(autouse=True)
    def _build_index(self, tmp_storage, sample_entries):
        self.idx = BM25Index(tmp_storage)
        self.db = _make_metadata_db(sample_entries)
        chunk_ids = [e[0] for e in sample_entries]
        self.idx.build_from_metadata_db(self.db, chunk_ids)

    def test_basic_search(self):
        # "total" only appears in chunk1, giving positive IDF
        results = self.idx.search("total items", k=2, metadata_db=self.db)
        assert len(results) > 0
        top_ids = [r[0] for r in results]
        assert "chunk1" in top_ids

    def test_cjk_search(self):
        results = self.idx.search("契約期間", k=2, metadata_db=self.db)
        assert len(results) > 0
        top_ids = [r[0] for r in results]
        assert "chunk4" in top_ids

    def test_search_with_filter_chunk_type(self):
        results = self.idx.search(
            "calculate", k=5,
            filters={"chunk_type": "class"},
            metadata_db=self.db,
        )
        for _, _, meta in results:
            assert meta["chunk_type"] == "class"

    def test_search_with_filter_file_pattern(self):
        results = self.idx.search(
            "user", k=5,
            filters={"file_pattern": ["auth"]},
            metadata_db=self.db,
        )
        for _, _, meta in results:
            assert "auth" in meta["relative_path"]

    def test_empty_query(self):
        results = self.idx.search("", k=5, metadata_db=self.db)
        assert results == []

    def test_no_match(self):
        results = self.idx.search("xyzzyplugh", k=5, metadata_db=self.db)
        assert results == []

    def test_search_returns_metadata(self):
        results = self.idx.search("database connection", k=1, metadata_db=self.db)
        assert len(results) >= 1
        cid, score, meta = results[0]
        assert isinstance(score, float)
        assert "relative_path" in meta


class TestBM25IndexPersistence:
    def test_save_and_load(self, tmp_storage, sample_entries):
        idx = BM25Index(tmp_storage)
        db = _make_metadata_db(sample_entries)
        chunk_ids = [e[0] for e in sample_entries]
        idx.build_from_metadata_db(db, chunk_ids)

        idx.save()
        assert (Path(tmp_storage) / "bm25.pkl").exists()

        # Load into a new instance
        idx2 = BM25Index(tmp_storage)
        assert idx2.load()
        assert len(idx2._chunk_ids) == len(idx._chunk_ids)

        # Search should work on loaded index ("total" has positive IDF)
        results = idx2.search("total items", k=2, metadata_db=db)
        assert len(results) > 0

    def test_load_missing_file(self, tmp_storage):
        idx = BM25Index(tmp_storage)
        assert not idx.load()


class TestBM25IndexLifecycle:
    def test_mark_dirty(self, tmp_storage, sample_entries):
        idx = BM25Index(tmp_storage)
        assert not idx.is_dirty
        idx.mark_dirty()
        assert idx.is_dirty

    def test_clear(self, tmp_storage, sample_entries):
        idx = BM25Index(tmp_storage)
        db = _make_metadata_db(sample_entries)
        chunk_ids = [e[0] for e in sample_entries]
        idx.build_from_metadata_db(db, chunk_ids)
        idx.save()

        assert (Path(tmp_storage) / "bm25.pkl").exists()

        idx.clear()
        assert idx._bm25 is None
        assert idx._chunk_ids == []
        assert not (Path(tmp_storage) / "bm25.pkl").exists()

    def test_ensure_loaded_from_pickle(self, tmp_storage, sample_entries):
        idx = BM25Index(tmp_storage)
        db = _make_metadata_db(sample_entries)
        chunk_ids = [e[0] for e in sample_entries]
        idx.build_from_metadata_db(db, chunk_ids)
        idx.save()

        # New instance — should load from pickle
        idx2 = BM25Index(tmp_storage)
        idx2.ensure_loaded(metadata_db=db, chunk_ids=chunk_ids)
        assert len(idx2._chunk_ids) == 4

    def test_ensure_loaded_rebuilds_when_dirty(self, tmp_storage, sample_entries):
        idx = BM25Index(tmp_storage)
        db = _make_metadata_db(sample_entries)
        chunk_ids = [e[0] for e in sample_entries]
        idx.build_from_metadata_db(db, chunk_ids)
        idx.save()

        # Mark dirty → should rebuild
        idx.mark_dirty()
        idx.ensure_loaded(metadata_db=db, chunk_ids=chunk_ids)
        assert not idx.is_dirty
