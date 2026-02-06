"""Unit tests for the cross-encoder reranker and searcher integration."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

from embeddings.reranker import CrossEncoderReranker, _MAX_DOC_CHARS
from search.searcher import IntelligentSearcher, SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(chunk_id: str, score: float = 0.5, name: str = None) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        similarity_score=score,
        content_preview=f"preview for {chunk_id}",
        file_path=f"/src/{chunk_id}.py",
        relative_path=f"src/{chunk_id}.py",
        folder_structure=["src"],
        chunk_type="function",
        name=name or chunk_id,
        parent_name=None,
        start_line=1,
        end_line=10,
        docstring=None,
        tags=[],
        context_info={},
    )


# ---------------------------------------------------------------------------
# CrossEncoderReranker unit tests
# ---------------------------------------------------------------------------

class TestCrossEncoderRerankerInit:
    def test_default_model_name(self):
        with patch.object(CrossEncoderReranker, '_resolve_device', return_value='cpu'):
            reranker = CrossEncoderReranker()
        assert reranker.model_name == "cl-nagoya/ruri-v3-reranker-310m"

    def test_custom_model_name(self):
        with patch.object(CrossEncoderReranker, '_resolve_device', return_value='cpu'):
            reranker = CrossEncoderReranker(model_name="custom/model")
        assert reranker.model_name == "custom/model"

    def test_device_resolution_cpu(self):
        reranker = CrossEncoderReranker(device="cpu")
        assert reranker._device == "cpu"


class TestCrossEncoderRerankerRerank:
    @pytest.fixture
    def reranker(self):
        """Create a reranker with a mocked CrossEncoder model."""
        with patch.object(CrossEncoderReranker, '_resolve_device', return_value='cpu'):
            r = CrossEncoderReranker()

        mock_model = MagicMock()
        # Inject mock model via cached_property override
        r.__dict__['model'] = mock_model
        r._model_loaded = True
        return r

    def test_rerank_sorts_by_score(self, reranker):
        """Results should be sorted by cross-encoder score descending."""
        reranker.model.predict.return_value = np.array([0.1, 0.9, 0.5])
        docs = [
            {"chunk_id": "a", "content": "doc a"},
            {"chunk_id": "b", "content": "doc b"},
            {"chunk_id": "c", "content": "doc c"},
        ]
        result = reranker.rerank("query", docs, k=3)
        assert [d["chunk_id"] for d in result] == ["b", "c", "a"]
        assert result[0]["reranker_score"] == pytest.approx(0.9)

    def test_rerank_returns_top_k(self, reranker):
        """Only top-k results should be returned."""
        reranker.model.predict.return_value = np.array([0.3, 0.9, 0.1, 0.7])
        docs = [
            {"chunk_id": "a", "content": "a"},
            {"chunk_id": "b", "content": "b"},
            {"chunk_id": "c", "content": "c"},
            {"chunk_id": "d", "content": "d"},
        ]
        result = reranker.rerank("query", docs, k=2)
        assert len(result) == 2
        assert result[0]["chunk_id"] == "b"
        assert result[1]["chunk_id"] == "d"

    def test_rerank_empty_documents(self, reranker):
        """Empty document list should return empty list."""
        result = reranker.rerank("query", [], k=5)
        assert result == []
        reranker.model.predict.assert_not_called()

    def test_rerank_content_preview_fallback(self, reranker):
        """When content is missing, content_preview should be used."""
        reranker.model.predict.return_value = np.array([0.8])
        docs = [{"chunk_id": "a", "content_preview": "preview text"}]
        result = reranker.rerank("query", docs, k=1)
        # Verify the pair passed to predict used content_preview
        call_args = reranker.model.predict.call_args[0][0]
        assert call_args[0] == ("query", "preview text")

    def test_rerank_truncates_long_content(self, reranker):
        """Documents longer than _MAX_DOC_CHARS should be truncated."""
        long_text = "x" * (_MAX_DOC_CHARS + 1000)
        reranker.model.predict.return_value = np.array([0.5])
        docs = [{"chunk_id": "a", "content": long_text}]
        reranker.rerank("query", docs, k=1)
        call_args = reranker.model.predict.call_args[0][0]
        _, doc_text = call_args[0]
        assert len(doc_text) == _MAX_DOC_CHARS

    def test_rerank_no_content_no_preview(self, reranker):
        """Document with neither content nor content_preview uses empty string."""
        reranker.model.predict.return_value = np.array([0.5])
        docs = [{"chunk_id": "a"}]
        reranker.rerank("query", docs, k=1)
        call_args = reranker.model.predict.call_args[0][0]
        assert call_args[0] == ("query", "")


class TestCrossEncoderRerankerModelInfo:
    def test_model_info_not_loaded(self):
        with patch.object(CrossEncoderReranker, '_resolve_device', return_value='cpu'):
            reranker = CrossEncoderReranker()
        info = reranker.get_model_info()
        assert info["status"] == "not_loaded"

    def test_model_info_loaded(self):
        with patch.object(CrossEncoderReranker, '_resolve_device', return_value='cpu'):
            reranker = CrossEncoderReranker()
        reranker._model_loaded = True
        info = reranker.get_model_info()
        assert info["status"] == "loaded"
        assert info["device"] == "cpu"


class TestDeviceResolution:
    def test_cpu_explicit(self):
        assert CrossEncoderReranker._resolve_device("cpu") == "cpu"

    def test_cuda_explicit(self):
        assert CrossEncoderReranker._resolve_device("cuda") == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_auto_selects_mps(self, mock_mps, mock_cuda):
        assert CrossEncoderReranker._resolve_device("auto") == "mps"

    @patch("torch.cuda.is_available", return_value=True)
    def test_auto_selects_cuda(self, mock_cuda):
        assert CrossEncoderReranker._resolve_device("auto") == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_auto_falls_back_to_cpu(self, mock_mps, mock_cuda):
        assert CrossEncoderReranker._resolve_device("auto") == "cpu"


# ---------------------------------------------------------------------------
# Searcher _rerank integration tests
# ---------------------------------------------------------------------------

class TestSearcherRerankIntegration:
    @pytest.fixture
    def mock_reranker(self):
        reranker = MagicMock()
        reranker.rerank.return_value = [
            {"chunk_id": "b", "reranker_score": 0.95, "content": "b"},
            {"chunk_id": "a", "reranker_score": 0.70, "content": "a"},
        ]
        return reranker

    @pytest.fixture
    def searcher_with_reranker(self, mock_reranker):
        mock_index = MagicMock()
        mock_index.search.return_value = [
            ("a", 0.9, {
                "content_preview": "func a", "file_path": "/a.py",
                "relative_path": "a.py", "folder_structure": [],
                "chunk_type": "function", "name": "a",
                "start_line": 1, "end_line": 5, "tags": [],
            }),
            ("b", 0.8, {
                "content_preview": "func b", "file_path": "/b.py",
                "relative_path": "b.py", "folder_structure": [],
                "chunk_type": "function", "name": "b",
                "start_line": 1, "end_line": 5, "tags": [],
            }),
        ]
        mock_index.search_bm25.return_value = []
        mock_index.get_stats.return_value = {"files_indexed": 2}
        mock_index.get_similar_chunks.return_value = []
        # metadata_db returns full content for reranking
        mock_index.metadata_db = {
            "a": {"metadata": {"content": "full content of a"}},
            "b": {"metadata": {"content": "full content of b"}},
        }

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.zeros(512, dtype=np.float32)

        return IntelligentSearcher(mock_index, mock_embedder, reranker=mock_reranker)

    def test_semantic_search_uses_reranker(self, searcher_with_reranker, mock_reranker):
        """Semantic search should apply reranking when reranker is present."""
        results = searcher_with_reranker.search("test", search_mode="semantic", k=2)
        mock_reranker.rerank.assert_called_once()
        # Reranker reordered: b first, a second
        assert results[0].chunk_id == "b"
        assert results[0].similarity_score == pytest.approx(0.95)

    def test_hybrid_search_uses_reranker(self, searcher_with_reranker, mock_reranker):
        """Hybrid search should apply reranking when reranker is present."""
        results = searcher_with_reranker.search("test", search_mode="hybrid", k=2)
        mock_reranker.rerank.assert_called_once()

    def test_keyword_search_skips_reranker(self, mock_reranker):
        """Keyword search should NOT use the reranker."""
        mock_index = MagicMock()
        mock_index.search_bm25.return_value = [
            ("chunk1", 3.5, {
                "content_preview": "def foo():", "file_path": "/foo.py",
                "relative_path": "foo.py", "folder_structure": [],
                "chunk_type": "function", "name": "foo",
                "start_line": 1, "end_line": 5, "tags": [],
            }),
        ]
        mock_index.get_stats.return_value = {"files_indexed": 1}
        mock_embedder = MagicMock()

        searcher = IntelligentSearcher(mock_index, mock_embedder, reranker=mock_reranker)
        results = searcher.search("foo", search_mode="keyword", k=5)
        mock_reranker.rerank.assert_not_called()
        assert len(results) == 1


class TestSearcherWithoutReranker:
    def test_no_reranker_backward_compatible(self):
        """Searcher without reranker should work as before."""
        mock_index = MagicMock()
        mock_index.search.return_value = [
            ("a", 0.9, {
                "content_preview": "func a", "file_path": "/a.py",
                "relative_path": "a.py", "folder_structure": [],
                "chunk_type": "function", "name": "a",
                "start_line": 1, "end_line": 5, "tags": [],
            }),
        ]
        mock_index.get_stats.return_value = {"files_indexed": 1}
        mock_index.get_similar_chunks.return_value = []

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.zeros(512, dtype=np.float32)

        # No reranker passed (default None)
        searcher = IntelligentSearcher(mock_index, mock_embedder)
        results = searcher.search("test", search_mode="semantic", k=5)
        assert len(results) == 1
        assert results[0].chunk_id == "a"


class TestRerankContentRetrieval:
    def test_rerank_fetches_content_from_metadata_db(self):
        """_rerank should fetch full content from metadata_db."""
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {"chunk_id": "a", "reranker_score": 0.9, "content": "full content"},
        ]

        mock_index = MagicMock()
        mock_index.metadata_db = MagicMock()
        mock_index.metadata_db.get.return_value = {
            "metadata": {"content": "full content from db"}
        }

        mock_embedder = MagicMock()
        searcher = IntelligentSearcher(mock_index, mock_embedder, reranker=mock_reranker)

        results_in = [_make_result("a", score=0.5)]
        out = searcher._rerank("query", results_in, k=1)

        # Verify metadata_db was consulted
        mock_index.metadata_db.get.assert_called_with("a")
        # Verify reranker received full content
        rerank_call_docs = mock_reranker.rerank.call_args[0][1]
        assert rerank_call_docs[0]["content"] == "full content from db"

    def test_rerank_falls_back_to_preview_on_db_miss(self):
        """When metadata_db has no entry, content_preview is used."""
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {"chunk_id": "a", "reranker_score": 0.8},
        ]

        mock_index = MagicMock()
        mock_index.metadata_db = MagicMock()
        mock_index.metadata_db.get.return_value = None

        mock_embedder = MagicMock()
        searcher = IntelligentSearcher(mock_index, mock_embedder, reranker=mock_reranker)

        results_in = [_make_result("a")]
        searcher._rerank("query", results_in, k=1)

        rerank_call_docs = mock_reranker.rerank.call_args[0][1]
        assert rerank_call_docs[0]["content"] == "preview for a"
