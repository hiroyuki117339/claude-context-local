"""Unit tests for hybrid search and RRF fusion in IntelligentSearcher."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from search.searcher import IntelligentSearcher, SearchResult


def _make_result(chunk_id: str, score: float = 0.5, name: str = None) -> SearchResult:
    """Helper to create a SearchResult for testing."""
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


class TestReciprocalRankFusion:
    def test_disjoint_lists(self):
        """Results from only one ranker get single-ranker RRF scores."""
        sem = [_make_result("a", 0.9), _make_result("b", 0.8)]
        bm25 = [_make_result("c", 5.0), _make_result("d", 4.0)]

        fused = IntelligentSearcher._reciprocal_rank_fusion(sem, bm25, k_rrf=60)

        ids = [r.chunk_id for r in fused]
        assert set(ids) == {"a", "b", "c", "d"}
        # Both rank-0 results should have same score: 1/(60+0)
        scores = {r.chunk_id: r.similarity_score for r in fused}
        assert scores["a"] == pytest.approx(scores["c"])

    def test_overlapping_results_boosted(self):
        """Results in both lists should score higher than single-list results."""
        sem = [_make_result("a", 0.9), _make_result("b", 0.8)]
        bm25 = [_make_result("a", 5.0), _make_result("c", 4.0)]

        fused = IntelligentSearcher._reciprocal_rank_fusion(sem, bm25, k_rrf=60)

        scores = {r.chunk_id: r.similarity_score for r in fused}
        # "a" appears in both → should have higher score
        assert scores["a"] > scores["b"]
        assert scores["a"] > scores["c"]

    def test_empty_lists(self):
        fused = IntelligentSearcher._reciprocal_rank_fusion([], [])
        assert fused == []

    def test_one_empty_list(self):
        sem = [_make_result("a")]
        fused = IntelligentSearcher._reciprocal_rank_fusion(sem, [])
        assert len(fused) == 1
        assert fused[0].chunk_id == "a"

    def test_preserves_semantic_result_when_overlap(self):
        """When chunk appears in both lists, semantic result (richer context) is kept."""
        sem_result = _make_result("a")
        sem_result.context_info = {"similar_chunks": [{"id": "x"}]}

        bm25_result = _make_result("a")
        bm25_result.context_info = {}

        fused = IntelligentSearcher._reciprocal_rank_fusion([sem_result], [bm25_result])
        assert fused[0].context_info == {"similar_chunks": [{"id": "x"}]}

    def test_rank_ordering_matters(self):
        """Higher-ranked items in input should get higher RRF scores."""
        sem = [_make_result("a"), _make_result("b"), _make_result("c")]

        fused = IntelligentSearcher._reciprocal_rank_fusion(sem, [], k_rrf=60)
        scores = [r.similarity_score for r in fused]
        # Scores should be strictly decreasing
        assert scores[0] > scores[1] > scores[2]


class TestSearchModeRouting:
    @pytest.fixture
    def searcher(self):
        """Create a searcher with mocked dependencies."""
        mock_index = MagicMock()
        mock_index.search.return_value = []
        mock_index.search_bm25.return_value = []
        mock_index.get_stats.return_value = {"total_chunks": 0}

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.zeros(512, dtype=np.float32)

        return IntelligentSearcher(mock_index, mock_embedder)

    def test_auto_resolves_to_hybrid(self, searcher):
        """search_mode='auto' should call hybrid search."""
        with patch.object(searcher, '_hybrid_search', return_value=[]) as mock_hybrid:
            searcher.search("test", search_mode="auto")
            mock_hybrid.assert_called_once()

    def test_semantic_mode(self, searcher):
        with patch.object(searcher, '_semantic_search', return_value=[]) as mock_sem:
            searcher.search("test", search_mode="semantic")
            mock_sem.assert_called_once()

    def test_keyword_mode(self, searcher):
        with patch.object(searcher, '_keyword_search', return_value=[]) as mock_kw:
            searcher.search("test", search_mode="keyword")
            mock_kw.assert_called_once()

    def test_hybrid_mode(self, searcher):
        with patch.object(searcher, '_hybrid_search', return_value=[]) as mock_hybrid:
            searcher.search("test", search_mode="hybrid")
            mock_hybrid.assert_called_once()

    def test_unknown_mode_falls_back_to_hybrid(self, searcher):
        with patch.object(searcher, '_hybrid_search', return_value=[]) as mock_hybrid:
            searcher.search("test", search_mode="foobar")
            mock_hybrid.assert_called_once()


class TestKeywordSearch:
    def test_keyword_search_returns_results(self):
        mock_index = MagicMock()
        mock_index.search_bm25.return_value = [
            ("chunk1", 3.5, {
                "content_preview": "def foo():",
                "file_path": "/src/foo.py",
                "relative_path": "src/foo.py",
                "folder_structure": ["src"],
                "chunk_type": "function",
                "name": "foo",
                "parent_name": None,
                "start_line": 1,
                "end_line": 5,
                "docstring": None,
                "tags": [],
            }),
        ]
        mock_index.get_stats.return_value = {"files_indexed": 1}

        mock_embedder = MagicMock()
        searcher = IntelligentSearcher(mock_index, mock_embedder)

        results = searcher.search("foo", search_mode="keyword", k=5)
        assert len(results) == 1
        assert results[0].chunk_id == "chunk1"
        assert results[0].similarity_score == 3.5


class TestHybridSearchIntegration:
    def test_hybrid_merges_both_arms(self):
        """Hybrid search should return results from both semantic and BM25."""
        mock_index = MagicMock()

        # Semantic returns chunk1, chunk2
        mock_index.search.return_value = [
            ("chunk1", 0.9, {
                "content_preview": "semantic result 1",
                "file_path": "/a.py", "relative_path": "a.py",
                "folder_structure": [], "chunk_type": "function",
                "name": "func_a", "start_line": 1, "end_line": 5,
                "tags": [],
            }),
            ("chunk2", 0.7, {
                "content_preview": "semantic result 2",
                "file_path": "/b.py", "relative_path": "b.py",
                "folder_structure": [], "chunk_type": "function",
                "name": "func_b", "start_line": 1, "end_line": 5,
                "tags": [],
            }),
        ]

        # BM25 returns chunk2, chunk3
        mock_index.search_bm25.return_value = [
            ("chunk2", 5.0, {
                "content_preview": "bm25 result 2",
                "file_path": "/b.py", "relative_path": "b.py",
                "folder_structure": [], "chunk_type": "function",
                "name": "func_b", "start_line": 1, "end_line": 5,
                "tags": [],
            }),
            ("chunk3", 3.0, {
                "content_preview": "bm25 result 3",
                "file_path": "/c.py", "relative_path": "c.py",
                "folder_structure": [], "chunk_type": "function",
                "name": "func_c", "start_line": 1, "end_line": 5,
                "tags": [],
            }),
        ]

        mock_index.get_stats.return_value = {"files_indexed": 3}
        mock_index.get_similar_chunks.return_value = []

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.zeros(512, dtype=np.float32)

        searcher = IntelligentSearcher(mock_index, mock_embedder)
        results = searcher.search("test query", search_mode="hybrid", k=10)

        result_ids = [r.chunk_id for r in results]
        # All three chunks should appear
        assert "chunk1" in result_ids
        assert "chunk2" in result_ids
        assert "chunk3" in result_ids

        # chunk2 (in both lists) should rank highest or near-highest
        scores = {r.chunk_id: r.similarity_score for r in results}
        assert scores["chunk2"] >= scores["chunk1"]
        assert scores["chunk2"] >= scores["chunk3"]
