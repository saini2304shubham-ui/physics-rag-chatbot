"""
tests/test_pipeline.py
────────────────────────────────────────────────────────────
Unit tests for the RAG pipeline components.
These tests use mocking so they run without a live vector store.

Usage:
    pytest tests/test_pipeline.py -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────
# INGESTION TESTS
# ─────────────────────────────────────────
class TestChunking:
    def test_chunk_short_text(self):
        from src.ingestion.ingest import chunk_pages
        pages = [{
            "page_num": 1,
            "text": "Newton's second law states F = ma.",
            "source": "test.pdf",
            "source_path": "/corpus/test.pdf",
        }]
        chunks = chunk_pages(pages)
        assert isinstance(chunks, list)
        # Short text may produce 0 or 1 chunk (≥50 chars required)

    def test_chunk_long_text(self):
        from src.ingestion.ingest import chunk_pages
        long_text = (
            "In classical mechanics, the motion of a particle is described by Newton's laws. "
            "The first law states that a body at rest remains at rest unless acted upon by a force. "
            "The second law states that the net force equals mass times acceleration: F = ma. "
            "The third law states that for every action there is an equal and opposite reaction. "
        ) * 20   # Repeat to exceed chunk size

        pages = [{
            "page_num": 1,
            "text": long_text,
            "source": "mechanics.pdf",
            "source_path": "/corpus/mechanics.pdf",
        }]
        chunks = chunk_pages(pages)
        assert len(chunks) > 1, "Long text should produce multiple chunks"
        for c in chunks:
            assert "source" in c["metadata"]
            assert "page_num" in c["metadata"]
            assert len(c["text"]) >= 50

    def test_chunk_metadata(self):
        from src.ingestion.ingest import chunk_pages
        pages = [{
            "page_num": 5,
            "text": "Thermodynamics is the study of heat and temperature and their relation to energy and work. " * 10,
            "source": "thermo.pdf",
            "source_path": "/corpus/thermo.pdf",
        }]
        chunks = chunk_pages(pages)
        assert len(chunks) >= 1
        assert chunks[0]["metadata"]["source"] == "thermo.pdf"
        assert chunks[0]["metadata"]["page_num"] == 5

    def test_chunk_ids_unique(self):
        from src.ingestion.ingest import chunk_pages
        long_text = "Physics is the natural science that studies matter, energy, and their interactions. " * 30
        pages = [{"page_num": 1, "text": long_text, "source": "phys.pdf", "source_path": "/x"}]
        chunks = chunk_pages(pages)
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_tiny_chunks_filtered(self):
        from src.ingestion.ingest import chunk_pages
        pages = [{"page_num": 1, "text": "Hi.", "source": "t.pdf", "source_path": "/x"}]
        chunks = chunk_pages(pages)
        # "Hi." is <50 chars → should be filtered out
        assert all(len(c["text"]) >= 50 for c in chunks)


# ─────────────────────────────────────────
# SCOPE DETECTION TESTS
# ─────────────────────────────────────────
class TestScopeDetector:
    def _make_doc(self, text="physics content"):
        from langchain.schema import Document
        return Document(page_content=text, metadata={})

    def test_physics_question_accepted(self):
        from src.retrieval.retriever import is_physics_question
        docs = [self._make_doc()]
        in_scope, conf = is_physics_question("What is Newton's second law F = ma?", docs)
        assert in_scope is True
        assert conf > 0.5

    def test_non_physics_rejected(self):
        from src.retrieval.retriever import is_physics_question
        in_scope, conf = is_physics_question("What is the capital of France?", [])
        assert in_scope is False
        assert conf < 0.5

    def test_cooking_rejected(self):
        from src.retrieval.retriever import is_physics_question
        in_scope, _ = is_physics_question("Give me a chocolate cake recipe.", [])
        assert in_scope is False

    def test_quantum_accepted(self):
        from src.retrieval.retriever import is_physics_question
        docs = [self._make_doc("quantum mechanics wave function")]
        in_scope, conf = is_physics_question(
            "Explain the quantum wave function and probability density.", docs
        )
        assert in_scope is True

    def test_electricity_accepted(self):
        from src.retrieval.retriever import is_physics_question
        docs = [self._make_doc()]
        in_scope, _ = is_physics_question(
            "What is the voltage across a capacitor?", docs
        )
        assert in_scope is True


# ─────────────────────────────────────────
# CONTEXT FORMATTER TESTS
# ─────────────────────────────────────────
class TestContextFormatter:
    def test_format_single_doc(self):
        from langchain.schema import Document
        from src.retrieval.retriever import format_context
        docs = [Document(
            page_content="The acceleration due to gravity on Earth is 9.8 m/s².",
            metadata={"source": "physics.pdf", "page_num": 12}
        )]
        ctx, sources = format_context(docs)
        assert "PASSAGE 1" in ctx
        assert "physics.pdf" in ctx
        assert len(sources) == 1
        assert sources[0]["page"] == 12

    def test_format_multiple_docs(self):
        from langchain.schema import Document
        from src.retrieval.retriever import format_context
        docs = [
            Document(page_content=f"Content {i}", metadata={"source": f"book{i}.pdf", "page_num": i})
            for i in range(1, 4)
        ]
        ctx, sources = format_context(docs)
        assert "PASSAGE 3" in ctx
        assert len(sources) == 3

    def test_snippet_truncation(self):
        from langchain.schema import Document
        from src.retrieval.retriever import format_context
        long_content = "A" * 500
        docs = [Document(page_content=long_content, metadata={"source": "x.pdf", "page_num": 1})]
        _, sources = format_context(docs)
        assert len(sources[0]["snippet"]) <= 305   # 300 + "…"


# ─────────────────────────────────────────
# CONFIDENCE SCORER TESTS
# ─────────────────────────────────────────
class TestConfidenceScorer:
    def _make_docs(self, n):
        from langchain.schema import Document
        return [Document(page_content=f"doc {i}", metadata={}) for i in range(n)]

    def test_no_docs_zero_confidence(self):
        from src.retrieval.retriever import compute_confidence
        assert compute_confidence([], None, True) == 0.0

    def test_out_of_scope_zero_confidence(self):
        from src.retrieval.retriever import compute_confidence
        docs = self._make_docs(5)
        assert compute_confidence(docs, [0.9]*5, False) == 0.0

    def test_high_scores_high_confidence(self):
        from src.retrieval.retriever import compute_confidence
        docs = self._make_docs(5)
        conf = compute_confidence(docs, [0.9, 0.88, 0.85, 0.82, 0.80], True)
        assert conf > 0.7

    def test_confidence_bounded(self):
        from src.retrieval.retriever import compute_confidence
        docs = self._make_docs(5)
        conf = compute_confidence(docs, [1.0]*5, True)
        assert 0.0 <= conf <= 1.0


# ─────────────────────────────────────────
# EVALUATION SUITE TESTS
# ─────────────────────────────────────────
class TestEvaluationSuite:
    def test_test_set_size(self):
        from src.evaluation.test_suite import IN_SCOPE_TESTS, OUT_OF_SCOPE_TESTS
        assert len(IN_SCOPE_TESTS) == 20
        assert len(OUT_OF_SCOPE_TESTS) == 10

    def test_all_in_scope_marked_physics(self):
        from src.evaluation.test_suite import IN_SCOPE_TESTS
        assert all(t.is_physics for t in IN_SCOPE_TESTS)

    def test_all_oos_marked_not_physics(self):
        from src.evaluation.test_suite import OUT_OF_SCOPE_TESTS
        assert all(not t.is_physics for t in OUT_OF_SCOPE_TESTS)

    def test_metrics_computation(self):
        from src.evaluation.test_suite import compute_metrics

        in_scope = [
            {"has_citation": True,  "hallucinated": False, "concept_score": 0.8, "sources_returned": True},
            {"has_citation": True,  "hallucinated": False, "concept_score": 0.9, "sources_returned": True},
            {"has_citation": False, "hallucinated": False, "concept_score": 0.7, "sources_returned": True},
            {"has_citation": True,  "hallucinated": True,  "concept_score": 0.5, "sources_returned": False},
        ]
        oos = [
            {"correctly_refused": True},
            {"correctly_refused": True},
            {"correctly_refused": False},
        ]
        m = compute_metrics(in_scope, oos)
        assert m["citation_accuracy"]  == pytest.approx(0.75, 0.01)
        assert m["hallucination_rate"] == pytest.approx(0.25, 0.01)
        assert m["oos_refusal_rate"]   == pytest.approx(0.667, 0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
