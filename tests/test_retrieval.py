"""Tests for policy ingestion and retrieval pipeline."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


# ---------------------------------------------------------------------------
# Ingestor tests
# ---------------------------------------------------------------------------

def test_load_policies_returns_list():
    from tools.policy_ingestor import load_policies
    policies = load_policies()
    assert isinstance(policies, list)
    assert len(policies) > 0


def test_load_policies_have_required_fields():
    from tools.policy_ingestor import load_policies
    policies = load_policies()
    required = {"policy_id", "name", "category", "sections"}
    for p in policies:
        assert required.issubset(p.keys()), f"Policy {p.get('policy_id')} missing fields"


def test_chunk_policies_produces_chunks():
    from tools.policy_ingestor import load_policies, chunk_policies
    policies = load_policies()
    chunks = chunk_policies(policies)
    assert len(chunks) > len(policies), "Should produce more chunks than policies"
    for c in chunks:
        assert "text" in c
        assert "policy_id" in c
        assert "section_title" in c


def test_chunk_text_under_size_limit():
    from tools.policy_ingestor import load_policies, chunk_policies, CHUNK_SIZE
    policies = load_policies()
    chunks = chunk_policies(policies)
    for c in chunks:
        assert len(c["text"]) <= CHUNK_SIZE + 50  # small tolerance for word boundaries


def test_build_index_returns_chunks_and_embeddings():
    from tools.policy_ingestor import build_index
    chunks, embeddings = build_index()
    assert len(chunks) > 0
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] > 0


def test_retrieval_returns_results():
    from tools.policy_ingestor import build_index
    from tools.dense_retrieval import hybrid_policy_retrieval

    build_index()

    state = MagicMock()
    state.user_query = "emergency room copay"
    state.retrieved_policies = []

    result = hybrid_policy_retrieval(state, config={"configurable": {"top_k": 5}})
    assert "retrieved_policies" in result
    assert len(result["retrieved_policies"]) > 0


def test_retrieval_scores_are_sorted():
    from tools.policy_ingestor import build_index
    from tools.dense_retrieval import hybrid_policy_retrieval

    build_index()

    state = MagicMock()
    state.user_query = "preauthorization for surgery"
    state.retrieved_policies = []

    result = hybrid_policy_retrieval(state, config={"configurable": {"top_k": 10}})
    scores = [c["retrieval_score"] for c in result["retrieved_policies"]]
    assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"


def test_retrieval_returns_relevant_category():
    from tools.policy_ingestor import build_index
    from tools.dense_retrieval import hybrid_policy_retrieval

    build_index()

    state = MagicMock()
    state.user_query = "insulin copay cap diabetes"
    state.retrieved_policies = []

    result = hybrid_policy_retrieval(state, config={"configurable": {"top_k": 5}})
    categories = {c["category"] for c in result["retrieved_policies"]}
    assert "prescription" in categories, "Should retrieve prescription-category policies for insulin query"


def test_empty_query_does_not_crash():
    from tools.policy_ingestor import build_index
    from tools.dense_retrieval import hybrid_policy_retrieval

    build_index()

    state = MagicMock()
    state.user_query = ""
    state.retrieved_policies = []

    result = hybrid_policy_retrieval(state, config={})
    assert "retrieved_policies" in result
