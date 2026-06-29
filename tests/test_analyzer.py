"""Tests for policy analysis and summarization."""
import pytest
from unittest.mock import MagicMock, patch


SAMPLE_CHUNKS = [
    {
        "policy_id": "POL-ER-001",
        "name": "Emergency Room Visit Coverage",
        "category": "emergency",
        "section_title": "Cost Sharing",
        "text": "Emergency room copay is $350 per visit. Annual deductible of $1,500 applies.",
        "combined_doc": "Emergency Room Visit Coverage - Cost Sharing: Emergency room copay is $350 per visit. Annual deductible of $1,500 applies.",
        "retrieval_score": 0.9,
        "cross_encoder_score": 8.5,
    },
    {
        "policy_id": "POL-OON-003",
        "name": "Emergency Out-of-Network Billing Policy",
        "category": "out_of_network",
        "section_title": "Cost Sharing",
        "text": "Emergency care received out-of-network is subject to in-network cost-sharing only. Member pays in-network emergency room copay of $350.",
        "combined_doc": "Emergency Out-of-Network Billing Policy - Cost Sharing: Emergency care received out-of-network is subject to in-network cost-sharing only.",
        "retrieval_score": 0.85,
        "cross_encoder_score": 7.2,
    },
]

CONFLICTING_CHUNKS = [
    {
        "policy_id": "POL-MH-001",
        "name": "Outpatient Mental Health Therapy",
        "category": "mental_health",
        "section_title": "Preauthorization",
        "text": "First 8 outpatient therapy sessions per calendar year do not require preauthorization.",
        "combined_doc": "First 8 sessions free of preauth.",
        "retrieval_score": 0.8,
        "cross_encoder_score": 7.0,
    },
    {
        "policy_id": "POL-MH-004",
        "name": "Telehealth Mental Health Services",
        "category": "mental_health",
        "section_title": "Preauthorization",
        "text": "No preauthorization required for telehealth mental health sessions 1-8 per year. Same preauthorization rules apply for sessions 9+ as in-person care.",
        "combined_doc": "Telehealth mental health: no preauth for sessions 1-8.",
        "retrieval_score": 0.75,
        "cross_encoder_score": 6.5,
    },
]


def test_format_chunks_for_llm_returns_string():
    from tools.policy_analyzer import _format_chunks_for_llm
    formatted = _format_chunks_for_llm(SAMPLE_CHUNKS)
    assert isinstance(formatted, str)
    assert "Emergency Room" in formatted
    assert "POL-ER-001" in formatted


def test_format_chunks_respects_max_chars():
    from tools.policy_analyzer import _format_chunks_for_llm
    formatted = _format_chunks_for_llm(SAMPLE_CHUNKS, max_chars=100)
    assert len(formatted) <= 200  # allow some overhead for the last chunk boundary


@patch("tools.policy_analyzer.analyze_policy_sections")
def test_analyze_policies_returns_analysis_and_conflicts(mock_analyze):
    mock_analyze.return_value = (
        "RELEVANT POLICIES: Emergency Room Visit Coverage\n"
        "KEY FINDINGS:\n- Copay is $350\n"
        "CONFLICTS DETECTED: POL-OON-003 uses in-network rates for OON emergency care\n"
        "RECOMMENDATION: Use in-network ER when possible."
    )

    from tools.policy_analyzer import analyze_policies

    state = MagicMock()
    state.user_query = "How much does emergency room care cost out of network?"
    state.reranked_policies = SAMPLE_CHUNKS

    result = analyze_policies(state, config={})
    assert "analysis" in result
    assert "conflicts" in result
    assert isinstance(result["conflicts"], list)
    assert len(result["conflicts"]) > 0


@patch("tools.policy_analyzer.analyze_policy_sections")
def test_no_conflicts_detected_when_policies_agree(mock_analyze):
    mock_analyze.return_value = (
        "RELEVANT POLICIES: Emergency Room Visit Coverage\n"
        "KEY FINDINGS:\n- Copay is $350\n"
        "CONFLICTS DETECTED: None detected\n"
        "RECOMMENDATION: Your ER copay is $350."
    )

    from tools.policy_analyzer import analyze_policies

    state = MagicMock()
    state.user_query = "What is my ER copay?"
    state.reranked_policies = [SAMPLE_CHUNKS[0]]

    result = analyze_policies(state, config={})
    assert result["conflicts"] == []


@patch("tools.policy_analyzer.analyze_policy_sections")
def test_empty_chunks_returns_graceful_fallback(mock_analyze):
    from tools.policy_analyzer import analyze_policies

    state = MagicMock()
    state.user_query = "some query"
    state.reranked_policies = []

    result = analyze_policies(state, config={})
    assert "analysis" in result
    assert "No relevant" in result["analysis"]
    mock_analyze.assert_not_called()


@patch("tools.policy_analyzer.summarize_analysis")
def test_summarize_policies_returns_string(mock_summarize):
    mock_summarize.return_value = "Your emergency room visit copay is $350 per visit."

    from tools.policy_analyzer import summarize_policies

    state = MagicMock()
    state.analysis = "RELEVANT POLICIES: ...\nKEY FINDINGS: - Copay $350"
    state.user_query = "ER cost?"

    result = summarize_policies(state, config={})
    assert "summary" in result
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0
