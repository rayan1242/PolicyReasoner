"""Tests for policy-to-code conversion."""
import pytest
from unittest.mock import patch, MagicMock


SAMPLE_POLICY_TEXT = """
Emergency Room Visit Coverage - Cost Sharing:
Emergency room copay is $350 per visit.
Annual deductible of $1,500 applies before copay.
After meeting the deductible, the plan pays 80% and the member pays 20% coinsurance.
Out-of-pocket maximum is $6,000 per individual per year.
Preauthorization is NOT required for true emergency room visits.
"""

SAMPLE_JSON_RESPONSE = """{
  "rule_id": "POL-ER-001",
  "policy_name": "Emergency Room Visit Coverage",
  "category": "emergency",
  "conditions": {
    "requires_preauth": false,
    "copay_amount": 350,
    "deductible": 1500,
    "coinsurance_plan_pct": 80,
    "oop_max": 6000
  },
  "actions": [
    {"trigger": "ER visit", "action": "apply_copay", "amount": "$350"}
  ],
  "exclusions": ["non-emergency visits billed as ER"],
  "notes": "No preauth required for true emergencies"
}"""

SAMPLE_PYTHON_RESPONSE = """def evaluate_claim(claim: dict) -> dict:
    \"\"\"Evaluate an emergency room claim against POL-ER-001.\"\"\"
    if not claim.get('is_emergency', False):
        return {'approved': False, 'action': 'deny', 'cost_to_member': '$0', 'reason': 'Not a true emergency'}
    copay = 350
    return {'approved': True, 'action': 'apply_copay', 'cost_to_member': f'${copay}', 'reason': 'ER copay applies'}
"""

SAMPLE_FEATURES_RESPONSE = """{
  "feature_set_name": "emergency_room_features",
  "binary_features": {
    "requires_preauth": "1 if preauthorization is required",
    "deductible_applies": "1 if annual deductible must be met first"
  },
  "numeric_features": {
    "copay_amount": {"description": "Fixed copay per ER visit", "unit": "dollars", "typical_range": "0-500"},
    "oop_max": {"description": "Out-of-pocket maximum per year", "unit": "dollars", "typical_range": "1000-10000"}
  },
  "categorical_features": {
    "network_type": ["in_network", "out_of_network"]
  },
  "target_variable": "claim_approval_decision"
}"""


@patch("tools.policy_converter.json_rules_chain")
def test_policy_to_json_rules_returns_dict(mock_chain):
    mock_response = MagicMock()
    mock_response.content = SAMPLE_JSON_RESPONSE
    mock_chain.invoke.return_value = mock_response

    from tools.policy_converter import policy_to_json_rules
    result = policy_to_json_rules(SAMPLE_POLICY_TEXT)

    assert isinstance(result, dict)
    assert "rule_id" in result or "policy_name" in result or "conditions" in result


@patch("tools.policy_converter.python_chain")
def test_policy_to_python_returns_function_string(mock_chain):
    mock_response = MagicMock()
    mock_response.content = SAMPLE_PYTHON_RESPONSE
    mock_chain.invoke.return_value = mock_response

    from tools.policy_converter import policy_to_python
    result = policy_to_python(SAMPLE_POLICY_TEXT)

    assert isinstance(result, str)
    assert "def evaluate_claim" in result
    assert "return" in result


@patch("tools.policy_converter.features_chain")
def test_policy_to_features_returns_dict(mock_chain):
    mock_response = MagicMock()
    mock_response.content = SAMPLE_FEATURES_RESPONSE
    mock_chain.invoke.return_value = mock_response

    from tools.policy_converter import policy_to_features
    result = policy_to_features(SAMPLE_POLICY_TEXT)

    assert isinstance(result, dict)
    assert "binary_features" in result or "numeric_features" in result or "feature_set_name" in result


@patch("tools.policy_converter.json_rules_chain")
def test_malformed_json_returns_parse_error_key(mock_chain):
    mock_response = MagicMock()
    mock_response.content = "This is not JSON at all."
    mock_chain.invoke.return_value = mock_response

    from tools.policy_converter import policy_to_json_rules
    result = policy_to_json_rules(SAMPLE_POLICY_TEXT)

    assert isinstance(result, dict)
    assert "parse_error" in result or "raw_output" in result


@patch("tools.policy_converter.json_rules_chain")
@patch("tools.policy_converter.python_chain")
@patch("tools.policy_converter.features_chain")
def test_convert_policy_node_all_formats(mock_feat, mock_py, mock_json):
    mock_json.invoke.return_value = MagicMock(content=SAMPLE_JSON_RESPONSE)
    mock_py.invoke.return_value = MagicMock(content=SAMPLE_PYTHON_RESPONSE)
    mock_feat.invoke.return_value = MagicMock(content=SAMPLE_FEATURES_RESPONSE)

    from tools.policy_converter import convert_policy

    chunk = {
        "policy_id": "POL-ER-001",
        "name": "Emergency Room Visit Coverage",
        "combined_doc": SAMPLE_POLICY_TEXT,
    }

    state = MagicMock()
    state.reranked_policies = [chunk]
    state.converted_output = {}

    result = convert_policy(state, config={"configurable": {"output_format": "all"}})

    assert "converted_output" in result
    out = result["converted_output"]
    assert "json_rules" in out
    assert "python_code" in out
    assert "ml_features" in out


def test_convert_policy_node_empty_chunks():
    from tools.policy_converter import convert_policy

    state = MagicMock()
    state.reranked_policies = []
    state.converted_output = {}

    result = convert_policy(state, config={})
    assert "error" in result["converted_output"]
