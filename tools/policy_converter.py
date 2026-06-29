import json
import logging
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

logger = logging.getLogger(__name__)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=1024,
    max_retries=3,
)

# --- JSON Rules conversion ---
json_rules_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a policy-to-rules converter for healthcare systems.
Extract the decision rules from the given policy text and return ONLY valid JSON.

Return a JSON object with this structure:
{{
  "rule_id": "string (use the policy_id if available)",
  "policy_name": "string",
  "category": "string",
  "conditions": {{
    "key": "value (use descriptive keys like requires_preauth, copay_amount, deductible_applies, etc.)"
  }},
  "actions": [
    {{"trigger": "condition description", "action": "what happens", "amount": "dollar amount or null"}}
  ],
  "exclusions": ["list of what is NOT covered"],
  "notes": "any important caveats"
}}

Return ONLY the JSON object. No markdown, no explanation."""),
    ("human", "Policy text:\n{policy_text}")
])

json_rules_chain = json_rules_prompt | llm

# --- Python function conversion ---
python_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a policy-to-code converter for healthcare claims systems.
Convert the given policy text into a Python function that evaluates a claim.

CRITICAL RULES:
- ONLY encode logic that is EXPLICITLY stated in the policy text
- Do NOT infer or add rules that are not in the text
- Every if/else branch must correspond to a specific policy clause
- Approval must require ALL conditions stated in the policy to be met, not just one
- If the policy says "preauthorization required", the function must check it AND deny if missing

The function must:
- Be named evaluate_claim(claim: dict) -> dict
- Return: approved (bool), action (str), cost_to_member (str), reason (str), policy_clause (str)
- policy_clause must quote the exact text from the policy that drove the decision
- Have a docstring listing what claim fields are expected

Return ONLY the Python function. No markdown fences, no explanation."""),
    ("human", "Policy text:\n{policy_text}")
])

python_chain = python_prompt | llm

# --- ML Features conversion ---
features_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a machine learning feature engineer for healthcare claims models.
Extract binary and numeric features from the given policy text that could be used in an ML model.

Return ONLY valid JSON with this structure:
{{
  "feature_set_name": "string",
  "binary_features": {{
    "feature_name": "description of what 1 means"
  }},
  "numeric_features": {{
    "feature_name": {{"description": "...", "unit": "dollars/days/count", "typical_range": "0-X"}}
  }},
  "categorical_features": {{
    "feature_name": ["possible_value_1", "possible_value_2"]
  }},
  "target_variable": "description of what this policy predicts/decides"
}}

Return ONLY the JSON object. No markdown, no explanation."""),
    ("human", "Policy text:\n{policy_text}")
])

features_chain = features_prompt | llm


def _safe_parse_json(raw: str) -> dict:
    raw = re.sub(r'```(?:json)?', '', raw).strip().rstrip('`').strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"raw_output": raw, "parse_error": "Could not parse JSON from LLM response"}


def policy_to_json_rules(policy_text: str) -> dict:
    response = json_rules_chain.invoke({"policy_text": policy_text})
    return _safe_parse_json(response.content)


def _validate_python(code: str) -> dict:
    """Compile + dry-run the generated function, return validation result."""
    result = {"valid": False, "error": None, "test_result": None}
    try:
        compile(code, "<generated>", "exec")
        result["valid"] = True
    except SyntaxError as e:
        result["error"] = f"SyntaxError: {e}"
        return result

    # Dry-run with a minimal claim dict
    try:
        namespace = {}
        exec(code, namespace)  # noqa: S102
        fn = namespace.get("evaluate_claim")
        if fn:
            test_claim = {
                "is_emergency": True,
                "has_preauth": False,
                "claim_amount": 1000,
                "procedure_type": "surgical",
                "in_network": True,
                "deductible_met": False,
                "preauthorization_date": None,
                "procedure_date": None,
            }
            out = fn(test_claim)
            result["test_result"] = out
    except Exception as e:
        result["error"] = f"Runtime error on test input: {e}"

    return result


def _score_json_rules(rules: dict) -> float:
    """Heuristic confidence on JSON rules completeness."""
    if "parse_error" in rules:
        return 0.1
    score = 0.5
    if "conditions" in rules and rules["conditions"]:
        score += 0.2
    if "actions" in rules and rules["actions"]:
        score += 0.15
    if "exclusions" in rules:
        score += 0.1
    if "rule_id" in rules:
        score += 0.05
    return min(1.0, score)


def policy_to_python(policy_text: str) -> dict:
    response = python_chain.invoke({"policy_text": policy_text})
    code = response.content.strip()
    code = re.sub(r'^```(?:python)?', '', code, flags=re.MULTILINE).strip()
    code = re.sub(r'```$', '', code, flags=re.MULTILINE).strip()
    validation = _validate_python(code)
    return {
        "code": code,
        "valid": validation["valid"],
        "validation_error": validation.get("error"),
        "test_output": validation.get("test_result"),
        "confidence": 0.9 if validation["valid"] and validation["test_result"] else
                      0.6 if validation["valid"] else 0.2,
    }


def policy_to_features(policy_text: str) -> dict:
    response = features_chain.invoke({"policy_text": policy_text})
    return _safe_parse_json(response.content)


def convert_policy(state, config) -> dict:
    cfg = config.get("configurable", {}) if isinstance(config, dict) else {}
    output_format = cfg.get("output_format", "all")

    chunks = state.reranked_policies
    if not chunks:
        state.converted_output = {"error": "No policy chunks available for conversion"}
        return {"converted_output": state.converted_output}

    top_policy_id = chunks[0]["policy_id"]
    top_policy_chunks = [c for c in chunks if c["policy_id"] == top_policy_id]
    policy_text = "\n".join(c["combined_doc"] for c in top_policy_chunks[:2])
    policy_text = policy_text[:3000]

    logger.info(f"Converting policy '{chunks[0]['name']}' to code (format: {output_format})...")

    result = {
        "policy_id": top_policy_id,
        "policy_name": chunks[0]["name"],
    }

    if output_format in ("json", "all"):
        rules = policy_to_json_rules(policy_text)
        result["json_rules"] = rules
        result["json_confidence"] = _score_json_rules(rules)

    if output_format in ("python", "all"):
        py = policy_to_python(policy_text)
        result["python_code"] = py["code"]
        result["python_valid"] = py["valid"]
        result["python_validation_error"] = py["validation_error"]
        result["python_test_output"] = py["test_output"]
        result["python_confidence"] = py["confidence"]

    if output_format in ("features", "all"):
        features = policy_to_features(policy_text)
        result["ml_features"] = features
        result["features_confidence"] = 0.85 if "parse_error" not in features else 0.1

    state.converted_output = result
    return {"converted_output": result}
