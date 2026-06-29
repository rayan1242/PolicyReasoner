import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=1024,
    max_retries=3,
)

# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

query_expansion_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a healthcare policy search expert.

Convert a user question into precise search tags for a vector database.

Output Format: tag1:tag2:tag3 (2-5 colon-separated lowercase tags)

Rules:
- Use healthcare-specific terms (copay, deductible, preauthorization, formulary)
- Use category names: emergency, surgical, mental_health, prescription, preventive, referral, lab, therapy, out_of_network, billing_contract
- Include clinical terms when relevant (MRI, bariatric, opioid, telehealth, buprenorphine)
- Output ONLY the tags, nothing else

Examples:
Input: "How much does an emergency room visit cost?"
Output: emergency:copay:deductible:cost-sharing

Input: "Do I need preauthorization for an MRI?"
Output: preauthorization:MRI:imaging:lab

Input: "Find policies about opioid prescriptions"
Output: prescription:opioid:prior-authorization:formulary
"""),
    ("human", "{query}")
])

expansion_chain = query_expansion_prompt | llm


def expand_policy_query(query: str, max_retries: int = 2) -> str:
    tag_pattern = r'^[a-z0-9_-]+(?::[a-z0-9_-]+){0,4}$'
    for _ in range(max_retries):
        response = expansion_chain.invoke({"query": query})
        tags = response.content.strip().lower()
        tags = re.sub(r'<think>.*?</think>', '', tags, flags=re.DOTALL).strip()
        if re.match(tag_pattern, tags):
            return tags
    return query.lower().replace(" ", "-")[:50]


# ---------------------------------------------------------------------------
# Analysis with grounding + structured conflicts
# ---------------------------------------------------------------------------

analysis_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a healthcare policy analyst. Analyze the retrieved policy sections and respond in EXACTLY this format:

RELEVANT POLICIES: [comma-separated list of policy IDs that directly answer the question]

KEY FINDINGS:
- [finding 1 — cite the policy ID in parentheses e.g. (POL-ER-001)]
- [finding 2 — cite the policy ID]
- [finding 3 — cite the policy ID]

EVIDENCE:
[policy_id] | [section title] | [exact sentence from the policy that supports the main finding] | [confidence 0.0-1.0]
[repeat for each key policy cited]

CONFLICTS DETECTED:
[If none: "None detected"]
[If conflicts exist, one per line:]
CONFLICT: [Policy A ID] vs [Policy B ID] — [describe the contradiction in one sentence] — PRIORITY: [which takes precedence and why: federal law > state law > payer contract > hospital policy]

RECOMMENDATION: [1-2 sentence plain-English guidance]
"""),
    ("human", "User question: {query}\n\nRetrieved policy sections:\n{policy_text}")
])

analysis_chain = analysis_prompt | llm


def analyze_policy_sections(query: str, policy_text: str) -> str:
    response = analysis_chain.invoke({"query": query, "policy_text": policy_text})
    return response.content.strip()


# ---------------------------------------------------------------------------
# Summary with confidence
# ---------------------------------------------------------------------------

summary_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a healthcare benefits advisor. Respond in EXACTLY this format:

SUMMARY: [3-4 sentence plain-English answer a patient can understand]

CONFIDENCE: [a number 0.0-1.0 reflecting how well the retrieved policies answered the question.
  1.0 = policies directly and completely answer the question
  0.7 = policies partially answer it
  0.4 = policies are tangentially related
  0.1 = no relevant policies found]

CONFIDENCE_REASON: [one sentence explaining the confidence score]
"""),
    ("human", "Analysis:\n{analysis}\n\nOriginal question: {query}")
])

summary_chain = summary_prompt | llm


def summarize_analysis(analysis: str, query: str) -> dict:
    response = summary_chain.invoke({"analysis": analysis, "query": query})
    text = response.content.strip()

    result = {"text": "", "confidence": 0.0, "confidence_reason": ""}

    for line in text.splitlines():
        if line.startswith("SUMMARY:"):
            result["text"] = line.split(":", 1)[-1].strip()
        elif line.startswith("CONFIDENCE:") and "REASON" not in line:
            try:
                result["confidence"] = float(re.search(r'[\d.]+', line).group())
            except (AttributeError, ValueError):
                result["confidence"] = 0.0
        elif line.startswith("CONFIDENCE_REASON:"):
            result["confidence_reason"] = line.split(":", 1)[-1].strip()

    if not result["text"]:
        result["text"] = text
    return result


# ---------------------------------------------------------------------------
# Match keyword explainer
# ---------------------------------------------------------------------------

match_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """Given a user query and a retrieved policy chunk, list exactly 3-5 specific terms or phrases
from the query that matched this policy chunk. Be precise — only list terms that actually appear
in or are semantically equivalent to terms in the policy text.

Output format: term1 | term2 | term3
Output ONLY the pipe-separated terms, nothing else."""),
    ("human", "Query: {query}\n\nPolicy text: {policy_text}")
])

match_chain = match_prompt | llm


def explain_match(query: str, policy_text: str) -> list[str]:
    try:
        response = match_chain.invoke({"query": query, "policy_text": policy_text[:600]})
        raw = response.content.strip()
        return [t.strip() for t in raw.split("|") if t.strip()][:5]
    except Exception:
        return []
