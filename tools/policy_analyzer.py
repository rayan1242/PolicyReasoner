import re
import logging
from tools.chat import analyze_policy_sections, summarize_analysis, explain_match

logger = logging.getLogger(__name__)

PRIORITY_CHAIN = ["federal law", "state law", "payer contract", "hospital policy"]


def _format_chunks_for_llm(chunks: list[dict], max_chars: int = 4000) -> str:
    parts = []
    total = 0
    for i, chunk in enumerate(chunks, 1):
        entry = (
            f"[{i}] Policy: {chunk['name']} (ID: {chunk['policy_id']}, "
            f"Category: {chunk['category']})\n"
            f"Section: {chunk['section_title']}\n"
            f"{chunk['text']}\n"
        )
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n---\n".join(parts)


def _parse_evidence(analysis_text: str) -> list[dict]:
    """Extract structured evidence blocks from analysis text."""
    evidence = []
    in_section = False
    for line in analysis_text.splitlines():
        if "EVIDENCE:" in line.upper():
            in_section = True
            continue
        if in_section:
            if any(line.upper().startswith(k) for k in ("CONFLICTS", "RECOMMENDATION", "KEY", "RELEVANT")):
                break
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                try:
                    conf = float(re.search(r'[\d.]+', parts[-1]).group()) if parts[-1] else 0.8
                except (AttributeError, ValueError):
                    conf = 0.8
                evidence.append({
                    "policy_id": parts[0],
                    "section": parts[1] if len(parts) > 1 else "",
                    "quote": parts[2] if len(parts) > 2 else "",
                    "confidence": min(1.0, max(0.0, conf)),
                })
    return evidence


def _parse_conflicts(analysis_text: str) -> list[dict]:
    """Extract structured conflict entries."""
    conflicts_raw = []
    structured = []
    in_section = False

    for line in analysis_text.splitlines():
        if "CONFLICTS DETECTED:" in line.upper():
            in_section = True
            remainder = line.split(":", 1)[-1].strip()
            if remainder and remainder.lower() not in ("none", "none detected"):
                conflicts_raw.append(remainder)
            continue
        if in_section:
            if any(line.upper().startswith(k) for k in ("RECOMMENDATION", "KEY FINDINGS", "RELEVANT", "EVIDENCE")):
                break
            stripped = line.strip()
            if stripped and stripped.lower() not in ("none", "none detected"):
                conflicts_raw.append(stripped)

    for raw in conflicts_raw:
        if not raw:
            continue
        # Parse: CONFLICT: POL-A vs POL-B — description — PRIORITY: ...
        match = re.match(r'(?:CONFLICT:\s*)?(.+?)\s+vs\s+(.+?)\s*[—-]\s*(.+?)(?:\s*[—-]\s*PRIORITY:\s*(.+))?$', raw, re.IGNORECASE)
        if match:
            structured.append({
                "policy_a": match.group(1).strip(),
                "policy_b": match.group(2).strip(),
                "description": match.group(3).strip(),
                "priority": match.group(4).strip() if match.group(4) else "Review required",
                "priority_chain": PRIORITY_CHAIN,
            })
        else:
            structured.append({
                "policy_a": "",
                "policy_b": "",
                "description": raw,
                "priority": "Review required",
                "priority_chain": PRIORITY_CHAIN,
            })

    return structured


def analyze_policies(state, config) -> dict:
    query = state.user_query
    chunks = state.reranked_policies

    if not chunks:
        state.analysis = "No relevant policies found for this query."
        state.conflicts = []
        state.grounding = []
        return {"analysis": state.analysis, "conflicts": [], "grounding": []}

    policy_text = _format_chunks_for_llm(chunks)
    logger.info(f"Analyzing {len(chunks)} policy chunks...")

    analysis = analyze_policy_sections(query, policy_text)
    conflicts = _parse_conflicts(analysis)
    grounding = _parse_evidence(analysis)

    # Attach match keywords to each retrieved chunk (top 4 only to save API calls)
    for chunk in chunks[:4]:
        chunk["match_keywords"] = explain_match(query, chunk["text"])

    logger.info(f"Analysis complete. Conflicts: {len(conflicts)}, Evidence items: {len(grounding)}")
    state.analysis = analysis
    state.conflicts = conflicts
    state.grounding = grounding
    return {"analysis": analysis, "conflicts": conflicts, "grounding": grounding}


def summarize_policies(state, config) -> dict:
    result = summarize_analysis(state.analysis, state.user_query)
    logger.info(f"Summary generated. Confidence: {result.get('confidence', '?')}")
    state.summary = result.get("text", "")
    state.summary_confidence = result.get("confidence", 0.0)
    state.summary_confidence_reason = result.get("confidence_reason", "")
    return {
        "summary": state.summary,
        "summary_confidence": state.summary_confidence,
        "summary_confidence_reason": state.summary_confidence_reason,
    }
