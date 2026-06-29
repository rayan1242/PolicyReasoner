# PolicyReasoner — Implementation Plan

## Phase 1: Data Layer
- [ ] Create `data/policies.json` with 40 mock healthcare policies
  - Each policy has: id, name, category, sections (title + content)
  - Categories: emergency, surgical, mental_health, prescription, preventive, referral, lab, therapy, out_of_network, billing_contract
  - 4 policies per category = 40 total

## Phase 2: Retrieval Infrastructure
- [ ] `tools/chat.py` — Groq LLM wrapper
  - Function: `expand_policy_query(query) → search_tags`
- [ ] `tools/policy_ingestor.py` — FAISS index builder
  - Load `policies.json`
  - Chunk each policy section into ~500 char chunks
  - Embed with sentence-transformers (all-mpnet-base-v2)
  - Build FAISS index, save to disk as `policy_index.pkl`
- [ ] `tools/dense_retrieval.py` — hybrid retrieval
  - Dense (FAISS cosine) + sparse (BM25) fusion at alpha=0.7
  - Input: `user_query`, FAISS index
  - Output: top-K policy chunks with scores
- [ ] `tools/cross_encoder_reranking.py` — re-rank
  - MiniLM cross-encoder (ms-marco-MiniLM-L-6-v2)
  - Input: query + chunks, Output: re-ranked top-N chunks

## Phase 3: Analysis (NEW)
- [ ] `tools/policy_analyzer.py`
  - `analyze_policies(query, chunks) → {analysis: str, conflicts: list, grounding: list}`
  - LLM prompt: given query + top policy chunks, identify:
    1. Which policies are most relevant
    2. Any contradictions between policies
    3. Key clauses that apply — with exact quoted evidence
  - Priority chain: federal law → state law → payer contract → hospital policy
  - `summarize_policies(analysis, chunks) → {summary, confidence, confidence_reason}`

## Phase 4: Conversion (NEW — key differentiator)
- [ ] `tools/policy_converter.py`
  - `policy_to_json_rules(policy_text) → dict` — decision rules with confidence score
  - `policy_to_python(policy_text) → dict` — Python function, compiled + dry-run validated
  - `policy_to_features(policy_text) → dict` — ML feature definitions
  - All use LLM prompts with structured output parsing

## Phase 5: Agent Orchestration
- [ ] `agent.py` — LangGraph StateGraph
  - Nodes: retrieve, rerank, analyze, summarize, convert
  - Linear pipeline
  - State: PolicyState dataclass
  - Config: PolicyAgentConfig (top_k, format, etc.)

## Phase 6: API Server
- [ ] `app.py` — FastAPI
  - `POST /analyze` — run full agent pipeline
  - `POST /convert` — convert specific policy text to code
  - `GET /policies` — list all policies in corpus
  - `GET /health` — health check

## Phase 7: Tests
- [ ] `tests/test_retrieval.py`
- [ ] `tests/test_analyzer.py`
- [ ] `tests/test_converter.py`

## Phase 8: Docker + CI
- [ ] `Dockerfile` — python:3.12-slim
- [ ] `.github/workflows/docker-image.yml`

## Decisions Made
1. Use `all-mpnet-base-v2` for embeddings — fast on CPU, high quality
2. Dense + BM25 hybrid at alpha=0.7/0.3
3. Use Groq (llama-3.1-8b-instant) for LLM
4. Output formats: JSON rules (primary), Python function (secondary), ML features (tertiary)
5. FAISS index built at startup, cached to disk for subsequent runs
6. Gradio UI for interactive demo

## API Response Format
```json
{
  "query": "Find all policies covering emergency surgery",
  "summary": "3 policies found covering emergency surgical procedures...",
  "retrieved_policies": [...],
  "conflicts": ["Policy A requires pre-auth, Policy C waives it for emergencies"],
  "analysis": "...",
  "converted_output": {
    "json_rules": {...},
    "python_code": "def evaluate_claim(...):",
    "features": {...}
  }
}
```
