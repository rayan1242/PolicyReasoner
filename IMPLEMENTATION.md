# PolicyReasoner — Implementation Plan

## Phase 1: Data Layer
- [ ] Create `data/policies.json` with 40 mock healthcare policies
  - Each policy has: id, name, category, sections (title + content)
  - Categories: emergency, surgical, mental_health, prescription, preventive, referral, lab, therapy, out_of_network, billing_contract
  - 4 policies per category = 40 total

## Phase 2: Retrieval Infrastructure (adapted from DeepGit)
- [ ] `tools/chat.py` — Groq LLM wrapper
  - Reuse DeepGit's `chat.py` but change system prompt to healthcare policy context
  - Function: `expand_policy_query(query) → search_tags`
- [ ] `tools/policy_ingestor.py` — FAISS index builder
  - Load `policies.json`
  - Chunk each policy section into ~500 char chunks
  - Embed with sentence-transformers (all-mpnet-base-v2, lighter than ColBERT for speed)
  - Build FAISS index, save to disk as `policy_index.faiss` + `policy_chunks.pkl`
- [ ] `tools/dense_retrieval.py` — hybrid retrieval
  - Adapt DeepGit's ColBERT+BM25 hybrid
  - Input: `user_query`, FAISS index
  - Output: top-K policy chunks with scores
- [ ] `tools/cross_encoder_reranking.py` — re-rank
  - Adapt DeepGit's cross-encoder code directly
  - Input: query + chunks, Output: re-ranked top-N chunks

## Phase 3: Analysis (NEW)
- [ ] `tools/policy_analyzer.py`
  - `analyze_policies(query, chunks) → {analysis: str, conflicts: list}`
  - LLM prompt: given query + top policy chunks, identify:
    1. Which policies are most relevant
    2. Any contradictions between policies (e.g., different deductibles for same procedure)
    3. Key clauses that apply
  - `summarize_policies(analysis, chunks) → summary: str`
  - Plain English summary for non-technical users

## Phase 4: Conversion (NEW — key differentiator)
- [ ] `tools/policy_converter.py`
  - `policy_to_json_rules(policy_text) → dict` — decision rules
  - `policy_to_python(policy_text) → str` — Python function
  - `policy_to_features(policy_text) → str` — ML feature definitions
  - All use LLM prompts with structured output parsing

## Phase 5: Agent Orchestration
- [ ] `agent.py` — LangGraph StateGraph
  - Nodes: expand_query, retrieve, rerank, analyze, summarize, convert
  - Linear pipeline (no parallel branches for simplicity)
  - State: PolicyState dataclass
  - Config: PolicyAgentConfig (top_k, format, model, etc.)

## Phase 6: API Server
- [ ] `app.py` — FastAPI
  - `POST /analyze` — run full agent pipeline, return analysis + summary
  - `POST /convert` — convert specific policy text to code
  - `GET /policies` — list all policies in corpus
  - `GET /health` — health check
  - Startup event: build FAISS index if not exists

## Phase 7: Tests
- [ ] `tests/test_retrieval.py` — 5 tests
  - test_ingestor_loads_policies
  - test_faiss_index_builds
  - test_retrieval_returns_results
  - test_reranking_improves_order
  - test_empty_query_handled
- [ ] `tests/test_analyzer.py` — 3 tests
  - test_analyze_returns_analysis_and_conflicts
  - test_summarize_returns_string
  - test_conflict_detected_when_policies_differ
- [ ] `tests/test_converter.py` — 4 tests
  - test_policy_to_json_returns_dict
  - test_policy_to_python_returns_function
  - test_policy_to_features_returns_dict
  - test_all_formats_generated

## Phase 8: Docker + CI
- [ ] `Dockerfile` — multi-stage, python:3.12-slim
- [ ] `.github/workflows/docker-image.yml` — adapted from DeepGit

## Decisions Made
1. Use `all-mpnet-base-v2` (not full ColBERT) for embeddings — faster on CPU, still high quality
2. Keep ColBERT for the dense_retrieval.py to match DeepGit's original approach (shows continuity)
3. Use Groq (llama-3.1-8b-instant) for LLM — already configured in DeepGit venv
4. Output formats: JSON rules (primary), Python function (secondary), ML features (tertiary)
5. FAISS index built at startup, cached to disk for subsequent runs
6. Gradio UI optional (add if time permits after core works)

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
