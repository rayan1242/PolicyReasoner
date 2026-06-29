# PolicyReasoner — CLAUDE.md

## Project Purpose
Cotiviti internship assessment submission. Topic 3: Content Management in Healthcare.
Agentic AI system for intelligent healthcare policy discovery, analysis, conflict detection,
and conversion of written policy into executable code (JSON rules / Python / ML features).

## What This Is
PolicyReasoner is a LangGraph-based agentic pipeline that:
1. Takes a natural language query about healthcare policies
2. Retrieves the most relevant policy sections from a FAISS vector store
3. Re-ranks them using a cross-encoder (sentence-transformers)
4. Analyzes the retrieved policies for conflicts and key findings (via LLM)
5. Summarizes the results in plain English
6. Converts a selected policy into executable code (JSON rules, Python, or ML features)

## Source: DeepGit (what was reused)
The DeepGit project (github.com/rayan1242/DeepGit2) is the parent codebase.
PolicyReasoner adapts the following components:
- `tools/chat.py` → LLM wrapper using Groq (llama-3.1-8b-instant)
- `tools/dense_retrieval.py` → ColBERT + BM25 hybrid FAISS retrieval
- `tools/cross_encoder_reranking.py` → MiniLM cross-encoder re-ranking
- `agent.py` → LangGraph StateGraph pattern
- `requirements.txt` → same base dependencies

New components added for PolicyReasoner:
- `data/policies.json` → 40 mock healthcare policy documents
- `tools/policy_ingestor.py` → loads, chunks, embeds policies into FAISS
- `tools/policy_analyzer.py` → LLM conflict detection + analysis
- `tools/policy_converter.py` → policy text → JSON rules / Python function / ML features
- `app.py` → FastAPI server with `/analyze`, `/convert`, `/policies` endpoints

## LangGraph Agent Flow
```
START
  → expand_query          (convert NL query to search tags via LLM)
  → retrieve_policies     (FAISS ColBERT + BM25 hybrid search)
  → rerank_policies       (cross-encoder MiniLM re-ranking)
  → analyze_policies      (LLM: identify conflicts, key clauses)
  → summarize_policies    (LLM: plain English summary)
  → convert_policy        (LLM: policy → JSON / Python / features)
END
```

## Key State Fields (PolicyState)
- `user_query` — raw user input
- `search_tags` — LLM-extracted search terms
- `retrieved_policies` — top-K policy chunks from FAISS
- `reranked_policies` — re-ranked by cross-encoder
- `analysis` — LLM findings (conflicts, key clauses)
- `conflicts` — list of detected policy conflicts
- `summary` — plain English summary
- `converted_output` — dict with json_rules, python_code, features
- `output_format` — "json" | "python" | "features" | "all"

## Tech Stack
- **LLM**: Groq API (llama-3.1-8b-instant) via langchain_groq
- **Embeddings**: ColBERT (colbert-ir/colbertv2.0) via HuggingFace
- **Sparse retrieval**: BM25 (rank_bm25)
- **Vector store**: FAISS (faiss-cpu)
- **Re-ranking**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Agent orchestration**: LangGraph
- **API server**: FastAPI + uvicorn
- **Tests**: pytest

## Environment Variables (.env)
- `GROQ_API_KEY` — Groq API key

## Policy Corpus (data/policies.json)
40 mock healthcare policies covering:
- Emergency room visits
- Surgical procedures
- Mental health / behavioral health
- Prescription drug coverage
- Preventive care
- Specialty referrals
- Lab tests / diagnostics
- Physical therapy / rehab
- Out-of-network coverage
- Payer-provider billing contracts

## Project Structure
```
PolicyReasoner/
├── CLAUDE.md               ← this file
├── IMPLEMENTATION.md       ← implementation plan
├── .env                    ← GROQ_API_KEY
├── .gitignore
├── requirements.txt
├── Dockerfile
├── langgraph.json
├── agent.py                ← LangGraph workflow
├── app.py                  ← FastAPI server
├── data/
│   └── policies.json       ← 40 mock healthcare policies
├── tools/
│   ├── __init__.py
│   ├── chat.py             ← Groq LLM wrapper (adapted from DeepGit)
│   ├── policy_ingestor.py  ← load + embed policies into FAISS
│   ├── dense_retrieval.py  ← ColBERT+BM25 hybrid retrieval
│   ├── cross_encoder_reranking.py ← MiniLM re-ranking
│   ├── policy_analyzer.py  ← conflict detection + analysis (NEW)
│   └── policy_converter.py ← policy → code (NEW)
└── tests/
    ├── __init__.py
    ├── test_retrieval.py
    ├── test_analyzer.py
    └── test_converter.py
```
