# PolicyReasoner

Agentic AI system for healthcare policy discovery, conflict detection, and automatic conversion of written policy into executable code.

Built as a proof of concept for the **Cotiviti Intern Assessment — Topic 3: Content Management in Health Care**.

---

## What It Does

PolicyReasoner takes a natural language question about healthcare policies and runs it through a six-stage LangGraph pipeline:

1. **Query Expansion** — LLM converts your question into precise healthcare search tags
2. **Hybrid Retrieval** — FAISS dense search + BM25 keyword search over 58 real policies (301 chunks)
3. **Cross-Encoder Re-ranking** — MiniLM reads query + document together for accurate relevance scores
4. **Policy Analysis** — LLM identifies key findings, extracts evidence with exact quotes, detects conflicts using a federal → state → payer → hospital priority chain
5. **Grounded Summary** — Plain English answer with a confidence score and traceable sources
6. **Policy → Code** — Converts the top policy into JSON decision rules, a Python `evaluate_claim()` function (compiled + dry-run tested), and ML feature definitions

---

## Architecture

```
User Query
    ↓
Query Expansion (Groq LLM)
    ↓
Hybrid Retrieval (FAISS cosine + BM25, α=0.7)
    ↓
Cross-Encoder Re-ranking (MiniLM, top 20 → top 8)
    ↓
Policy Analysis (LLM → findings + evidence + conflicts)
    ↓
Summarization (LLM → plain English + confidence score)
    ↓
Policy → Code (JSON rules + Python function + ML features, validated)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq API — `llama-3.1-8b-instant` |
| Embeddings | `all-mpnet-base-v2` via HuggingFace |
| Sparse retrieval | BM25 (`rank_bm25`) |
| Vector store | FAISS (`faiss-cpu`) |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Agent orchestration | LangGraph |
| UI | Gradio |
| API server | FastAPI + uvicorn |
| Tests | pytest |

---

## Policy Corpus

58 policies across 10 categories:
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

40 mock policies in `data/policies.json`. Run `fetch_real_policies.py` to pull real data from **OpenFDA** and **Wikipedia** and merge into `data/policies_combined.json` (58 total).

---

## Setup

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/) (free tier works)

### Install

```bash
git clone https://github.com/rayan1242/PolicyReasoner.git
cd PolicyReasoner

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Configure

```bash
# Create .env file
echo GROQ_API_KEY=your_key_here > .env
```

### (Optional) Fetch real policies

```bash
python fetch_real_policies.py
```

### Run the UI

```bash
python ui.py
# Open http://127.0.0.1:7860
```

### Run the API server

```bash
python app.py
# Open http://127.0.0.1:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/analyze` | Run full pipeline on a query |
| `POST` | `/convert` | Convert a policy to code |
| `GET` | `/policies` | List all indexed policies |
| `GET` | `/health` | Health check |

---

## Example Queries

```
Do I need preauthorization for emergency surgery and how much will it cost?

What are the prior authorization requirements for buprenorphine?

Find all billing policies covering out-of-network emergency care and detect conflicts.

Compare preauthorization requirements for MRI vs routine lab tests.
```

---

## Tests

```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
PolicyReasoner/
├── agent.py                 ← LangGraph StateGraph workflow
├── app.py                   ← FastAPI REST server
├── ui.py                    ← Gradio web UI
├── fetch_real_policies.py   ← Fetches OpenFDA + Wikipedia policies
├── requirements.txt
├── data/
│   └── policies.json        ← 40 mock healthcare policies
└── tools/
    ├── chat.py              ← Groq LLM wrapper + prompts
    ├── policy_ingestor.py   ← FAISS index builder
    ├── dense_retrieval.py   ← Hybrid retrieval node
    ├── cross_encoder_reranking.py ← MiniLM re-ranking node
    ├── policy_analyzer.py   ← Conflict detection + analysis node
    └── policy_converter.py  ← Policy → code node
```
## Demo Video

[![PolicyReasoner Demo](https://img.youtube.com/vi/jXh4IEm9z-I/maxresdefault.jpg)](https://youtu.be/jXh4IEm9z-I)

---

## Author

**Rayyan Maindargi**  
Illinois Institute of Technology  
rayyanmaindargi12@gmail.com
