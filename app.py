import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tools.policy_ingestor import build_index, get_all_policy_names
from tools.policy_converter import policy_to_json_rules, policy_to_python, policy_to_features
from agent import graph, PolicyStateInput

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building FAISS policy index on startup...")
    build_index()
    logger.info("Policy index ready.")
    yield


app = FastAPI(
    title="PolicyReasoner",
    description="Agentic AI for healthcare policy discovery, analysis, and code generation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    query: str = Field(..., description="Natural language question about healthcare policies")
    top_k: int = Field(20, ge=5, le=50, description="Number of chunks to retrieve")
    rerank_top_n: int = Field(8, ge=3, le=20, description="Top N after re-ranking")
    output_format: Literal["json", "python", "features", "all"] = Field("all")


class PolicyChunk(BaseModel):
    policy_id: str
    name: str
    category: str
    section_title: str
    text: str
    retrieval_score: float = 0.0
    cross_encoder_score: float = 0.0


class AnalyzeResponse(BaseModel):
    query: str
    summary: str
    analysis: str
    conflicts: list[str]
    top_policies: list[PolicyChunk]
    converted_output: dict


class ConvertRequest(BaseModel):
    policy_text: str = Field(..., description="Raw policy text to convert")
    format: Literal["json", "python", "features"] = Field("json")


class ConvertResponse(BaseModel):
    format: str
    output: dict | str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "PolicyReasoner"}


@app.get("/policies")
def list_policies(category: str | None = None):
    """List all policies in the corpus, optionally filtered by category."""
    policies = get_all_policy_names()
    if category:
        policies = [p for p in policies if p["category"] == category]
    return {"total": len(policies), "policies": policies}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Run the full PolicyReasoner pipeline:
    retrieve → rerank → analyze → summarize → convert
    """
    try:
        result = graph.invoke(
            PolicyStateInput(user_query=req.query),
            config={
                "configurable": {
                    "top_k": req.top_k,
                    "rerank_top_n": req.rerank_top_n,
                    "output_format": req.output_format,
                }
            },
        )
    except Exception as e:
        logger.error(f"Agent pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    top_policies = [
        PolicyChunk(
            policy_id=c.get("policy_id", ""),
            name=c.get("name", ""),
            category=c.get("category", ""),
            section_title=c.get("section_title", ""),
            text=c.get("text", ""),
            retrieval_score=c.get("retrieval_score", 0.0),
            cross_encoder_score=c.get("cross_encoder_score", 0.0),
        )
        for c in result.get("reranked_policies", [])
    ]

    return AnalyzeResponse(
        query=req.query,
        summary=result.get("summary", ""),
        analysis=result.get("analysis", ""),
        conflicts=result.get("conflicts", []),
        top_policies=top_policies,
        converted_output=result.get("converted_output", {}),
    )


@app.post("/convert", response_model=ConvertResponse)
def convert(req: ConvertRequest):
    """Convert raw policy text to JSON rules, Python code, or ML features."""
    try:
        if req.format == "json":
            output = policy_to_json_rules(req.policy_text)
        elif req.format == "python":
            output = policy_to_python(req.policy_text)
        elif req.format == "features":
            output = policy_to_features(req.policy_text)
        else:
            raise HTTPException(status_code=400, detail="Unknown format")
    except Exception as e:
        logger.error(f"Conversion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return ConvertResponse(format=req.format, output=output)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
