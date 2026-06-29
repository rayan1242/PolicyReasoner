import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph

from tools.dense_retrieval import hybrid_policy_retrieval
from tools.cross_encoder_reranking import rerank_policies
from tools.policy_analyzer import analyze_policies, summarize_policies
from tools.policy_converter import convert_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

dotenv_path = Path(__file__).resolve().parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class PolicyState:
    user_query: str = field(default="")
    search_tags: str = field(default="")
    retrieved_policies: list[Any] = field(default_factory=list)
    reranked_policies: list[Any] = field(default_factory=list)
    analysis: str = field(default="")
    conflicts: list[Any] = field(default_factory=list)
    grounding: list[Any] = field(default_factory=list)
    summary: str = field(default="")
    summary_confidence: float = field(default=0.0)
    summary_confidence_reason: str = field(default="")
    converted_output: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class PolicyStateInput:
    user_query: str = field(default="")


@dataclass(kw_only=True)
class PolicyStateOutput:
    summary: str = field(default="")
    summary_confidence: float = field(default=0.0)
    summary_confidence_reason: str = field(default="")
    analysis: str = field(default="")
    conflicts: list[Any] = field(default_factory=list)
    grounding: list[Any] = field(default_factory=list)
    reranked_policies: list[Any] = field(default_factory=list)
    converted_output: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class PolicyAgentConfig(BaseModel):
    top_k: int = Field(20, description="Number of chunks to retrieve from FAISS")
    rerank_top_n: int = Field(8, description="Top N chunks after cross-encoder re-ranking")
    retrieval_alpha: float = Field(0.7, description="Weight for dense vs sparse retrieval (1=dense only)")
    output_format: str = Field("all", description="Conversion format: json | python | features | all")

    @classmethod
    def from_runnable_config(cls, config: Any = None) -> "PolicyAgentConfig":
        cfg = (config or {}).get("configurable", {})
        return cls(**{k: cfg[k] for k in cls.model_fields if k in cfg})


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

builder = StateGraph(
    PolicyState,
    input=PolicyStateInput,
    output=PolicyStateOutput,
    config_schema=PolicyAgentConfig,
)

builder.add_node("retrieve_policies", hybrid_policy_retrieval)
builder.add_node("rerank_policies", rerank_policies)
builder.add_node("analyze_policies", analyze_policies)
builder.add_node("summarize_policies", summarize_policies)
builder.add_node("convert_policy", convert_policy)

builder.add_edge(START, "retrieve_policies")
builder.add_edge("retrieve_policies", "rerank_policies")
builder.add_edge("rerank_policies", "analyze_policies")
builder.add_edge("analyze_policies", "summarize_policies")
builder.add_edge("summarize_policies", "convert_policy")
builder.add_edge("convert_policy", END)

graph = builder.compile()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = graph.invoke(
        PolicyStateInput(
            user_query="Do I need preauthorization for emergency surgery, and how much will it cost me?"
        )
    )
    print("\n=== SUMMARY ===")
    print(result["summary"])
    print("\n=== ANALYSIS ===")
    print(result["analysis"])
    print("\n=== CONFLICTS ===")
    for c in result["conflicts"]:
        print(f"  - {c}")
    print("\n=== CONVERTED OUTPUT ===")
    import json
    out = dict(result["converted_output"])
    out.pop("python_code", None)
    print(json.dumps(out, indent=2))
