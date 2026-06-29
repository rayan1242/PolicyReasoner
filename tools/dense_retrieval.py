import logging
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from tools.policy_ingestor import build_index

logger = logging.getLogger(__name__)

EMBED_MODEL = "all-mpnet-base-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def hybrid_policy_retrieval(state, config) -> dict:
    cfg = config.get("configurable", {}) if isinstance(config, dict) else {}
    top_k = int(cfg.get("top_k", 20))
    alpha = float(cfg.get("retrieval_alpha", 0.7))

    query = state.user_query
    chunks, embeddings = build_index()

    model = _get_model()
    query_embedding = model.encode([query], normalize_embeddings=True)[0]

    # Dense scores (cosine similarity since embeddings are normalized)
    dense_scores = np.dot(embeddings, query_embedding)

    # BM25 sparse scores
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = np.array(bm25.get_scores(query.lower().split()))

    # Normalize both
    def normalize(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-10)

    combined = alpha * normalize(dense_scores) + (1 - alpha) * normalize(bm25_scores)

    top_indices = np.argsort(combined)[::-1][:top_k]
    retrieved = []
    for idx in top_indices:
        chunk = dict(chunks[idx])
        chunk["retrieval_score"] = float(combined[idx])
        retrieved.append(chunk)

    logger.info(f"Retrieved {len(retrieved)} policy chunks for query: '{query[:60]}...'")
    state.retrieved_policies = retrieved
    return {"retrieved_policies": retrieved}
