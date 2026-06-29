import logging
import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CE_MODEL)
    return _cross_encoder


def rerank_policies(state, config) -> dict:
    cfg = config.get("configurable", {}) if isinstance(config, dict) else {}
    top_n = int(cfg.get("rerank_top_n", 8))

    query = state.user_query
    candidates = state.retrieved_policies

    if not candidates:
        state.reranked_policies = []
        return {"reranked_policies": []}

    ce = _get_cross_encoder()
    pairs = [[query, c["text"]] for c in candidates]

    try:
        scores = ce.predict(pairs, show_progress_bar=False)
    except Exception as e:
        logger.error(f"Cross-encoder scoring failed: {e}")
        state.reranked_policies = candidates[:top_n]
        return {"reranked_policies": candidates[:top_n]}

    # Shift negative scores
    min_score = float(np.min(scores))
    if min_score < 0:
        scores = scores - min_score

    for i, chunk in enumerate(candidates):
        chunk["cross_encoder_score"] = float(scores[i])

    reranked = sorted(candidates, key=lambda x: x["cross_encoder_score"], reverse=True)[:top_n]
    logger.info(f"Re-ranked to top {len(reranked)} policy chunks.")
    state.reranked_policies = reranked
    return {"reranked_policies": reranked}
