import json
import pickle
import logging
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Use combined corpus (mock + real) if available, else fall back to mock only
POLICIES_PATH = (
    _DATA_DIR / "policies_combined.json"
    if (_DATA_DIR / "policies_combined.json").exists()
    else _DATA_DIR / "policies.json"
)
INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "policy_index.pkl"

EMBED_MODEL = "all-mpnet-base-v2"
CHUNK_SIZE = 600


def load_policies() -> list[dict]:
    with open(POLICIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_policies(policies: list[dict]) -> list[dict]:
    chunks = []
    for policy in policies:
        base = {
            "policy_id": policy["policy_id"],
            "name": policy["name"],
            "category": policy["category"],
            "version": policy["version"],
            "effective_date": policy["effective_date"],
        }
        for section in policy.get("sections", []):
            text = f"{policy['name']} - {section['title']}: {section['content']}"
            # split if over chunk size
            for i in range(0, len(text), CHUNK_SIZE):
                chunk_text = text[i: i + CHUNK_SIZE]
                chunks.append({
                    **base,
                    "section_title": section["title"],
                    "text": chunk_text,
                    "combined_doc": text,
                })
    return chunks


def build_index(force: bool = False) -> tuple[list[dict], np.ndarray]:
    if not force and INDEX_PATH.exists():
        logger.info("Loading cached policy index from disk...")
        with open(INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        return data["chunks"], data["embeddings"]

    logger.info("Building policy index from scratch...")
    policies = load_policies()
    chunks = chunk_policies(policies)

    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)

    logger.info(f"Index built: {len(chunks)} chunks from {len(policies)} policies.")
    return chunks, embeddings


def get_all_policy_names() -> list[dict]:
    policies = load_policies()
    return [
        {
            "policy_id": p["policy_id"],
            "name": p["name"],
            "category": p["category"],
            "version": p["version"],
            "effective_date": p["effective_date"],
        }
        for p in policies
    ]
