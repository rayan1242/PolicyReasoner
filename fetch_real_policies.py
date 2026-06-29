"""
fetch_real_policies.py

Fetches real healthcare policy content from two public sources:
  1. OpenFDA drug label API  — real payer-relevant drug coverage data
  2. Wikipedia REST API       — real healthcare coverage policy articles

Appends fetched policies to data/real_policies.json and rebuilds the FAISS index.

Usage:
    python fetch_real_policies.py
"""

import json
import time
import logging
import urllib.request
import urllib.parse
import urllib.error
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "real_policies.json"
INDEX_PATH  = Path(__file__).resolve().parent / "data" / "policy_index.pkl"


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def fetch(url: str, accept: str = "application/json") -> bytes | None:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "PolicyReasoner/1.0 (research; contact: rayyanmaindargi12@gmail.com)",
            "Accept": accept,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        logger.warning(f"HTTP {e.code} — {url}")
    except Exception as e:
        logger.warning(f"Fetch failed ({e}) — {url}")
    return None


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[\d+\]', '', text)        # remove Wikipedia footnotes [1]
    text = re.sub(r'={2,}[^=]+=+', '', text)   # remove wiki section headers
    return text.strip()


# ---------------------------------------------------------------------------
# Source 1: OpenFDA drug labels
# ---------------------------------------------------------------------------

FDA_DRUGS = [
    ("metformin",      "Metformin (Diabetes)",              "prescription"),
    ("lisinopril",     "Lisinopril (Hypertension)",         "prescription"),
    ("atorvastatin",   "Atorvastatin (High Cholesterol)",   "prescription"),
    ("buprenorphine",  "Buprenorphine (Opioid Use Disorder)","prescription"),
    ("insulin glargine","Insulin Glargine (Diabetes)",       "prescription"),
    ("adalimumab",     "Adalimumab/Humira (Autoimmune)",    "prescription"),
    ("sertraline",     "Sertraline (Depression/Anxiety)",   "mental_health"),
    ("albuterol",      "Albuterol (Asthma)",                "preventive"),
]

SECTIONS_MAP = {
    "indications_and_usage":        "Indications and Usage",
    "contraindications":            "Contraindications",
    "warnings_and_cautions":        "Warnings and Cautions",
    "dosage_and_administration":    "Dosage and Administration",
    "adverse_reactions":            "Adverse Reactions",
    "drug_interactions":            "Drug Interactions",
    "use_in_specific_populations":  "Use in Specific Populations",
}


def fetch_fda_policy(drug_name: str, display_name: str, category: str) -> dict | None:
    encoded = urllib.parse.quote(drug_name)
    url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{encoded}&limit=1"
    raw = fetch(url)
    if not raw:
        return None

    try:
        data = json.loads(raw)
        results = data.get("results", [])
        if not results:
            return None
        label = results[0]
    except (json.JSONDecodeError, KeyError):
        return None

    openfda = label.get("openfda", {})
    brand   = openfda.get("brand_name", [display_name])[0]
    generic = openfda.get("generic_name", [drug_name])[0]
    mfr     = openfda.get("manufacturer_name", ["Unknown"])[0]

    sections = []
    for key, title in SECTIONS_MAP.items():
        val = label.get(key)
        if isinstance(val, list) and val:
            content = clean_text(val[0])[:800]
            if len(content) > 60:
                sections.append({"title": title, "content": content})

    # Coverage-specific section derived from indications
    indications = label.get("indications_and_usage", [""])[0]
    if indications:
        sections.append({
            "title": "Coverage Eligibility Notes",
            "content": (
                f"{brand} ({generic}) is typically covered under the pharmacy benefit "
                f"(Tier 2–4 depending on formulary). Prior authorization may be required "
                f"for brand-name prescribing when generics exist. "
                f"Manufactured by {mfr}. "
                + clean_text(indications)[:400]
            )
        })

    if not sections:
        return None

    policy_id = f"FDA-{drug_name.upper().replace(' ', '-')[:20]}-001"
    logger.info(f"  ✓ FDA: {policy_id} — {brand}")
    return {
        "policy_id":      policy_id,
        "name":           f"{display_name} — Drug Label & Coverage",
        "category":       category,
        "version":        "FDA-Label",
        "effective_date": "2024-01-01",
        "source":         "OpenFDA",
        "source_url":     url,
        "sections":       sections,
    }


# ---------------------------------------------------------------------------
# Source 2: Wikipedia healthcare policy articles
# ---------------------------------------------------------------------------

WIKI_ARTICLES = [
    ("Managed_care",                          "Managed Care Overview",                    "billing_contract"),
    ("Prior_authorization",                   "Prior Authorization in Healthcare",        "billing_contract"),
    ("Copayment",                             "Copayment and Cost Sharing Policy",        "billing_contract"),
    ("Deductible",                            "Deductible Policy",                        "billing_contract"),
    ("Out-of-pocket_maximum",                 "Out-of-Pocket Maximum Policy",             "out_of_network"),
    ("Emergency_Medical_Treatment_and_Labor_Act", "EMTALA Emergency Coverage Law",        "emergency"),
    ("No_Surprises_Act",                      "No Surprises Act (Surprise Billing)",      "out_of_network"),
    ("Mental_Health_Parity_Act",              "Mental Health Parity Act",                 "mental_health"),
    ("Affordable_Care_Act",                   "ACA Essential Health Benefits",            "preventive"),
    ("Medicare_(United_States)",              "Medicare Coverage Overview",               "billing_contract"),
    ("Medicaid",                              "Medicaid Coverage Policy",                 "billing_contract"),
    ("Health_maintenance_organization",       "HMO Referral and Network Policy",          "referral"),
    ("Preferred_provider_organization",       "PPO Out-of-Network Policy",                "out_of_network"),
    ("Pre-existing_condition",                "Pre-existing Condition Coverage",          "billing_contract"),
    ("Drug_formulary",                        "Prescription Drug Formulary Policy",       "prescription"),
    ("Step_therapy",                          "Step Therapy Policy",                      "prescription"),
    ("Physical_therapy",                      "Physical Therapy Coverage",                "therapy"),
    ("Telemedicine",                          "Telehealth Coverage Policy",               "referral"),
    ("Ambulance_services_in_the_United_States","Ambulance and Emergency Transport Coverage","emergency"),
    ("Diagnostic_imaging",                    "Diagnostic Imaging Coverage (MRI/CT)",     "lab"),
]

WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
WIKI_CONTENT_API = "https://en.wikipedia.org/w/api.php?action=query&titles={}&prop=extracts&exintro=false&explaintext=true&format=json"


def split_into_sections(text: str, article_title: str) -> list[dict]:
    """Split Wikipedia plain text into logical sections of ~600 chars."""
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 80]
    if not paragraphs:
        return []

    sections = []
    chunk = ""
    section_num = 1

    for para in paragraphs:
        if len(chunk) + len(para) < 700:
            chunk += " " + para
        else:
            if chunk.strip():
                sections.append({
                    "title": f"Section {section_num}",
                    "content": clean_text(chunk)[:700],
                })
                section_num += 1
            chunk = para

    if chunk.strip():
        sections.append({
            "title": f"Section {section_num}",
            "content": clean_text(chunk)[:700],
        })

    # Keep max 6 sections per article
    return sections[:6]


def fetch_wikipedia_policy(article_title: str, display_name: str, category: str) -> dict | None:
    # Get full article text
    url = WIKI_CONTENT_API.format(urllib.parse.quote(article_title))
    raw = fetch(url)
    if not raw:
        return None

    try:
        data = json.loads(raw)
        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        text = page.get("extract", "")
    except (json.JSONDecodeError, StopIteration, KeyError):
        return None

    if len(text) < 200:
        return None

    sections = split_into_sections(text, display_name)
    if not sections:
        return None

    policy_id = f"WIKI-{article_title[:20].upper().replace('_', '-')}-001"
    logger.info(f"  ✓ Wiki: {policy_id} — {display_name}")
    return {
        "policy_id":      policy_id,
        "name":           display_name,
        "category":       category,
        "version":        "Wikipedia",
        "effective_date": "2024-01-01",
        "source":         "Wikipedia",
        "source_url":     f"https://en.wikipedia.org/wiki/{article_title}",
        "sections":       sections,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    policies = []

    logger.info("=== Fetching from OpenFDA ===")
    for drug_name, display, cat in FDA_DRUGS:
        p = fetch_fda_policy(drug_name, display, cat)
        if p:
            policies.append(p)
        time.sleep(0.3)

    logger.info("=== Fetching from Wikipedia ===")
    for article, display, cat in WIKI_ARTICLES:
        p = fetch_wikipedia_policy(article, display, cat)
        if p:
            policies.append(p)
        time.sleep(0.5)

    logger.info(f"\nFetched {len(policies)} real policies total.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(policies, f, indent=2)
    logger.info(f"Saved to {OUTPUT_PATH}")

    # Merge with mock policies and rebuild index
    mock_path = Path(__file__).resolve().parent / "data" / "policies.json"
    combined_path = Path(__file__).resolve().parent / "data" / "policies_combined.json"

    with open(mock_path, "r", encoding="utf-8") as f:
        mock = json.load(f)

    combined = mock + policies
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"Combined corpus: {len(combined)} policies → {combined_path}")

    # Delete cached index so it rebuilds on next run
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
        logger.info("Cleared FAISS index cache — will rebuild on next startup.")

    logger.info("\nDone! Restart the app (python ui.py) to use the real policies.")


if __name__ == "__main__":
    main()
