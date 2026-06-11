"""
Download FDA drug labels (XML) from DailyMed.
"""

import os
import re
import json
import zipfile
import io
import time
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drug_labels")

DRUGS = [
    ("metformin", "tablet extended release"),
    ("glipizide", "tablet"),
    ("dapagliflozin", "tablet"),
    ("sitagliptin", "tablet"),
    ("amlodipine", "tablet"),
    ("lisinopril", "tablet"),
    ("losartan", "tablet"),
    ("hydrochlorothiazide", "tablet"),
    ("atorvastatin", "tablet"),
    ("rosuvastatin", "tablet"),
    ("metoprolol", "tablet"),
    ("aspirin", "tablet"),
    ("levothyroxine", "tablet"),
    ("albuterol", "aerosol"),
    ("alendronate", "tablet"),
]

BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

_retry = Retry(total=4, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
_session = requests.Session()
_session.mount("https://", HTTPAdapter(max_retries=_retry))


def get(url, binary=False):
    resp = _session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content if binary else resp.text


def search_spl(drug_name, dosage_form):
    """Search DailyMed for the most recently published HUMAN PRESCRIPTION DRUG
    LABEL whose title mentions the desired dosage form (tablet, or aerosol for
    albuterol), and return its setid.

    Note: the search response only exposes "published_date" (no "updated_date"
    field exists in the v2 API), so we treat that as the recency signal.
    """
    from urllib.parse import quote

    # tablet / aerosol -> the keyword we look for as a whole word in the title
    form_keyword = "AEROSOL" if "aerosol" in dosage_form.lower() else "TABLET"
    pattern = re.compile(rf"\b{form_keyword}\b")

    matches = []
    for page in range(1, 6):  # scan up to 500 results before giving up
        url = (
            f"{BASE}/spls.json?drug_name={quote(drug_name)}"
            "&labeltypes=HUMAN%20PRESCRIPTION%20DRUG%20LABEL"
            f"&pagesize=100&page={page}"
        )
        data = json.loads(get(url))
        results = data.get("data", [])
        if not results:
            break

        matches.extend(item for item in results if pattern.search(item.get("title", "")))

        if matches and page >= 2:
            break

    if not matches:
        return None

    # Prefer single-ingredient labels (title starts with drug name), e.g. exclude
    # "GLYBURIDE AND METFORMIN TABLET" when searching for metformin.
    drug_upper = drug_name.upper()
    single = [m for m in matches if m.get("title", "").upper().startswith(drug_upper)]
    candidates = single if single else matches

    candidates.sort(key=lambda item: datetime.strptime(item["published_date"], "%b %d, %Y"), reverse=True)
    return candidates[0]["setid"]


def download_xml(setid, drug_name):
    url = f"https://dailymed.nlm.nih.gov/dailymed/downloadzipfile.cfm?setId={setid}"
    content = get(url, binary=True)

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        xml_files = [n for n in zf.namelist() if n.endswith(".xml")]
        if not xml_files:
            raise ValueError("No XML in zip")
        xml_content = zf.read(xml_files[0])

    out_path = os.path.join(OUTPUT_DIR, f"{drug_name.replace(' ', '_')}.xml")
    with open(out_path, "wb") as f:
        f.write(xml_content)
    return out_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving to: {OUTPUT_DIR}\n")

    success, failed = [], []

    for drug_name, dosage_form in DRUGS:
        print(f"[{drug_name}] searching...", end=" ", flush=True)
        try:
            setid = search_spl(drug_name, dosage_form)
            if not setid:
                print("NOT FOUND")
                failed.append((drug_name, "no result"))
                continue
            path = download_xml(setid, drug_name)
            kb = os.path.getsize(path) // 1024
            print(f"OK ({kb} KB)")
            success.append(drug_name)
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append((drug_name, str(e)))
        time.sleep(0.3)

    print(f"\nDone: {len(success)}/{len(DRUGS)} downloaded.")
    if failed:
        print("Failed:")
        for name, reason in failed:
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()
