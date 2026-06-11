"""
Step 2: Parse DailyMed SPL XML files and extract section-level chunks.

SPL XML structure (HL7 v3, namespace urn:hl7-org:v3):
  document → component → structuredBody → component → section (recursive)

Each <section> has:
  <code code="LOINC_CODE"/>   — section type identifier
  <title>...</title>           — section title (sometimes absent)
  <text>...</text>             — clinical text content
  <section>...</section>       — subsections (recursive)

Output: chunks/all_chunks.jsonl
Each line: {"drug", "loinc_code", "section", "text", "char_count", "chunk_index"}
"""

import os
import re
import json
import xml.etree.ElementTree as ET

NS = "urn:hl7-org:v3"

# Skip these LOINC section codes — duplicates or packaging artwork
SKIP_CODES = {
    "51945-4",  # PRINCIPAL DISPLAY PANEL
    "48780-1",  # SPL Highlights (summary that duplicates full content)
}

# Fallback human-readable names for sections that lack a <title> element
LOINC_NAMES = {
    "34066-1": "Boxed Warning",
    "34067-9": "Indications and Usage",
    "34068-7": "Dosage and Administration",
    "43678-2": "Dosage Forms and Strengths",
    "34070-3": "Contraindications",
    "43685-7": "Warnings and Precautions",
    "34084-4": "Adverse Reactions",
    "34073-7": "Drug Interactions",
    "43684-0": "Use in Specific Populations",
    "42228-7": "Pregnancy",
    "34081-0": "Pediatric Use",
    "34082-8": "Geriatric Use",
    "34088-5": "Overdosage",
    "34089-3": "Description",
    "34090-1": "Clinical Pharmacology",
    "43679-0": "Mechanism of Action",
    "43682-4": "Pharmacokinetics",
    "43680-8": "Nonclinical Toxicology",
    "34092-7": "Clinical Studies",
    "34069-5": "How Supplied / Storage",
    "34076-0": "Patient Counseling Information",
    "44425-7": "Storage and Handling",
    "42229-5": "Unclassified Section",
    "42230-3": "Patient Information",
    "42231-1": "Medication Guide",
    "88436-1": "Patient Counseling",
    "34077-8": "Drug Interactions",
    "34083-6": "Precautions",
}

DRUG_LABELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drug_labels")
CHUNKS_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chunks")

MIN_CHARS  = 40    # skip chunks shorter than this
MAX_CHARS  = 2000  # split chunks larger than this
OVERLAP    = 200   # character overlap between split sub-chunks


def clean_text(node):
    """Extract all text from an XML element and normalize whitespace."""
    parts = [t.strip() for t in node.itertext() if t.strip()]
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def split_large_chunk(text, max_chars=MAX_CHARS, overlap=OVERLAP):
    """Split a long text into overlapping sub-chunks at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    # Split on sentence endings followed by a space (preserve sentence integrity)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sub_chunks, current, current_len = [], [], 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > max_chars and current:
            sub_chunks.append(" ".join(current))
            # Keep last few sentences as overlap context
            overlap_text = " ".join(current)[-overlap:]
            current = [overlap_text, sent] if overlap_text else [sent]
            current_len = len(overlap_text) + sent_len + 1
        else:
            current.append(sent)
            current_len += sent_len + 1

    if current:
        sub_chunks.append(" ".join(current))
    return sub_chunks


def iter_chunks(node, ancestor_titles=None):
    """Recursively walk the element tree, yielding a chunk for every <section>
    that has substantive text content.

    SPL XML wraps sections inside component/structuredBody/component, so we
    must traverse ALL child elements (not only <section> direct children) to
    reach the actual section nodes.
    """
    if ancestor_titles is None:
        ancestor_titles = []

    for child in node:
        if child.tag != f"{{{NS}}}section":
            # Not a section — keep descending to find sections inside
            yield from iter_chunks(child, ancestor_titles)
            continue

        code_el = child.find(f"{{{NS}}}code")
        code    = code_el.attrib.get("code", "") if code_el is not None else ""

        if code in SKIP_CODES:
            continue

        # Resolve section title: prefer explicit <title>, fall back to LOINC name
        title_el = child.find(f"{{{NS}}}title")
        if title_el is not None:
            title = clean_text(title_el).strip()
        else:
            title = LOINC_NAMES.get(code, "")

        crumbs = ancestor_titles + [title] if title else ancestor_titles

        # Extract only the direct <text> of this section (exclude nested sections)
        text_el = child.find(f"{{{NS}}}text")
        if text_el is not None:
            raw_parts = []
            for el in text_el.iter():
                if el.tag == f"{{{NS}}}section":
                    continue
                if el.text and el.text.strip():
                    raw_parts.append(el.text.strip())
                if el.tail and el.tail.strip():
                    raw_parts.append(el.tail.strip())
            text = re.sub(r"\s+", " ", " ".join(raw_parts)).strip()
        else:
            text = ""

        if text and len(text) >= MIN_CHARS:
            section_path = " > ".join(crumbs) if crumbs else "(no section)"
            for i, sub_text in enumerate(split_large_chunk(text)):
                yield {
                    "drug":        "",          # filled by caller
                    "loinc_code":  code,
                    "section":     section_path,
                    "chunk_index": i,
                    "text":        sub_text,
                    "char_count":  len(sub_text),
                }

        yield from iter_chunks(child, crumbs)


def parse_drug(xml_path, drug_name):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    chunks = []
    for chunk in iter_chunks(root):
        chunk["drug"] = drug_name
        chunks.append(chunk)
    return chunks


def main():
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    out_path = os.path.join(CHUNKS_DIR, "all_chunks.jsonl")

    total_chunks = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for fname in sorted(os.listdir(DRUG_LABELS_DIR)):
            if not fname.endswith(".xml"):
                continue
            drug_name = fname[:-4]
            xml_path  = os.path.join(DRUG_LABELS_DIR, fname)
            chunks    = parse_drug(xml_path, drug_name)

            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            total_chunks += len(chunks)
            avg = sum(c["char_count"] for c in chunks) // max(len(chunks), 1)
            print(f"  {drug_name:<22}  {len(chunks):>4} chunks  avg {avg} chars")

    print(f"\nTotal: {total_chunks} chunks → {out_path}")


if __name__ == "__main__":
    main()
