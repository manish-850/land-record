import os
import json
from google import genai

# ==========================================
# Gemini setup
# ==========================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=api_key)


# ==========================================
# Fields we want
# ==========================================

FIELDS = [
    "district",
    "circle",
    "halka",
    "village",
    "thana_no",
    "khata_no",
    "khesra_no",
    "raiyat_name",
    "father_husband_name",
    "address",
    "land_type",
    "area",
    "area_unit",
    "possession",
    "north_boundary",
    "south_boundary",
    "east_boundary",
    "west_boundary",
    "lagaan",
    "mutation_details",
    "remarks"
]


# ==========================================
# Extract paragraph using Gemini
# ==========================================

def extract_paragraph(text, ocr_confidence=0.0):
    """
    Extract paragraph information with confidence.

    Confidence is NOT a calibrated Gemini probability.
    It combines:
      - OCR confidence of the source text: 50%
      - extraction evidence/field presence: 30%
      - JSON/schema validity: 20%

    Gemini itself does not provide a reliable calibrated
    per-field probability, so the score is an engineering
    confidence score based on available evidence.
    """

    prompt = f"""
You are extracting structured information from an Indian land record.

Extract ONLY information explicitly present in the OCR text.

Return exactly this JSON structure:

{{
    "district": null,
    "circle": null,
    "halka": null,
    "village": null,
    "thana_no": null,
    "khata_no": null,
    "khesra_no": null,
    "raiyat_name": null,
    "father_husband_name": null,
    "address": null,
    "land_type": null,
    "area": null,
    "area_unit": null,
    "possession": null,
    "north_boundary": null,
    "south_boundary": null,
    "east_boundary": null,
    "west_boundary": null,
    "lagaan": null,
    "mutation_details": null,
    "remarks": null
}}

Rules:

- Do NOT invent information.
- If a field is not present, return null.
- Preserve names, places, numbers and dates from the OCR.
- Correct only obvious OCR errors when the intended value is clear.
- Do not use outside knowledge.
- Put miscellaneous relevant information in "remarks".
- Return JSON only.

OCR TEXT:
{text}
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        raw = (response.text or "").strip()

        # Handle ```json ... ```
        if raw.startswith("```"):
            raw = raw.replace("```json", "")
            raw = raw.replace("```", "")
            raw = raw.strip()

        result = json.loads(raw)

        extracted = {
            field: result.get(field)
            for field in FIELDS
        }

        # Schema validity: all requested keys survived.
        schema_confidence = 100.0 if all(
            field in result for field in FIELDS
        ) else 60.0

        return extracted, schema_confidence

    except Exception as exc:
        print(f"LLM extraction error: {exc}")

        return {
            field: None
            for field in FIELDS
        }, 0.0


def clamp(value, low=0.0, high=100.0):
    return round(
        max(low, min(high, float(value))),
        2
    )


def confidence_status(confidence):
    if confidence >= 85:
        return "HIGH"
    if confidence >= 65:
        return "MEDIUM"
    if confidence >= 40:
        return "LOW"
    return "VERY_LOW"


def calculate_field_confidence(
    value,
    ocr_confidence,
    schema_confidence
):
    """
    Engineering confidence score.

    Components:
      50% OCR/source quality
      30% extraction evidence (field is actually present)
      20% valid structured response

    A non-null field gets stronger extraction evidence.
    A null field is not treated as an extraction failure;
    it simply has no field-level confidence because no value
    was found.
    """

    if value is None or str(value).strip() == "":
        return None

    extraction_evidence = 100.0

    score = (
        0.50 * ocr_confidence
        + 0.30 * extraction_evidence
        + 0.20 * schema_confidence
    )

    return clamp(score)


def get_region_ocr_confidence(region):
    """
    Average OCR confidence across all words in the paragraph
    region. Falls back to block confidence when word-level
    confidence is unavailable.
    """

    word_confidences = []
    block_confidences = []

    for block in region.get("ocr_blocks", []):
        if block.get("confidence") is not None:
            try:
                block_confidences.append(
                    float(block.get("confidence"))
                )
            except (TypeError, ValueError):
                pass

        for word in block.get("words", []):
            try:
                word_confidences.append(
                    float(word.get("confidence", 0) or 0)
                )
            except (TypeError, ValueError):
                pass

    if word_confidences:
        return clamp(
            sum(word_confidences) / len(word_confidences)
        )

    if block_confidences:
        return clamp(
            sum(block_confidences) / len(block_confidences)
        )

    return 0.0


# ==========================================
# Read your extraction JSON
# ==========================================

INPUT_FILE = 'C:/Users/nikhi/OneDrive/Desktop/New folder/bhumisetu/output/output_of_classifier/bhumi_rasid_extraction.json'

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


# ==========================================
# Process ONLY paragraph regions
# ==========================================

results = []

for region in data.get("regions", []):

    if region.get("route") != "paragraph":
        continue

    text = region.get("text", "").strip()

    if not text:
        continue

    print("\n" + "=" * 60)
    print("REGION:", region["region_id"])
    print("OCR:", text)

    ocr_confidence = get_region_ocr_confidence(region)

    extracted, schema_confidence = extract_paragraph(
        text,
        ocr_confidence
    )

    field_confidence = {}
    confidence_statuses = {}

    for field in FIELDS:
        score = calculate_field_confidence(
            extracted.get(field),
            ocr_confidence,
            schema_confidence
        )

        if score is not None:
            field_confidence[field] = score
            confidence_statuses[field] = confidence_status(score)

    # Overall LLM stage confidence is based on fields that
    # were actually extracted.
    extracted_scores = list(field_confidence.values())

    stage_confidence = (
        round(
            sum(extracted_scores) / len(extracted_scores),
            2
        )
        if extracted_scores
        else 0.0
    )

    stage_status = confidence_status(
        stage_confidence
    )

    print("\nEXTRACTED:")
    print(json.dumps(
        extracted,
        ensure_ascii=False,
        indent=4
    ))

    print("\nFIELD CONFIDENCE:")
    for field, score in field_confidence.items():
        print(
            f"{field:22}: "
            f"{score:6.2f} "
            f"({confidence_statuses[field]})"
        )

    print(
        "\nOCR confidence:",
        ocr_confidence
    )

    print(
        "LLM stage confidence:",
        stage_confidence,
        f"({stage_status})"
    )

    results.append({
        "region_id": region["region_id"],
        "ocr_text": text,

        "ocr_confidence": ocr_confidence,

        "extracted": extracted,

        "field_confidence": field_confidence,

        "confidence_status": confidence_statuses,

        "stage_confidence": {
            "score": stage_confidence,
            "status": stage_status
        }
    })


# ==========================================
# Save results
# ==========================================

OUTPUT_FILE = 'C:/Users/nikhi/OneDrive/Desktop/New folder/bhumisetu/output/output_of_extractors/paragraph_extraction.json'

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=4
    )

print("\nSaved:", OUTPUT_FILE)