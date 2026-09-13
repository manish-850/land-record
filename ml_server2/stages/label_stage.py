import os
import json
import re
from pathlib import Path


FIELDS = [
    "district", "circle", "halka", "village", "thana_no",
    "khata_no", "khesra_no", "raiyat_name",
    "father_husband_name", "address", "land_type", "area",
    "area_unit", "possession", "north_boundary",
    "south_boundary", "east_boundary", "west_boundary",
    "lagaan", "mutation_details", "remarks"
]

ALIASES = {
    "district": ["जिला", "district"],
    "circle": ["अंचल", "circle"],
    "halka": ["हल्का", "halka"],
    "village": ["मौजा", "मौजा नाम", "ग्राम", "गांव", "गाँव", "village", "mouza"],
    "thana_no": [
        "मौजा/थाना संख्या", "मौजा / थाना संख्या",
        "थाना संख्या", "थाना नं", "थाना न", "thana number", "thana no"
    ],
    "khata_no": ["खाता संख्या", "खाता नं", "खाता न", "khata number", "khata no"],
    "khesra_no": [
        "खेसरा संख्या", "खेसरा/संख्या", "खेसरा / संख्या",
        "खसरा संख्या", "khesra number", "khesra no", "khasra number"
    ],
    "raiyat_name": [
        "जमाबंदी रेयत का नाम", "जमाबन्दी रेयत का नाम",
        "रेयत का नाम", "रैयत का नाम", "raiyat name"
    ],
    "father_husband_name": [
        "अभिभावक का नाम", "पिता/पति का नाम",
        "पिता / पति का नाम", "पिता का नाम", "पति का नाम",
        "father name", "husband name"
    ],
    "address": ["पता", "address"],
    "land_type": ["भूमि का प्रकार", "भूमि का वर्ग", "भूमि वर्ग", "किस्म", "land type"],
    "area": ["रकबा", "क्षेत्रफल", "area"],
    "area_unit": ["डिसमिल", "डिसमील", "एकड़", "एकर", "हेक्टर", "हेक्टेयर", "area unit"],
    "possession": ["दखल", "कब्जा", "दखल कब्जा", "possession"],
    "north_boundary": ["उत्तर सीमा", "उत्तर", "north boundary", "north"],
    "south_boundary": ["दक्षिण सीमा", "दक्षिण", "south boundary", "south"],
    "east_boundary": ["पूर्व सीमा", "पूर्व", "east boundary", "east"],
    "west_boundary": ["पश्चिम सीमा", "पश्चिम", "west boundary", "west"],
    "lagaan": ["लगान", "जमाबंदी लगान", "भूमि लगान", "land revenue"],
    "mutation_details": ["नामांतरण", "दाखिल खारिज", "म्यूटेशन", "mutation"],
    "remarks": ["टिप्पणी", "अभियुक्ति", "कैफियत", "remarks", "remark"],
}

def clean(text):
    text = str(text or "")
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    text = text.translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    return re.sub(r"\s+", " ", text).strip(" :-,;|\"'`")

# Build one regex for all known labels. Longer labels first so
# "मौजा/थाना संख्या" is matched before "मौजा".
LABEL_ENTRIES = []
for field, aliases in ALIASES.items():
    for alias in aliases:
        LABEL_ENTRIES.append((field, alias))
LABEL_ENTRIES.sort(key=lambda x: len(x[1]), reverse=True)

LABEL_PATTERN = re.compile(
    r"(?P<label>"
    + "|".join(re.escape(alias) for _, alias in LABEL_ENTRIES)
    + r")\s*:\s*-?\s*",
    flags=re.IGNORECASE
)

ALIAS_TO_FIELD = {alias.lower(): field for field, alias in LABEL_ENTRIES}

def clamp_confidence(value):
    return round(
        max(0.0, min(100.0, float(value))),
        2
    )


def confidence_status(score):
    if score >= 85:
        return "HIGH"
    if score >= 65:
        return "MEDIUM"
    if score >= 40:
        return "LOW"
    return "VERY_LOW"


def get_supporting_words(block, value):
    """
    Find OCR words that contribute to the extracted value.
    Uses token overlap first; falls back to the block's words.
    """
    if not block:
        return []

    value_tokens = set(
        clean(value).lower().split()
    )

    words = block.get("words", [])

    matched = [
        word
        for word in words
        if clean(word.get("text", "")).lower()
        in value_tokens
    ]

    return matched if matched else words


def average_ocr_confidence(words, block=None):
    values = []

    for word in words:
        try:
            values.append(
                float(word.get("confidence", 0))
            )
        except (TypeError, ValueError):
            pass

    if values:
        return clamp_confidence(
            sum(values) / len(values)
        )

    try:
        return clamp_confidence(
            float(block.get("confidence", 0))
        )
    except (AttributeError, TypeError, ValueError):
        return 0.0


def parse_inline(line, block=None):
    """
    Parses multiple labels in one OCR line.

    Returns dictionaries instead of tuples so each extraction
    carries its confidence information.
    """
    line = clean(line)
    matches = list(LABEL_PATTERN.finditer(line))
    out = []

    for i, m in enumerate(matches):
        raw_label = m.group("label")

        field = ALIAS_TO_FIELD.get(
            raw_label.lower()
        )

        if not field:
            field = next(
                (
                    f
                    for f, a in LABEL_ENTRIES
                    if a.lower() == raw_label.lower()
                ),
                None
            )

        if not field:
            continue

        start = m.end()

        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(line)
        )

        value = clean(
            line[start:end]
        )

        # "मौजा/थाना संख्या" is THANA NUMBER, not village.
        if field == "thana_no":
            number = re.search(
                r"\d+",
                value
            )
            value = (
                number.group(0)
                if number
                else None
            )

        if not value:
            continue

        supporting_words = get_supporting_words(
            block,
            value
        )

        ocr_confidence = average_ocr_confidence(
            supporting_words,
            block
        )

        # Extraction confidence combines:
        #   70% OCR evidence
        #   30% explicit label match
        #
        # Explicit label matching is deterministic, so it gets
        # full 100 confidence here. This is NOT a model probability.
        confidence = clamp_confidence(
            0.70 * ocr_confidence
            +
            0.30 * 100.0
        )

        out.append({
            "field": field,
            "value": value,
            "confidence": confidence,
            "confidence_status": confidence_status(
                confidence
            ),
            "ocr_confidence": ocr_confidence,
            "source": "label_extraction"
        })

    return out

def extract_region(region):
    result = {}
    result_confidence = {}

    blocks = []

    for block in region.get(
        "ocr_blocks",
        []
    ):
        text = clean(
            block.get(
                "text",
                ""
            )
        )

        if text:
            blocks.append(
                block
            )

    if not blocks:
        # Fallback when only region text exists.
        blocks = [
            {
                "text": clean(line),
                "words": [],
                "confidence": region.get(
                    "confidence",
                    0
                )
            }
            for line in region.get(
                "text",
                ""
            ).splitlines()
            if clean(line)
        ]

    field_details = []

    for block in blocks:

        parsed_items = parse_inline(
            block.get("text", ""),
            block=block
        )

        for item in parsed_items:

            field = item["field"]
            value = item["value"]
            confidence = item["confidence"]

            field_details.append(item)

            # Keep highest-confidence occurrence.
            if (
                field not in result
                or
                confidence
                >
                result_confidence.get(
                    field,
                    -1
                )
            ):
                result[field] = value
                result_confidence[field] = confidence

    return result, result_confidence, field_details

def extract_labels(classified_data):
    """
    Runs label (key:value) extraction over every region routed as
    "label" by the classifier stage.

    Parameters:
        classified_data: the dict returned by
            stages.classifier_stage.classify_document
            (must contain a "regions" list).

    Returns:
        {
            "fields": {field: value, ...},
            "field_confidence": {field: {"confidence":..., "confidence_status":...}, ...},
            "stage_confidence": float,
            "regions": [ ... per-region extraction detail ... ]
        }
    """
    data = classified_data

    final_fields = {}
    final_confidence = {}
    regions = []

    for region in data.get(
        "regions",
        []
    ):
        if region.get("route") != "label":
            continue

        (
            extracted,
            extracted_confidence,
            field_details
        ) = extract_region(
            region
        )

        regions.append({
            "region_id": region.get(
                "region_id"
            ),
            "source_text": region.get(
                "text",
                ""
            ),
            "region_confidence": region.get(
                "confidence",
                0.0
            ),
            "extracted": {
                field: {
                    "value": value,
                    "confidence": extracted_confidence.get(
                        field,
                        0.0
                    ),
                    "confidence_status": confidence_status(
                        extracted_confidence.get(
                            field,
                            0.0
                        )
                    )
                }
                for field, value in extracted.items()
            },
            "field_details": field_details
        })

        for field, value in extracted.items():

            confidence = extracted_confidence.get(
                field,
                0.0
            )

            if (
                field not in final_fields
                or
                confidence
                >
                final_confidence.get(
                    field,
                    -1
                )
            ):
                final_fields[field] = value
                final_confidence[field] = confidence

    output = {
        "fields": final_fields,

        "field_confidence": {
            field: {
                "confidence": confidence,
                "confidence_status": confidence_status(
                    confidence
                )
            }
            for field, confidence
            in final_confidence.items()
        },

        "stage_confidence": (
            round(
                sum(
                    final_confidence.values()
                )
                /
                len(
                    final_confidence
                ),
                2
            )
            if final_confidence
            else 0.0
        ),

        "regions": regions
    }

    return output


# ============================================================
# CLI (kept for standalone / debugging use)
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run label extraction on classifier output")
    parser.add_argument("classifier_json_path")
    parser.add_argument("--output", default="label_extraction_with_confidence.json")
    args = parser.parse_args()

    with open(args.classifier_json_path, "r", encoding="utf-8") as f:
        classified_data = json.load(f)

    output = extract_labels(classified_data)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print("\n" + "=" * 70)
    print("LABEL EXTRACTION + CONFIDENCE")
    print("=" * 70)

    for field in FIELDS:
        value = output["fields"].get(field)
        conf_info = output["field_confidence"].get(field, {"confidence": 0.0, "confidence_status": "VERY_LOW"})
        print(
            f"{field:25} : {value} "
            f"[{conf_info['confidence']:.2f}% {conf_info['confidence_status']}]"
        )

    print("\nStage confidence:", output["stage_confidence"])
    print("\nSaved:", args.output)
