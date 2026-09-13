import json
from pathlib import Path
import os

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

LABEL_FILE = Path('C:/Users/nikhi/OneDrive/Desktop/New folder/bhumisetu/label_extraction_with_confidence.json')
TABLE_FILE = Path('C:/Users/nikhi/OneDrive/Desktop/New folder/bhumisetu/table_extraction.json')
LLM_FILE = Path('C:/Users/nikhi/OneDrive/Desktop/New folder/bhumisetu/output/output_of_extractors/paragraph_extraction.json')
OUTPUT_FILE = Path('C:/Users/nikhi/OneDrive/Desktop/New folder/bhumisetu/output/final_land_record_with_confidence.json')

FIELDS = [
    "district", "circle", "halka", "village", "thana_no",
    "khata_no", "khesra_no", "raiyat_name", "father_husband_name",
    "address", "land_type", "area", "area_unit", "possession",
    "north_boundary", "south_boundary", "east_boundary",
    "west_boundary", "lagaan", "mutation_details", "remarks"
]

# Higher = more trusted when sources disagree.
SOURCE_PRIORITY = {
    "label": 3,
    "table": 2,
    "llm": 1
}

# Used only when a source file does not contain confidence yet.
DEFAULT_CONFIDENCE = {
    "label": 70.0,
    "table": 70.0,
    "llm": 60.0
}


# ============================================================
# CONFIDENCE HELPERS
# ============================================================

def clamp(value, low=0.0, high=100.0):
    return round(max(low, min(high, float(value))), 2)


def confidence_status(confidence):
    if confidence >= 85:
        return "HIGH"
    if confidence >= 65:
        return "MEDIUM"
    if confidence >= 40:
        return "LOW"
    return "VERY_LOW"


def numeric_confidence(value, default=0.0):
    try:
        return clamp(value)
    except (TypeError, ValueError):
        return default


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# NORMALIZATION
# ============================================================

def clean(value):
    if value is None:
        return None

    if isinstance(value, str):
        value = " ".join(value.strip().split())
        return value or None

    return value


def compare_value(value):
    value = clean(value)

    if value is None:
        return None

    return str(value).lower().strip()


# ============================================================
# CONFIDENCE EXTRACTION
# ============================================================

def get_confidence_from_dict(data, field, source):
    """
    Supports confidence formats produced by:
      label extractor
      table extractor
      LLM extractor

    Examples:
      {
        "field_confidence": {"district": 91.2}
      }

      {
        "confidence": {"district": 91.2}
      }

      {
        "extracted": {...},
        "field_confidence": {...}
      }
    """

    if not isinstance(data, dict):
        return DEFAULT_CONFIDENCE[source]

    field_conf = data.get("field_confidence")

    if isinstance(field_conf, dict):
        value = field_conf.get(field)
        if value is not None:
            return numeric_confidence(
                value,
                DEFAULT_CONFIDENCE[source]
            )

    # Some versions may store confidence directly under "confidence".
    confidence = data.get("confidence")

    if isinstance(confidence, dict):
        value = confidence.get(field)
        if value is not None:
            return numeric_confidence(
                value,
                DEFAULT_CONFIDENCE[source]
            )

    # If the source has one stage confidence, use it as fallback.
    for key in ("stage_confidence", "ocr_confidence"):
        value = data.get(key)

        if isinstance(value, dict):
            value = value.get("score")

        if value is not None:
            return numeric_confidence(
                value,
                DEFAULT_CONFIDENCE[source]
            )

    return DEFAULT_CONFIDENCE[source]


# ============================================================
# FIELD UNWRAPPING
# ============================================================

def unwrap_fields(data, source):
    """
    Returns:
        {
            field: {
                "value": ...,
                "confidence": ...,
                "source": source
            }
        }

    Handles:
      1. {fields:{...}}
      2. direct dictionaries
      3. paragraph result lists
      4. table region lists
    """

    result = {}

    if not data:
        return result

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(data, dict):

        if isinstance(data.get("fields"), dict):
            fields = data["fields"]

            for field in FIELDS:
                value = clean(fields.get(field))

                if value is not None:
                    result[field] = {
                        "value": value,
                        "confidence": get_confidence_from_dict(
                            data, field, source
                        ),
                        "source": source
                    }

            return result

        if isinstance(data.get("extracted"), dict):
            fields = data["extracted"]

            for field in FIELDS:
                value = clean(fields.get(field))

                if value is not None:
                    result[field] = {
                        "value": value,
                        "confidence": get_confidence_from_dict(
                            data, field, source
                        ),
                        "source": source
                    }

            return result

        # Direct field dictionary
        for field in FIELDS:
            value = clean(data.get(field))

            if value is not None:
                result[field] = {
                    "value": value,
                    "confidence": get_confidence_from_dict(
                        data, field, source
                    ),
                    "source": source
                }

        return result

    # --------------------------------------------------------
    # List
    # --------------------------------------------------------

    if isinstance(data, list):

        # Keep the strongest/most trusted candidate from each
        # source for every field.
        for item in data:

            if not isinstance(item, dict):
                continue

            extracted = item.get("extracted", item)

            if not isinstance(extracted, dict):
                continue

            for field in FIELDS:

                value = clean(extracted.get(field))

                if value is None:
                    continue

                candidate = {
                    "value": value,
                    "confidence": get_confidence_from_dict(
                        item, field, source
                    ),
                    "source": source
                }

                if (
                    field not in result
                    or candidate["confidence"]
                    > result[field]["confidence"]
                ):
                    result[field] = candidate

        return result

    return result


# ============================================================
# TABLE FIELDS
# ============================================================

def table_fields(data):
    """
    Extract candidates from all table regions.

    Important:
      We keep the confidence of the best table region for each
      field instead of discarding confidence while merging regions.
    """

    if not data:
        return {}, []

    candidates = {}

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(data, dict):

        if isinstance(data.get("fields"), dict):
            source_data = data["fields"]

            for field in FIELDS:
                value = clean(source_data.get(field))

                if value is not None:
                    candidates[field] = {
                        "value": value,
                        "confidence": get_confidence_from_dict(
                            data, field, "table"
                        ),
                        "source": "table"
                    }

            return candidates, []

        if isinstance(data.get("tables"), list):
            data = data["tables"]

        else:
            for field in FIELDS:
                value = clean(data.get(field))

                if value is not None:
                    candidates[field] = {
                        "value": value,
                        "confidence": get_confidence_from_dict(
                            data, field, "table"
                        ),
                        "source": "table"
                    }

            return candidates, []

    if not isinstance(data, list):
        return {}, []

    # --------------------------------------------------------
    # Multiple table regions
    # --------------------------------------------------------

    for table in data:

        if not isinstance(table, dict):
            continue

        for field in FIELDS:

            value = clean(table.get(field))

            if value is None:
                continue

            confidence = get_confidence_from_dict(
                table, field, "table"
            )

            candidate = {
                "value": value,
                "confidence": confidence,
                "source": "table",
                "region_id": table.get("region_id")
            }

            # Keep highest-confidence value from the table source.
            if (
                field not in candidates
                or confidence > candidates[field]["confidence"]
            ):
                candidates[field] = candidate

    return candidates, data


# ============================================================
# BUILD CANDIDATES
# ============================================================

def build_candidates(label, table, llm):
    candidates = {
        field: []
        for field in FIELDS
    }

    for source_values in (label, table, llm):

        for field in FIELDS:

            candidate = source_values.get(field)

            if not candidate:
                continue

            candidates[field].append(candidate.copy())

    return candidates


# ============================================================
# MERGE ONE FIELD
# ============================================================

def merge_field(candidates):
    if not candidates:
        return {
            "value": None,
            "source": None,
            "confidence": 0.0,
            "confidence_status": "VERY_LOW",
            "status": "NOT_FOUND",
            "candidates": []
        }

    # --------------------------------------------------------
    # Normalize candidate confidences
    # --------------------------------------------------------

    normalized_candidates = []

    for candidate in candidates:

        item = dict(candidate)

        item["value"] = clean(item.get("value"))
        item["confidence"] = numeric_confidence(
            item.get("confidence"),
            0.0
        )

        normalized_candidates.append(item)

    # --------------------------------------------------------
    # Check agreement
    # --------------------------------------------------------

    unique_values = {
        compare_value(c["value"])
        for c in normalized_candidates
        if c.get("value") is not None
    }

    agreed = len(unique_values) == 1

    # --------------------------------------------------------
    # Select final candidate
    #
    # Preserve your original source-priority rule:
    # label > table > llm
    #
    # Confidence is then attached to the selected candidate.
    # --------------------------------------------------------

    best = max(
        normalized_candidates,
        key=lambda c: (
            SOURCE_PRIORITY.get(c.get("source"), 0),
            c.get("confidence", 0.0)
        )
    )

    final_confidence = best["confidence"]

    # Conflicting sources reduce confidence because the system
    # does not have complete agreement.
    if not agreed and len(normalized_candidates) > 1:
        final_confidence *= 0.70

    final_confidence = clamp(final_confidence)

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    status = "AGREED" if agreed else "CONFLICT"

    if status == "CONFLICT":
        # A conflict should never be presented as HIGH confidence.
        final_confidence = min(final_confidence, 79.99)

    return {
        "value": best["value"],
        "source": best["source"],
        "confidence": final_confidence,
        "confidence_status": confidence_status(final_confidence),
        "status": status,
        "candidates": normalized_candidates
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL LAND RECORD MERGER")
    print("=" * 70)

    label_data = load_json(LABEL_FILE)
    table_data = load_json(TABLE_FILE)
    llm_data = load_json(LLM_FILE)

    label = unwrap_fields(
        label_data,
        "label"
    )

    table, table_regions = table_fields(
        table_data
    )

    llm = unwrap_fields(
        llm_data,
        "llm"
    )

    candidates = build_candidates(
        label,
        table,
        llm
    )

    final_fields = {
        field: merge_field(
            candidates[field]
        )
        for field in FIELDS
    }

    # --------------------------------------------------------
    # Overall confidence
    # --------------------------------------------------------

    found_confidences = [
        result["confidence"]
        for result in final_fields.values()
        if result["value"] is not None
    ]

    overall_confidence = (
        round(
            sum(found_confidences) /
            len(found_confidences),
            2
        )
        if found_confidences
        else 0.0
    )

    conflicts = [
        field
        for field, result in final_fields.items()
        if result["status"] == "CONFLICT"
    ]

    not_found = [
        field
        for field, result in final_fields.items()
        if result["status"] == "NOT_FOUND"
    ]

    # Overall confidence is reduced when conflicts exist.
    overall_confidence = clamp(overall_confidence)

    if conflicts:
        overall_confidence = min(
            overall_confidence,
            79.99
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = {
        "fields": final_fields,

        "summary": {
            "overall_confidence": overall_confidence,
            "confidence_status": confidence_status(
                overall_confidence
            ),
            "total_fields": len(FIELDS),
            "fields_found": len(FIELDS) - len(not_found),
            "fields_not_found": len(not_found),
            "conflict_count": len(conflicts),
            "conflicts": conflicts
        },

        "table_regions": table_regions
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=4
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    for field, result in final_fields.items():

        print(
            f"{field:25} : "
            f"{result['value']} "
            f"[{result['source']}] "
            f"{result['status']} "
            f"CONF={result['confidence']} "
            f"{result['confidence_status']}"
        )

    print("\n" + "-" * 70)

    print(
        "Overall confidence:",
        overall_confidence,
        confidence_status(overall_confidence)
    )

    print(
        "Conflicts:",
        len(conflicts)
    )

    if conflicts:
        for field in conflicts:
            print(" -", field)

    print(
        "Not found:",
        len(not_found)
    )

    print(
        "\nSaved:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
