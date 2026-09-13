import json
import re
from pathlib import Path
import os
# ============================================================
# GENERIC LAND-RECORD TABLE EXTRACTOR
#
# Input:
#   bhumi_rasid_extraction.json
#
# Output:
#   table_extraction.json
#
# Goals:
#   1. Process ONLY regions routed as "table".
#   2. Do NOT assume every table has the same layout.
#   3. Reconstruct OCR rows from word coordinates.
#   4. Extract khata/khesra/area using header positions when possible.
#   5. Extract lagaan from the row/column belonging to "लगान",
#      never from years such as 2021-2022.
#   6. Preserve raw rows for debugging.
# ============================================================


BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = Path('C:/Users/nikhi/OneDrive/Desktop/New folder/bhumisetu/output/output_of_classifier/bhumi_rasid_extraction.json')
OUTPUT_FILE = Path('C:/Users/nikhi/OneDrive/Desktop/New folder/bhumisetu/table_extraction.json')


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    text = str(text)

    text = text.replace("\u200b", "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")
    text = text.replace("\ufeff", "")

    text = text.translate(
        str.maketrans(
            "०१२३४५६७८९",
            "0123456789"
        )
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def normalize_key(text):
    text = clean_text(text).lower()

    replacements = {
        "खेसरा/संख्या": "खेसरा संख्या",
        "खेसरा/संख्या,": "खेसरा संख्या",
        "रकबा/डिसमिल": "रकबा डिसमिल",
        "रकबा/डिसमील": "रकबा डिसमील",
        "खाता/संख्या": "खाता संख्या",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(
        r"[^a-z0-9\u0900-\u097f ]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def number_from_text(text):
    """
    Return the first normal numeric value.

    Important:
    This deliberately ignores financial-year strings such as
    2021-2022.
    """
    text = clean_text(text)

    # Decimal/integer that is NOT part of a year range.
    matches = re.findall(
        r"(?<![\d-])\d+(?:\.\d+)?(?![\d-])",
        text
    )

    return matches[0] if matches else None


def all_numbers(text):
    text = clean_text(text)

    return re.findall(
        r"(?<![\d-])\d+(?:\.\d+)?(?![\d-])",
        text
    )


# ============================================================
# OCR WORDS
# ============================================================

def get_words(region):
    words = []

    for block in region.get(
        "ocr_blocks",
        []
    ):
        for word in block.get(
            "words",
            []
        ):

            text = clean_text(
                word.get("text", "")
            )

            if not text:
                continue

            words.append({
                "text": text,
                "left": int(
                    word.get("left", 0)
                ),
                "top": int(
                    word.get("top", 0)
                ),
                "width": int(
                    word.get("width", 0)
                ),
                "height": int(
                    word.get("height", 0)
                ),
                "confidence": float(
                    word.get(
                        "confidence",
                        0
                    )
                )
            })

    return words


# ============================================================
# ROW RECONSTRUCTION
# ============================================================

def group_into_rows(
    words,
    y_tolerance=12
):
    words = sorted(
        words,
        key=lambda w: (
            w["top"],
            w["left"]
        )
    )

    rows = []

    for word in words:

        center_y = (
            word["top"]
            +
            word["height"] / 2
        )

        best = None
        best_distance = None

        for row in rows:

            distance = abs(
                center_y -
                row["center_y"]
            )

            if distance <= y_tolerance:

                if (
                    best_distance is None
                    or
                    distance < best_distance
                ):
                    best = row
                    best_distance = distance

        if best is None:

            rows.append({
                "center_y": center_y,
                "words": [word]
            })

        else:

            best["words"].append(word)

            best["center_y"] = sum(
                w["top"] + w["height"] / 2
                for w in best["words"]
            ) / len(best["words"])

    rows.sort(
        key=lambda r: r["center_y"]
    )

    for row in rows:
        row["words"].sort(
            key=lambda w: w["left"]
        )

        row["text"] = " ".join(
            w["text"]
            for w in row["words"]
        )

    return rows


# ============================================================
# HEADER / COLUMN HELPERS
# ============================================================

FIELD_ALIASES = {
    "khata_no": [
        "खाता",
        "खाता संख्या",
        "khata",
        "khata no",
        "khata number",
    ],
    "khesra_no": [
        "खेसरा",
        "खेसरा संख्या",
        "खेसरा/संख्या",
        "khesra",
        "khesra no",
        "plot",
    ],
    "area": [
        "रकबा",
        "क्षेत्रफल",
        "area",
        "रकबा डिसमिल",
    ],
    "area_unit": [
        "डिसमिल",
        "डिसमील",
        "एकड़",
        "हेक्टर",
        "hectare",
        "acre",
    ],
    "lagaan": [
        "लगान",
        "जमाबंदी लगान",
        "land revenue",
        "revenue",
    ],
}


def word_center_x(word):
    return (
        word["left"]
        +
        word["width"] / 2
    )


def find_header_centers(rows):
    """
    Search early rows for known land-record headers.

    Returns:
        field -> approximate X center
    """

    header_centers = {}

    # Header normally appears near the beginning of a region.
    candidate_rows = rows[:8]

    for row in candidate_rows:

        text = normalize_key(
            row["text"]
        )

        for field, aliases in FIELD_ALIASES.items():

            if field in header_centers:
                continue

            for alias in aliases:

                alias_normalized = normalize_key(
                    alias
                )

                if (
                    alias_normalized
                    and
                    alias_normalized in text
                ):

                    # Find the first word belonging to the alias.
                    for word in row["words"]:

                        word_norm = normalize_key(
                            word["text"]
                        )

                        if (
                            alias_normalized == word_norm
                            or
                            alias_normalized in word_norm
                            or
                            word_norm in alias_normalized
                        ):
                            header_centers[field] = (
                                word_center_x(word)
                            )
                            break

                    if field in header_centers:
                        break

    return header_centers


def nearest_word_to_x(
    words,
    x,
    max_distance=120
):
    if not words:
        return None

    best = min(
        words,
        key=lambda w:
        abs(word_center_x(w) - x)
    )

    if (
        abs(
            word_center_x(best) - x
        )
        <= max_distance
    ):
        return best

    return None


# ============================================================
# CONFIDENCE HELPERS
# ============================================================

def clamp_confidence(value):
    return round(max(0.0, min(100.0, float(value))), 2)


def confidence_status(score):
    if score >= 85:
        return "HIGH"
    if score >= 65:
        return "MEDIUM"
    if score >= 40:
        return "LOW"
    return "VERY_LOW"


def average(values):
    values = [float(v) for v in values if v is not None]
    return round(sum(values) / len(values), 2) if values else 0.0


def find_word_for_value(rows, value):
    """
    Find the OCR word that produced an extracted value.
    Prefer exact text match.
    """
    if value is None:
        return None

    value = str(value)

    for row in rows:
        for word in row["words"]:
            if str(word["text"]) == value:
                return word

    return None


def land_field_confidences(rows, fields):
    """
    Confidence for land-table fields.

    70% = source OCR confidence
    30% = extraction structure confidence

    Structure confidence is higher when a field was found using
    header/column information and lower for fallback extraction.
    """
    header_centers = find_header_centers(rows)

    confidences = {}
    evidence = {}

    for field in [
        "khata_no",
        "khesra_no",
        "area",
        "area_unit"
    ]:
        value = fields.get(field)

        if value is None:
            confidences[field] = 0.0
            evidence[field] = None
            continue

        word = find_word_for_value(rows, value)

        if word is None:
            # The value exists, but we cannot tie it back to a
            # specific OCR token. Keep confidence conservative.
            ocr_conf = 50.0
        else:
            ocr_conf = float(word.get("confidence", 0) or 0)

        if field in header_centers:
            structural = 95.0
        elif field == "area_unit":
            structural = 90.0
        else:
            # Numeric fallback was used.
            structural = 60.0

        score = (
            0.70 * ocr_conf
            + 0.30 * structural
        )

        confidences[field] = clamp_confidence(score)
        evidence[field] = word

    return confidences, evidence


def lagaan_confidence(rows, extracted_value):
    """
    Confidence for the existing lagaan extraction logic.

    The extractor already avoids financial years and prefers
    decimal monetary values. Confidence reflects:
      - OCR confidence of the selected number
      - strength of the row structure
      - whether the candidate is decimal
      - whether multiple monetary candidates remain ambiguous
    """
    if extracted_value is None:
        return 0.0, None, "No reliable lagaan value extracted."

    candidate_rows = []

    for row in rows:
        normalized = normalize_key(row["text"])

        if "लगान" in normalized or "जमाबंदी लगान" in normalized:
            candidate_rows.append(row)

    if not candidate_rows:
        return 0.0, None, "No lagaan-labelled row found."

    # Prefer the same row type used by extract_lagaan().
    jamabandi_rows = [
        row
        for row in candidate_rows
        if "जमाबंदी" in normalize_key(row["text"])
    ]

    rows_to_check = jamabandi_rows if jamabandi_rows else candidate_rows

    for row in rows_to_check:
        matches = []

        for word in row["words"]:
            value = number_from_text(word["text"])

            if value != str(extracted_value):
                continue

            # Do not use a 4-digit year as money.
            if re.fullmatch(r"\d{4}", value):
                year = int(value)
                if 1900 <= year <= 2100:
                    continue

            matches.append(word)

        if not matches:
            continue

        selected = matches[0]
        ocr_conf = float(selected.get("confidence", 0) or 0)

        decimal_count = 0
        numeric_count = 0

        for word in row["words"]:
            value = number_from_text(word["text"])
            if value is None:
                continue

            if re.fullmatch(r"\d{4}", value):
                year = int(value)
                if 1900 <= year <= 2100:
                    continue

            numeric_count += 1
            if "." in value:
                decimal_count += 1

        # Structural confidence mirrors the extractor's actual
        # decision rule.
        if decimal_count >= 5:
            structural = 90.0
        elif decimal_count >= 3:
            structural = 75.0
        elif decimal_count >= 1:
            structural = 65.0
        else:
            structural = 50.0

        # More candidates = more ambiguity.
        if numeric_count > 5:
            structural -= 10.0

        score = clamp_confidence(
            0.70 * ocr_conf + 0.30 * structural
        )

        reason = (
            "Decimal monetary candidate selected from lagaan row."
            if decimal_count > 0
            else "Numeric monetary candidate selected from lagaan row."
        )

        return score, selected, reason

    return 0.0, None, "Extracted lagaan value could not be matched to OCR evidence."


def label_field_confidences(region, label_fields):
    """
    Confidence for the existing regex-based label/value fallback.

    Label extraction is deterministic, so structural confidence is
    high when a regex matched. OCR confidence is estimated from the
    words in the matched value.
    """
    rows = group_into_rows(get_words(region))
    text = region.get("text", "") or ""

    confidences = {}
    evidence = {}

    for field, value in label_fields.items():
        if value is None:
            confidences[field] = 0.0
            evidence[field] = None
            continue

        value_words = []

        for row in rows:
            row_text = row["text"]
            if str(value) in row_text:
                value_words.extend(row["words"])

        if value_words:
            ocr_conf = average(
                [w.get("confidence", 0) for w in value_words]
            )
        else:
            ocr_conf = 50.0

        score = clamp_confidence(
            0.70 * ocr_conf
            + 0.30 * 100.0
        )

        confidences[field] = score
        evidence[field] = value_words

    return confidences, evidence


# ============================================================
# LAND TABLE EXTRACTION
# ============================================================

def extract_land_table(rows):
    """
    Dynamically uses the header positions.

    Falls back to numeric ordering ONLY when no useful header
    information is available.
    """

    result = {
        "khata_no": None,
        "khesra_no": None,
        "area": None,
        "area_unit": None,
    }

    header_centers = find_header_centers(
        rows
    )

    # Find candidate data rows.
    for row_index, row in enumerate(rows):

        text = clean_text(
            row["text"]
        )

        # Skip obvious headers.
        normalized = normalize_key(
            text
        )

        if any(
            key in normalized
            for key in [
                "खाता संख्या",
                "खेसरा संख्या",
                "रकबा डिसमिल"
            ]
        ):
            continue

        numbers = []

        for word in row["words"]:

            value = number_from_text(
                word["text"]
            )

            if value is not None:

                numbers.append({
                    "value": value,
                    "word": word
                })

        if not numbers:
            continue

        # ----------------------------------------------------
        # Header-position based extraction
        # ----------------------------------------------------

        if "khata_no" in header_centers:

            candidate = min(
                numbers,
                key=lambda n:
                abs(
                    word_center_x(
                        n["word"]
                    )
                    -
                    header_centers[
                        "khata_no"
                    ]
                )
            )

            result["khata_no"] = candidate[
                "value"
            ]

        if "khesra_no" in header_centers:

            candidate = min(
                numbers,
                key=lambda n:
                abs(
                    word_center_x(
                        n["word"]
                    )
                    -
                    header_centers[
                        "khesra_no"
                    ]
                )
            )

            result["khesra_no"] = candidate[
                "value"
            ]

        if "area" in header_centers:

            area_candidates = [
                n
                for n in numbers
                if (
                    "." in n["value"]
                    or
                    any(
                        unit in text
                        for unit in [
                            "डिसमिल",
                            "डिसमील",
                            "एकड़",
                            "हेक्टर"
                        ]
                    )
                )
            ]

            if area_candidates:

                candidate = min(
                    area_candidates,
                    key=lambda n:
                    abs(
                        word_center_x(
                            n["word"]
                        )
                        -
                        header_centers[
                            "area"
                        ]
                    )
                )

                result["area"] = (
                    candidate["value"]
                )

        # Area unit
        for i, word in enumerate(
            row["words"]
        ):

            unit = word["text"].lower()

            if any(
                u in unit
                for u in [
                    "डिसम",
                    "एकड़",
                    "हेक्टर",
                    "dismil"
                ]
            ):

                result["area_unit"] = (
                    word["text"]
                )

                # If area wasn't found from header
                # positioning, search immediately around
                # the unit.
                if result["area"] is None:

                    for previous in reversed(
                        row["words"][:i]
                    ):

                        value = number_from_text(
                            previous["text"]
                        )

                        if (
                            value is not None
                            and
                            "." in value
                        ):
                            result["area"] = value
                            break

        # One good data row is enough for this table.
        if (
            result["khata_no"] is not None
            and
            result["khesra_no"] is not None
            and
            result["area"] is not None
        ):
            break

    # --------------------------------------------------------
    # Fallback for tables where header OCR is poor
    # --------------------------------------------------------

    if (
        result["khata_no"] is None
        or
        result["khesra_no"] is None
        or
        result["area"] is None
    ):

        for row in rows:

            text = row["text"]

            if not (
                "डिसम" in text
                or
                "एकड़" in text
                or
                "हेक्टर" in text
            ):
                continue

            numeric = []

            for word in row["words"]:

                value = number_from_text(
                    word["text"]
                )

                if value is not None:
                    numeric.append(
                        (value, word)
                    )

            if len(numeric) < 2:
                continue

            # Only use the numeric-position fallback
            # when this actually resembles a small land table.
            if result["khata_no"] is None:
                result["khata_no"] = numeric[0][0]

            if result["khesra_no"] is None:
                result["khesra_no"] = numeric[1][0]

            if result["area"] is None:

                decimal_candidates = [
                    value
                    for value, word in numeric
                    if "." in value
                ]

                if decimal_candidates:
                    result["area"] = (
                        decimal_candidates[-1]
                    )

            if result["area_unit"] is None:

                for word in row["words"]:

                    if any(
                        u in word["text"].lower()
                        for u in [
                            "डिसम",
                            "एकड़",
                            "हेक्टर",
                            "dismil"
                        ]
                    ):
                        result["area_unit"] = (
                            word["text"]
                        )
                        break

            break

    return result


# ============================================================
# LAGAAN EXTRACTION
# ============================================================

def extract_lagaan(rows):
    """
    IMPORTANT:
    Never take the first number in a row containing "लगान".

    In these records, years such as 2021-2022 appear in the
    same row. The extractor therefore:

      1. Finds rows explicitly describing lagaan.
      2. Ignores financial-year tokens.
      3. Uses the numeric columns associated with the row.
      4. Prefers the "कुल" amount when a clear total column
         exists.
      5. For "जमाबंदी लगान", uses its monetary row value,
         not the year.

    Returns None when a reliable monetary value cannot be
    determined.
    """

    lagaan_rows = []

    for row in rows:

        normalized = normalize_key(
            row["text"]
        )

        if (
            "लगान" in normalized
            or
            "जमाबंदी लगान" in normalized
        ):
            lagaan_rows.append(row)

    if not lagaan_rows:
        return None

    # Prefer an actual "जमाबंदी लगान" row over a generic
    # "सेस" row.
    jamabandi_rows = [
        row
        for row in lagaan_rows
        if "जमाबंदी" in normalize_key(
            row["text"]
        )
    ]

    candidate_rows = (
        jamabandi_rows
        if jamabandi_rows
        else lagaan_rows
    )

    for row in candidate_rows:

        numeric_words = []

        for word in row["words"]:

            value = number_from_text(
                word["text"]
            )

            if value is None:
                continue

            # Do not treat 2021-2022 etc. as money.
            if re.fullmatch(
                r"\d{4}",
                value
            ):
                year = int(value)

                if 1900 <= year <= 2100:
                    continue

            numeric_words.append(
                (
                    value,
                    word
                )
            )

        if not numeric_words:
            continue

        # Numbers in these financial tables are monetary values.
        # Prefer decimals because they are much less likely to be
        # a record/year identifier.
        decimal_values = [
            value
            for value, word in numeric_words
            if "." in value
        ]

        if decimal_values:
            # For a "जमाबंदी लगान" row in the common structure:
            #
            # annual rate | arrears | current | interest | total
            #
            # the first/current/total values may all be valid
            # monetary values. For the canonical single "lagaan"
            # field, use the TOTAL when there are >= 5 amounts;
            # otherwise use the first monetary value.
            if len(decimal_values) >= 5:
                return decimal_values[4]

            if len(decimal_values) >= 3:
                return decimal_values[-1]

            return decimal_values[0]

        # Integer monetary value fallback.
        if numeric_words:
            return numeric_words[0][0]

    return None


# ============================================================
# LABEL/VALUE FALLBACK
# ============================================================

LABEL_PATTERNS = {
    "district": [
        r"जिला\s*[:\-]+\s*(.+?)(?=\s+अंचल|\s+हल्का|\s+मौजा|$)"
    ],
    "circle": [
        r"अंचल\s*[:\-]+\s*(.+?)(?=\s+हल्का|\s+मौजा|$)"
    ],
    "halka": [
        r"हल्का\s*[:\-]+\s*(.+?)(?=\s+मौजा|$)"
    ],
    "village": [
        r"मौजा\s*[:\-]+\s*(.+?)(?=\s+जमाबंदी|\s+मौजा/थाना|\s+थाना|$)"
    ],
    "raiyat_name": [
        r"जमाबंदी\s*रेयत\s*का\s*नाम\s*[:\-]+\s*(.+)$",
        r"रेयत\s*का\s*नाम\s*[:\-]+\s*(.+)$",
        r"रैयत\s*का\s*नाम\s*[:\-]+\s*(.+)$",
    ],
    "father_husband_name": [
        r"अभिभावक\s*का\s*नाम\s*[:\-]+\s*(.+)$",
        r"पिता\s*/?\s*पति\s*का\s*नाम\s*[:\-]+\s*(.+)$",
    ],
    "address": [
        r"पता\s*[:\-]+\s*(.+)$"
    ],
}


def extract_label_values(region):
    text = region.get(
        "text",
        ""
    )

    if not text:
        return {}

    result = {}

    for field, patterns in LABEL_PATTERNS.items():

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE | re.MULTILINE
            )

            if match:

                value = clean_text(
                    match.group(1)
                )

                if value:
                    result[field] = value
                    break

    return result


# ============================================================
# ONE TABLE REGION
# ============================================================

def extract_table(region):

    words = get_words(
        region
    )

    if not words:
        return None

    rows = group_into_rows(
        words
    )

    raw_rows = []

    for row in rows:

        raw_rows.append({
            "text": row["text"],
            "words": row["words"]
        })

    # --------------------------------------------------------
    # Existing extraction logic
    # --------------------------------------------------------

    land_fields = extract_land_table(
        rows
    )

    lagaan = extract_lagaan(
        rows
    )

    # Keep label/value information if a DETR table candidate
    # actually contains a bordered metadata block.
    label_fields = extract_label_values(
        region
    )

    # --------------------------------------------------------
    # Confidence for extracted land fields
    # --------------------------------------------------------

    land_confidence, land_evidence = land_field_confidences(
        rows,
        land_fields
    )

    # --------------------------------------------------------
    # Confidence for lagaan
    # --------------------------------------------------------

    lagaan_score, lagaan_evidence, lagaan_reason = lagaan_confidence(
        rows,
        lagaan
    )

    # --------------------------------------------------------
    # Confidence for label/value fallback
    # --------------------------------------------------------

    label_confidence, label_evidence = label_field_confidences(
        region,
        label_fields
    )

    # --------------------------------------------------------
    # Existing output fields are preserved.
    # New confidence metadata is added alongside them.
    # --------------------------------------------------------

    result = {
        "region_id": region.get(
            "region_id"
        ),

        "khata_no": land_fields[
            "khata_no"
        ],

        "khesra_no": land_fields[
            "khesra_no"
        ],

        "area": land_fields[
            "area"
        ],

        "area_unit": land_fields[
            "area_unit"
        ],

        "lagaan": lagaan,

        "raw_rows": raw_rows,

        # ----------------------------------------------------
        # FIELD-LEVEL CONFIDENCE
        # ----------------------------------------------------

        "field_confidence": {
            "khata_no": land_confidence["khata_no"],
            "khesra_no": land_confidence["khesra_no"],
            "area": land_confidence["area"],
            "area_unit": land_confidence["area_unit"],
            "lagaan": lagaan_score,

            # Label/value fallback fields
            **label_confidence
        },

        "confidence_status": {},

        "confidence_reason": {
            "lagaan": lagaan_reason
        },

        "field_evidence": {
            "khata_no": land_evidence["khata_no"],
            "khesra_no": land_evidence["khesra_no"],
            "area": land_evidence["area"],
            "area_unit": land_evidence["area_unit"],
            "lagaan": lagaan_evidence,

            **label_evidence
        }
    }

    # Add regex label/value fields exactly as before.
    result.update(
        label_fields
    )

    # --------------------------------------------------------
    # Status for every field
    # --------------------------------------------------------

    for field, score in result["field_confidence"].items():

        result["confidence_status"][field] = (
            confidence_status(score)
        )

    # --------------------------------------------------------
    # Overall TABLE EXTRACTION confidence
    #
    # Only fields that were actually extracted are included.
    # This prevents null fields from artificially lowering the
    # score.
    # --------------------------------------------------------

    extracted_confidences = []

    for field, score in result["field_confidence"].items():

        if result.get(field) is not None:
            extracted_confidences.append(score)

    overall_score = average(
        extracted_confidences
    )

    result["stage_confidence"] = {
        "score": overall_score,
        "status": confidence_status(
            overall_score
        ),
        "method": (
            "Average of confidence scores for successfully "
            "extracted fields."
        )
    }

    return result


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("GENERIC TABLE EXTRACTION")
print("=" * 70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Classifier output not found:\n{INPUT_FILE}"
    )

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


table_results = []

for region in data.get(
    "regions",
    []
):

    if region.get(
        "route"
    ) != "table":
        continue

    result = extract_table(
        region
    )

    if result is None:
        continue

    # Ignore completely empty candidates.
    if not any(
        result.get(field) is not None
        for field in [
            "khata_no",
            "khesra_no",
            "area",
            "area_unit",
            "lagaan",
            "district",
            "circle",
            "halka",
            "village",
            "raiyat_name",
            "father_husband_name",
            "address",
        ]
    ):
        continue

    table_results.append(
        result
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        table_results,
        f,
        ensure_ascii=False,
        indent=4
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    f"\nTable regions extracted: {len(table_results)}"
)

for result in table_results:

    print("\n" + "-" * 60)
    print(
        "Region:",
        result.get("region_id")
    )
    print(
        "Khata :",
        result.get("khata_no")
    )
    print(
        "Khesra:",
        result.get("khesra_no")
    )
    print(
        "Area  :",
        result.get("area")
    )
    print(
        "Unit  :",
        result.get("area_unit")
    )
    print(
        "Lagaan:",
        result.get("lagaan")
    )

    for field in [
        "district",
        "circle",
        "halka",
        "village",
        "raiyat_name",
        "father_husband_name",
        "address",
    ]:
        if result.get(field) is not None:
            print(
                f"{field:22}:",
                result[field]
            )

print("\nSaved:", OUTPUT_FILE)
