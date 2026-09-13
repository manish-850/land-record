# confidence_utils.py
#
# Common confidence helpers for the BhumiSetu pipeline.
#
# All confidence values are on a 0-100 scale.
#
# Important:
# - OCR confidence comes from Tesseract.
# - Layout confidence comes from DETR.
# - Extraction confidence is an evidence score derived from
#   the OCR confidence supporting the extracted value.
# - LLM extraction does NOT expose a reliable per-field probability
#   through the current Gemini response, so its score is explicitly
#   an evidence/heuristic confidence, not a claimed model probability.


def clamp(value, low=0.0, high=100.0):
    return max(low, min(high, float(value)))


def average_confidences(items):
    """
    Accepts numbers or dictionaries containing 'confidence'.
    Returns 0 when nothing is available.
    """
    values = []

    for item in items:
        if isinstance(item, dict):
            value = item.get("confidence")
        else:
            value = item

        try:
            if value is not None:
                values.append(float(value))
        except (TypeError, ValueError):
            pass

    if not values:
        return 0.0

    return clamp(sum(values) / len(values))


def weighted_confidence(
    ocr_confidence,
    structure_confidence=0.0,
    extraction_confidence=0.0,
    weights=(0.50, 0.30, 0.20),
):
    """
    Generic evidence confidence.

    Default:
        50% OCR evidence
        30% structural evidence
        20% extraction/matching evidence
    """
    a, b, c = weights

    score = (
        a * float(ocr_confidence)
        + b * float(structure_confidence)
        + c * float(extraction_confidence)
    )

    return round(clamp(score), 2)


def value_confidence(
    supporting_words,
    structure_confidence=0.0,
    match_confidence=100.0,
):
    """
    Confidence for a field extracted from OCR.

    supporting_words:
        OCR word dictionaries used to produce the value.

    structure_confidence:
        0-100 confidence in the structure used for extraction,
        e.g. table/header alignment.

    match_confidence:
        0-100 confidence that the label/header matched the
        canonical field.
    """
    ocr = average_confidences(
        supporting_words
    )

    return weighted_confidence(
        ocr_confidence=ocr,
        structure_confidence=structure_confidence,
        extraction_confidence=match_confidence,
    )


def stage_summary(confidences):
    """
    Summary confidence for one processing stage.
    """
    return round(
        average_confidences(confidences),
        2
    )


def confidence_status(score):
    score = float(score)

    if score >= 85:
        return "HIGH"
    if score >= 65:
        return "MEDIUM"
    if score >= 40:
        return "LOW"

    return "VERY_LOW"
