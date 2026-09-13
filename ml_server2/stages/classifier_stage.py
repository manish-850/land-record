"""
Stage 2: Layout classification + routing.

Refactored from Classifier_of_OCR.py.

Given the OCR-stage output (from stages.ocr_stage.run_ocr) and the path to
the source page image, this runs a DETR layout model, then decides -- using
the *internal structure* of each region, not just the DETR label -- whether
a region is really a table, a label (key:value) block, a paragraph, or a
header/footer/etc.

The heavy layout model is loaded once (lazily, on first use) and cached at
module level, so repeated calls / API requests do not reload it.

Public API:
    load_model()                          -> (model, processor, device)
    classify_document(ocr_result, image_path, model=None, processor=None,
                       device=None)        -> dict  (same shape as the
                                                       original script's
                                                       OUTPUT_PATH JSON)
"""

import threading

import cv2
import numpy as np
import re
import torch
from PIL import Image
from transformers import AutoImageProcessor, DetrForSegmentation


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "cmarkea/detr-layout-detection"

CONFIDENCE_THRESHOLD = 0.30


# ============================================================
# TABLE STRUCTURE SETTINGS
# ============================================================

ROW_TOLERANCE_FACTOR = 0.75
COLUMN_TOLERANCE_FACTOR = 0.10

# A compact region containing a clear table header and a data row
# should be accepted even if it has only two rows.
MIN_HEADER_MATCHES = 2
MIN_ALIGNED_COLUMNS = 2

# Very large DETR regions are often parent/page detections.
HUGE_REGION_RATIO = 0.70

# Compact key:value metadata blocks can be labels.
MAX_LABEL_REGION_RATIO = 0.45

HEADER_ALIASES = {
    "khata": [
        "खाता",
        "खाता संख्या",
        "खाता नं",
        "khata",
        "khata no",
        "account no",
    ],
    "khesra": [
        "खेसरा",
        "खेसरा संख्या",
        "खेसरा/संख्या",
        "खसरा",
        "खसरा संख्या",
        "khesra",
        "khasra",
        "plot no",
        "survey no",
    ],
    "area": [
        "रकबा",
        "क्षेत्रफल",
        "रकबा/डिसमिल",
        "रकबा/डिसमील",
        "area",
        "extent",
    ],
    "lagaan": [
        "लगान",
        "जमाबंदी लगान",
        "भूमि लगान",
        "land revenue",
        "land tax",
        "revenue",
    ],
    "total": [
        "कुल",
        "कुल राशि",
        "कुल रकम",
        "कुल वसूली",
        "total",
        "total amount",
        "grand total",
    ],
}

# ============================================================
# MODEL LOADING (lazy singleton)
# ============================================================

_model_lock = threading.Lock()
_model_state = {"model": None, "processor": None, "device": None}


def load_model():
    """
    Loads (and caches) the DETR layout model. Safe to call repeatedly --
    the model is only loaded once per process.
    """
    if _model_state["model"] is not None:
        return _model_state["model"], _model_state["processor"], _model_state["device"]

    with _model_lock:
        if _model_state["model"] is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
            model = DetrForSegmentation.from_pretrained(MODEL_NAME)
            model.to(device)
            model.eval()

            _model_state["model"] = model
            _model_state["processor"] = processor
            _model_state["device"] = device

    return _model_state["model"], _model_state["processor"], _model_state["device"]

# ============================================================
# GEOMETRY / PURE HELPERS
# ============================================================

def bbox_area(bbox):
    x1, y1, x2, y2 = bbox

    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_area_ratio(bbox, image_width, image_height):
    return (
        bbox_area(bbox)
        /
        float(image_width * image_height)
    )


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x2 <= x1 or y2 <= y1:
        return 0

    return (x2 - x1) * (y2 - y1)


def contains_point(x, y, bbox):
    x1, y1, x2, y2 = bbox

    return (
        x1 <= x <= x2
        and
        y1 <= y <= y2
    )

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

            text = word.get(
                "text",
                ""
            ).strip()

            if not text:
                continue

            words.append({
                "text": text,
                "left": float(
                    word.get("left", 0)
                ),
                "top": float(
                    word.get("top", 0)
                ),
                "width": float(
                    word.get("width", 0)
                ),
                "height": float(
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

def normalize_match(text):

    text = str(text or "").lower()

    text = (
        text
        .replace("०", "0")
        .replace("१", "1")
        .replace("२", "2")
        .replace("३", "3")
        .replace("४", "4")
        .replace("५", "5")
        .replace("६", "6")
        .replace("७", "7")
        .replace("८", "8")
        .replace("९", "9")
    )

    text = re.sub(
        r"[/,:;|()\-]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def is_numeric(text):
    text = str(text).strip()

    return bool(
        re.fullmatch(
            r"\d+(?:\.\d+)?",
            text
        )
    )


def count_year_tokens(text):
    return len(
        re.findall(
            r"\b(?:19|20)\d{2}\b",
            str(text)
        )
    )

def group_rows(words):

    if not words:
        return []

    heights = [
        w["height"]
        for w in words
        if w["height"] > 0
    ]

    if heights:

        heights.sort()

        mid = len(heights) // 2

        if len(heights) % 2:
            median_height = heights[mid]
        else:
            median_height = (
                heights[mid - 1]
                +
                heights[mid]
            ) / 2

    else:

        median_height = 15

    tolerance = max(
        6,
        median_height * ROW_TOLERANCE_FACTOR
    )

    rows = []

    for word in sorted(
        words,
        key=lambda w: (
            w["top"],
            w["left"]
        )
    ):

        cy = (
            word["top"]
            +
            word["height"] / 2
        )

        best = None
        best_distance = None

        for row in rows:

            distance = abs(
                cy - row["center_y"]
            )

            if distance <= tolerance:

                if (
                    best_distance is None
                    or
                    distance < best_distance
                ):
                    best = row
                    best_distance = distance

        if best is None:

            rows.append({
                "center_y": cy,
                "words": [word]
            })

        else:

            best["words"].append(
                word
            )

            best["center_y"] = sum(
                w["top"] + w["height"] / 2
                for w in best["words"]
            ) / len(
                best["words"]
            )

    rows.sort(
        key=lambda row:
        row["center_y"]
    )

    for row in rows:

        row["words"].sort(
            key=lambda w:
            w["left"]
        )

        row["text"] = " ".join(
            w["text"]
            for w in row["words"]
        )

    return rows

def header_matches(text):

    normalized = normalize_match(
        text
    )

    matches = set()

    for field, aliases in HEADER_ALIASES.items():

        for alias in aliases:

            alias_normalized = normalize_match(
                alias
            )

            if (
                alias_normalized
                and
                alias_normalized in normalized
            ):

                matches.add(field)
                break

    return matches


def find_header_rows(rows):

    header_rows = []

    for index, row in enumerate(rows):

        matches = header_matches(
            row["text"]
        )

        numeric_count = sum(
            1
            for w in row["words"]
            if is_numeric(w["text"])
        )

        # Strong header:
        # at least two known semantic concepts and
        # little/no numeric data.
        if (
            len(matches)
            >= MIN_HEADER_MATCHES
            and
            numeric_count <= 1
        ):

            header_rows.append(
                index
            )

    return header_rows

def infer_columns(rows, image_width):

    if len(rows) < 2:
        return []

    # Word starting positions are useful because OCR word boxes
    # preserve the original horizontal layout.
    starts = []

    for row in rows:

        if not row["words"]:
            continue

        starts.append([
            w["left"]
            for w in row["words"]
        ])

    if len(starts) < 2:
        return []

    all_x = sorted(
        x
        for row in starts
        for x in row
    )

    if not all_x:
        return []

    page_width = float(image_width)

    tolerance = max(
        25,
        page_width * COLUMN_TOLERANCE_FACTOR
    )

    clusters = []

    for x in all_x:

        placed = False

        for cluster in clusters:

            if abs(
                x
                -
                cluster["center"]
            ) <= tolerance:

                cluster["values"].append(x)

                cluster["center"] = (
                    sum(
                        cluster["values"]
                    )
                    /
                    len(
                        cluster["values"]
                    )
                )

                placed = True
                break

        if not placed:

            clusters.append({
                "center": float(x),
                "values": [x]
            })

    # Only retain positions appearing in multiple rows.
    repeated = []

    for cluster in clusters:

        row_hits = 0

        for row_positions in starts:

            if any(
                abs(
                    x
                    -
                    cluster["center"]
                ) <= tolerance
                for x in row_positions
            ):
                row_hits += 1

        if row_hits >= 2:

            repeated.append(
                cluster["center"]
            )

    return sorted(
        repeated
    )

def detect_grid_strength(region, image_width, image_height, cv_image):

    x1, y1, x2, y2 = map(
        int,
        region["bbox"]
    )

    x1 = max(
        0,
        min(
            image_width - 1,
            x1
        )
    )

    y1 = max(
        0,
        min(
            image_height - 1,
            y1
        )
    )

    x2 = max(
        0,
        min(
            image_width,
            x2
        )
    )

    y2 = max(
        0,
        min(
            image_height,
            y2
        )
    )

    if x2 <= x1 or y2 <= y1:
        return {
            "horizontal": 0,
            "vertical": 0
        }

    crop = cv_image[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return {
            "horizontal": 0,
            "vertical": 0
        }

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10
    )

    h, w = binary.shape

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(15, w // 8),
            1
        )
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            1,
            max(15, h // 8)
        )
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    h_count = 0

    num, _, stats, _ = cv2.connectedComponentsWithStats(
        horizontal,
        8
    )

    for i in range(1, num):

        width = stats[
            i,
            cv2.CC_STAT_WIDTH
        ]

        if width >= w * 0.35:
            h_count += 1

    v_count = 0

    num, _, stats, _ = cv2.connectedComponentsWithStats(
        vertical,
        8
    )

    for i in range(1, num):

        height = stats[
            i,
            cv2.CC_STAT_HEIGHT
        ]

        if height >= h * 0.25:
            v_count += 1

    return {
        "horizontal": h_count,
        "vertical": v_count
    }

def extract_key_value_lines(text):
    """
    Splits OCR text into key:value pairs.

    IMPORTANT FIX:
    The document's two-column layout (e.g. "जिला" beside "अंचल",
    "हल्का" beside "मौजा") sometimes gets merged onto a single OCR
    text line. The original version of this function only ever
    produced ONE key:value pair per line, so the "value" of the
    first field greedily swallowed the second field's "label:value"
    text too (e.g. district became "Rohtas अंचल :- Dehri" instead
    of just "Rohtas").

    This version anchors on KNOWN label keywords (kept in sync with
    label_extractor.py's FIELD_ALIASES) and splits a line into
    separate pairs at each recognized label boundary, instead of
    assuming one line = one pair. As a side benefit, this also fixes
    labels that had OCR garbage glued onto their front (e.g.
    "aor 3 मौजा/थाना संख्या"), since we anchor on the label
    substring itself rather than "everything before the first
    separator".
    """

    KNOWN_LABELS = [
        "जिला", "district",
        "अंचल", "circle",
        "हल्का", "halka",
        "मौजा/थाना संख्या", "थाना संख्या", "thana no",
        "मौजा", "ग्राम", "गांव", "village",
        "जमाबन्दी संख्या", "जमाबंदी संख्या",
        "कंप्यूटरीकृत जमाबन्दी संख्या", "कंप्यूटरीकृत जमाबंदी संख्या",
        "खाता संख्या", "खाता नं", "khata no",
        "खेसरा संख्या", "खेसरा नं", "खसरा संख्या", "khesra no", "khasra no",
        "जमाबंदी रेयत का नाम", "रेयत का नाम", "रैयत का नाम", "raiyat name",
        "अभिभावक का नाम", "पिता/पति का नाम", "पिता का नाम", "पति का नाम",
        "father/husband name", "father husband name",
        "पता", "address",
        "भूमि का प्रकार", "भूमि वर्ग", "land type",
        "रकबा", "क्षेत्रफल", "area",
        "दखल", "कब्जा", "possession",
        "उत्तर सीमा", "north boundary",
        "दक्षिण सीमा", "south boundary",
        "पूर्व सीमा", "east boundary",
        "पश्चिम सीमा", "west boundary",
        "लगान", "lagaan",
        "नामांतरण", "दाखिल खारिज", "mutation details",
        "टिप्पणी", "अभियुक्ति", "remarks",
        "तिथि", "date",
    ]
    # Longest-first so multi-word labels are matched before shorter
    # substrings of themselves (e.g. "मौजा/थाना संख्या" before "मौजा").
    KNOWN_LABELS = sorted(set(KNOWN_LABELS), key=len, reverse=True)

    SEPARATOR_AFTER_LABEL = re.compile(r"^\s*(?::-|:|–|—)")

    pairs = []

    for line in str(text).splitlines():

        line = line.strip()

        if not line:
            continue

        # Find every position where a KNOWN label occurs, immediately
        # followed (after optional whitespace) by a separator — this
        # avoids false positives where a label word merely appears
        # inside some other value's text.
        label_positions = []

        for label in KNOWN_LABELS:
            for m in re.finditer(re.escape(label), line, flags=re.IGNORECASE):
                after = line[m.end():m.end() + 6]
                if SEPARATOR_AFTER_LABEL.match(after):
                    label_positions.append((m.start(), m.end(), label))

        if not label_positions:
            # No known label matched this line at all — fall back to
            # the original single-pair behavior so unrelated lines
            # (e.g. genuinely free-text lines) still get a best-effort
            # split rather than being silently dropped.
            match = re.match(
                r"^\s*(.+?)\s*(?::-|:|–|—|-)\s*(.+?)\s*$",
                line
            )
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                if key and value:
                    pairs.append({"key": key, "value": value})
            continue

        # Remove overlapping matches (keep earliest at each position).
        label_positions.sort(key=lambda t: t[0])
        filtered = []
        last_end = -1
        for start, end, label in label_positions:
            if start >= last_end:
                filtered.append((start, end, label))
                last_end = end

        # Split the line into segments BETWEEN consecutive recognized
        # labels, so each label only claims the text up to the next
        # recognized label (or end of line), never bleeding past it.
        for i, (start, end, label) in enumerate(filtered):
            segment_end = (
                filtered[i + 1][0] if i + 1 < len(filtered) else len(line)
            )
            segment = line[end:segment_end]
            value = re.sub(r"^\s*(?::-|:|–|—|-)\s*", "", segment).strip()

            if value:
                pairs.append({"key": label, "value": value})

    return pairs

def validate_table(region, image_width, image_height, cv_image):

    words = get_words(
        region
    )

    rows = group_rows(
        words
    )

    area_ratio = bbox_area_ratio(
        region["bbox"], image_width, image_height
    )

    if not rows:

        return {
            "is_table": False,
            "reason": "no_ocr",
            "score": 0.0,
            "rows": 0,
            "columns": 0,
            "header_matches": 0,
            "grid": {}
        }

    header_indices = find_header_rows(
        rows
    )

    header_fields = set()

    for index in header_indices:

        header_fields.update(
            header_matches(
                rows[index]["text"]
            )
        )

    columns = infer_columns(
        rows, image_width
    )

    grid = detect_grid_strength(
        region, image_width, image_height, cv_image
    )

    key_values = extract_key_value_lines(
        region.get("text", "")
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # A compact region with explicit multiple key:value lines
    # is a label block, not a table.
    # --------------------------------------------------------

    key_value_ratio = (
        len(key_values)
        /
        max(
            1,
            len(
                [
                    line
                    for line in region.get(
                        "text",
                        ""
                    ).splitlines()
                    if line.strip()
                ]
            )
        )
    )

    compact_label_block = (
        area_ratio < MAX_LABEL_REGION_RATIO
        and
        len(key_values) >= 2
        and
        key_value_ratio >= 0.45
    )

    if compact_label_block:

        return {
            "is_table": False,
            "reason": "key_value_block",
            "score": 0.0,
            "rows": len(rows),
            "columns": len(columns),
            "header_matches": len(
                header_fields
            ),
            "grid": grid
        }

    # --------------------------------------------------------
    # Strong TABLE signals
    # --------------------------------------------------------

    score = 0.0

    # Header semantics
    if len(header_fields) >= 3:
        score += 0.40
    elif len(header_fields) >= 2:
        score += 0.32
    elif len(header_fields) >= 1:
        score += 0.12

    # Rows
    if len(rows) >= 5:
        score += 0.25
    elif len(rows) >= 3:
        score += 0.20
    elif len(rows) >= 2:
        score += 0.12

    # Repeated columns
    if len(columns) >= 4:
        score += 0.25
    elif len(columns) >= 3:
        score += 0.22
    elif len(columns) >= 2:
        score += 0.16

    # Grid
    grid_count = (
        grid["horizontal"]
        +
        grid["vertical"]
    )

    if grid_count >= 6:
        score += 0.20
    elif grid_count >= 3:
        score += 0.15
    elif grid_count >= 1:
        score += 0.08

    # Numeric content
    numeric_words = sum(
        1
        for w in words
        if is_numeric(w["text"])
    )

    if numeric_words >= 4:
        score += 0.08
    elif numeric_words >= 2:
        score += 0.04

    # --------------------------------------------------------
    # Obvious small table rescue
    #
    # This is the critical fix for forms like:
    #
    #   खाता संख्या | खेसरा संख्या | रकबा/डिसमिल
    #   79          | 345          | 2.5 डिसमील
    #
    # It needs only:
    #   - >= 2 semantic headers
    #   - >= 2 rows
    #   - >= 2 aligned columns
    #   - numeric data
    # --------------------------------------------------------

    # Strong grid + repeated columns can identify a table even when
    # OCR misses one or more semantic headers.
    geometric_table = (
        len(rows) >= 2
        and
        len(columns) >= MIN_ALIGNED_COLUMNS
        and
        (
            grid_count >= 2
            or
            (
                len(header_fields) >= 2
                and
                numeric_words >= 2
            )
        )
        and
        area_ratio < HUGE_REGION_RATIO
    )

    obvious_small_table = (
        len(header_fields) >= 2
        and
        len(rows) >= 2
        and
        len(columns) >= MIN_ALIGNED_COLUMNS
        and
        numeric_words >= 2
        and
        area_ratio < HUGE_REGION_RATIO
    )

    # --------------------------------------------------------
    # Large parent/page detections
    # --------------------------------------------------------

    if area_ratio >= HUGE_REGION_RATIO:

        # A huge region can still be a genuine table if it has
        # exceptionally strong internal table structure.
        strong_huge_table = (
            len(header_fields) >= 3
            and
            len(columns) >= 4
            and
            len(rows) >= 5
            and
            grid_count >= 3
        )

        if not strong_huge_table:

            return {
                "is_table": False,
                "reason": "oversized_parent_region",
                "score": round(score, 3),
                "rows": len(rows),
                "columns": len(columns),
                "header_matches": len(
                    header_fields
                ),
                "grid": grid
            }

    if obvious_small_table:

        return {
            "is_table": True,
            "reason": "obvious_header_data_table",
            "score": round(
                max(
                    score,
                    0.70
                ),
                3
            ),
            "rows": len(rows),
            "columns": len(columns),
            "header_matches": len(
                header_fields
            ),
            "grid": grid
        }

    if geometric_table:

        return {
            "is_table": True,
            "reason": "geometric_table_structure",
            "score": round(
                max(
                    score,
                    0.62
                ),
                3
            ),
            "rows": len(rows),
            "columns": len(columns),
            "header_matches": len(
                header_fields
            ),
            "grid": grid
        }

    return {
        "is_table": score >= 0.55,
        "reason": (
            "strong_table_structure"
            if score >= 0.55
            else "weak_table_structure"
        ),
        "score": round(
            score,
            3
        ),
        "rows": len(rows),
        "columns": len(columns),
        "header_matches": len(
            header_fields
        ),
        "grid": grid
    }

def classify_non_table(
    region
):

    text = region.get(
        "text",
        ""
    )

    if not text.strip():

        return {
            "route": "ignore"
        }

    key_values = extract_key_value_lines(
        text
    )

    if key_values:

        return {
            "route": "label",
            "fields": key_values
        }

    return {
        "route": "paragraph"
    }

# ============================================================
# MAIN ENTRY POINT
# ============================================================

def classify_document(ocr_result, image_path, model=None, processor=None, device=None, verbose=False):
    """
    Runs layout detection + table/label/paragraph routing for a single
    OCR-stage document result (see stages.ocr_stage.run_ocr).

    Returns a dict:
        {
            "document_id": ...,
            "file_hash": ...,
            "image_path": ...,
            "regions": [ {region_id, label, confidence, bbox, ocr_blocks,
                          text, route, ...}, ... ]
        }
    """

    if model is None or processor is None or device is None:
        model, processor, device = load_model()

    page = ocr_result["pages"][0]
    ocr_blocks = page.get("ocr_blocks", [])

    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if verbose:
        print("Running layout detection...")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[image_height, image_width]], device=device)

    results = processor.post_process_object_detection(
        outputs,
        threshold=CONFIDENCE_THRESHOLD,
        target_sizes=target_sizes
    )[0]


    regions = []

    for region_id, (
        score,
        label_id,
        box
    ) in enumerate(
        zip(
            results["scores"],
            results["labels"],
            results["boxes"]
        )
    ):

        score = float(
            score.detach().cpu()
        )

        label_id = int(
            label_id.detach().cpu()
        )

        box = [
            int(round(float(x)))
            for x in box.detach().cpu().tolist()
        ]

        label = model.config.id2label[
            label_id
        ]

        regions.append({
            "region_id": region_id,
            "label": label,
            "confidence": round(score, 3),
            "bbox": box
        })


    # ============================================================
    # ASSIGN OCR BLOCKS
    # ============================================================

    for region in regions:

        region["ocr_blocks"] = []
        region["text"] = ""


    for block in ocr_blocks:

        block_bbox = block.get("bbox")

        if not block_bbox:
            continue

        cx, cy = bbox_center(
            block_bbox
        )

        matching = []

        for region in regions:

            if contains_point(
                cx,
                cy,
                region["bbox"]
            ):
                matching.append(region)

        if not matching:
            continue

        # Smallest containing region keeps the OCR block.
        best = min(
            matching,
            key=lambda r: bbox_area(
                r["bbox"]
            )
        )

        best["ocr_blocks"].append(
            block
        )


    for region in regions:

        region["ocr_blocks"].sort(
            key=lambda block: (
                block["bbox"][1],
                block["bbox"][0]
            )
        )

        region["text"] = "\n".join(
            block.get("text", "")
            for block in region["ocr_blocks"]
            if block.get("text")
        )

    for region in regions:

        layout_label = region[
            "label"
        ].lower()

        # --------------------------------------------------------
        # TABLE-LIKE TEST
        #
        # Run this for both DETR Table and DETR Text candidates.
        # A genuine table wins over a misleading "Text" classification.
        # --------------------------------------------------------

        if layout_label in [
            "table",
            "text"
        ]:

            validation = validate_table(
                region, image_width, image_height, cv_image
            )

            region[
                "table_validation"
            ] = validation

            if validation["is_table"]:

                region["route"] = "table"

                continue

        # --------------------------------------------------------
        # NON-TABLE TEXT
        # --------------------------------------------------------

        if layout_label == "text":

            region.update(
                classify_non_table(
                    region
                )
            )

        # --------------------------------------------------------
        # DETR TABLE THAT FAILED TABLE TEST
        #
        # Only a compact, genuinely key:value region can become
        # a label. Otherwise ignore it.
        # --------------------------------------------------------

        elif layout_label == "table":

            validation = region.get(
                "table_validation",
                {}
            )

            text = region.get(
                "text",
                ""
            )

            key_values = extract_key_value_lines(
                text
            )

            line_count = len([
                line
                for line in text.splitlines()
                if line.strip()
            ])

            key_value_ratio = (
                len(key_values)
                /
                max(1, line_count)
            )

            area_ratio = bbox_area_ratio(
                region["bbox"], image_width, image_height
        )

            # A compact region with clear key:value structure is label.
            if (
                len(key_values) >= 2
                and
                key_value_ratio >= 0.45
                and
                area_ratio < MAX_LABEL_REGION_RATIO
            ):

                region["route"] = "label"
                region["fields"] = key_values

            else:

                # Large/mixed/weak candidates are not useful as tables.
                region["route"] = "ignore"

        # --------------------------------------------------------
        # OTHER VISUAL REGIONS
        # --------------------------------------------------------

        elif layout_label in [
            "title",
            "section-header",
            "page-header"
        ]:

            region["route"] = "header"

        elif layout_label == "page-footer":

            region["route"] = "footer"

        elif layout_label == "picture":

            region["route"] = "picture"

        elif layout_label == "caption":

            region["route"] = "caption"

        elif layout_label == "footnote":

            region["route"] = "footnote"

        elif layout_label == "list-item":

            region["route"] = "list"

        elif layout_label == "formula":

            region["route"] = "formula"

        else:

            region["route"] = "ignore"

    # ============================================================
    # TABLE / PARAGRAPH OVERLAP RESOLUTION
    #
    # DETR sometimes produces two overlapping boxes over the same
    # block of narrative prose: one gets validated as a "table"
    # (e.g. because of underlines/ruled lines in the scan), the
    # other falls through to "paragraph". When that happens, most
    # OCR lines get assigned to whichever box is smaller, which can
    # silently strand real narrative content (containing khata/area/
    # village numbers written as prose, not a grid) inside a "table"
    # region -- so it never reaches the LLM extractor.
    #
    # Paragraph wins on significant overlap: the table region's OCR
    # blocks are folded into the paragraph region (so no text is
    # lost) and the table region is dropped.
    # ============================================================

    TABLE_PARAGRAPH_OVERLAP_THRESHOLD = 0.5

    paragraph_regions = [
        r for r in regions if r.get("route") == "paragraph"
    ]

    for region in regions:

        if region.get("route") != "table":
            continue

        region_area = bbox_area(region["bbox"])

        if region_area <= 0:
            continue

        for other in paragraph_regions:

            overlap = intersection_area(region["bbox"], other["bbox"])

            overlap_ratio = overlap / float(region_area)

            if overlap_ratio >= TABLE_PARAGRAPH_OVERLAP_THRESHOLD:

                # Fold the table region's OCR blocks into the
                # paragraph region and rebuild its text in reading
                # order, then drop the table region.
                other["ocr_blocks"].extend(region["ocr_blocks"])

                other["ocr_blocks"].sort(
                    key=lambda block: (
                        block["bbox"][1],
                        block["bbox"][0]
                    )
                )

                other["text"] = "\n".join(
                    block.get("text", "")
                    for block in other["ocr_blocks"]
                    if block.get("text")
                )

                region["route"] = "ignore"
                region["ignored_reason"] = "absorbed_into_overlapping_paragraph"

                break

    accepted_tables = [
        r
        for r in regions
        if r.get("route") == "table"
    ]

    for region in list(accepted_tables):

        region_area = bbox_area(
            region["bbox"]
        )

        region_ratio = bbox_area_ratio(
            region["bbox"], image_width, image_height
        )

        for other in accepted_tables:

            if (
                region["region_id"]
                ==
                other["region_id"]
            ):
                continue

            other_area = bbox_area(
                other["bbox"]
            )

            if other_area <= 0:
                continue

            overlap = intersection_area(
                region["bbox"],
                other["bbox"]
            )

            other_coverage = (
                overlap
                /
                float(other_area)
            )

            other_ratio = bbox_area_ratio(
                other["bbox"], image_width, image_height
        )

            # A giant region containing a smaller table fragment is
            # usually a parent/background detection.
            if (
                other_coverage >= 0.90
                and
                region_area > other_area * 2.0
                and
                region_ratio > other_ratio
            ):

                region["route"] = "ignore"

                if "table_validation" in region:

                    region[
                        "table_validation"
                    ]["reason"] += (
                        ",parent_of_smaller_table"
                    )

                break

    accepted_tables = [
        r
        for r in regions
        if r.get("route") == "table"
    ]

    for region in regions:

        if region.get("route") != "table":
            continue

        area = bbox_area(
            region["bbox"]
        )

        for other in accepted_tables:

            if (
                region["region_id"]
                ==
                other["region_id"]
            ):
                continue

            other_area = bbox_area(
                other["bbox"]
            )

            if other_area <= 0:
                continue

            overlap = intersection_area(
                region["bbox"],
                other["bbox"]
            )

            other_coverage = (
                overlap
                /
                float(other_area)
            )

            # Current region is a much larger parent containing
            # almost all of the smaller real table.
            if (
                other_coverage >= 0.90
                and
                area > other_area * 2.0
                and
                bbox_area_ratio(
                    region["bbox"], image_width, image_height
        ) > bbox_area_ratio(
                    other["bbox"], image_width, image_height
        )
            ):

                region["route"] = "ignore"

                if "table_validation" in region:

                    region[
                        "table_validation"
                    ]["reason"] += (
                        ",parent_of_smaller_table"
                    )

                break

    output = {
        "document_id": ocr_result.get(
            "document_id"
        ),
        "file_hash": ocr_result.get(
            "file_hash"
        ),
        "image_path": image_path,
        "regions": regions
    }

    return output


# ============================================================
# CLI (kept for standalone / debugging use)
# ============================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Classify OCR'd document regions")
    parser.add_argument("ocr_json_path")
    parser.add_argument("image_path")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.ocr_json_path, "r", encoding="utf-8") as f:
        ocr_result = json.load(f)

    result = classify_document(ocr_result, args.image_path, verbose=True)

    out_path = args.output or "classifier_output.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Regions:", len(result["regions"]))
    for region in result["regions"]:
        print(f"  [{region['region_id']}] {region['label']:15} -> {region['route']}")

    print("Saved:", out_path)