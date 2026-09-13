"""
Stage 1: Document Ingestion + OCR

Refactored from OCR_Reader.py so it can be imported by the pipeline / API
instead of running as a standalone script against hardcoded paths.

Public API:
    run_ocr(image_path, document_id=None, lang="hin+eng") -> dict

Output shape (unchanged from the original script):

{
    "document_id": "...",
    "file_hash": "...",
    "pages": [
        {
            "page_number": 1,
            "original_image_path": "...",
            "language": "hin+eng",
            "ocr_text": "...",
            "ocr_blocks": [
                {
                    "text": "...",
                    "confidence": 95.4,
                    "bbox": [x1, y1, x2, y2],
                    "words": [...]
                }
            ]
        }
    ]
}
"""

import os
import hashlib
import uuid

import pytesseract
from PIL import Image

# ============================================================
# TESSERACT CONFIGURATION
#
# Configurable via environment variables so the same code works
# on Windows (where tesseract.exe has to be pointed at explicitly)
# and on Linux/Docker (where "tesseract" is usually already on PATH).
# ============================================================

TESSERACT_PATH = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA_PATH = os.getenv("TESSDATA_PREFIX", r"C:\Program Files\Tesseract-OCR\tessdata")

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

if os.path.isdir(TESSDATA_PATH):
    os.environ["TESSDATA_PREFIX"] = TESSDATA_PATH


def check_tesseract():
    """Returns (version, available_languages). Raises if tesseract is not usable."""
    version = pytesseract.get_tesseract_version()
    langs = pytesseract.get_languages(config="")
    return version, langs


# ============================================================
# FILE HASH
# ============================================================

def compute_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


# ============================================================
# OCR
# ============================================================

def run_tesseract(image_path, lang="hin+eng"):
    """
    Runs Tesseract OCR.

    Returns:
        full_text, ocr_blocks

    Each OCR block contains:
        text, confidence, bbox, words
    """

    image = Image.open(image_path)

    data = pytesseract.image_to_data(
        image,
        lang=lang,
        output_type=pytesseract.Output.DICT
    )

    n = len(data["text"])

    # Group words by: block -> paragraph -> line
    lines = {}

    for i in range(n):
        word = data["text"][i].strip()

        if word == "":
            continue

        conf = float(data["conf"][i])

        if conf < 0:
            continue

        key = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i]
        )

        lines.setdefault(key, []).append({
            "text": word,
            "confidence": conf,
            "left": data["left"][i],
            "top": data["top"][i],
            "width": data["width"][i],
            "height": data["height"][i],
        })

    # ========================================================
    # LINE-LEVEL OCR BLOCKS
    # ========================================================

    ocr_blocks = []

    for key, words in sorted(
        lines.items(),
        key=lambda item: (item[1][0]["top"], item[1][0]["left"])
    ):
        line_text = " ".join(word["text"] for word in words)

        avg_conf = sum(word["confidence"] for word in words) / len(words)

        x0 = min(word["left"] for word in words)
        y0 = min(word["top"] for word in words)
        x1 = max(word["left"] + word["width"] for word in words)
        y1 = max(word["top"] + word["height"] for word in words)

        ocr_blocks.append({
            "text": line_text,
            "confidence": round(avg_conf, 2),
            "bbox": [x0, y0, x1, y1],
            "words": words
        })

    full_text = "\n".join(block["text"] for block in ocr_blocks)

    return full_text, ocr_blocks


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def run_ocr(image_path, document_id=None, lang="hin+eng"):
    """
    Runs OCR on a single-page document image and returns the
    OCR-stage JSON structure expected by the classifier stage.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if document_id is None:
        document_id = uuid.uuid4().hex[:12]

    file_hash = compute_file_hash(image_path)

    full_text, ocr_blocks = run_tesseract(image_path, lang=lang)

    return {
        "document_id": document_id,
        "file_hash": file_hash,
        "pages": [
            {
                "page_number": 1,
                "original_image_path": image_path,
                "language": lang,
                "ocr_text": full_text,
                "ocr_blocks": ocr_blocks
            }
        ]
    }


# ============================================================
# CLI (kept for standalone / debugging use)
# ============================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run OCR on an image")
    parser.add_argument("image_path")
    parser.add_argument("--lang", default="hin+eng")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    version, langs = check_tesseract()
    print("Tesseract version:", version)
    print("Available languages:", langs)

    result = run_ocr(args.image_path, lang=args.lang)

    out_path = args.output or (os.path.splitext(args.image_path)[0] + "_ocr.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nOCR lines: {len(result['pages'][0]['ocr_blocks'])}")
    print("Saved:", out_path)
