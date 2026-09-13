"""
Step 0: Document Ingestion + OCR

Produces:

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
import json
import hashlib

import pytesseract
from PIL import Image

img_path =r"bhumisetu\files_for_extraction\nikhilkajameen.png"
output_path = img_path[:-5] + "_json"


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA_PATH = r"C:\Program Files\Tesseract-OCR\tessdata"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
os.environ["TESSDATA_PREFIX"] = TESSDATA_PATH


# Check Tesseract
print("Tesseract version:")
print(pytesseract.get_tesseract_version())

print("\nAvailable languages:")
print(pytesseract.get_languages(config=""))


# ============================================================
# FILE HASH
# ============================================================

def compute_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


# ============================================================
# OCR
# ============================================================

def run_ocr(image_path, lang="hin+eng"):
    """
    Runs Tesseract OCR.

    Returns:
        full_text
        ocr_blocks

    Each OCR block contains:
        text
        confidence
        bbox
        words
    """

    image = Image.open(image_path)

    data = pytesseract.image_to_data(
        image,
        lang=lang,
        output_type=pytesseract.Output.DICT
    )

    n = len(data["text"])

    # Group words by:
    # block -> paragraph -> line
    lines = {}

    for i in range(n):

        word = data["text"][i].strip()

        if word == "":
            continue

        # Tesseract confidence can be a float
        conf = float(data["conf"][i])

        # Ignore invalid confidence
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
    # CREATE LINE-LEVEL OCR BLOCKS
    # ========================================================

    ocr_blocks = []

    for key, words in sorted(
        lines.items(),
        key=lambda item: (
            item[1][0]["top"],
            item[1][0]["left"]
        )
    ):

        line_text = " ".join(
            word["text"]
            for word in words
        )

        avg_conf = (
            sum(word["confidence"] for word in words)
            / len(words)
        )

        x0 = min(
            word["left"]
            for word in words
        )

        y0 = min(
            word["top"]
            for word in words
        )

        x1 = max(
            word["left"] + word["width"]
            for word in words
        )

        y1 = max(
            word["top"] + word["height"]
            for word in words
        )

        ocr_blocks.append({
            "text": line_text,
            "confidence": round(avg_conf, 2),
            "bbox": [x0, y0, x1, y1],
            "words": words
        })


    # ========================================================
    # FULL TEXT
    # ========================================================

    full_text = "\n".join(
        block["text"]
        for block in ocr_blocks
    )

    return full_text, ocr_blocks


# ============================================================
# PROCESS DOCUMENT
# ============================================================

def process_document(
    image_path,
    document_id,
    lang="hin+eng"
):

    file_hash = compute_file_hash(image_path)

    full_text, ocr_blocks = run_ocr(
        image_path,
        lang=lang
    )

    result = {
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

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # image_path = (
    #     r"C:\Users\nikhi\OneDrive\Desktop\nikhilkajaminha.jpg"
    # )

    # output_path = (
    #     r"C:\Users\nikhi\OneDrive\Desktop\nikhilkajameen.json"
    # )

    result = process_document(
        img_path,
        document_id="DOC001",
        lang="hin+eng"
    )


    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n===================================")
    print("OCR COMPLETE")
    print("===================================")

    print(
        f"Document ID: {result['document_id']}"
    )

    print(
        f"File hash: {result['file_hash']}"
    )

    print(
        f"Total OCR lines: "
        f"{len(result['pages'][0]['ocr_blocks'])}"
    )

    print(
        f"\nJSON saved to:\n{output_path}"
    )


    # ========================================================
    # PRINT OCR
    # ========================================================

    print(
        "\n=== LINE-BY-LINE OCR ===\n"
    )

    for block in result["pages"][0]["ocr_blocks"]:

        print(
            f"[{block['confidence']:.1f}%] "
            f"{block['text']}"
        )