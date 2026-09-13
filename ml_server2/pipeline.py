"""
BhumiSetu extraction pipeline.

    1. OCR reading            (stages.ocr_stage)
    2. Layout classification  (stages.classifier_stage)
    3. Field extraction       (stages.label_stage, stages.table_stage, stages.llm_stage)
    4. Merge                  (stages.merger_stage)
    5. Output                 -> the flat FIELDS schema below

Usage as a library:

    from pipeline import run_pipeline

    result = run_pipeline("path/to/page.jpeg")
    print(result["fields"])

Usage from the command line:

    python pipeline.py path/to/page.jpeg --output result.json

Notes on the LLM stage:
    It requires GEMINI_API_KEY to be set and network access. If it is not
    configured, run_pipeline() will skip it automatically (a warning is
    printed) rather than failing the whole pipeline -- the label and table
    extractors alone are often enough for structured Hindi land records.
"""

import os
import json
import time

from stages import ocr_stage
from stages import classifier_stage
from stages import label_stage
from stages import table_stage
from stages import llm_stage
from stages import merger_stage

# The canonical field schema for the final output.
FIELDS = [
    "district", "circle", "halka", "village", "thana_no",
    "khata_no", "khesra_no", "raiyat_name", "father_husband_name",
    "address", "land_type", "area", "area_unit", "possession",
    "north_boundary", "south_boundary", "east_boundary",
    "west_boundary", "lagaan", "mutation_details", "remarks"
]


def _llm_available():
    return bool(os.getenv("GEMINI_API_KEY"))


def run_pipeline(
    image_path,
    document_id=None,
    lang="hin+eng",
    use_llm=None,
    save_intermediate_dir=None,
    verbose=False,
):
    """
    Runs the full BhumiSetu pipeline on a single page image and returns
    the final merged record.

    Parameters:
        image_path: path to the page image (jpeg/png/etc).
        document_id: optional id; a random one is generated if omitted.
        lang: tesseract language string, default "hin+eng".
        use_llm: True/False to force the LLM stage on/off. If None
            (default), the LLM stage runs only when GEMINI_API_KEY is set.
        save_intermediate_dir: if given, each stage's raw JSON output is
            also written to this directory (ocr.json, classifier.json,
            label.json, table.json, llm.json, final.json) for debugging.
        verbose: print stage-by-stage progress.

    Returns:
        {
            "document_id": ...,
            "fields": { field: value_or_None, ... },          # flat, easy to consume
            "field_details": { field: {value, source, confidence,
                                        confidence_status, status}, ... },
            "summary": {...},          # from the merger stage
            "timings": {stage: seconds, ...}
        }
    """

    timings = {}

    def _log(msg):
        if verbose:
            print(msg)

    def _save(name, data):
        if save_intermediate_dir:
            os.makedirs(save_intermediate_dir, exist_ok=True)
            path = os.path.join(save_intermediate_dir, name)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    if use_llm is None:
        use_llm = _llm_available()

    # --------------------------------------------------------
    # 1. OCR
    # --------------------------------------------------------
    _log("[1/5] Running OCR...")
    t0 = time.time()
    ocr_result = ocr_stage.run_ocr(image_path, document_id=document_id, lang=lang)
    timings["ocr"] = round(time.time() - t0, 2)
    _save("1_ocr.json", ocr_result)

    document_id = ocr_result["document_id"]

    # --------------------------------------------------------
    # 2. Classifier / layout routing
    # --------------------------------------------------------
    _log("[2/5] Classifying regions...")
    t0 = time.time()
    classified = classifier_stage.classify_document(
        ocr_result,
        image_path,
        verbose=verbose,
    )
    timings["classifier"] = round(time.time() - t0, 2)
    _save("2_classifier.json", classified)

    # --------------------------------------------------------
    # 3. Extractors (label, table, LLM)
    # --------------------------------------------------------
    _log("[3/5] Extracting fields (label)...")
    t0 = time.time()
    label_result = label_stage.extract_labels(classified)
    timings["label_extractor"] = round(time.time() - t0, 2)
    _save("3_label.json", label_result)

    _log("[3/5] Extracting fields (table)...")
    t0 = time.time()
    table_result = table_stage.extract_tables(classified, verbose=verbose)
    timings["table_extractor"] = round(time.time() - t0, 2)
    _save("3_table.json", table_result)

    llm_result = []
    if use_llm:
        _log("[3/5] Extracting fields (LLM)...")
        t0 = time.time()
        try:
            llm_result = llm_stage.extract_llm(classified, verbose=verbose)
        except Exception as exc:
            _log(f"    LLM stage failed, continuing without it: {exc}")
            llm_result = []
        timings["llm_extractor"] = round(time.time() - t0, 2)
    else:
        _log("[3/5] Skipping LLM stage (GEMINI_API_KEY not set / use_llm=False).")
    _save("3_llm.json", llm_result)

    # --------------------------------------------------------
    # 4. Merge
    # --------------------------------------------------------
    _log("[4/5] Merging extraction results...")
    t0 = time.time()
    merged = merger_stage.merge_all(label_result, table_result, llm_result)
    timings["merger"] = round(time.time() - t0, 2)

    # --------------------------------------------------------
    # 5. Output in the flat FIELDS schema
    # --------------------------------------------------------
    _log("[5/5] Building final output...")

    fields = {field: merged["fields"][field]["value"] for field in FIELDS}

    field_details = {
        field: {
            "value": merged["fields"][field]["value"],
            "source": merged["fields"][field]["source"],
            "confidence": merged["fields"][field]["confidence"],
            "confidence_status": merged["fields"][field]["confidence_status"],
            "status": merged["fields"][field]["status"],
        }
        for field in FIELDS
    }

    result = {
        "document_id": document_id,
        "fields": fields,
        "field_details": field_details,
        "summary": merged["summary"],
        "timings": timings,
    }

    _save("4_final.json", result)

    return result


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full BhumiSetu extraction pipeline on an image")
    parser.add_argument("image_path")
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--lang", default="hin+eng")
    parser.add_argument("--no-llm", action="store_true", help="Force-disable the LLM stage")
    parser.add_argument("--save-intermediate", default=None, help="Directory to dump each stage's raw JSON")
    parser.add_argument("--output", default="final_land_record.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = run_pipeline(
        args.image_path,
        document_id=args.document_id,
        lang=args.lang,
        use_llm=(False if args.no_llm else None),
        save_intermediate_dir=args.save_intermediate,
        verbose=not args.quiet,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("FINAL LAND RECORD")
    print("=" * 70)

    for field in FIELDS:
        detail = result["field_details"][field]
        print(
            f"{field:22}: {detail['value']!s:30} "
            f"[{detail['source']}] {detail['confidence']:.1f}% {detail['confidence_status']}"
        )

    print("\nOverall confidence:", result["summary"]["overall_confidence"],
          result["summary"]["confidence_status"])
    print("Timings:", result["timings"])
    print("\nSaved:", args.output)
