# BhumiSetu extraction pipeline

Turns a scanned Hindi/English land-record page image into a structured
JSON record with 21 fields, each carrying a confidence score and the
source ("label", "table", or "llm") it came from.

```
image
  │
  ▼
1. OCR (Tesseract)                 stages/ocr_stage.py
  │
  ▼
2. Layout classification (DETR)    stages/classifier_stage.py
  │  routes each region into: table / label / paragraph / header / etc.
  │
  ├──▶ 3a. Label extractor  ─┐     stages/label_stage.py
  ├──▶ 3b. Table extractor  ─┼──▶ 4. Merge   stages/merger_stage.py
  └──▶ 3c. LLM extractor    ─┘
  │
  ▼
5. Final record (21 fields, with confidence + source per field)
```

## Files

```
bhumisetu/
├── stages/
│   ├── ocr_stage.py          OCR (Tesseract)
│   ├── classifier_stage.py   DETR layout model + table/label/paragraph routing
│   ├── label_stage.py        key:value label extraction
│   ├── table_stage.py        table/grid extraction (khata, khesra, area, lagaan, ...)
│   ├── llm_stage.py          Gemini-based extraction for free-text paragraphs
│   ├── merger_stage.py       merges the three extractors into one record
│   └── confidence_utils.py   shared confidence-scoring helpers
├── pipeline.py                orchestrates all 5 stages end-to-end
├── app.py                     FastAPI wrapper around pipeline.py
├── requirements.txt
└── README.md
```

This is a refactor of the original standalone scripts
(`OCR_Reader.py`, `Classifier_of_OCR.py`, `label_extractor_with_confidence.py`,
`table_extractor_with_confidence_exact.py`, `LLM_extractor_with_confidence.py`,
`extractiokn_mergeer_with_confidence.py`) into importable functions that pass
data in memory instead of hardcoded Windows file paths. The extraction logic
itself (regex label matching, table row/column geometry, confidence
weighting, source-priority merge rules) is unchanged.

## Output schema

Every result (from `pipeline.run_pipeline()` or `POST /extract`) has a
`fields` dict with exactly these 21 keys (value is `null` if not found):

```python
[
    "district", "circle", "halka", "village", "thana_no",
    "khata_no", "khesra_no", "raiyat_name", "father_husband_name",
    "address", "land_type", "area", "area_unit", "possession",
    "north_boundary", "south_boundary", "east_boundary",
    "west_boundary", "lagaan", "mutation_details", "remarks"
]
```

`field_details` gives the same fields with `value`, `source`
(`label`/`table`/`llm`), `confidence` (0-100), `confidence_status`
(`HIGH`/`MEDIUM`/`LOW`/`VERY_LOW`), and `status` (`AGREED`/`CONFLICT`/`NOT_FOUND`).

## Setup

```bash
pip install -r requirements.txt
```

**Tesseract** must be installed separately (it's a system binary, not a
pip package):
- Windows: install from https://github.com/UB-Mannheim/tesseract/wiki,
  then either add it to PATH or set:
  ```
  set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
  set TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata
  ```
- Linux: `apt-get install tesseract-ocr tesseract-ocr-hin`

**Gemini (optional, for the LLM stage on free-text paragraphs):**
```bash
export GEMINI_API_KEY=your_key_here
```
If this isn't set, the pipeline automatically skips the LLM stage and
relies on the label + table extractors only.

## Running as a script

```bash
python pipeline.py path/to/page.jpg --output result.json
```

Useful flags:
- `--no-llm` — force-disable the LLM stage even if `GEMINI_API_KEY` is set
- `--save-intermediate ./debug` — dump each stage's raw JSON to `./debug/`
  (`1_ocr.json`, `2_classifier.json`, `3_label.json`, `3_table.json`,
  `3_llm.json`, `4_final.json`) for debugging a specific stage
- `--lang hin+eng` — Tesseract language string (default)

You can also run any single stage standalone, e.g.:
```bash
python -m stages.ocr_stage page.jpg --output ocr.json
python -m stages.classifier_stage ocr.json page.jpg --output classified.json
python -m stages.label_stage classified.json --output labels.json
python -m stages.table_stage classified.json --output tables.json
python -m stages.llm_stage classified.json --output llm.json
python -m stages.merger_stage labels.json tables.json llm.json --output final.json
```

## Running as an API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then:

```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@page.jpg" \
  -F "lang=hin+eng"
```

Endpoints:
| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness check |
| `GET /fields` | the 21-field schema |
| `POST /extract` | full pipeline → final merged record |
| `POST /extract/ocr` | OCR only |
| `POST /extract/classify` | OCR + classifier only |
| `POST /extract/label` | OCR + classifier + label extractor |
| `POST /extract/table` | OCR + classifier + table extractor |
| `POST /extract/llm` | OCR + classifier + LLM extractor |

The DETR layout model is loaded once at server startup (not per request)
and reused; requests are serialized with a lock around model inference
for safety. For higher throughput, run multiple `uvicorn` worker
processes rather than removing the lock.

## Known limitations carried over from the original scripts

- The classifier stage currently processes a **single page** per call
  (`ocr_result["pages"][0]`); multi-page documents need one call per page.
- The LLM stage's confidence score is an engineering/evidence heuristic
  (OCR quality + field presence + JSON schema validity), not a calibrated
  model probability — Gemini doesn't expose one.
- `table_extractor.py` (the non-confidence version) was not wired into
  the pipeline; `table_extractor_with_confidence_exact.py` is used since
  it's the more complete implementation.
