# Land Record

Digitizes scanned Hindi/English land-record documents (khatiyan, khesra,
lagaan receipts, etc.) into structured, searchable data — combining OCR,
layout detection, and LLM-assisted extraction.

**Live demo:** [land-record-gold.vercel.app](https://land-record-gold.vercel.app)

>  **Note to maintainer:** this README was drafted from the repo's folder
> structure (`client`, `server`, `ml_server`, `ml_server2`) since I didn't
> have direct access to read file contents inside each service. Sections
> marked with  are placeholders — please confirm/edit the exact framework,
> ports, and env vars used in `client`, `server`, and `ml_server2` before
> publishing. The `ml_server` section describing the OCR/extraction pipeline
> reflects what we built together and should be accurate.

---

## What it does

A scanned page of a land record goes in; a structured JSON record comes out
with fields like district, village, khata number, khesra number, area,
lagaan (land revenue), boundaries, and more — each with a confidence score
so you know how much to trust it.

```
scanned page image
      │
      ▼
 OCR (Tesseract)              — reads Hindi + English text off the page
      │
      ▼
 Layout classification (DETR) — figures out which part of the page is a
      │                         table, a label block, or free-text prose
      ▼
 ┌─────────────┬─────────────┬─────────────┐
 │   Label      │   Table     │    LLM      │  — three specialized
 │  extractor   │  extractor  │  extractor  │    extractors, one per
 └─────────────┴─────────────┴─────────────┘    region type
      │
      ▼
 Merge                        — reconciles the three extractors into
      │                         one final record, flags conflicts
      ▼
 Structured JSON record
```

## Repository structure

```
land-record/
├── client/            frontend — <framework: React / Next.js?>
├── server/            backend API — <framework: Node/Express?>
├── ml_server/        Python OCR + extraction pipeline (see below)
├── ml_server2/        <purpose — second ML service?>
└── .gitignore
```

 *Maintainer: replace the above with the actual purpose of each folder —
e.g. is `ml_server2` a newer version of `ml_server`, a separate model, or
something else entirely?*

## The extraction pipeline (`ml_server`)

```
ml_server/
├── stages/
│   ├── ocr_stage.py          OCR (Tesseract) — reads the page image
│   ├── classifier_stage.py   DETR layout model + table/label/paragraph routing
│   ├── label_stage.py        key:value ("district: सिवान") extraction
│   ├── table_stage.py        grid/table extraction (khata, khesra, area, lagaan)
│   ├── llm_stage.py          Gemini-based extraction for free-text paragraphs
│   ├── merger_stage.py       merges label + table + LLM results into one record
│   └── confidence_utils.py   shared confidence-scoring helpers
├── pipeline.py       orchestrates all 5 stages end-to-end
├── app.py            FastAPI wrapper exposing the pipeline as an HTTP API
└── requirements.txt
```

### Output schema

Every extraction returns exactly these 21 fields (`null` if not found),
each with a `confidence` score, a `confidence_status`
(`HIGH`/`MEDIUM`/`LOW`/`VERY_LOW`), and the extractor `source` it came from
(`label` / `table` / `llm`):

```python
[
    "district", "circle", "halka", "village", "thana_no",
    "khata_no", "khesra_no", "raiyat_name", "father_husband_name",
    "address", "land_type", "area", "area_unit", "possession",
    "north_boundary", "south_boundary", "east_boundary",
    "west_boundary", "lagaan", "mutation_details", "remarks"
]
```

## Getting started

### Prerequisites

- Python 3.11+ (for `ml_server`)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed
  and on your PATH (or pointed to via `TESSERACT_CMD`)
- A [Gemini API key](https://aistudio.google.com/apikey) (optional — only
  needed for the LLM extraction stage; the pipeline runs fine without it
  using just the label + table extractors)
-  Node.js `<version>` (for `client` / `server`)

### 1. Clone the repo

```bash
git clone https://github.com/manish-850/land-record.git
cd land-record
```

### 2. Set up the ML server

```bash
cd ml_server
python -m venv venv

# Windows
venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your Gemini key (if using the LLM
stage):

```bash
cp .env.example .env
```

Run the API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Or run the pipeline directly on one image from the command line:

```bash
python pipeline.py path/to/page.jpg --output result.json
```

### 3.  Set up the server

```bash
cd server
npm install
npm run dev
```

### 4.  Set up the client

```bash
cd client
npm install
npm run dev
```

## API reference (`ml_server`)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/fields` | GET | The 21-field output schema |
| `/extract` | POST | Full pipeline — upload an image, get the final merged record |
| `/extract/ocr` | POST | OCR only |
| `/extract/classify` | POST | OCR + layout classification only |
| `/extract/label` | POST | OCR + classifier + label extractor |
| `/extract/table` | POST | OCR + classifier + table extractor |
| `/extract/llm` | POST | OCR + classifier + LLM extractor |

Example:

```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@page.jpg" \
  -F "lang=hin+eng"
```

 *Maintainer: add `server`'s API routes here if the client talks to
`server` rather than directly to `ml_server`.*

## Environment variables

| Variable | Used by | Required | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | ml_server | Optional | Enables the LLM extraction stage |
| `GEMINI_MODEL` | ml_server | Optional | Defaults to `gemini-3.6-flash` |
| `TESSERACT_CMD` | ml_server | Optional (Windows) | Path to `tesseract.exe` if not on PATH |
| `TESSDATA_PREFIX` | ml_server | Optional (Windows) | Path to Tesseract's `tessdata` folder |
|  | server | | |
|  | client | | |

## Deployment

The client is deployed on [Vercel](https://land-record-gold.vercel.app).

 *Maintainer: document where `server`, `ml_server`, and `ml_server2` are
hosted (Render/Railway/a VM/Docker?), and any build steps specific to
deployment (e.g. baking Tesseract into a Docker image, since it's a system
binary that `pip` alone won't install).*

## Known limitations

- The classifier currently processes one page per call; multi-page PDFs
  need to be split first.
- LLM-stage confidence scores are an engineering/evidence heuristic (OCR
  quality + field presence + JSON schema validity) — Gemini doesn't expose
  a calibrated per-field probability.
- Tesseract must be installed separately on any machine running
  `ml_server`; it isn't bundled by `pip install`.

## Contributing

 *Add contribution guidelines here (branch naming, PR process, etc.) if
this is intended to accept outside contributions.*

## License

*Add a license (MIT, Apache-2.0, etc.) — currently unspecified.*
