"""
BhumiSetu extraction API.

Run with:

    uvicorn app:app --host 0.0.0.0 --port 8000

Endpoints:

    GET  /health
        Basic liveness check.

    POST /extract
        multipart/form-data with a single file field named "file"
        (the scanned land-record page image).
        Optional form fields:
            lang       (default "hin+eng")
            use_llm    ("true"/"false" - default: on iff GEMINI_API_KEY is set)
        Returns the final merged land-record JSON (same shape as
        pipeline.run_pipeline()).

    POST /extract/ocr, /extract/classify, /extract/label,
    /extract/table, /extract/llm
        Same file upload, but returns only that single stage's raw
        output. Handy for debugging one stage of the pipeline without
        re-running the whole thing.

The DETR layout model is loaded once, lazily, on first use (or eagerly
at startup, see the startup event below) and reused across requests.
Because the layout model is not guaranteed to be safe under concurrent
inference calls, requests are serialized with a lock; this is a simple,
correct default for a small/medium-traffic internal tool. If you need
higher throughput, run multiple worker processes instead (each gets
its own model copy) rather than removing the lock.
"""

import os
import shutil
import tempfile
import threading
import traceback
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pipeline import run_pipeline, FIELDS
from stages import ocr_stage, classifier_stage, label_stage, table_stage, llm_stage

app = FastAPI(
    title="BhumiSetu Extraction API",
    description="OCR -> layout classification -> label/table/LLM extraction -> merge, for Hindi land records.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline_lock = threading.Lock()

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@app.on_event("startup")
def _warm_up_model():
    
    """
    Loads the DETR layout model at startup instead of on the first
    request, so the first real request isn't slow.
    """
    try:
        classifier_stage.load_model()
    except Exception as exc:
        # Don't crash the server if the model can't be downloaded/loaded
        # in this environment (e.g. no network) -- it will just fail
        # (with a clear error) on first actual use instead.
        print(f"[startup] Warning: could not pre-load layout model: {exc}")


def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[1].lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    tmp_dir = tempfile.mkdtemp(prefix="bhumisetu_")
    tmp_path = os.path.join(tmp_dir, f"page{suffix}")

    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)

    return tmp_path


def _cleanup(path: str):
    try:
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)
    except Exception:
        pass


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    lang: str = Form("hin+eng"),
    use_llm: str = Form(None),
):
    """
    Runs the full pipeline on the uploaded page image and returns the
    final merged land-record JSON:

        {
            "document_id": ...,
            "fields": {field: value, ...},
            "field_details": {field: {value, source, confidence,
                                       confidence_status, status}, ...},
            "summary": {...},
            "timings": {...}
        }
    """

    tmp_path = _save_upload_to_temp(file)

    use_llm_flag = None
    if use_llm is not None:
        use_llm_flag = use_llm.strip().lower() in ("1", "true", "yes", "on")

    try:
        with _pipeline_lock:
            result = run_pipeline(
                tmp_path,
                document_id=uuid.uuid4().hex[:12],
                lang=lang,
                use_llm=use_llm_flag,
                verbose=False,
            )
        return JSONResponse(content=result)

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}")

    finally:
        _cleanup(tmp_path)


def _single_stage_response(stage_name, fn):
    async def handler(file: UploadFile = File(...), lang: str = Form("hin+eng")):
        tmp_path = _save_upload_to_temp(file)
        try:
            with _pipeline_lock:
                data = fn(tmp_path, lang)
            return JSONResponse(content=data)
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"{stage_name} stage failed: {exc}")
        finally:
            _cleanup(tmp_path)

    return handler


@app.post("/extract/ocr")
async def extract_ocr(file: UploadFile = File(...), lang: str = Form("hin+eng")):
    tmp_path = _save_upload_to_temp(file)
    try:
        return JSONResponse(content=ocr_stage.run_ocr(tmp_path, lang=lang))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OCR stage failed: {exc}")
    finally:
        _cleanup(tmp_path)


@app.post("/extract/classify")
async def extract_classify(file: UploadFile = File(...), lang: str = Form("hin+eng")):
    tmp_path = _save_upload_to_temp(file)
    try:
        with _pipeline_lock:
            ocr_result = ocr_stage.run_ocr(tmp_path, lang=lang)
            classified = classifier_stage.classify_document(ocr_result, tmp_path)
        return JSONResponse(content=classified)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Classifier stage failed: {exc}")
    finally:
        _cleanup(tmp_path)


@app.post("/extract/label")
async def extract_label(file: UploadFile = File(...), lang: str = Form("hin+eng")):
    tmp_path = _save_upload_to_temp(file)
    try:
        with _pipeline_lock:
            ocr_result = ocr_stage.run_ocr(tmp_path, lang=lang)
            classified = classifier_stage.classify_document(ocr_result, tmp_path)
            labels = label_stage.extract_labels(classified)
        return JSONResponse(content=labels)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Label stage failed: {exc}")
    finally:
        _cleanup(tmp_path)


@app.post("/extract/table")
async def extract_table(file: UploadFile = File(...), lang: str = Form("hin+eng")):
    tmp_path = _save_upload_to_temp(file)
    try:
        with _pipeline_lock:
            ocr_result = ocr_stage.run_ocr(tmp_path, lang=lang)
            classified = classifier_stage.classify_document(ocr_result, tmp_path)
            tables = table_stage.extract_tables(classified)
        return JSONResponse(content=tables)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Table stage failed: {exc}")
    finally:
        _cleanup(tmp_path)


@app.post("/extract/llm")
async def extract_llm_endpoint(file: UploadFile = File(...), lang: str = Form("hin+eng")):
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY is not configured on the server.")

    tmp_path = _save_upload_to_temp(file)
    try:
        with _pipeline_lock:
            ocr_result = ocr_stage.run_ocr(tmp_path, lang=lang)
            classified = classifier_stage.classify_document(ocr_result, tmp_path)
            llm_results = llm_stage.extract_llm(classified)
        return JSONResponse(content=llm_results)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"LLM stage failed: {exc}")
    finally:
        _cleanup(tmp_path)


@app.get("/fields")
def get_fields():
    """Returns the canonical output field schema."""
    return {"fields": FIELDS}
