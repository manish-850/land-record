# syntax=docker/dockerfile:1

FROM python:3.11-slim

# --------------------------------------------------------------
# System dependencies:
#   - tesseract-ocr + hindi language pack (the OCR stage)
#   - libgl1 / libglib2.0-0 (needed by opencv-python at runtime)
# --------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-hin \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so Docker can cache this layer
# across deploys where only application code changed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY . .

# Render sets $PORT at runtime; default to 8000 for local testing.
ENV PORT=8000
EXPOSE 8000

# Tesseract is on PATH inside this image, so ocr_stage.py's default
# Windows paths are simply ignored (see the os.path.exists() checks).
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
