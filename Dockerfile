# ─────────────────────────────────────────────────────────────
# Stage 1: Builder — install all Python dependencies
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# System libs needed to build OpenCV, PaddlePaddle, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

COPY requirements.txt .

# Install into an isolated prefix so we can copy cleanly
RUN pip install --no-cache-dir --prefix=/install/deps -r requirements.txt


# ─────────────────────────────────────────────────────────────
# Stage 2: Runtime image
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

LABEL maintainer="FleetCodes"
LABEL description="FleetCodes OCR Processor — ANPR & Document OCR API"

# Runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install/deps /usr/local

WORKDIR /app

# Copy application source
COPY anpr.py .
COPY doc_ocr.py .
COPY server.py .
COPY config.yaml .

# Copy YOLO model
COPY models/ models/

# Output directory (JSON results written here)
RUN mkdir -p output/anpr output/doc

# PaddleOCR model cache will be stored here so it persists across restarts
# when mounted as a volume: -v paddlex_cache:/root/.paddlex
ENV PADDLEX_HOME=/root/.paddlex
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# AWS credentials — injected at runtime via Dokploy env vars, NOT baked in
# ENV AWS_ACCESS_KEY_ID=...
# ENV AWS_SECRET_ACCESS_KEY=...
# ENV AWS_DEFAULT_REGION=ap-south-1

EXPOSE 8000

# Health check for Dokploy
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "server.py"]
