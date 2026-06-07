import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from dotenv import load_dotenv
load_dotenv()  # loads AWS_* and other vars from .env

import uuid
import tempfile
import urllib.request
import urllib.parse
from contextlib import asynccontextmanager

import yaml
import cv2
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from anpr import load_models as load_anpr_models, read_plate, is_valid_indian_plate, get_rejection_reason
from doc_ocr import load_ocr, extract_header, normalize_data


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

def load_config(path: str = "config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# GLOBAL MODEL STORE  (loaded once at startup)
# ─────────────────────────────────────────────

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML models once when the server starts."""
    print("\n[SERVER] Loading models …\n")
    cfg = load_config()
    yolo_model, ocr_model = load_anpr_models(cfg)
    doc_ocr_model = load_ocr(cfg)

    models["config"] = cfg
    models["yolo"] = yolo_model
    models["anpr_ocr"] = ocr_model
    models["doc_ocr"] = doc_ocr_model

    print("\n[SERVER] Models ready. Server is up.\n")
    yield
    models.clear()


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="FleetCodes OCR Processor",
    description="ANPR & Document OCR via S3 image URL",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────

VALID_TYPES = {"anpr", "doc", "both"}


class ProcessRequest(BaseModel):
    s3_url: str
    type: str  # anpr | doc | both

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        v = v.lower().strip()
        if v not in VALID_TYPES:
            raise ValueError(f"type must be one of: {', '.join(VALID_TYPES)}")
        return v

    @field_validator("s3_url")
    @classmethod
    def validate_url(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("s3_url must not be empty")
        return v


class ProcessLocalRequest(BaseModel):
    local_path: str
    type: str  # anpr | doc | both

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        v = v.lower().strip()
        if v not in VALID_TYPES:
            raise ValueError(f"type must be one of: {', '.join(VALID_TYPES)}")
        return v

    @field_validator("local_path")
    @classmethod
    def validate_local_path(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("local_path must not be empty")
        if not os.path.isfile(v):
            raise ValueError(f"File not found: {v}")
        return v


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def download_image(url: str) -> str:
    """
    Download an image from a URL to a temporary file.
    Supports:
      - s3://bucket/key  → downloaded via boto3
      - http/https       → downloaded via urllib
    Returns the local file path.
    """
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        suffix = os.path.splitext(key)[-1] or ".jpg"
        tmp_dir = tempfile.mkdtemp()
        local_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}{suffix}")
        try:
            s3 = boto3.client("s3")
            s3.download_file(bucket, key, local_path)
        except (BotoCoreError, ClientError) as e:
            raise HTTPException(status_code=400, detail=f"Failed to download from S3: {e}")
        return local_path

    else:
        suffix = os.path.splitext(parsed.path)[-1] or ".jpg"
        tmp_dir = tempfile.mkdtemp()
        local_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}{suffix}")
        try:
            urllib.request.urlretrieve(url, local_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")
        return local_path


def run_anpr_on_image(image_path: str) -> list:
    """Run YOLO plate detection + OCR on a local image file."""
    config = models["config"]
    yolo_model = models["yolo"]
    ocr_model = models["anpr_ocr"]

    img = cv2.imread(image_path)
    if img is None:
        raise HTTPException(status_code=422, detail="Downloaded file is not a valid image.")

    results = yolo_model(
        image_path,
        conf=config["anpr"]["detection"]["conf_threshold"]
    )

    all_boxes = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        all_boxes.extend(boxes)

    results_list = []

    if len(all_boxes) == 0:
        results_list.append({
            "status": "rejected",
            "reason": "NO_PLATE_DETECTED",
            "message": "No license plate detected in image"
        })
        return results_list

    for box in all_boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        plate_text = read_plate(crop, ocr_model, config)

        if is_valid_indian_plate(plate_text, config):
            results_list.append({
                "status": "valid",
                "plate_text": plate_text,
                "bbox": [x1, y1, x2, y2]
            })
        else:
            reason = get_rejection_reason(plate_text)
            results_list.append({
                "status": "rejected",
                "raw_text": plate_text,
                "reason": reason,
                "bbox": [x1, y1, x2, y2]
            })

    return results_list


def run_doc_on_image(image_path: str) -> dict:
    """Run document OCR on a local image file."""
    ocr_model = models["doc_ocr"]
    extracted = extract_header(image_path, ocr_model)
    extracted = normalize_data(extracted)
    return extracted


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bool(models)}


@app.post("/process")
def process(req: ProcessRequest):
    """
    Download image from s3_url and process based on type:
      - anpr  → license plate detection & OCR
      - doc   → document header OCR
      - both  → both pipelines
    """
    local_path = download_image(req.s3_url)

    try:
        response = {"s3_url": req.s3_url, "type": req.type}

        if req.type in ("anpr", "both"):
            response["anpr"] = run_anpr_on_image(local_path)

        if req.type in ("doc", "both"):
            response["doc"] = run_doc_on_image(local_path)

        return JSONResponse(content=response)

    finally:
        # clean up temp file
        try:
            os.remove(local_path)
            os.rmdir(os.path.dirname(local_path))
        except Exception:
            pass


@app.post("/process-local")
def process_local(req: ProcessLocalRequest):
    """
    Process a locally accessible image file based on type:
      - anpr  → license plate detection & OCR
      - doc   → document header OCR
      - both  → both pipelines
    """
    response = {"local_path": req.local_path, "type": req.type}

    if req.type in ("anpr", "both"):
        response["anpr"] = run_anpr_on_image(req.local_path)

    if req.type in ("doc", "both"):
        response["doc"] = run_doc_on_image(req.local_path)

    return JSONResponse(content=response)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
