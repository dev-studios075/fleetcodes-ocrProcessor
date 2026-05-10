### ANPR + Document OCR Pipeline

A modular computer vision pipeline that performs:

1. Automatic Number Plate Recognition (ANPR)           (FEED BACK iteration in Progress)
2. Logistics Document OCR (header field extraction)    (IN PROGRESS - Under Development)


--------------------------------------------------

### FEATURES

ANPR (Vehicle Plate Recognition)

- YOLO26 nano based license plate detection
- PaddleOCR V5 based text recognition
- Supports both:
  - PP-OCRv5_mobile (fast)
  - PP-OCRv5_server (high accuracy)
- Removes small unwanted text (IND, logos, etc.)
- Validates Indian license plate format
- Automatic OCR correction for common mistakes:
  - O ↔ 0
  - B ↔ 8
  - S ↔ 5
  - Z ↔ 2
- Handles single-line and double-line plates
- Debug image generation
- Structured JSON output

Example output:

```json
[
  {
    "status": "valid",
    "plate_text": "GJ17XX5852",
    "bbox": [937, 2206, 1538, 2580]
  }
]

```
--------------------------------------------------

DOCUMENT OCR (Logistics Header Extraction)

Extracts structured fields from logistics documents:

- Route
- STD (Scheduled Time)
- ATD (Actual Time of Departure)
- Print Date
- Prepared At (Location)
- Vendor
- ETA
- Transit Hours
- Vehicle Number
- Driver Name
- Sheet ID

Includes strong normalization logic for OCR mistakes.

Example output:
```json
{
    "route": "BL011-AML11",
    "std": "5:00",
    "atd": "12/02/2026 16:30",
    "vendor": "ABC LOGISTICS LTD",
    "vehicle_no": "GJ17XX5852"
}
```
--------------------------------------------------

PROJECT STRUCTURE

```text
ANPR_DOC_System
│
├── anpr.py              -> ANPR pipeline
├── doc_ocr.py           -> Document OCR pipeline
├── main.py              -> Main controller
├── config.yaml          -> Configuration file
├── requirements.txt     -> Dependencies list
├── README.md
├── .gitignore
│
├── models/
│   └── yolo26n.pt       -> trained YOLO model
│
├── Test_imgs/
│   └── sample images
│
└── output/
    └── json results saved here
```

--------------------------------------------------

### ENV SETUP


1. Create virtual environment

```bash
python -m venv anprdoc_env
```

activate environment

Windows:
```bash
anprdoc_env\Scripts\activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

--------------------------------------------------

USAGE

All behaviour is controlled using:

config.yaml

--------------------------------------------------

Select pipeline mode

mode: anpr      -> runs number plate recognition only
mode: doc       -> runs document OCR only
mode: both      -> runs both pipelines

--------------------------------------------------

Run pipeline

```bash
python main.py
```

--------------------------------------------------

### CONFIGURATION

Example config.yaml:

```yaml
mode: anpr

input:
  anpr_image: Test_imgs/plate1.jpg
  doc_image: Test_imgs/doc1.jpg

models:
  yolo_path: models/yolo26n.pt


OCR model options:

ocr:
  model: mobile     # fast inference
  model: server     # higher accuracy

```
--------------------------------------------------

### OUTPUT
```text
Results are saved inside:

output/

ANPR result:

output/anpr_result.json

Document OCR result:

output/doc_result.json
```
--------------------------------------------------

### TECH STACK
```text
TOTAL ENV - ~2.5 GB

YOLO (Ultralytics v8.4.33)
custom YOLO26 Nano license plate detection model          ~6 MB

PaddleOCR v3.4.1  & PaddlePaddle v3.2.0
text detection + recognition (PP-OCRv5)

  OCR Models Used:

  PP-OCRv5_mobile (fast)
    PP-OCRv5_mobile_det        text detection             ~4.47 MB
    en_PP-OCRv5_mobile_rec     text recognition           ~7.41 MB

  PP-OCRv5_server (high accuracy)
    PP-OCRv5_server_det        text detection             ~83.8 MB
    PP-OCRv5_server_rec        text recognition           ~80.5 MB

PyTorch v2.5.1
deep learning backend for YOLO

OpenCV v4.9.0.80
image preprocessing & visualization

NumPy v1.26.4
matrix operations

PyYAML v6.0.2
config-driven pipeline
```
--------------------------------------------------

