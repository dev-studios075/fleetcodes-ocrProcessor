# Quick Guide

## Config (config.yaml)

Set pipeline mode:
mode: doc             # anpr | doc | both


Set input image paths: From TEST_img folder
input:
  anpr_image: Test_imgs/Trucks2.jpg
  doc_image: Test_imgs/3.jpeg

Select OCR model:
ocr:
  model: server        # mobile (Small) | server(Large) (PaddleOCR v5)

---

## Run System

python main.py

---

## Output

Results are saved automatically:

output/anpr/   → number plate results
output/doc/    → document OCR results

Each run creates new files (no overwrite).