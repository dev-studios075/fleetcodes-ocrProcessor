from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import re
import numpy as np
import os
import json
from datetime import datetime


VALID_STATE_CODES = {
'AP','AR','AS','BR','CG','CH','DD','DL','DN','GA',
'GJ','HR','HP','JH','JK','KA','KL','LA','LD','MH',
'ML','MN','MP','MZ','NL','OD','PB','PY','RJ','SK',
'TN','TR','TS','UK','UP','WB'
}


def load_models(config):

    yolo_model = YOLO(
        config["models"]["yolo_path"]
    )

    model_type = config["ocr"]["model"]


    if model_type == "mobile":

        print("\nOCR MODEL LOADING: PP-OCRv5_mobile\n")

        det_model = "PP-OCRv5_mobile_det"
        rec_model = "en_PP-OCRv5_mobile_rec"


    elif model_type == "server":

        print("\nOCR MODEL LOADING: PP-OCRv5_server\n")

        det_model = "PP-OCRv5_server_det"
        rec_model = "PP-OCRv5_server_rec"


    else:

        raise ValueError("config ocr.model must be mobile or server")


    ocr_model = PaddleOCR(

        text_detection_model_name = det_model,
        text_recognition_model_name = rec_model,

        use_doc_orientation_classify = False,
        use_doc_unwarping = False,
        use_textline_orientation = False
    )

    return yolo_model, ocr_model



# OCR PREPROCESS

def preprocess(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    gray = cv2.equalizeHist(gray)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def get_text_height(box):
    """
    box format from PaddleOCR:
    [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    top_y = box[0][1]
    bottom_y = box[2][1]

    return abs(bottom_y - top_y)


def clean_text(text):

    text = text.upper()

    # keep only letters and numbers
    text = re.sub(r'[^A-Z0-9]', '', text)

    return text


def correct_by_position(text):

    if len(text) < 8:
        return text

    text = list(text)

    for i, c in enumerate(text):

        # first 2 letters → state code
        if i < 2:

            if c in {'0','1','2','5','6','8'}:

                text[i] = {
                    '0':'O',
                    '1':'I',
                    '2':'Z',
                    '5':'S',
                    '6':'G',
                    '8':'B'
                }[c]


        # next 2 digits → RTO code
        elif i < 4:

            if c in {'O','I','Z','S','B','G'}:

                text[i] = {
                    'O':'0',
                    'I':'1',
                    'Z':'2',
                    'S':'5',
                    'B':'8',
                    'G':'6'
                }[c]


        # last 4 digits → vehicle number
        elif i >= len(text)-4:

            if c in {'O','I','Z','S','B','G'}:

                text[i] = {
                    'O':'0',
                    'I':'1',
                    'Z':'2',
                    'S':'5',
                    'B':'8',
                    'G':'6'
                }[c]


        # middle letters → series
        else:

            if c in {'0','1','2','5','6','8'}:

                text[i] = {
                    '0':'O',
                    '1':'I',
                    '2':'Z',
                    '5':'S',
                    '6':'G',
                    '8':'B'
                }[c]


    return "".join(text)


def is_valid_indian_plate(text, config):
    
    if len(text) < config["anpr"]["plate_filter"]["min_plate_length"]:
        return False

    state_code = text[:2]

    if state_code not in VALID_STATE_CODES:
        return False

    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{3,4}$'

    return bool(re.match(pattern, text))


def read_plate(crop_img, ocr_model, config):

    processed = preprocess(crop_img)

    result = ocr_model.predict(processed)  
     
    if not result:
        return "NOT_READABLE"

    words_data = []

    for page in result:

        texts = page["rec_texts"]
        boxes = page["dt_polys"]

        for text, box in zip(texts, boxes):
            
            if not text.strip():
                continue

            height = get_text_height(box)

            words_data.append({
                "text": text,
                "height": height,
                "box": box
            })


    if len(words_data) == 0:
        return "NOT_READABLE"


    # find tallest text (main plate characters)
    max_height = max(w["height"] for w in words_data)


    # keep only large text (remove IND etc)
    filtered_words = []

    ratio = config["anpr"]["plate_filter"]["min_text_height_ratio"]
    
    for w in words_data:
        
        if w["height"] > ratio * max_height:
            
            filtered_words.append(w)


    # sort TOP→BOTTOM then LEFT→RIGHT (important for 2-line plates)
    filtered_words = sorted(
        filtered_words,
        key=lambda x: (
            (x["box"][0][1] + x["box"][2][1]) / 2,   # sort by vertical center (line order)
            (x["box"][0][0] + x["box"][2][0]) / 2    # then sort left→right
        )
    )


    # combine
    combined_text = ""

    for w in filtered_words:

        combined_text += clean_text(w["text"])


    combined_text = correct_by_position(combined_text)

    return combined_text if combined_text else "NOT_READABLE"


def get_rejection_reason(text):

    if text == "NOT_READABLE":
        return "OCR_FAILED"

    if len(text) < 8:
        return "TEXT_TOO_SHORT"

    state_code = text[:2]

    if state_code not in VALID_STATE_CODES:
        return "INVALID_STATE_CODE"

    return "INVALID_FORMAT"

# MAIN ANPR FUNCTION
def run_anpr(config):

    img_path = config["input"]["anpr_image"]

    yolo_model, ocr_model = load_models(config)

    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    
    results_list = []

    results = yolo_model(
        img_path,
        conf=config["anpr"]["detection"]["conf_threshold"]
    )

    
    
    all_boxes = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        all_boxes.extend(boxes)


    # CASE 1 — no plate detected
    if len(all_boxes) == 0:

        results_list.append({
            "status": "rejected",
            "reason": "NO_PLATE_DETECTED",
            "message": "No license plate detected in image"
        })


    # CASE 2 — process detected plates
    for box in all_boxes:

        x1, y1, x2, y2 = map(int, box)

        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        plate_text = read_plate(
            crop,
            ocr_model,
            config
        )


        if is_valid_indian_plate(plate_text, config):

            print("VALID Plate:", plate_text)

            results_list.append({
                "status": "valid",
                "plate_text": plate_text,
                "bbox": [x1, y1, x2, y2]
            })

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                2
            )

            cv2.putText(
                img,
                plate_text,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )


        else:

            reason = get_rejection_reason(plate_text)

            print("Rejected OCR:", plate_text, "| reason:", reason)

            results_list.append({
                "status": "rejected",
                "raw_text": plate_text,
                "reason": reason,
                "bbox": [x1, y1, x2, y2]
            })
          
        
    # save json
    if config["anpr"]["output"]["save_json"]:

        os.makedirs("output/anpr", exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")

        img_name = os.path.basename(img_path).split(".")[0]

        base_json_path = config["anpr"]["output"]["json_path"]

        json_path = f"output/anpr/{img_name}_{timestamp}_anpr.json"

        with open(json_path, "w") as f:
            json.dump(results_list, f, indent=4)

        print(f"Saved ANPR json → {json_path}")
    
    # debug display
    if config["debug"]["show_image"]:

        display_img = img.copy()

        max_width = 1200
        max_height = 800

        h, w = display_img.shape[:2]

        scale = min(max_width/w, max_height/h)

        if scale < 1:
            display_img = cv2.resize(
                display_img,
                (int(w*scale), int(h*scale))
            )

        cv2.imshow("Plate Detection", display_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    # save debug image
    if config["debug"].get("save_image", False):

        base_path = config["debug"]["image_path"]

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")

        img_name = os.path.basename(img_path).split(".")[0]

        output_path = f"output/anpr/{img_name}_{timestamp}_anpr.jpg"

        cv2.imwrite(output_path, img)

        print(f"Saved debug image → {output_path}")