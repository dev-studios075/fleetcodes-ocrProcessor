from paddleocr import PaddleOCR
import cv2
import os
import re
import json
from datetime import datetime


def load_ocr(config):

    model_type = config["ocr"]["model"]

    if model_type == "mobile":

        print("\nDOC OCR MODEL LOADING: PP-OCRv5_mobile (EN)\n")

        det_model = "PP-OCRv5_mobile_det"
        rec_model = "en_PP-OCRv5_mobile_rec"


    elif model_type == "server":

        print("\nDOC OCR MODEL LOADING: PP-OCRv5_server\n")

        det_model = "PP-OCRv5_server_det"
        rec_model = "PP-OCRv5_server_rec"


    else:

        raise ValueError("config ocr.model must be mobile or server")


    ocr = PaddleOCR(

        text_detection_model_name = det_model,
        text_recognition_model_name = rec_model,

        use_doc_orientation_classify = True,
        use_doc_unwarping = True,
        use_textline_orientation = False
    )

    return ocr


# REGEX PATTERNS

patterns = {

    # Route 
    "route": r"\b([A-Z]{3}[0-9IO]{2}(?:[-\s][A-Z]{3}[0-9IO]{2}){0,2})\b",
    
    # STD
    "std": r"\bS[T7][D0]\s*[:\-]?\s*([0-9OIlSB]{1,2}:[0-9OIlSB]{1,2}|MARKET)",

    # ATD
    "atd": r"ATD\s*[:\-]?\s*([0-9]{2}/[0-9]{2}/[0-9]{4}\s*[0-9OIlSB]{1,2}:[0-9OIlSB]{1,2})",

    # PRINTDATE
    "print_date": r"\bPR[I1l]NT\s*D[A4]TE\s*[:\-]?\s*([0-9]{2}/[0-9]{2}/[0-9]{4}\s*[0-9OIlSB]{1,2}:[0-9OIlSB]{1,2})",

    # Pre-Pared At
    "prepared_at": r"PREPARED\s*AT[:\s]+([A-Z\s]{3,30}(?:\s+(?:HUB|OUTBOUND|HOB|HOR|HUS|OUTBO\w*))?[-\s]?\d{0,2})",
    
    # Vendor
    "vendor": r"VENDOR\s*[:\-]?\s*(.+?)(?=\s*(ETA|TRANSIT|VEHICLE|DRIVER|ATD|STD|ROUTE|PRINT)\b|$)",
    
    # ETA
    "eta": r"\bET[A4]\s*[:\-]?\s*([0-9OIlSB]{1,2}:[0-9OIlSB]{1,2})",
     
    # Transit Hrs 
    "transit_hrs": r"TRANSIT\s*HR[S5]\s*[:\-]?\s*([0-9OIlSB]{1,3}:[0-9OIlSB]{2})",
     
    # Vehicle Number
    "vehicle_no": r"VEHICLE\s*NO[:\s]+([A-Z]{2}[0-9O]{1,2}[A-Z0-9]{1,3}[0-9O]{3,4})",

    # Driver name
    "driver": r"DRIVER\s*[:\-]?\s*([A-Z\s\.]{3,40})?(?=\s*(VEHICLE|VEHICLENO|S\.?NO|MANIFEST|TRANSIT|ETA)\b|$)"
              
    }


def normalize_data(data):

   
    # ROUTE Normalization
    if data["route"]:

        # normalize separator
        data["route"] = re.sub(r"\s+", "-", data["route"])

        segments = data["route"].split("-")

        fixed_segments = []

        for seg in segments:

            seg = seg.upper()

            if len(seg) == 5:

                chars = list(seg)

                # first 3 MUST be letters
                for i in range(3):

                    if chars[i].isdigit():

                        # convert digit that looks like letter
                        chars[i] = (
                            chars[i]
                            .replace("0","O")
                            .replace("1","I")
                            .replace("5","S")
                            .replace("8","B")
                            .replace("2","Z")
                            .replace("6","G")
                        )

                # last 2 MUST be digits
                for i in [3,4]:

                    chars[i] = (
                        chars[i]
                        .replace("O","0")
                        .replace("I","1")
                        .replace("L","1")
                        .replace("S","5")
                        .replace("B","8")
                        .replace("Z","2")
                    )

                seg = "".join(chars)

            fixed_segments.append(seg)

        # keep max 3 segments
        data["route"] = "-".join(fixed_segments[:3])

        # remove duplicate dash
        data["route"] = re.sub(r"-{2,}", "-", data["route"])
        
        
    # VEHICLE normalization
    
    if data["vehicle_no"]:

        v = data["vehicle_no"].upper()

        chars = list(v)

        for i, c in enumerate(chars):

            # first 2 must be letters
            if i < 2:

                chars[i] = (
                    c.replace("0","O")
                    .replace("1","I")
                    .replace("2","Z")
                    .replace("5","S")
                    .replace("8","B")
                )

            # RTO digits
            elif i < 4:

                chars[i] = (
                    c.replace("O","0")
                    .replace("I","1")
                    .replace("Z","2")
                    .replace("S","5")
                    .replace("B","8")
                )

            # last 4 digits
            elif i >= len(chars)-4:

                chars[i] = (
                    c.replace("O","0")
                    .replace("I","1")
                    .replace("Z","2")
                    .replace("S","5")
                    .replace("B","8")
                )

            # middle letters
            else:

                chars[i] = (
                    c.replace("0","O")
                    .replace("1","I")
                    .replace("2","Z")
                    .replace("5","S")
                    .replace("8","B")
                )

        v = "".join(chars)

        match = re.search(
            r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$',
            v
        )

        data["vehicle_no"] = match.group(0) if match else None

    
    # ETA, STD, Transit Hrs & ATD Shared correction logic
    def fix_time(val):
        
        if not val:
            return val

        val = (
            val
            .replace("O","0")
            .replace("I","1")
            .replace("l","1")
            .replace("S","5")
            .replace("B","8")
        )

        h,m = val.split(":")

        return f"{h.zfill(2)}:{m.zfill(2)}"

    # ETA normalization
    if data["eta"]:
        data["eta"] = fix_time(data["eta"])
        
    # TRANSIT HRS normalization
    if data["transit_hrs"]:
        data["transit_hrs"] = fix_time(data["transit_hrs"])

    # STD normalization
    if data["std"]:

        v = data["std"].upper()

        # MARKET based departure (valid case)
        if "MARKET" in v:

            data["std"] = "MARKET"

        else:

            match = re.search(r'\d{1,2}:\d{1,2}', v)

            data["std"] = fix_time(match.group(0)) if match else None
            
    
    
    # ATD Normalization
    
    if data["atd"]:

        val = data["atd"]

        # fix missing space
        val = re.sub(
            r'(\d{4})(\d{1,2}:[0-9OIlSB]{1,2})',
            r'\1 \2',
            val
        )

        # fix OCR mistakes in time
        match = re.search(r'([0-9OIlSB]{1,2}:[0-9OIlSB]{1,2})', val)

        if match:

            fixed_time = fix_time(match.group(1))

            val = re.sub(
                r'[0-9OIlSB]{1,2}:[0-9OIlSB]{1,2}',
                fixed_time,
                val
            )

        data["atd"] = val


    # Print Date Normalization
    
    if data["print_date"]:

        val = data["print_date"]

        # fix missing space
        val = re.sub(
            r'(\d{4})(\d{1,2}:[0-9OIlSB]{1,2})',
            r'\1 \2',
            val
        )

        # fix OCR mistakes in time
        match = re.search(r'([0-9OIlSB]{1,2}:[0-9OIlSB]{1,2})', val)

        if match:

            fixed_time = fix_time(match.group(1))

            val = re.sub(
                r'[0-9OIlSB]{1,2}:[0-9OIlSB]{1,2}',
                fixed_time,
                val
            )

        data["print_date"] = val
            
    
    # Pre-Pared at (Normalization)

    if data["prepared_at"]:

        val = data["prepared_at"].upper()
        
        # remove accidental next-field words
        val = re.split(
            r'\b(ROUTE|STD|ATD|VENDOR|ETA|TRANSIT|VEHICLE|PRINT|DRIVER)\b',
            val
        )[0]

        # fix truncated OUTBOUND words
        val = re.sub(
            r'OUTBO\w*',
            'OUTBOUND',
            val
        )

        # fix HUB OCR mistakes
        val = re.sub(
            r'\b(HOB|HOR|HUS|HUE|HUBB)\b',
            'HUB',
            val
        )

        # remove number suffix like -11 or -1
        val = re.sub(
            r'[-\s]*\d{1,2}$',
            '',
            val
        )

        # remove trailing dash
        val = re.sub(
            r'-+$',
            '',
            val
        )

        # remove very large OCR garbage numbers
        val = re.sub(
            r'\s*\d{5,}',
            '',
            val
        )

        # clean spacing
        val = re.sub(
            r'\s{2,}',
            ' ',
            val
        ).strip()

        data["prepared_at"] = val
        
    
    # DRIVER normalization
     
    if data["driver"]:

        d = data["driver"].strip()

        d = re.sub(r'\s{2,}', ' ', d)

        d = d.strip(" -:.")

        if len(d) < 3:
            d = None

        data["driver"] = d
        
        

    # Vendor Normalization
    
    if data["vendor"]:

        v = data["vendor"]

        # cut text belonging to next fields
        v = re.split(
            r'\b(ETA|ATD|STA|TRANSIT|DATE|DRIVER|VEHICLE)\b',
            v
        )[0]

        # remove trailing code after last dash like "11 TC", "18 TC", "I1 T"
        v = re.sub(
            # r'\s*-\s*[0-9I]{1,3}\s*[A-Z]{0,3}$',
            r'\s*-\s*[^-]*$',
            '',
            v
        )

        # remove trailing number
        v = re.sub(
            r'\b\d{1,3}\b$',
            '',
            v
        )
        
        # remove OCR unicode junk like 新中
        v = re.sub(
            r'[^\x00-\x7F]+',
            '',
            v
        )

        # normalize spaces
        v = re.sub(r'\s{2,}', ' ', v)

        # remove trailing dash
        v = v.strip(" -")

        data["vendor"] = v
        
    return data
        
        
def extract_sheet_id_from_prepared_at(ocr_result, img):

    height, width = img.shape[:2]

    prepared_y = None


    for page in ocr_result:

        texts = page["rec_texts"]
        boxes = page["dt_polys"]

        for text, box in zip(texts, boxes):

            if "PREPARED" in text.upper():

                prepared_y = sum(p[1] for p in box) / 4


    if prepared_y is None:
        return None


    candidates = []


    for page in ocr_result:

        texts = page["rec_texts"]
        boxes = page["dt_polys"]

        for text, box in zip(texts, boxes):

            text = text.strip()

            if not re.fullmatch(r"\d{10}", text):
                continue


            x_center = sum(p[0] for p in box) / 4
            y_center = sum(p[1] for p in box) / 4


            if x_center < width * 0.5:
                continue

            if y_center < prepared_y:
                continue

            if y_center > height * 0.55:
                continue


            candidates.append((y_center, text))


    if candidates:

        candidates.sort()

        return candidates[0][1]


    return None

        
# OCR FUNCTION

def extract_header(image_path, ocr):

    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    result = ocr.predict(img)
    
    if not result:
     return {key: None for key in patterns}

    full_text = " ".join(

        text

        for page in result
        for text in page["rec_texts"]

        if text.strip()
    )

    data = {}

    for key, pattern in patterns.items():

        match = re.search(pattern, full_text)

        data[key] = match.group(1).strip() if (match and match.group(1)) else None
    
    data["sheet_id"] = extract_sheet_id_from_prepared_at(result, img)

    return data


# MAIN DOC OCR FUNCTION

def run_doc_ocr(config):

    image_path = config["input"]["doc_image"]

    ocr = load_ocr(config)

    extracted = extract_header(image_path, ocr)

    extracted = normalize_data(extracted)


    print("\nDOC OCR RESULT:\n")

    print(json.dumps(extracted, indent=4))


    # save json
    if config["doc_ocr"]["output"]["save_json"]:

        os.makedirs("output/doc", exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")

        img_name = os.path.basename(image_path).split(".")[0]

        output_path = f"output/doc/{img_name}_{timestamp}_doc.json"

        with open(output_path, "w") as f:
            json.dump(extracted, f, indent=4)

        print(f"Saved DOC json → {output_path}")
        
            
    # save debug image
    if config["debug"].get("save_image", False):

        base_path = config["debug"]["image_path"]

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")

        img_name = os.path.basename(image_path).split(".")[0]

        output_path = f"output/doc/{img_name}_{timestamp}_doc.jpg"

        img = cv2.imread(image_path)

        cv2.imwrite(output_path, img)

        print(f"Saved debug image → {output_path}")