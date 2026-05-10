import os
import json
import yaml
import cv2

from anpr import (
    load_models,
    read_plate,
    is_valid_indian_plate,
    get_rejection_reason
)

# ======================
# PATH
# ======================

PLATE_FOLDER = r"D:/WORK/ANPR_DOC_System/Test_imgs/New folder"
OUTPUT_FOLDER = "ocr_only_output"

VALID_EXT = [".jpg",".jpeg",".png",".bmp"]


# ======================
# LOAD CONFIG
# ======================

def load_config(path="config.yaml"):

    with open(path,"r") as f:
        return yaml.safe_load(f)


# ======================
# DRAW RESULT
# ======================

def draw_text(img,text,status):

    if status=="valid":
        color=(0,255,0)
    else:
        color=(0,0,255)

    cv2.putText(
        img,
        text,
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    return img


# ======================
# RUN OCR ONLY
# ======================

def run_ocr_only():

    config = load_config()

    os.makedirs(OUTPUT_FOLDER,exist_ok=True)

    # load OCR model only
    _, ocr_model = load_models(config)

    summary=[]

    img_files = [

        f for f in os.listdir(PLATE_FOLDER)
        if os.path.splitext(f)[1].lower() in VALID_EXT
    ]


    print(f"\nFound {len(img_files)} plate crops\n")


    for img_name in img_files:

        print("Processing:",img_name)

        img_path = os.path.join(
            PLATE_FOLDER,
            img_name
        )

        img = cv2.imread(img_path)

        plate_text = read_plate(
            img,
            ocr_model,
            config
        )


        if is_valid_indian_plate(
            plate_text,
            config
        ):

            result = {

                "status":"valid",
                "plate_text":plate_text
            }

            img = draw_text(
                img,
                plate_text,
                "valid"
            )


        else:

            reason = get_rejection_reason(
                plate_text
            )

            result = {

                "status":"rejected",
                "raw_text":plate_text,
                "reason":reason
            }

            img = draw_text(
                img,
                f"{plate_text} | {reason}",
                "rejected"
            )


        # save image
        save_img_path = os.path.join(
            OUTPUT_FOLDER,
            img_name
        )

        cv2.imwrite(
            save_img_path,
            img
        )


        # save json
        json_name = os.path.splitext(
            img_name
        )[0] + ".json"

        with open(

            os.path.join(
                OUTPUT_FOLDER,
                json_name
            ),
            "w"

        ) as f:

            json.dump(
                result,
                f,
                indent=4
            )


        summary.append({

            "image":img_name,
            "result":result

        })


    # save summary

    with open(

        os.path.join(
            OUTPUT_FOLDER,
            "SUMMARY.json"
        ),
        "w"

    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )


    print("\nDone.")
    print("Results saved in:",OUTPUT_FOLDER)



if __name__=="__main__":

    run_ocr_only()