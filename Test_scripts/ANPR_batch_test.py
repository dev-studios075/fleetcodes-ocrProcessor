import os
import sys
import json
import yaml
import cv2


# =====================================
# SET PROJECT ROOT (EDIT THIS PATH)
# =====================================

PROJECT_ROOT = r"D:\WORK\ANPR_DOC_System"

sys.path.append(PROJECT_ROOT)


from anpr import (
    load_models,
    read_plate,
    is_valid_indian_plate,
    get_rejection_reason
)


# ======================
# PATHS
# ======================

IMAGE_FOLDER = r"D:\WORK\Data ANPR & Product\ANPR DATA\Cropped_Truck_plate - zoomed in"

OUTPUT_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "Anpr_V5zooomed_plate_test"
)

VALID_EXT = [".jpg",".jpeg",".png",".bmp"]


# ======================
# LOAD CONFIG
# ======================

def load_config():

    config_path = os.path.join(
        PROJECT_ROOT,
        "config.yaml"
    )

    with open(config_path, "r") as f:

        return yaml.safe_load(f)


# ======================
# DRAW RESULT
# ======================

def draw_result(img, box, text, status):

    x1,y1,x2,y2 = map(int, box)

    if status == "valid":

        color = (0,255,0)

    else:

        color = (0,0,255)

    cv2.rectangle(img,(x1,y1),(x2,y2),color,2)

    cv2.putText(
        img,
        text,
        (x1, max(25,y1-10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    return img


# ======================
# RUN BATCH
# ======================

def run_batch():

    config = load_config()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    yolo_model, ocr_model = load_models(config)

    summary = []

    # ======================
    # STATS
    # ======================

    total_images = 0
    passed_images = 0
    failed_images = 0

    total_plates = 0
    valid_plates = 0
    rejected_plates = 0


    img_files = sorted([

        f for f in os.listdir(IMAGE_FOLDER)

        if os.path.splitext(f)[1].lower() in VALID_EXT

    ])


    for img_name in img_files:

        total_images += 1

        print(f"\nProcessing {img_name}")

        img_path = os.path.join(IMAGE_FOLDER, img_name)

        img = cv2.imread(img_path)

        if img is None:

            print("Image failed to load")

            summary.append({

                "image": img_name,
                "error": "IMAGE_LOAD_FAILED"

            })

            failed_images += 1

            continue


        results = yolo_model(

            img_path,

            conf=config["anpr"]["detection"]["conf_threshold"]

        )


        all_boxes = []

        for r in results:

            boxes = r.boxes.xyxy.cpu().numpy()

            for b in boxes:

                all_boxes.append(list(map(int,b)))


        image_result = []

        image_has_valid_plate = False


        # ======================
        # CASE: no plate detected
        # ======================

        if len(all_boxes) == 0:

            reason = "NO_PLATE_DETECTED"

            image_result.append({

                "status":"rejected",
                "reason":reason

            })

            cv2.putText(

                img,
                reason,
                (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2

            )


        # ======================
        # PROCESS PLATES
        # ======================

        for box in all_boxes:

            total_plates += 1

            x1,y1,x2,y2 = box

            crop = img[y1:y2,x1:x2]

            if crop.size == 0:

                continue


            plate_text = read_plate(

                crop,
                ocr_model,
                config

            )


            # VALID PLATE
            if is_valid_indian_plate(plate_text,config):

                valid_plates += 1

                image_has_valid_plate = True

                image_result.append({

                    "status":"valid",
                    "plate_text":plate_text,
                    "bbox":[x1,y1,x2,y2]

                })


                img = draw_result(

                    img,
                    (x1,y1,x2,y2),
                    plate_text,
                    "valid"

                )


            # REJECTED PLATE
            else:

                rejected_plates += 1

                reason = get_rejection_reason(plate_text)

                image_result.append({

                    "status":"rejected",
                    "raw_text":plate_text,
                    "reason":reason,
                    "bbox":[x1,y1,x2,y2]

                })


                img = draw_result(

                    img,
                    (x1,y1,x2,y2),
                    f"{plate_text} | {reason}",
                    "rejected"

                )


        # ======================
        # IMAGE PASS / FAIL
        # ======================

        if image_has_valid_plate:

            passed_images += 1

        else:

            failed_images += 1


        # ======================
        # SAVE IMAGE
        # ======================

        save_img_path = os.path.join(

            OUTPUT_FOLDER,
            img_name

        )

        cv2.imwrite(save_img_path,img)


        # ======================
        # SAVE JSON
        # ======================

        json_name = os.path.splitext(img_name)[0] + ".json"

        with open(

            os.path.join(OUTPUT_FOLDER,json_name),
            "w"

        ) as f:

            json.dump(image_result,f,indent=4)


        summary.append({

            "image":img_name,
            "result":image_result

        })


    # ======================
    # SAVE SUMMARY JSON
    # ======================

    with open(

        os.path.join(OUTPUT_FOLDER,"SUMMARY.json"),
        "w"

    ) as f:

        json.dump(summary,f,indent=4)


    # ======================
    # FINAL STATS
    # ======================

    success_rate = 0

    if total_images > 0:

        success_rate = (passed_images / total_images) * 100


    stats = {

        "total_images": total_images,
        "passed_images": passed_images,
        "failed_images": failed_images,
        "success_rate": success_rate,

        "total_plates": total_plates,
        "valid_plates": valid_plates,
        "rejected_plates": rejected_plates
    }


    with open(

        os.path.join(OUTPUT_FOLDER,"STATS.json"),
        "w"

    ) as f:

        json.dump(stats,f,indent=4)


    print("\n==============================")
    print("BATCH SUMMARY")
    print("==============================\n")

    print(f"Total Images        : {total_images}")
    print(f"Passed Images       : {passed_images}")
    print(f"Failed Images       : {failed_images}")
    print(f"Success Rate        : {success_rate:.2f} %\n")

    print(f"Total Plates Found  : {total_plates}")
    print(f"Valid Plates        : {valid_plates}")
    print(f"Rejected Plates     : {rejected_plates}\n")

    print("Results saved in:")
    print(OUTPUT_FOLDER)



if __name__ == "__main__":

    run_batch()