import os
import random
import shutil
import json
import sys


# allow imports from project root
sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

from doc_ocr import load_ocr, extract_header, normalize_data


ROOT = r"D:\WORK\Data ANPR & Product\ANPR DATA\Documents scan\data\data"

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "doc_ocr_test_samples"
)

N_SAMPLES = 100



def collect_images():

    image_paths = []

    for folder in os.listdir(ROOT):

        trip_path = os.path.join(ROOT, folder, "trip-sheets")

        if not os.path.exists(trip_path):
            continue


        for f in os.listdir(trip_path):

            if f.lower().endswith((".jpg", ".jpeg", ".png")):

                image_paths.append(
                    os.path.join(trip_path, f)
                )

    return image_paths




def find_txt_for_image(sheet_id, img_path):

    if not sheet_id:
        return None


    trip_folder = os.path.dirname(img_path)


    for f in os.listdir(trip_folder):

        if f.startswith(sheet_id) and f.endswith((".txt", ".log")):

            return os.path.join(trip_folder, f)


    return None




def run_test(config):

    os.makedirs(OUTPUT_DIR, exist_ok=True)


    images = collect_images()


    if not images:

        print("No images found")
        return


    # true randomness every run
    random.seed(os.urandom(32))
    random.shuffle(images)


    sample_imgs = images[:min(N_SAMPLES, len(images))]


    ocr = load_ocr(config)


    for i, img_path in enumerate(sample_imgs):

        print("\n-----------------------------")
        print(f"{i+1}/{len(sample_imgs)}")

        data = extract_header(img_path, ocr)

        data = normalize_data(data)


        sheet_id = data.get("sheet_id")


        txt_path = find_txt_for_image(sheet_id, img_path)


        base = f"{i:02d}_" + os.path.splitext(os.path.basename(img_path))[0]


        print("sheet_id   :", sheet_id)
        print("route      :", data.get("route"))
        print("prepared_at:", data.get("prepared_at"))
        print("txt found  :", bool(txt_path))


        # copy image
        shutil.copy(
            img_path,
            os.path.join(OUTPUT_DIR, f"{base}.jpg")
        )


        # copy dataset txt
        if txt_path:

            shutil.copy(
                txt_path,
                os.path.join(OUTPUT_DIR, f"{base}.txt")
            )


        # save OCR json
        with open(
            os.path.join(OUTPUT_DIR, f"{base}.json"),
            "w"
        ) as f:

            json.dump(data, f, indent=4)



    print("\n=============================")
    print("DONE")
    print("saved in:", OUTPUT_DIR)
    print("=============================")




if __name__ == "__main__":

    import yaml

    CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config.yaml"
    )

    with open(CONFIG_PATH) as f:

        config = yaml.safe_load(f)


    run_test(config)