import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
import yaml
from anpr import run_anpr
from doc_ocr import run_doc_ocr   



# LOAD CONFIG

def load_config(path="config.yaml"):

    with open(path, "r") as f:

        config = yaml.safe_load(f)

    return config



# MAIN CONTROLLER

def main():

    config = load_config()

    mode = config["mode"]


    if mode == "anpr":

        print("\nRunning ANPR...\n")

        run_anpr(config)


    elif mode == "doc":

        print("\nRunning Document OCR...\n")

        run_doc_ocr(config)


    elif mode == "both":

        print("\nRunning ANPR + Document OCR...\n")

        run_anpr(config)

        run_doc_ocr(config)


    else:

        raise ValueError("Invalid mode in config.yaml")




# ENTRY POINT

if __name__ == "__main__":

    main()