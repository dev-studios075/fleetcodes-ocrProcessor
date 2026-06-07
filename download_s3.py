"""
S3 Recursive Downloader
Usage: python download_s3.py <s3_prefix> <local_output_dir>

Example:
  python download_s3.py organisations/AVR0001/trip/ d:/Data/fleetcodes/fleetcodes-avr001-trips
"""

import os
import sys
import boto3
from dotenv import load_dotenv

load_dotenv()

BUCKET = "fleetcodes-storage-bucket"


def long_path(p: str) -> str:
    """Add Windows extended-length path prefix to bypass 260-char limit."""
    p = os.path.abspath(p)
    if sys.platform == "win32" and not p.startswith("\\\\?\\"):
        p = "\\\\?\\" + p
    return p


def download(prefix: str, out_dir: str):

    profile = os.environ.get("AWS_PROFILE")
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    s3 = session.client("s3")

    paginator = s3.get_paginator("list_objects_v2")

    total = skipped = errors = 0

    print(f"\nBucket  : {BUCKET}")
    print(f"Prefix  : {prefix}")
    print(f"Out dir : {out_dir}\n")

    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):]

            if not rel:          # skip the folder placeholder itself
                continue

            local = os.path.join(out_dir, rel.replace("/", os.sep))
            lpath = long_path(local)

            if os.path.exists(lpath):
                skipped += 1
                continue

            os.makedirs(long_path(os.path.dirname(local)), exist_ok=True)

            try:
                print(f"  ↓  {key}")
                s3.download_file(BUCKET, key, lpath)
                total += 1
            except Exception as e:
                print(f"  ✗  FAILED {key}: {e}")
                errors += 1

    print(f"\n✅  Done. {total} downloaded | {skipped} skipped | {errors} errors")
    print(f"   Saved to: {out_dir}\n")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    download(
        prefix=sys.argv[1].lstrip("s3://").split("/", 1)[-1] if sys.argv[1].startswith("s3://") else sys.argv[1],
        out_dir=sys.argv[2],
    )
