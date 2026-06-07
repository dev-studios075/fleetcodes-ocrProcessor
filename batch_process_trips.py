"""
Batch RPS Sheet Processor
Walks every <trip-id>/trip-sheets/ folder under the trips root,
finds image files, sends each to POST /process-local?type=doc,
and writes the response into output.log in that same folder.

Usage:
    python batch_process_trips.py [trips_root] [server_url]

Defaults:
    trips_root  = d:/Data/fleetcodes/fleetcodes-avr001-trips
    server_url  = http://localhost:8000
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

TRIPS_ROOT  = sys.argv[1] if len(sys.argv) > 1 else "d:/Data/fleetcodes/fleetcodes-avr001-trips"
SERVER_URL  = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
ENDPOINT    = f"{SERVER_URL}/process-local"

IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Files to skip (parsed logs, PDFs, etc.)
SKIP_EXTS   = {".log", ".pdf", ".txt", ".csv", ".json"}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def is_image(filename: str) -> bool:
    ext = os.path.splitext(filename)[-1].lower()
    if ext in SKIP_EXTS:
        return False
    # if no extension, try to treat as image (many S3 keys have no ext)
    return ext in IMAGE_EXTS or ext == ""


def call_process_local(local_path: str) -> dict:
    payload = json.dumps({"local_path": local_path, "type": "doc"}).encode()
    req = urllib.request.Request(
        ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return {"error": f"HTTP {e.code}", "detail": body}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():

    print(f"\nTrips root : {TRIPS_ROOT}")
    print(f"Server     : {SERVER_URL}\n")

    trip_dirs = sorted([
        d for d in os.listdir(TRIPS_ROOT)
        if os.path.isdir(os.path.join(TRIPS_ROOT, d))
    ])

    total_trips = len(trip_dirs)
    total_images = processed = skipped = errors = 0

    for idx, trip_id in enumerate(trip_dirs, 1):

        sheets_dir = os.path.join(TRIPS_ROOT, trip_id, "trip-sheets")

        if not os.path.isdir(sheets_dir):
            continue

        files = sorted(os.listdir(sheets_dir))
        images = [f for f in files if is_image(f)]

        if not images:
            continue

        log_path = os.path.join(sheets_dir, "output.log")
        log_entries = []

        print(f"[{idx}/{total_trips}] Trip: {trip_id}  ({len(images)} image(s))")

        for fname in images:
            img_path = os.path.join(sheets_dir, fname)
            total_images += 1

            print(f"    → {fname}")
            result = call_process_local(img_path)

            if "error" in result:
                errors += 1
                print(f"      ✗ {result}")
            else:
                processed += 1

            log_entries.append({
                "file": fname,
                "result": result
            })

            time.sleep(0.05)   # small pause to avoid overloading server

        # write output.log
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_entries, f, indent=2, ensure_ascii=False)

        print(f"    ✅ Saved → {log_path}\n")

    print("═" * 60)
    print(f"Trips scanned  : {total_trips}")
    print(f"Images found   : {total_images}")
    print(f"Processed OK   : {processed}")
    print(f"Errors         : {errors}")
    print("═" * 60)


if __name__ == "__main__":
    main()
