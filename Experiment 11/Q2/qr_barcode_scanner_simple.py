"""
Simple QR / Barcode Scanner (Minimal UI)
================================================
- Opens default webcam
- Detects QR codes and common barcodes in real‑time
- Draws a green rectangle around each detection
- Logs every unique scan to inventory_log.csv with a timestamp
- Emits a Windows beep on each successful scan
- Press Q or ESC to quit
"""

import csv
import os
import time
from datetime import datetime
import winsound

import cv2
from pyzbar import pyzbar

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
CAMERA_INDEX = 0               # default webcam
CSV_FILE = "inventory_log.csv"
COOLDOWN_SEC = 2.0             # seconds before same code can be logged again
WINDOW_TITLE = "Simple QR/Barcode Scanner"

# ---------------------------------------------------------------------
def init_csv(path: str) -> None:
    """Create CSV with header if it does not exist."""
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Type", "Data"])
        print(f"[INFO] CSV created -> {os.path.abspath(path)}")
    else:
        print(f"[INFO] Using existing CSV -> {os.path.abspath(path)}")


def log_to_csv(path: str, code_type: str, data: str) -> None:
    """Append a scan entry to the CSV file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, code_type, data])


def draw_rect(frame, barcode) -> None:
    """Draw a simple green rectangle around the detected barcode."""
    x, y, w, h = barcode.rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # optional label on top
    label = f"{barcode.type}"
    cv2.putText(frame, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1)


def main() -> None:
    init_csv(CSV_FILE)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_INDEX.")
        return

    # reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # state for duplicate‑filtering
    last_seen: dict[str, float] = {}

    print("[INFO] Scanner started – press Q or ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed – retrying...")
            time.sleep(0.05)
            continue

        # decode barcodes in the whole frame
        barcodes = pyzbar.decode(frame)
        now = time.time()
        for bc in barcodes:
            draw_rect(frame, bc)
            data = bc.data.decode("utf-8", errors="replace")
            key = f"{bc.type}:{data}"
            if now - last_seen.get(key, 0) >= COOLDOWN_SEC:
                last_seen[key] = now
                log_to_csv(CSV_FILE, bc.type, data)
                # beep (frequency 1000Hz, duration 120ms)
                try:
                    winsound.Beep(1000, 120)
                except Exception:
                    pass
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[SCAN {ts}] {bc.type:<12} {data}")

        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Scanner stopped.")
    print(f"[INFO] CSV location -> {os.path.abspath(CSV_FILE)}")

if __name__ == "__main__":
    main()
