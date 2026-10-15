"""
Live Motion & Security Alarm Detector
=======================================
- Uses cv2.absdiff() to compare consecutive frames
- If motion above MOTION_THRESHOLD pixels is detected, triggers a RED on-screen alert
- Saves an emergency snapshot to disk when alert fires
- Press Q or ESC to quit

Key Skills: cv2.absdiff(), frame subtraction, contour area thresholding
"""

import cv2
import os
import time
from datetime import datetime

# ─────────────────── CONFIG ──────────────────────
CAMERA_INDEX     = 0        # 0 = default webcam
MOTION_THRESHOLD = 5000     # minimum contour area (pixels) to count as motion
BLUR_SIZE        = 21       # Gaussian blur kernel (must be odd)
SNAPSHOT_COOLDOWN = 3       # seconds between consecutive snapshots
SNAPSHOT_DIR     = "snapshots"
WINDOW_TITLE     = "Motion & Security Alarm Detector"
# ─────────────────────────────────────────────────


def ensure_snapshot_dir():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def save_snapshot(frame) -> str:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = os.path.join(SNAPSHOT_DIR, f"motion_{ts}.jpg")
    cv2.imwrite(name, frame)
    return name


def draw_alert(frame):
    """Draw a bold red ALERT banner at the top of the frame."""
    h, w = frame.shape[:2]
    # semi-transparent red overlay at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "!! MOTION DETECTED -- ALERT !!",
                (10, 42), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)


def draw_status(frame, motion: bool, total: int):
    """Draw bottom status bar."""
    h, w = frame.shape[:2]
    bar_color = (0, 0, 60) if motion else (20, 20, 20)
    cv2.rectangle(frame, (0, h - 36), (w, h), bar_color, -1)
    ts  = datetime.now().strftime("%H:%M:%S")
    msg = f"  Snapshots saved: {total}   |   Time: {ts}   |   Q/ESC = Quit"
    cv2.putText(frame, msg, (8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def main():
    ensure_snapshot_dir()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_INDEX.")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("[INFO] Motion detector running. Press Q or ESC to quit.")
    print(f"[INFO] Motion threshold : {MOTION_THRESHOLD} px^2")
    print(f"[INFO] Snapshots saved to: {os.path.abspath(SNAPSHOT_DIR)}")

    prev_frame      = None
    last_snap_time  = 0
    total_snaps     = 0
    motion_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed - retrying...")
            time.sleep(0.05)
            continue

        # ── Pre-processing ───────────────────────────────────────
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)

        motion_detected = False

        if prev_frame is not None:
            # ── Frame Difference ─────────────────────────────────
            diff    = cv2.absdiff(prev_frame, blurred)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            # small noise removal
            thresh  = cv2.dilate(thresh, None, iterations=2)

            # ── Contour Detection ────────────────────────────────
            contours, _ = cv2.findContours(thresh,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MOTION_THRESHOLD:
                    continue  # too small – ignore

                motion_detected = True
                # draw green bounding box around moving region
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # ── Alert & Snapshot ─────────────────────────────────
            if motion_detected:
                draw_alert(frame)
                now = time.time()
                if now - last_snap_time >= SNAPSHOT_COOLDOWN:
                    last_snap_time = now
                    path = save_snapshot(frame)
                    total_snaps += 1
                    print(f"[SNAP #{total_snaps}] Saved -> {path}")

        draw_status(frame, motion_detected, total_snaps)
        cv2.imshow(WINDOW_TITLE, frame)

        prev_frame = blurred

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Total snapshots saved: {total_snaps}")
    print(f"[INFO] Snapshots folder: {os.path.abspath(SNAPSHOT_DIR)}")


if __name__ == "__main__":
    main()
