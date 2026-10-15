"""Real-Time AR Filter Overlay
==============================

A lightweight script that captures video from a webcam (or a phone stream) and
adds a virtual filter – for example a pair of glasses – onto the user’s face in
real time. The implementation uses **MediaPipe Face Mesh** to obtain facial
landmarks and then blends a transparent PNG onto the frame.

Dependencies
------------
```powershell
pip install mediapipe opencv-python numpy
```
Place a PNG with an alpha channel (e.g., ``glasses.png``) in the same folder.
You can replace it with any overlay (hats, masks, etc.) as long as the image
has transparency.

Usage
-----
```powershell
cd "e:\\BTCS301p\\Experiment 11\\Q10"
python ar_filter.py
```
Press **q** to quit.

---
"""

import cv2
import mediapipe as mp
import numpy as np
import os

# ---------------------------------------------------------------------------
# Load overlay image (transparent PNG). Adjust filename as needed.
# The image should be oriented such that its width matches the distance between
# the left and right eye corners.
# ---------------------------------------------------------------------------
OVERLAY_PATH = os.path.join(os.path.dirname(__file__), "glasses.png")


def create_default_glasses():
    """Create a simple transparent glasses overlay when no PNG is provided."""
    image = np.zeros((140, 400, 4), dtype=np.uint8)
    color = (40, 220, 255, 255)
    cv2.ellipse(image, (105, 68), (88, 48), 0, 0, 360, color, 8, cv2.LINE_AA)
    cv2.ellipse(image, (295, 68), (88, 48), 0, 0, 360, color, 8, cv2.LINE_AA)
    cv2.line(image, (190, 62), (210, 62), color, 8, cv2.LINE_AA)
    cv2.line(image, (17, 58), (0, 48), color, 7, cv2.LINE_AA)
    cv2.line(image, (383, 58), (399, 48), color, 7, cv2.LINE_AA)
    return image


overlay_img = (
    cv2.imread(OVERLAY_PATH, cv2.IMREAD_UNCHANGED)
    if os.path.isfile(OVERLAY_PATH)
    else None
)
if overlay_img is None:
    print(f"Overlay image unavailable at {OVERLAY_PATH}; using built-in glasses.")
    overlay_img = create_default_glasses()

# ---------------------------------------------------------------------------
# Helper to overlay a PNG with alpha onto a BGR frame.
# ---------------------------------------------------------------------------
def overlay_transparent(background, overlay, x, y, overlay_size=None):
    bh, bw = background.shape[:2]
    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_AREA)
    h, w = overlay.shape[:2]

    if x >= bw or y >= bh:
        return background
    # Clip overlay if it goes out of bounds
    if x + w > bw:
        w = bw - x
        overlay = overlay[:, :w]
    if y + h > bh:
        h = bh - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        # No alpha channel – treat as opaque
        overlay_img = overlay
        mask = np.ones((h, w), dtype=np.uint8) * 255
    else:
        overlay_img = overlay[:, :, :3]
        mask = overlay[:, :, 3]

    mask_inv = cv2.bitwise_not(mask)
    roi = background[y:y+h, x:x+w]
    # Black‑out the area of overlay in ROI
    bg_clean = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # Take only region of overlay
    fg = cv2.bitwise_and(overlay_img, overlay_img, mask=mask)
    # Combine and replace in background
    combined = cv2.add(bg_clean, fg)
    background[y:y+h, x:x+w] = combined
    return background

# ---------------------------------------------------------------------------
# MediaPipe Face Mesh setup – we need eye corner landmarks for positioning.
# ---------------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
# Landmark indices for left/right eye outer corners (from MediaPipe docs)
LEFT_OUTER = 33   # left eye outer corner
RIGHT_OUTER = 263 # right eye outer corner


def main():
    cap = cv2.VideoCapture(0)  # replace with phone stream URL if needed
    if not cap.isOpened():
        print('Error: could not open video source')
        return

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]
                # Get pixel coordinates of eye corners
                left_pt = np.array([int(landmarks[LEFT_OUTER].x * w), int(landmarks[LEFT_OUTER].y * h)])
                right_pt = np.array([int(landmarks[RIGHT_OUTER].x * w), int(landmarks[RIGHT_OUTER].y * h)])
                # Compute center point between eyes
                center = ((left_pt + right_pt) // 2).astype(int)
                # Width of overlay based on distance between eye corners
                eye_width = np.linalg.norm(right_pt - left_pt)
                overlay_width = int(eye_width * 2.0)  # make glasses a bit wider than eyes
                overlay_height = int(overlay_width * overlay_img.shape[0] / overlay_img.shape[1])
                # Top‑left corner for overlay
                x = int(center[0] - overlay_width / 2)
                y = int(center[1] - overlay_height / 2)
                frame = overlay_transparent(frame, overlay_img, x, y, (overlay_width, overlay_height))

            cv2.imshow('AR Filter Overlay (q=quit)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
