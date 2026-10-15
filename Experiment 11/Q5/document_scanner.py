"""Document Scanner with Perspective Transform
================================================

Capture a photo of a document (e.g., a sheet of paper) using a webcam or the camera
feed of a phone streamed to the PC. The script automatically:

1. Enhances contrast and combines adaptive Canny and threshold-based edges.
2. Finds the largest plausible 4‑point contour (the document).
3. Computes a perspective transform to obtain a top‑down view.
4. Converts the warped image to a high‑contrast black‑and‑white scan.

The result is displayed in a window and saved to ``scanned_YYYYMMDD_HHMMSS.png``.

Dependencies
------------
```powershell
pip install opencv-python numpy
```

Usage
-----
```powershell
cd "e:\\BTCS301p\\Experiment 11\\Q5"
python document_scanner.py                 # default webcam (camera 0)
python document_scanner.py --source 1      # another webcam
python document_scanner.py --source "http://192.168.1.42:8080/video"  # phone stream
```
Press **s** to save the current scan, **m** to switch between color and
black-and-white output, **e** to show/hide the edge-detection view, and **q**
to quit.
For best detection, place the whole page in view against a contrasting,
non-patterned background and avoid glare or strong shadows.

---
"""

import cv2
import numpy as np
import os
import argparse
from datetime import datetime

# ---------------------------------------------------------------------------
# Helper: order the four corner points consistently (top‑left, top‑right,
# bottom‑right, bottom‑left).
# ---------------------------------------------------------------------------

def order_points(pts):
    """Return points ordered as TL, TR, BR, BL.
    ``pts`` is a NumPy array of shape (4, 2).
    """
    rect = np.zeros((4, 2), dtype="float32")
    # sum and diff of coordinates give top‑left/bottom‑right etc.
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]       # top‑left has smallest sum
    rect[2] = pts[np.argmax(s)]       # bottom‑right has largest sum
    rect[1] = pts[np.argmin(diff)]    # top‑right has smallest diff
    rect[3] = pts[np.argmax(diff)]    # bottom‑left has largest diff
    return rect

# ---------------------------------------------------------------------------
# Helper: perform the perspective warp and binarize the result.
# ---------------------------------------------------------------------------

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # Compute width and height of the new image
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


def find_document_contour(image):
    """Find the largest plausible document quadrilateral in a BGR image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    median = float(np.median(blurred))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, max(lower + 30, 1.33 * median)))
    edges = cv2.Canny(blurred, lower, upper)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )

    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    adaptive = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
    )

    min_area = image.shape[0] * image.shape[1] * 0.04
    candidates = []
    for source in (edges, adaptive):
        contours, _ = cv2.findContours(
            source, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:40]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                break
            perimeter = cv2.arcLength(contour, True)
            for epsilon_fraction in (0.015, 0.02, 0.03, 0.04):
                approx = cv2.approxPolyDP(
                    contour, epsilon_fraction * perimeter, True
                )
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    x, y, width, height = cv2.boundingRect(approx)
                    if (
                        x <= 1
                        or y <= 1
                        or x + width >= image.shape[1] - 1
                        or y + height >= image.shape[0] - 1
                    ):
                        continue
                    candidates.append((area, approx))
                    break

    document = max(candidates, key=lambda item: item[0])[1] if candidates else None
    return document, edges


def parse_video_source(source):
    """Use integer camera indexes as webcams and other strings as stream URLs."""
    try:
        return int(source)
    except ValueError:
        return source


def open_video_source(source):
    """Open a webcam or live video URL, falling back to OpenCV's default backend."""
    if isinstance(source, int) and os.name == "nt":
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if capture.isOpened():
            return capture
        capture.release()

    return cv2.VideoCapture(source)


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find and scan a document from a live webcam or phone video stream."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="webcam index (0, 1, ...) or live video stream URL",
    )
    args = parser.parse_args()
    source = parse_video_source(args.source)

    cap = open_video_source(source)
    if not cap.isOpened():
        print(f'Error: Could not open video source: {args.source}')
        print('Check camera permissions, or pass a reachable phone stream URL with --source.')
        cap.release()
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f'Live document search started using video source: {args.source}')

    color_mode = False
    show_edges = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Error: Could not read a frame from the video source.')
                break

            # Resize for faster detection while keeping aspect ratio.
            ratio = min(1.0, 800.0 / frame.shape[1])
            orig = frame.copy()
            preview = cv2.resize(
                frame, (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio))
            )

            document, edges = find_document_contour(preview)

            scanned = None
            if document is not None:
                cv2.drawContours(preview, [document], -1, (0, 220, 0), 3)
                points = document.reshape(4, 2) / ratio
                warped = four_point_transform(orig, points)
                if color_mode:
                    scanned = warped
                else:
                    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    scanned = cv2.adaptiveThreshold(
                        warped_gray, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        11, 2
                    )

            cv2.putText(
                preview,
                ('DOCUMENT FOUND' if document is not None else 'SEARCHING FOR DOCUMENT')
                + ' | S: save  M: color/B&W  E: edges  Q: quit',
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 0) if document is not None else (0, 180, 255),
                2,
            )
            cv2.imshow('Document Scanner', preview)
            if show_edges:
                cv2.imshow('Detection Edges', edges)
            if scanned is not None:
                cv2.imshow('Scan Preview', scanned)
            else:
                waiting = np.zeros((120, 420), dtype=np.uint8)
                cv2.putText(
                    waiting,
                    'Show a clear document to the camera',
                    (12, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    255,
                    1,
                )
                cv2.imshow('Scan Preview', waiting)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('m'):
                color_mode = not color_mode
                print(f"Scan mode: {'color' if color_mode else 'black-and-white'}")
            if key == ord('e'):
                show_edges = not show_edges
                if not show_edges:
                    cv2.destroyWindow('Detection Edges')
            if key == ord('s'):
                if scanned is None:
                    print('No document detected; cannot save a scan yet.')
                    continue
                output_dir = os.path.join(os.path.dirname(__file__), 'scans')
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                out_path = os.path.join(output_dir, f'scanned_{timestamp}.png')
                if not cv2.imwrite(out_path, scanned):
                    print(f'Error: could not save scan to {out_path}')
                else:
                    print(f'Saved scan to {out_path}')
    except KeyboardInterrupt:
        print('\nInterrupted; closing the document scanner.')
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
