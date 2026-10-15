"""Real-Time OCR Text Reader
=============================

Capture video from a webcam (or a phone camera stream) and run Tesseract OCR on each
frame. The recognized text is overlaid on the video feed in real time.

Dependencies
------------
```powershell
pip install opencv-python pytesseract
```
You also need the Tesseract engine installed and its binary on the system PATH.
- Windows installer: https://github.com/UB-Mannheim/tesseract/wiki
- After installing, verify with `tesseract --version`.

Usage
-----
```powershell
cd "e:\\BTCS301p\\Experiment 11\\Q8"
python ocr_reader.py
```
Press **q** to quit.

---
"""

import os
import shutil
from pathlib import Path

import cv2
import pytesseract


def configure_tesseract():
    """Configure the Tesseract executable or report how to install it."""
    executable = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if executable is None:
        windows_executable = (
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "Tesseract-OCR"
            / "tesseract.exe"
        )
        if windows_executable.is_file():
            executable = str(windows_executable)

    if executable is None:
        print(
            "Error: Tesseract OCR engine was not found. Install it from "
            "https://github.com/UB-Mannheim/tesseract/wiki, then add it to PATH "
            "or set the TESSERACT_CMD environment variable to tesseract.exe."
        )
        return False

    pytesseract.pytesseract.tesseract_cmd = executable
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        print(
            f"Error: Tesseract executable was not found at {executable!r}. "
            "Check the TESSERACT_CMD environment variable or your PATH."
        )
        return False
    return True

def preprocess(frame):
    """Convert frame to grayscale and apply thresholding for better OCR accuracy."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold to handle varying lighting
    th = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    return th

def main():
    if not configure_tesseract():
        return

    cap = cv2.VideoCapture(0)  # replace with phone stream URL if needed
    if not cap.isOpened():
        print('Error: could not open video source')
        cap.release()
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: could not read a frame from the video source.")
                break

            processed = preprocess(frame)
            # Run OCR – limit to a region of interest if desired (here full frame)
            text = pytesseract.image_to_string(processed)
            # Overlay text (first few lines) on the original frame
            for i, line in enumerate(text.splitlines()[:5]):
                line = line.strip()
                if line:
                    cv2.putText(frame, line[:100], (10, 30 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('Real-Time OCR (q=quit)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
