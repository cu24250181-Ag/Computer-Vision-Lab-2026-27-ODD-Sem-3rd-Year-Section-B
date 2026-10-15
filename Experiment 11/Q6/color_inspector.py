"""Portable Color & Defect Inspector
=====================================

A lightweight script that captures video from a webcam (or a phone stream) and isolates
a user‑defined HSV color range. The resulting mask is displayed alongside the original
frame, making it easy to spot objects of a particular colour – useful for fruit ripeness
checks, defect detection on small parts, etc.

Dependencies
------------
```powershell
pip install opencv-python numpy
```

Usage
-----
```powershell
cd "e:\\BTCS301p\\Experiment 11\\Q6"
python color_inspector.py
```
- Adjust the ``LOWER_HSV`` and ``UPPER_HSV`` values in the script to match the colour you
  want to detect.
- Press **q** to quit.

---
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# USER SETTINGS – edit these to target a specific colour.
# Example for ripe red apples (HSV approx):
#   LOWER_HSV = np.array([0, 120, 70])
#   UPPER_HSV = np.array([10, 255, 255])
# ---------------------------------------------------------------------------
LOWER_HSV = np.array([35, 50, 50])   # lower bound (example: green)
UPPER_HSV = np.array([85, 255, 255])  # upper bound

def main():
    # Change to a phone stream URL if needed, e.g. "http://192.168.1.42:8080/video"
    cap = None
    try:
        # DirectShow usually opens Windows webcams more reliably than the
        # default backend.
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print('Error: could not open video source. Check that the webcam is connected and not in use.')
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print('Error: could not read a frame from the video source.')
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
            # Apply mask to original image for visualisation
            result = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow('Original', frame)
            cv2.imshow('Mask', mask)
            cv2.imshow('Detected Colour', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print('\nInterrupted; closing the color inspector.')
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
