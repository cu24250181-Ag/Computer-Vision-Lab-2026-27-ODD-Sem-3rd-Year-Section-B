# Portable Color & Defect Inspector (Q6)

## Overview
A lightweight script that isolates a user‑defined HSV colour range from a video stream. Useful for detecting ripe fruit, defective parts, etc.

## Folder Structure
```
Q6/
│   color_inspector.py
│   README.md
│   requirements.txt
```

## Setup & Run
```powershell
cd "e:\\BTCS301p\\Experiment 11\\Q6"
pip install -r requirements.txt
python color_inspector.py
```
- Edit `LOWER_HSV` and `UPPER_HSV` in `color_inspector.py` to match the colour you want to detect.
- Press **`q`** to quit.
- Press **Ctrl+C** in the terminal to stop cleanly if webcam startup or capture
  takes too long.

## Customisation
- Adjust the HSV thresholds for different lighting conditions.
- Replace the mask visualisation with contour drawing if you prefer.
- For a batch of images, modify the script to read from a directory instead of a live stream.
