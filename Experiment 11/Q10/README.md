# Real-Time AR Filter Overlay (Q10)

## Overview
A lightweight script that captures video from a webcam (or a phone stream) and overlays a virtual filter—such as glasses—onto the detected face in real time. It uses **MediaPipe Face Mesh** for facial landmarks and blends a transparent PNG onto the video feed.

## Folder Structure
```
Q10/
│   ar_filter.py
│   README.md
│   requirements.txt
│   glasses.png   # optional: provide your own PNG with an alpha channel
```

## Setup & Run
```powershell
cd "e:\\BTCS301p\\Experiment 11\\Q10"
pip install -r requirements.txt
python ar_filter.py
```
- Press **`q`** to quit.

## Adding an Overlay Image
1. Choose a PNG image with a transparent background (e.g., a pair of glasses, a hat, a mask).
2. Save it as `glasses.png` in this folder (or rename your image accordingly).
3. The script will automatically resize and position the overlay based on the distance between the left‑ and right‑eye outer corners. If no readable `glasses.png` is present, the script uses a simple built-in glasses overlay.

## Customisation
- **Overlay file** – change the `OVERLAY_PATH` variable in `ar_filter.py` to point to a different image.
- **Scale factor** – modify `overlay_width = int(eye_width * 2.0)` to make the overlay larger or smaller.
- **Multiple filters** – you can load additional PNGs and switch between them with a key press.

## Notes
- The script expects the overlay PNG to contain an alpha channel for proper blending.
- For best results, ensure good lighting and keep the face relatively centered.

Feel free to adapt this example for more complex AR effects, such as 3‑D models or animated masks.
