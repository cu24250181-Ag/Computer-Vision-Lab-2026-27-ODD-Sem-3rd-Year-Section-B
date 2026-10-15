# Virtual Whiteboard / Air Painting App (Q7)

## Overview
This script turns your index fingertip into a virtual brush. Draw with a bare
index finger or hold a marker cap in your hand; the app tracks the hand rather
than the marker's color, so the cap does not need to be bright or blue.

## Folder Structure
```
Q7/
│   whiteboard.py
│   README.md
│   requirements.txt
```

## Setup & Run
```powershell
cd "e:\\BTCS301p\\Experiment 11\\Q7"
pip install -r requirements.txt
python whiteboard.py
```
- Press **`c`** to clear the canvas.
- Press **`q`** to quit.
- Extend your index finger and curl your middle, ring, and little fingers to draw.

## Customisation
- **Line style:** Change the colour or thickness inside the `cv2.line` call.
- **Persistence:** Replace the NumPy canvas with a saved image file if you want the drawing to survive between runs.
