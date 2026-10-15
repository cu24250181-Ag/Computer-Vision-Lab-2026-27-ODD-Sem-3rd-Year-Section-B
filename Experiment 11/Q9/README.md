# Mobile Driver Drowsiness / Blink Monitor (Q9)

## Overview
A simple driver‑monitoring script that watches a face via webcam (or phone stream), computes the Eye Aspect Ratio (EAR) using MediaPipe Face Mesh and sounds an alarm when the eyes stay closed longer than a configurable threshold.

## Folder Structure
```
Q9/
│   drowsiness_monitor.py
│   README.md
│   requirements.txt
```

## Setup & Run
```powershell
cd "e:\\BTCS301p\\Experiment 11\\Q9"
pip install -r requirements.txt
python drowsiness_monitor.py
```
- Press **`q`** to quit.
- Press **Ctrl+C** in the terminal to stop cleanly if webcam startup or capture
  takes too long.

## How it works
1. **Face Mesh** provides 468 facial landmarks.
2. Six landmarks per eye are extracted (indices defined in the script).
3. EAR = (‖p2‑p6‖ + ‖p3‑p5‖) / (2·‖p1‑p4‖).  When EAR falls below `EAR_THRESHOLD` for `CONSEC_FRAMES` frames, an alarm (Windows `winsound.Beep`) is triggered.

## Customisation
- **EAR_THRESHOLD** – lower values make detection stricter.
- **CONSEC_FRAMES** – adjust the duration (higher = longer eye closure required).
- **Alarm** – replace `winsound.Beep` with a custom audio file via `pygame` or `playsound` if you prefer.

## Notes
- The script uses the built‑in `winsound` module on Windows; no extra audio library is required.
- For non‑Windows platforms you can swap the alarm section with any cross‑platform audio playback method.
