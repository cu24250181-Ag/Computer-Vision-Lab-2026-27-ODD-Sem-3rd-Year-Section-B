# Hand Gesture Desktop Controller (Q3)

## Overview
This project provides a simple hand‑gesture controller that maps finger counts to common desktop actions (volume control, slide navigation, scrolling). It uses **MediaPipe Hands** for landmark detection and **pyautogui** to trigger system commands.

## Folder Structure
```
Q3/
│   hand_gesture_controller.py   # main script
```

## Setup
```powershell
cd "e:\BTCS301p\Experiment 11\Q3"
pip install mediapipe opencv-python pyautogui
```

## Usage
```powershell
python hand_gesture_controller.py
```
- A webcam window opens. Show your hand to the camera.
- Press **q** to quit.

## Customising Mappings
Edit the `if count == …` block in `hand_gesture_controller.py` to change which gesture triggers which action.
