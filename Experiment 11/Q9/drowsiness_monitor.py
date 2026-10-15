"""Mobile Driver Drowsiness / Blink Monitor
==========================================

A lightweight script that monitors a driver (or any person) via a webcam/phone feed.
It uses **MediaPipe Face Mesh** to obtain eye landmarks, computes the Eye Aspect
Ratio (EAR) and triggers an audible alarm when the eyes stay closed longer than a
configurable duration.

Dependencies
------------
```powershell
pip install mediapipe opencv-python numpy
```
For the alarm we use the built‑in Windows ``winsound`` module – no extra packages
required.

Usage
-----
```powershell
cd "e:\\BTCS301p\\Experiment 11\\Q9"
python drowsiness_monitor.py
```
Press **q** to quit.

---
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import winsound

# ---------------------------------------------------------------------------
# EAR calculation – uses six landmarks per eye (indices from MediaPipe Face Mesh).
# ---------------------------------------------------------------------------

def euclidean(a, b):
    return np.linalg.norm(a - b)

def eye_aspect_ratio(landmarks, eye_indices):
    # eye_indices: list of six landmark indices (order: p1-p6 as per typical EAR formula)
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    # vertical distances
    vert1 = euclidean(p2, p6)
    vert2 = euclidean(p3, p5)
    # horizontal distance
    horiz = euclidean(p1, p4)
    ear = (vert1 + vert2) / (2.0 * horiz)
    return ear

# Indices for left and right eye from MediaPipe Face Mesh (0‑based)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Thresholds – adjust per person/environment
EAR_THRESHOLD = 0.25  # below this means eye closed
CONSEC_FRAMES = 20     # number of consecutive frames below threshold to trigger alarm


def main():
    cap = cv2.VideoCapture(0)  # replace with phone stream URL if needed
    if not cap.isOpened():
        print('Error: could not open video source')
        return

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as face_mesh:
        closed_frames = 0
        alarm_on = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                # Convert normalized landmarks to pixel coordinates
                h, w = frame.shape[:2]
                pts = np.array([(int(l.x * w), int(l.y * h)) for l in landmarks])
                left_ear = eye_aspect_ratio(pts, LEFT_EYE)
                right_ear = eye_aspect_ratio(pts, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0

                cv2.putText(frame, f'EAR: {ear:.2f}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if ear < EAR_THRESHOLD:
                    closed_frames += 1
                else:
                    closed_frames = 0
                    alarm_on = False

                if closed_frames >= CONSEC_FRAMES and not alarm_on:
                    # Trigger alarm – a simple beep
                    winsound.Beep(1000, 500)  # frequency Hz, duration ms
                    alarm_on = True
                    print('Drowsiness detected! Alarm triggered.')

            cv2.imshow('Drowsiness Monitor (q=quit)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
