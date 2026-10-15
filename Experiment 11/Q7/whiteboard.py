"""Virtual Whiteboard / Air Painting App
=======================================

Use the webcam to draw with your index finger, either bare or while holding a
marker cap. Extend your index finger and curl the other fingers to draw.

Dependencies
------------
```powershell
pip install -r requirements.txt
```

Usage
-----
```powershell
cd "e:\\BTCS301p\\Experiment 11\\Q7"
python whiteboard.py
```
Press **q** to quit. Press **c** to clear the canvas.
"""

import cv2
import mediapipe as mp
import numpy as np


def is_index_only(hand_landmarks) -> bool:
    """Return whether the index finger is extended and the other fingers curled."""
    landmarks = hand_landmarks.landmark
    index_extended = landmarks[8].y < landmarks[6].y
    other_fingers_curled = all(
        landmarks[tip].y > landmarks[pip].y
        for tip, pip in ((12, 10), (16, 14), (20, 18))
    )
    return index_extended and other_fingers_curled


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open video source")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: cannot read first frame")
        cap.release()
        return

    canvas = np.zeros_like(frame)
    prev_center = None
    hands_module = mp.solutions.hands

    try:
        with hands_module.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        ) as hands:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                center = None

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    index_tip = hand_landmarks.landmark[8]
                    center = (int(index_tip.x * width), int(index_tip.y * height))
                    cv2.circle(frame, center, 8, (0, 255, 0), -1)

                    if is_index_only(hand_landmarks):
                        if prev_center is not None:
                            cv2.line(canvas, prev_center, center, (0, 0, 255), 5)
                        prev_center = center
                    else:
                        prev_center = None
                else:
                    prev_center = None

                combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
                cv2.imshow("Virtual Whiteboard (c=clear, q=quit)", combined)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    canvas[:] = 0

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
