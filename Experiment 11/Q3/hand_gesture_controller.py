"""Hand Gesture Desktop Controller
===================================

This script uses **MediaPipe Hands** to detect hand landmarks from a webcam feed (you can point a mobile phone at your hand and stream it to the PC – for a quick prototype a normal webcam works too).

Detected finger counts are mapped to common desktop actions via **pyautogui**:

- **5 fingers (open palm)**   → Increase master volume
- **0 fingers (fist)**        → Mute / Unmute toggle
- **2 fingers**               → Next slide (right‑arrow key)
- **3 fingers**               → Previous slide (left‑arrow key)
- **1 finger (point)**        → Scroll down
- **4 fingers**               → Scroll up

The mapping follows the *default* option you selected.

Requirements
------------
```bash
pip install -r requirements.txt
```

Usage
-----
```bash
python hand_gesture_controller.py
```
Press **q** in the video window to quit.

Implementation notes
--------------------
* MediaPipe provides 21 hand landmarks. Finger joint angles and direction relative
  to the palm are used to avoid counting curled fingers as extended.
* The thumb is checked for straight joints and separation from the index finger.
* Up to two hands are tracked and annotated independently.
* PyAutoGUI performs the system actions.  Volume control works on Windows via
  the `keyboard` library (which PyAutoGUI calls under the hood).  If you need more
  precise control you can replace the helper functions.
"""

import cv2
import math
import mediapipe as mp
import pyautogui
from pathlib import Path
import sys
import time

# MediaPipe 1.x removed the legacy Solutions API used by this controller.
if not hasattr(mp, 'solutions'):
    requirements = Path(__file__).with_name('requirements.txt')
    raise SystemExit(
        f'MediaPipe {getattr(mp, "__version__", "unknown")} does not provide '
        f'mp.solutions in the active interpreter:\n{sys.executable}\n'
        'Install this project\'s compatible dependencies with:\n'
        f'"{sys.executable}" -m pip install -r "{requirements}"'
    )

# ---------------------------------------------------------------------------
# Helper functions for system actions
# ---------------------------------------------------------------------------

def increase_volume(step: int = 5):
    """Increase the master volume by ``step`` percent.
    On Windows ``pyautogui.press('volumeup')`` sends a single volume‑up key press.
    We repeat it to achieve a noticeable change.
    """
    for _ in range(step // 2):  # each press ~2% on many systems
        pyautogui.press('volumeup')
        time.sleep(0.01)


def toggle_mute():
    """Toggle mute / unmute. """
    pyautogui.press('volumemute')


def next_slide():
    pyautogui.press('right')


def previous_slide():
    pyautogui.press('left')


def scroll(amount: int):
    """Scroll vertically. Positive ``amount`` scrolls up, negative down."""
    pyautogui.scroll(amount)

# ---------------------------------------------------------------------------
# Hand detection and finger counting
# ---------------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Indices of landmarks for each finger's base, joints, and tip
FINGER_LANDMARKS = [
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
]


def _joint_angle(first, joint, last) -> float:
    """Return the 2D angle at ``joint`` in degrees."""
    first_vector = (first.x - joint.x, first.y - joint.y)
    last_vector = (last.x - joint.x, last.y - joint.y)
    first_length = math.hypot(*first_vector)
    last_length = math.hypot(*last_vector)
    if first_length == 0 or last_length == 0:
        return 0

    cosine = sum(a * b for a, b in zip(first_vector, last_vector))
    cosine /= first_length * last_length
    return math.degrees(math.acos(max(-1, min(1, cosine))))


def count_extended_fingers(hand_landmarks) -> int:
    """Count fingers that are straight and extend away from the palm."""
    landmarks = hand_landmarks.landmark
    count = 0

    # The thumb moves sideways, so check that it is straight and spread from
    # the index-finger base rather than comparing x-coordinates.
    palm_length = math.hypot(
        landmarks[9].x - landmarks[0].x,
        landmarks[9].y - landmarks[0].y,
    )
    thumb_spread = math.hypot(
        landmarks[4].x - landmarks[5].x,
        landmarks[4].y - landmarks[5].y,
    )
    if (
        _joint_angle(landmarks[1], landmarks[2], landmarks[3]) >= 145
        and _joint_angle(landmarks[2], landmarks[3], landmarks[4]) >= 145
        and thumb_spread > palm_length * 0.35
    ):
        count += 1

    palm_axis = (landmarks[9].x - landmarks[0].x, landmarks[9].y - landmarks[0].y)
    palm_axis_length = math.hypot(*palm_axis)
    for mcp_idx, pip_idx, dip_idx, tip_idx in FINGER_LANDMARKS:
        finger_axis = (
            landmarks[tip_idx].x - landmarks[mcp_idx].x,
            landmarks[tip_idx].y - landmarks[mcp_idx].y,
        )
        finger_axis_length = math.hypot(*finger_axis)
        aligned_with_palm = False
        if palm_axis_length and finger_axis_length:
            alignment = sum(a * b for a, b in zip(palm_axis, finger_axis))
            aligned_with_palm = alignment / (palm_axis_length * finger_axis_length) > 0.45

        if (
            _joint_angle(landmarks[mcp_idx], landmarks[pip_idx], landmarks[dip_idx]) >= 155
            and _joint_angle(landmarks[pip_idx], landmarks[dip_idx], landmarks[tip_idx]) >= 145
            and aligned_with_palm
        ):
            count += 1
    return count

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def open_camera(index: int = 0):
    """Open the webcam using Windows capture backends."""
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def main():
    cap = open_camera()
    if cap is None:
        print(
            'Error: Could not open webcam (camera index 0). '
            'Close apps using the camera and check Windows camera permissions.'
        )
        return

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        previous_counts = {}
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Error: Webcam opened but no frame could be read.')
                break

            # Flip for a selfie view and convert to RGB
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            current_counts = {}
            triggered_counts = set()
            hand_landmarks_list = result.multi_hand_landmarks or []
            handedness_list = result.multi_handedness or []
            for index, hand_landmarks in enumerate(hand_landmarks_list):
                label = f'Hand {index + 1}'
                if index < len(handedness_list):
                    label = handedness_list[index].classification[0].label
                current_counts[label] = count_extended_fingers(hand_landmarks)
                count = current_counts[label]

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                )
                wrist = hand_landmarks.landmark[0]
                position = (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))
                cv2.putText(
                    frame,
                    f'{label}: {count}',
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if count != previous_counts.get(label) and count not in triggered_counts:
                    print(f'{label} finger count: {count}')
                    if count == 5:
                        increase_volume()
                    elif count == 0:
                        toggle_mute()
                    elif count == 2:
                        next_slide()
                    elif count == 3:
                        previous_slide()
                    elif count == 1:
                        scroll(-30)  # scroll down
                    elif count == 4:
                        scroll(30)   # scroll up
                    triggered_counts.add(count)
            previous_counts = current_counts

            # Show the annotated frame
            cv2.putText(
                frame,
                ' | '.join(f'{label}: {count}' for label, count in current_counts.items())
                or 'No hands detected',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow('Hand Gesture Controller (press q to quit)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
