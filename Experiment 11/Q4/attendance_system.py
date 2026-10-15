r"""Smart Attendance System via Face Recognition
===================================================

Place a phone (or any webcam) at the entryway. The script captures the video
stream, detects faces, matches them against a local directory of known people
and automatically logs attendance.

**How it works**
-----------------
1. A CSV file (``students.csv``) stores the *name* of each person and the path
   to a reference image containing a clear frontal face.
2. At start‑up the script detects faces in the reference images and trains an
   OpenCV LBPH face recognizer.
3. For each video frame OpenCV detects faces and compares them with the
   trained recognizer.
4. When a face matches, the person’s name is displayed and an entry is
   appended to ``attendance_log.csv`` (once per day).
5. Adding a new student is as simple as appending a line to ``students.csv``
   and dropping the reference image into the same folder – the next run will
   pick it up automatically.

**Dependencies**
-----------------
```powershell
pip install -r requirements.txt
```
* ``opencv-contrib-python`` provides the LBPH face recognizer used here.

**Folder layout** (created under ``Q4``)
```
Q4/
│   attendance_system.py
│   students.csv            # list of known people
│   attendance_log.csv      # attendance records (auto‑generated)
└─ known_faces/             # place reference images here
```

**Running**
```powershell
cd "e:\BTCS301p\Experiment 11\Q4"
python attendance_system.py
```
Press **q** to quit the video window.

---
"""

import csv
import os
from datetime import datetime

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration – adjust paths if you change the layout
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STUDENTS_CSV = os.path.join(BASE_DIR, "students.csv")
ATTENDANCE_CSV = os.path.join(BASE_DIR, "attendance_log.csv")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")

# ---------------------------------------------------------------------------
# Helper: load known faces and compute encodings
# ---------------------------------------------------------------------------
def load_known_faces(face_detector):
    """Train and return an LBPH recognizer and its label-to-name mapping."""
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    if not os.path.exists(STUDENTS_CSV):
        with open(STUDENTS_CSV, "w", newline="", encoding="utf-8") as students_file:
            csv.writer(students_file).writerow(["name", "image_path"])
        print(f"Created student registry: {STUDENTS_CSV}")
        print(
            "Add students to this CSV using the columns 'name' and 'image_path', "
            "and place their face images in the known_faces folder."
        )
        return None, {}

    training_faces = []
    training_labels = []
    names_by_label = {}
    with open(STUDENTS_CSV, newline="", encoding="utf-8-sig") as students_file:
        reader = csv.DictReader(students_file)
        required_columns = {"name", "image_path"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "Students CSV must contain 'name' and 'image_path' columns."
            )

        for row_number, row in enumerate(reader, start=2):
            name = (row.get("name") or "").strip()
            image_path = (row.get("image_path") or "").strip()
            if not name or not image_path:
                print(f"Warning: skipping incomplete students.csv row {row_number}")
                continue

            img_path = os.path.join(BASE_DIR, image_path)
            if not os.path.isfile(img_path):
                print(f"Warning: image not found for {name}: {img_path}")
                continue
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: could not read image for {name}: {img_path}")
                continue
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                grayscale,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )
            if len(faces) == 0:
                print(f"Warning: no face detected in {img_path} for {name}")
                continue
            # A reference image can contain multiple faces; train only on the
            # largest one to avoid assigning someone else's face to this name.
            x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
            face = cv2.resize(grayscale[y:y + height, x:x + width], (200, 200))
            face = cv2.equalizeHist(face)
            label = next(
                (
                    existing_label
                    for existing_label, existing_name in names_by_label.items()
                    if existing_name == name
                ),
                len(names_by_label),
            )
            names_by_label[label] = name
            training_faces.append(face)
            training_labels.append(label)

    if not training_faces:
        return None, names_by_label

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(training_faces, np.array(training_labels, dtype=np.int32))
    return recognizer, names_by_label


def match_face(face_image, recognizer, names_by_label, confidence_threshold: float = 70.0):
    """Return the predicted name when the LBPH confidence is within threshold."""
    normalized_face = cv2.resize(face_image, (200, 200))
    normalized_face = cv2.equalizeHist(normalized_face)
    label, confidence = recognizer.predict(normalized_face)
    if confidence <= confidence_threshold:
        return names_by_label[label]
    return "Unknown"


# ---------------------------------------------------------------------------
# Helper: log attendance (once per day per person)
# ---------------------------------------------------------------------------
def log_attendance(name: str):
    """Append a row ``date, time, name`` to ``attendance_log.csv`` if the
    person hasn't been logged today yet.
    """
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    fieldnames = ["date", "time", "name"]

    with open(ATTENDANCE_CSV, "a+", newline="", encoding="utf-8") as log_file:
        log_file.seek(0)
        reader = csv.DictReader(log_file)
        if reader.fieldnames:
            if not set(fieldnames).issubset(reader.fieldnames):
                raise ValueError(
                    "Attendance log must contain 'date', 'time', and 'name' columns."
                )
            if any(
                row.get("date") == today_str and row.get("name") == name
                for row in reader
            ):
                return

        log_file.seek(0, os.SEEK_END)
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        if not reader.fieldnames:
            writer.writeheader()
        writer.writerow(
            {
                "date": today_str,
                "time": now.strftime("%H:%M:%S"),
                "name": name,
            }
        )
    print(f"Logged attendance: {name} at {now.time()}")

# ---------------------------------------------------------------------------
# Main loop – video capture, detection, matching, logging
# ---------------------------------------------------------------------------
def main():
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_detector.empty():
        raise RuntimeError("Could not load OpenCV's frontal-face cascade classifier.")

    recognizer, names_by_label = load_known_faces(face_detector)
    if recognizer is None:
        print("No known faces loaded – aborting.")
        return

    # Use default webcam (0). To use a phone stream replace with the URL, e.g.:
    # cap = cv2.VideoCapture('http://192.168.1.42:8080/video')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open video source.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: could not read a frame from the video source.")
                break
            # Resize for speed (optional)
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            grayscale = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            face_locations = face_detector.detectMultiScale(
                grayscale,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )

            for left, top, width, height in face_locations:
                name = match_face(
                    grayscale[top:top + height, left:left + width],
                    recognizer,
                    names_by_label,
                )
                left *= 2
                top *= 2
                width *= 2
                height *= 2
                right = left + width
                bottom = top + height
                if name != "Unknown":
                    log_attendance(name)

                # Draw rectangle and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(
                    frame,
                    name,
                    (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (0, 0, 0),
                    1,
                )

            cv2.imshow('Smart Attendance (press q to quit)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
