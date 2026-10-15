# Smart Attendance System (Q4)

## Overview
A Python‑based attendance system that recognises faces from a video stream (e.g., a phone camera placed at an entryway) and marks attendance automatically.

## How it works
1. **Student registry** – `students.csv` lists each person’s name and a reference image path.
2. **Encoding** – At start‑up the script loads each image, computes a 128‑dim face embedding using the `face_recognition` library, and stores the encodings.
3. **Recognition** – For every video frame faces are detected, encoded and compared against the known list.
4. **Logging** – When a match is found the person’s name is displayed and an entry is appended to `attendance_log.csv` (once per day per person).
5. **Extensible** – Add a new student by appending a row to `students.csv` and placing the image in the `known_faces/` directory – no code changes required.

## Folder layout
```
Q4/
│   attendance_system.py   # main script
│   students.csv            # name, image_path
│   attendance_log.csv      # generated log (date, time, name)
└─ known_faces/             # reference images for each student
```

## Setup
```powershell
cd "e:\BTCS301p\Experiment 11\Q4"
pip install face_recognition opencv-python pandas
```
> **Note:** `face_recognition` depends on `dlib`. On Windows you may need a pre‑built wheel.

## Adding students
Edit `students.csv` (comma‑separated) with two columns:
```csv
name,image_path
Alice,known_faces/alice.jpg
Bob,known_faces/bob.png
```
Place the referenced images inside the `known_faces` folder.

On the first run, if `students.csv` is missing, the script creates it with the
required column headers and creates the `known_faces` folder. Add at least one
student and a clear face image, then run the script again.

## Running the system
```powershell
python attendance_system.py
```
A window shows the live video feed with recognized names overlayed. Press **`q`** to quit.

## Customisation
- Adjust `tolerance` in `face_recognition.compare_faces` for stricter/looser matching.
- Change logging format or add a database backend.
- Replace the video source with a phone stream URL (e.g., `http://192.168.1.42:8080/video`).
