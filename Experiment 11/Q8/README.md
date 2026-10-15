# Real-Time OCR Text Reader (Q8)

## Overview
A script that captures video from a webcam (or a phone camera stream) and runs Tesseract OCR on each frame. Recognised text is overlaid on the video feed in real time.

## Folder Structure
```
Q8/
│   ocr_reader.py
│   README.md
│   requirements.txt
```

## Setup
1. **Install Python dependencies**
   ```powershell
   cd "e:\\BTCS301p\\Experiment 11\\Q8"
   pip install -r requirements.txt
   ```
2. **Install Tesseract OCR Engine** (Windows)
   - Download the installer from the UB‑Mannheim repository: https://github.com/UB-Mannheim/tesseract/wiki
   - Run the installer and make sure the installation folder (e.g., `C:\Program Files\Tesseract-OCR`) is added to your system `PATH`.
   - If it is not on `PATH`, set `TESSERACT_CMD` to the executable before running:
     ```powershell
     $env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
     ```
   - Verify the installation by running `tesseract --version` in a command prompt.

## Usage
```powershell
python ocr_reader.py
```
- A window shows the live video with the first few recognised lines displayed on top.
- Press **`q`** to quit.

## Customisation
- **Region of Interest (ROI):** Modify the `preprocess` function or select a sub‑region of `frame` before passing it to Tesseract to improve speed.
- **Language Support:** Install additional language packs for Tesseract and specify them via `pytesseract.image_to_string(..., lang='eng+spa')`.
- **Performance:** Reduce the resolution of `frame` or process every nth frame for higher frame rates.

Feel free to adapt the script for specific use‑cases such as translating signage, reading receipts, or assisting visually‑impaired users.
