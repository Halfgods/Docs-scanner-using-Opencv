
<p align="center">
  <h1 align="center">📄 Interactive OpenCV Document Scanner</h1>
  <p align="center">
    Automatic document detection with manual corner adjustment and precise perspective correction.
  </p>
  <p align="center">
    <b>OpenCV · Geometry · Human-in-the-loop CV</b>
  </p>
</p>

---

## 🎥 Demo (Click to Play)
<video width="320" height="240" controls>
  <source src="final_video.mp4v" type="video/mp4">
</video>

---

## 🧭 What This Project Does

An end-to-end **document scanning pipeline** that combines:

- **Automatic corner detection**
- **Interactive manual correction**
- **Mathematically correct four-point perspective transform**

Designed with **robustness and debuggability** in mind—not just happy-path automation.

---

## 🔁 Demo Flow

<p align="center">
Load Image → Auto Detect → Adjust Corners → Perspective Warp → Final Scan
</p>

---

## 🚀 Features

### 🟢 Automatic Corner Detection
- Otsu Thresholding for clean binarization  
- Largest contour extraction  
- Iterative `approxPolyDP` for reliable 4-point detection  

---

### 🖱️ Manual Corner Adjustment (Human-in-the-Loop)
- Drag & reposition corners in real time  
- Visual feedback during interaction  
- Recovery from imperfect auto-detection  

---

### 📐 Perspective Transformation
- Strict geometric point ordering  
- Prevents bowtie / twisted warps  
- Produces a clean top-down scan  

---

### 🛡️ Fail-Safe Design
- Fallback to image-bound corners if detection fails  
- Pipeline never crashes silently  

---

## 🛠️ Tech Stack

<p align="center">

| Tool | Purpose |
|-----|--------|
| **Python 3** | Core language |
| **OpenCV** | Image processing & geometry |
| **NumPy** | Coordinate math |
| **Imutils** | Resizing & rotation |

</p>

---

## ⚙️ Installation

```bash
git clone https://github.com/Halfgods/Docs-scanner-using-Opencv.git
cd Docs-scanner-using-Opencv
pip install opencv-python numpy imutils
````

---

## ▶️ Usage

```bash
python main.py --image ./hello.png
```

---

## 🎮 Controls

<p align="center">

| Action       | Input             |
| ------------ | ----------------- |
| Drag corners | Left Click + Drag |
| Scan         | `s`               |
| Reset        | `r`               |
| Quit         | `q`               |

</p>

---

## 🧠 Engineering Highlights

### ① Correct Point Ordering (Critical)

Perspective transforms **require strict ordering**:

* Top-Left → smallest `(x + y)`
* Bottom-Right → largest `(x + y)`
* Top-Right → smallest `(x − y)`
* Bottom-Left → largest `(x − y)`

Prevents geometric distortion.

---

### ② Adaptive Polygon Approximation

Instead of fixed epsilon:

```python
for eps in np.linspace(0.01, 0.10, 10):
    approx = cv2.approxPolyDP(...)
```

Improves robustness across resolutions and noise levels.

---

### ③ Data Quality > Parameter Tuning

Focuses on:

* Clean thresholding
* External contours
* Area-based filtering

Rather than endless edge tuning.

---

### ④ Human-Correctable CV Pipelines

Fully automatic CV systems fail silently.

This design:

* Surfaces uncertainty
* Lets the user fix it
* Mirrors real production CV workflows

---

## 🗂️ Code Structure

```text
scanner.py
│
├── basic.py                # Its how we basically make a Docscanner
├── multiplemethods.py      # 4 multiple methods for finding the contours
├── transform.py            # A method to warp perspective
├── sortingcontours.py      # Sorting and naming them wrt to left-right , bottom to right and viceversa
└── main.py                 # Main/final logic
```

---

## 🔮 Future Improvements

* Semantic segmentation (MobileNet / U-Net)
* OCR integration (Tesseract / EasyOCR)
* Mobile camera distortion correction
* Batch scanning support

---

<p align="center">
  <b>Author</b><br>
  Halfgods<br>
  <i>Built as part of a deep dive into practical computer vision and geometry.</i>
</p>


