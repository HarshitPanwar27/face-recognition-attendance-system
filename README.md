# Face Recognition Attendance System

**Developed by Harshit Panwar**

## Files in This Project

| File | Purpose |
|---|---|
| `main.py` | **Run this** — PyQt5 GUI (sidebar navigation + Dashboard) |
| `core.py` | All backend logic (capture, train, attendance, CSV) |
| `courses.py` | Course catalog — edit to add/change subjects |
| `styles.py` | Theme constants (palette, fonts, component styles) |
| `requirements.txt` | Python dependencies |

---

## One-Time Setup

### 1. Install Python 3.8+
Download from https://python.org — check **"Add Python to PATH"**

### 2. Install all dependencies
```
pip install PyQt5 opencv-contrib-python numpy matplotlib
```

> **Important:** Use `opencv-contrib-python`, NOT `opencv-python`.
> If you already have `opencv-python`:
> ```
> pip uninstall opencv-python
> pip install opencv-contrib-python
> ```

### 3. Verify
```
python -c "import PyQt5, cv2, numpy, matplotlib; print('All OK'); print(cv2.face)"
```
Should print `All OK` and `<module 'cv2.face'>` with no errors.

### 4. Run
```
python main.py
```

---

## How to Use the GUI

The app opens with a **left sidebar** for navigation and a **Dashboard** as
the landing page. Click any sidebar item to switch pages — your place in
the workflow is always shown by the highlighted nav item.

### 🏠 Dashboard
The landing page. Shows at a glance:
- Total registered students, trained class groups, and attendance sessions
- A "Sessions Today" counter
- A table of the most recent attendance sessions
- A department breakdown chart
- Quick-action buttons that jump straight to Capture / Train / Attendance / Reports

### 📷 Capture Dataset
Register each student and capture their face images.

**Fill in:**
- Student ID (unique number per student, e.g. 1–10)
- Full Name
- Department: `ECE` or `CSE`
- Semester: 1–8
- Batch:
  - ECE → `A1`, `A2`, or `A3`
  - CSE → `B1`, `B2`, or `B3`

Click **Start Capture** → webcam opens → student looks at camera → 100 photos captured automatically.

Repeat for every student in the batch.

---

### 🧠 Train Model
Click **Start Training** after all students are captured.

- Trains one LBPH model per class group (Dept × Semester × Batch)
- Live training log shown on screen
- Re-run anytime a new student is added

---

### ✅ Take Attendance
Run this at the start of every class.

1. Select Department, Semester, Batch
2. Select Subject from the dropdown (auto-populated from course catalog)
3. Click **Start Session**
4. Students walk past the webcam one by one

**What appears on screen:**

| Box Colour | Meaning |
|---|---|
| 🟢 Green | Student recognised — MARKED PRESENT |
| Dark green | Student already marked this session |
| 🟠 Orange | Student from wrong class — NOT ENROLLED |
| 🔴 Red | Face not recognised — Unknown |

Click **End Session & Save** when done.

---

### 📊 Reports
View all past sessions with:
- Full attendance table (Present / Absent per student)
- Pie chart showing present vs absent breakdown
- Filter by Department and Batch

---

## Attendance Folder Structure

```
attendance/
└── ECE/
    └── A1/
        └── VLSI_Design/
            └── 2026-04-25_09-15-00.csv
└── CSE/
    └── B2/
        └── Cloud_Computing/
            └── 2026-04-25_10-00-00.csv
```

Each CSV contains every student in the batch:
- Present students → time of recognition
- Absent students → `--:--:--`

---

## Batches

| Department | Batches |
|---|---|
| ECE | A1, A2, A3 |
| CSE | B1, B2, B3 |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named PyQt5` | `pip install PyQt5` |
| `No module named cv2.face` | `pip install opencv-contrib-python` |
| Webcam not opening | Check Windows Settings → Privacy → Camera |
| Too many Unknown detections | Raise `CONFIDENCE_THRESHOLD` in `core.py` (try 75) |
| Wrong students recognised | Lower `CONFIDENCE_THRESHOLD` in `core.py` (try 55) |
| No subjects in dropdown | Check `courses.py` for that dept/semester |
| Model not found error | Run the Train Model page first |
| Black window on startup | Update graphics drivers or try `app.setStyle("Windows")` in `main.py` |

---

## Quick Start Commands

```bash
# Install everything
pip install PyQt5 opencv-contrib-python numpy matplotlib

# Launch the application
python main.py
```