import cv2
import numpy as np
import os
import csv
import json
from datetime import datetime

from courses import get_subjects, get_batches, DEPARTMENTS


BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR   = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR   = os.path.join(BASE_DIR, "trainer")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")
REGISTRY_FILE  = os.path.join(BASE_DIR, "student_registry.json")
CASCADE_PATH   = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

SAMPLES_PER_STUDENT  = 100
CONFIDENCE_THRESHOLD = 65   # lower = stricter

CV_GREEN  = (0, 200, 0)
CV_RED    = (0, 0, 210)
CV_ORANGE = (0, 140, 255)
CV_WHITE  = (255, 255, 255)
CV_YELLOW = (0, 215, 255)
CV_GRAY   = (150, 150, 150)
CV_DARK   = (25, 25, 25)


def load_registry():
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}


def save_registry(registry):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=4)


def student_exists(sid: int):
    reg = load_registry()
    return str(sid) in reg


def add_student(sid: int, name: str, department: str, semester: int, batch: str):
    reg = load_registry()
    reg[str(sid)] = {
        "name":       name,
        "department": department,
        "semester":   int(semester),
        "batch":      batch,
    }
    save_registry(reg)


def get_all_students():
    return load_registry()


def get_batch_students(department, semester, batch):
    """Return list of student IDs in a specific dept/sem/batch."""
    reg = load_registry()
    return [
        sid for sid, info in reg.items()
        if info["department"] == department
        and int(info["semester"]) == int(semester)
        and info["batch"] == batch
    ]


def dataset_path(department, semester, batch, sid):
    return os.path.join(DATASET_DIR, department, str(semester), batch, str(sid))


def model_path(department, semester, batch):
    return os.path.join(TRAINER_DIR, department, str(semester), batch, "trainer.yml")


def attendance_path(department, batch, subject):
    """Returns directory: attendance/<dept>/<batch>/<subject_safe>/"""
    safe_subj = subject.replace(" ", "_").replace("/", "-").replace("&", "and")
    return os.path.join(ATTENDANCE_DIR, department, batch, safe_subj)



def capture_faces(sid: int, name: str, department: str, semester: int, batch: str,
                  on_progress=None, on_done=None, on_error=None):
    """
    Opens webcam, captures SAMPLES_PER_STUDENT face images.
    Callbacks (all optional):
      on_progress(count, total, frame_bgr)
      on_done(count)
      on_error(msg)
    Blocking — run in a QThread.
    """
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        if on_error:
            on_error("Haar cascade file not found.")
        return

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        if on_error:
            on_error("Cannot open webcam. Check camera connection.")
        return

    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    save_dir = dataset_path(department, semester, batch, sid)
    os.makedirs(save_dir, exist_ok=True)

    count    = 0
    frame_no = 0

    while count < SAMPLES_PER_STUDENT:
        ret, frame = cam.read()
        if not ret:
            continue

        frame_no += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
        )

        display = frame.copy()

        for (x, y, w, h) in faces:
            if frame_no % 2 == 0 and count < SAMPLES_PER_STUDENT:
                count += 1
                face_img = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                cv2.imwrite(os.path.join(save_dir, f"{count}.jpg"), face_img)

            cv2.rectangle(display, (x, y), (x+w, y+h), CV_GREEN, 2)
            cv2.putText(display, name, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, CV_YELLOW, 1, cv2.LINE_AA)

        # Progress bar
        _draw_progress(display, count, SAMPLES_PER_STUDENT, name, department, semester, batch)

        if on_progress:
            on_progress(count, SAMPLES_PER_STUDENT, display)

        # Safety: let thread be stopped externally via on_progress returning False
        if on_progress and on_progress(count, SAMPLES_PER_STUDENT, display) is False:
            break

    cam.release()

    if on_done:
        on_done(count)


def _draw_progress(frame, count, total, name, dept, sem, batch):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 32), CV_DARK, -1)
    cv2.putText(frame, f"Capturing: {name}  [{dept} | Sem-{sem} | Batch-{batch}]",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.50, CV_YELLOW, 1, cv2.LINE_AA)
    # progress bar
    cv2.rectangle(frame, (8, h - 36), (w - 8, h - 14), (50, 50, 50), -1)
    fill = int((w - 16) * count / total)
    cv2.rectangle(frame, (8, h - 36), (8 + fill, h - 14), CV_GREEN, -1)
    cv2.putText(frame, f"{count}/{total}", (w // 2 - 20, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, CV_WHITE, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_all(on_log=None, on_done=None, on_error=None):
    """
    Train one LBPH model per (dept, semester, batch) group.
    Callbacks: on_log(msg), on_done(groups_trained), on_error(msg)
    Blocking — run in a QThread.
    """
    reg = load_registry()
    if not reg:
        if on_error:
            on_error("No students in registry. Capture faces first.")
        return

    # Group students
    groups = {}
    for sid, info in reg.items():
        key = (info["department"], str(info["semester"]), info["batch"])
        groups.setdefault(key, []).append(sid)

    if on_log:
        on_log(f"Found {len(groups)} class group(s) to train.\n")

    trained = 0
    for (dept, sem, batch), sids in sorted(groups.items()):
        if on_log:
            on_log(f"── Training: {dept} | Sem-{sem} | Batch-{batch} ({len(sids)} student(s))")

        faces, labels = [], []
        for sid in sids:
            folder = dataset_path(dept, int(sem), batch, sid)
            if not os.path.isdir(folder):
                if on_log:
                    on_log(f"   [SKIP] No images for ID {sid}.")
                continue
            imgs = [f for f in os.listdir(folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            for img_f in imgs:
                img = cv2.imread(os.path.join(folder, img_f), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                faces.append(cv2.resize(img, (200, 200)))
                labels.append(int(sid))
            if on_log:
                on_log(f"   ID {sid}: {len(imgs)} images")

        if not faces:
            if on_log:
                on_log(f"   [SKIP] No images found.\n")
            continue

        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8
        )
        recognizer.train(
            [np.array(f, dtype=np.uint8) for f in faces],
            np.array(labels, dtype=np.int32)
        )

        mp = model_path(dept, int(sem), batch)
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        recognizer.write(mp)
        trained += 1
        if on_log:
            on_log(f"   Model saved → {mp}\n")

    if on_done:
        on_done(trained)


# ══════════════════════════════════════════════════════════════════════════════
#  ATTENDANCE SESSION
# ══════════════════════════════════════════════════════════════════════════════

def run_attendance_session(department, semester, batch, subject,
                           on_frame=None, on_marked=None,
                           on_wrong_class=None, stop_flag=None):
    """
    Opens webcam. Runs until stop_flag() returns True.
    Callbacks:
      on_frame(bgr_frame)               — every rendered frame
      on_marked(sid_str, name, time)    — when a student is first marked present
      on_wrong_class(name, info)        — student from wrong class detected
    Returns: (present_ids dict, filepath)
    """
    mp = model_path(department, semester, batch)
    if not os.path.exists(mp):
        return None, f"Model not found: {mp}"

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(mp)

    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        return None, "Haar cascade not found."

    reg = load_registry()
    batch_sids = set(int(s) for s in get_batch_students(department, semester, batch))
    all_registered_ids = set(int(s) for s in reg.keys())

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return None, "Cannot open webcam."

    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  820)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    session_start = datetime.now()
    present_ids   = {}   # sid_str → datetime
    flash_msgs    = {}   # sid_str → (text, colour, expire_frame)
    frame_count   = 0

    while True:
        if stop_flag and stop_flag():
            break

        ret, frame = cam.read()
        if not ret:
            continue

        frame_count += 1
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        faces = detector.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            sid_int, conf = recognizer.predict(face_roi)
            sid_str = str(sid_int)

            if conf >= CONFIDENCE_THRESHOLD:
                _face_box(frame, x, y, w, h, "Unknown",
                          f"Conf: {conf:.1f}", None, CV_RED)
                continue

            info = reg.get(sid_str, {})
            name = info.get("name", f"ID_{sid_int}")

            if sid_int not in batch_sids:
                s_dept = info.get("department", "?")
                s_sem  = info.get("semester", "?")
                s_bat  = info.get("batch", "?")
                _face_box(frame, x, y, w, h, name,
                          "NOT IN THIS CLASS",
                          f"{s_dept} Sem-{s_sem} {s_bat}", CV_ORANGE)
                if on_wrong_class:
                    on_wrong_class(name, info)
            else:
                if sid_str not in present_ids:
                    now = datetime.now()
                    present_ids[sid_str] = now
                    flash_msgs[sid_str] = (
                        f"MARKED PRESENT  {now.strftime('%H:%M:%S')}",
                        CV_GREEN, frame_count + 90
                    )
                    if on_marked:
                        on_marked(sid_str, name, now)

                if sid_str in flash_msgs and frame_count < flash_msgs[sid_str][2]:
                    badge = flash_msgs[sid_str][0]
                    box_c = CV_GREEN
                else:
                    badge = "ALREADY PRESENT"
                    box_c = (0, 140, 0)

                _face_box(frame, x, y, w, h, name,
                          f"ID:{sid_int}  Conf:{conf:.1f}", badge, box_c)

        # Top bar
        bar_w = frame.shape[1] - 230
        cv2.rectangle(frame, (0, 0), (bar_w, 30), CV_DARK, -1)
        cv2.putText(frame,
            f"{department} Sem-{semester} {batch}  |  {subject}"
            f"  |  {datetime.now().strftime('%H:%M:%S')}",
            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CV_WHITE, 1, cv2.LINE_AA)

        # Side panel
        _side_panel(frame, department, semester, batch, subject,
                    session_start, present_ids,
                    get_batch_students(department, semester, batch), reg)

        if on_frame:
            on_frame(frame.copy())

    cam.release()

    # Save CSV
    all_batch = get_batch_students(department, semester, batch)
    filepath = _save_csv(department, semester, batch, subject,
                         session_start, present_ids, all_batch, reg)

    return present_ids, filepath


def _face_box(frame, x, y, w, h, top, bot, badge, colour):
    cv2.rectangle(frame, (x, y), (x+w, y+h), colour, 2)
    ly1 = max(0, y - 40)
    ly2 = max(40, y)
    cv2.rectangle(frame, (x, ly1), (x+w, ly2), colour, -1)
    cv2.putText(frame, top, (x+4, ly1+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, CV_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, bot, (x+4, ly1+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, CV_WHITE, 1, cv2.LINE_AA)
    if badge:
        cv2.putText(frame, badge, (x+4, y+h-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, CV_YELLOW, 1, cv2.LINE_AA)


def _side_panel(frame, dept, sem, batch, subject,
                session_start, present_ids, all_batch, reg):
    h, w    = frame.shape[:2]
    pw      = 230
    x0      = w - pw + 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - pw, 0), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    present = len(present_ids)
    absent  = len(all_batch) - present

    lines = [
        ("SESSION INFO",          CV_YELLOW, 0.48, True),
        (f"Dept  : {dept}",       CV_WHITE,  0.40, False),
        (f"Sem   : {sem}",        CV_WHITE,  0.40, False),
        (f"Batch : {batch}",      CV_WHITE,  0.40, False),
        (f"Subj  : {subject[:16]}", CV_WHITE, 0.38, False),
        ("",                      CV_WHITE,  0.38, False),
        (f"Total  : {len(all_batch)}", CV_WHITE,  0.40, False),
        (f"Present: {present}",   CV_GREEN,  0.40, False),
        (f"Absent : {absent}",    CV_RED,    0.40, False),
        ("",                      CV_WHITE,  0.38, False),
        ("MARKED PRESENT",        CV_GREEN,  0.42, True),
    ]

    y = 20
    for text, col, scale, bold in lines:
        if text == "":
            y += 8
            continue
        thick = 2 if bold else 1
        cv2.putText(frame, text, (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)
        y += 20

    cv2.line(frame, (w - pw, y), (w, y), CV_GRAY, 1)
    y += 12

    for sid_str in list(reversed(list(present_ids.keys())))[:8]:
        info = reg.get(sid_str, {})
        name = info.get("name", sid_str)[:18]
        ts   = present_ids[sid_str].strftime("%H:%M")
        cv2.putText(frame, f"✓ {name}", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, CV_GREEN, 1, cv2.LINE_AA)
        y += 13
        cv2.putText(frame, f"  {ts}", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, CV_GRAY, 1, cv2.LINE_AA)
        y += 15
        if y > h - 44:
            break

    cv2.line(frame, (w - pw, h - 38), (w, h - 38), CV_GRAY, 1)
    cv2.putText(frame, "Stop button to end session",
                (x0, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.34, CV_YELLOW, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  CSV SAVING
# ══════════════════════════════════════════════════════════════════════════════

def _save_csv(dept, sem, batch, subject, session_start,
              present_ids, all_batch, reg):
    """
    Save attendance CSV in:
      attendance/<DEPT>/<BATCH>/<Subject>/<date>_<time>.csv
    Every batch student listed. Present with time, Absent with --:--:--.
    """
    date_str = session_start.strftime("%Y-%m-%d")
    time_str = session_start.strftime("%H-%M-%S")
    save_dir = attendance_path(dept, batch, subject)
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{date_str}_{time_str}.csv"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Student_ID", "Student_Name", "Department",
                         "Semester", "Batch", "Subject", "Date", "Time", "Status"])
        for sid in sorted(all_batch, key=lambda x: int(x)):
            info   = reg.get(str(sid), {})
            name   = info.get("name", f"ID_{sid}")
            if str(sid) in present_ids:
                ts     = present_ids[str(sid)].strftime("%H:%M:%S")
                status = "Present"
            else:
                ts     = "--:--:--"
                status = "Absent"
            writer.writerow([sid, name, dept, sem, batch, subject,
                             date_str, ts, status])
    return filepath


def read_csv_records(filepath):
    """Read a session CSV and return list of row dicts."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_all_sessions(department=None, batch=None, subject=None):
    """
    Walk attendance/ directory tree and return all session CSV paths.
    Optional filters: department, batch, subject.
    """
    sessions = []
    base = ATTENDANCE_DIR
    if not os.path.isdir(base):
        return sessions

    for dept_dir in os.listdir(base):
        if department and dept_dir != department:
            continue
        dept_path = os.path.join(base, dept_dir)
        if not os.path.isdir(dept_path):
            continue
        for bat_dir in os.listdir(dept_path):
            if batch and bat_dir != batch:
                continue
            bat_path = os.path.join(dept_path, bat_dir)
            if not os.path.isdir(bat_path):
                continue
            for subj_dir in os.listdir(bat_path):
                if subject and subj_dir.replace("_", " ") != subject:
                    continue
                subj_path = os.path.join(bat_path, subj_dir)
                if not os.path.isdir(subj_path):
                    continue
                for fname in os.listdir(subj_path):
                    if fname.endswith(".csv"):
                        sessions.append({
                            "department": dept_dir,
                            "batch":      bat_dir,
                            "subject":    subj_dir.replace("_", " "),
                            "filename":   fname,
                            "filepath":   os.path.join(subj_path, fname),
                        })
    return sorted(sessions, key=lambda x: x["filename"], reverse=True)