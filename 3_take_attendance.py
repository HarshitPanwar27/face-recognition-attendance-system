import cv2
import numpy as np
import os
import csv
import json
from datetime import datetime

from courses import get_subjects, list_departments, list_semesters

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
TRAINER_DIR    = os.path.join(BASE_DIR, "trainer")
REGISTRY_FILE  = os.path.join(BASE_DIR, "student_registry.json")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")
CASCADE_PATH   = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ── Recognition tuning ─────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 65   # lower = stricter. Recommended: 55–75

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
GREEN   = (0, 200, 0)
RED     = (0, 0, 210)
ORANGE  = (0, 140, 255)
WHITE   = (255, 255, 255)
YELLOW  = (0, 215, 255)
GRAY    = (150, 150, 150)
DARK    = (25, 25, 25)
CYAN    = (255, 210, 0)

# ── Registry ───────────────────────────────────────────────────────────────────

def load_registry():
    if not os.path.exists(REGISTRY_FILE):
        print("\n  [ERROR] student_registry.json not found.")
        print("  Run 1_capture_dataset.py first.\n")
        raise SystemExit
    with open(REGISTRY_FILE, "r") as f:
        return json.load(f)

# ── Terminal menu helpers ──────────────────────────────────────────────────────

def pick_from_list(prompt, options):
    print(f"\n  {prompt}")
    for i, opt in enumerate(options, 1):
        print(f"    [{i}] {opt}")
    while True:
        try:
            choice = int(input("  Enter number: ").strip())
            if 1 <= choice <= len(options):
                return options[choice - 1]
            print(f"  [!] Enter a number between 1 and {len(options)}.")
        except ValueError:
            print("  [!] Please enter a valid number.")


# ── CSV ────────────────────────────────────────────────────────────────────────

def save_csv(dept, sem, batch, subject, session_start, present_ids, all_batch_students, registry):
    """
    Write ONE CSV for the session.
    Every student in the batch appears exactly once:
      - present_ids  → Status = Present  (with timestamp)
      - others       → Status = Absent
    """
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    date_str     = session_start.strftime("%Y-%m-%d")
    time_str     = session_start.strftime("%H-%M-%S")
    safe_subject = subject.replace(" ", "_").replace("/", "-").replace("&", "and")
    filename     = f"Attendance_{dept}_Sem{sem}_{batch}_{safe_subject}_{date_str}_{time_str}.csv"
    filepath     = os.path.join(ATTENDANCE_DIR, filename)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Student_ID", "Student_Name", "Department",
                         "Semester", "Batch", "Subject", "Date", "Time", "Status"])
        for sid in sorted(all_batch_students, key=lambda x: int(x)):
            info   = registry.get(str(sid), {})
            name   = info.get("name", f"ID_{sid}")
            if sid in present_ids:
                ts     = present_ids[sid]          # datetime object
                status = "Present"
                t_str  = ts.strftime("%H:%M:%S")
            else:
                status = "Absent"
                t_str  = "--:--:--"
            writer.writerow([sid, name, dept, sem, batch, subject,
                             date_str, t_str, status])
    return filepath


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(dept, sem, batch, subject, present_ids, all_batch_students, registry, filepath):
    print("\n" + "="*62)
    print(f"  ATTENDANCE SUMMARY")
    print(f"  {dept} | Semester {sem} | Batch {batch} | {subject}")
    print("="*62)
    print(f"  {'ID':<8} {'Name':<24} {'Status':<10} Time")
    print("  " + "-"*56)
    for sid in sorted(all_batch_students, key=lambda x: int(x)):
        name = registry.get(str(sid), {}).get("name", f"ID_{sid}")
        if sid in present_ids:
            ts   = present_ids[sid].strftime("%H:%M:%S")
            status = "Present"
        else:
            ts   = "--:--:--"
            status = "Absent"
        marker = "✓" if status == "Present" else "✗"
        print(f"  {marker} {sid:<7} {name:<24} {status:<10} {ts}")
    print("  " + "-"*56)
    total   = len(all_batch_students)
    present = len(present_ids)
    absent  = total - present
    print(f"  Present : {present}/{total}    Absent : {absent}/{total}")
    print(f"\n  CSV saved → {filepath}")
    print("="*62 + "\n")


# ── Side panel drawing ────────────────────────────────────────────────────────

def draw_side_panel(frame, dept, sem, batch, subject,
                    session_start, present_ids, all_batch_students, registry, fps):
    h, w    = frame.shape[:2]
    panel_w = 230
    x0      = w - panel_w + 8

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Header
    cv2.putText(frame, "SESSION INFO", (x0, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, YELLOW, 1, cv2.LINE_AA)
    cv2.line(frame, (w - panel_w, 28), (w, 28), YELLOW, 1)

    total   = len(all_batch_students)
    present = len(present_ids)
    absent  = total - present

    info_lines = [
        (f"Dept   : {dept}",    WHITE),
        (f"Sem    : {sem}",     WHITE),
        (f"Batch  : {batch}",   WHITE),
        (f"Subj   : {subject[:18]}", WHITE),
        (f"Time   : {session_start.strftime('%H:%M:%S')}", GRAY),
        (f"FPS    : {fps:.1f}", GRAY),
        ("", WHITE),
        (f"Total  : {total}",   WHITE),
        (f"Present: {present}", GREEN),
        (f"Absent : {absent}",  RED),
    ]

    y = 46
    for text, col in info_lines:
        cv2.putText(frame, text, (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
        y += 18

    cv2.line(frame, (w - panel_w, y + 2), (w, y + 2), GRAY, 1)
    y += 14
    cv2.putText(frame, "MARKED PRESENT", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREEN, 1, cv2.LINE_AA)
    y += 18

    # List present students (most recent first, up to 10)
    recent_present = list(reversed(list(present_ids.keys())))[:10]
    for sid in recent_present:
        name = registry.get(str(sid), {}).get("name", f"ID_{sid}")
        name_short = name[:20]
        ts   = present_ids[sid].strftime("%H:%M")
        cv2.putText(frame, f"✓ {name_short}", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, GREEN, 1, cv2.LINE_AA)
        y += 13
        cv2.putText(frame, f"  {ts}", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, GRAY, 1, cv2.LINE_AA)
        y += 16
        if y > h - 50:
            break

    # Controls
    cv2.line(frame, (w - panel_w, h - 44), (w, h - 44), GRAY, 1)
    cv2.putText(frame, "Q=Save & Quit", (x0, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, YELLOW, 1, cv2.LINE_AA)
    cv2.putText(frame, "S=Save Summary", (x0, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, YELLOW, 1, cv2.LINE_AA)


# ── Face label drawing ────────────────────────────────────────────────────────

def draw_face_label(frame, x, y, w, h, top_text, bot_text, badge_text, box_color):
    # Face rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

    # Label background above face
    label_h = 40
    ly1 = max(0, y - label_h)
    ly2 = max(label_h, y)
    cv2.rectangle(frame, (x, ly1), (x + w, ly2), box_color, -1)

    cv2.putText(frame, top_text, (x + 4, ly1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, bot_text, (x + 4, ly1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, WHITE, 1, cv2.LINE_AA)

    if badge_text:
        cv2.putText(frame, badge_text, (x + 4, y + h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*58)
    print("  BATCH ATTENDANCE SYSTEM — Face Attendance System v2")
    print("="*58)

    # ── Load registry ─────────────────────────────────────────────────────────
    registry = load_registry()
    if not registry:
        print("\n  [ERROR] No students registered. Run 1_capture_dataset.py.\n")
        return

    # ── Teacher selects session parameters ────────────────────────────────────
    depts = list_departments()
    dept  = pick_from_list("Select Department:", depts)

    sems = [str(s) for s in list_semesters()]
    sem  = int(pick_from_list("Select Semester:", sems))

    # Find all batches registered for this dept+sem
    batches_available = sorted(set(
        info["batch"]
        for info in registry.values()
        if info["department"] == dept and int(info["semester"]) == sem
    ))
    if not batches_available:
        print(f"\n  [ERROR] No students registered for {dept} Sem-{sem}.")
        print("  Run 1_capture_dataset.py to add students.\n")
        return

    batch = pick_from_list("Select Batch:", batches_available)

    # ── Subject selection ─────────────────────────────────────────────────────
    subjects = get_subjects(dept, sem)
    if not subjects:
        print(f"\n  [ERROR] No subjects found for {dept} Sem-{sem} in courses.py.\n")
        return
    subject = pick_from_list("Select Subject:", subjects)

    # ── Load trained model ────────────────────────────────────────────────────
    model_path = os.path.join(TRAINER_DIR, dept, str(sem), batch, "trainer.yml")
    if not os.path.exists(model_path):
        print(f"\n  [ERROR] Trained model not found for {dept} Sem-{sem} Batch-{batch}.")
        print(f"  Expected: {model_path}")
        print("  Run 2_train_model.py first.\n")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    # ── Collect all students in this batch ─────────────────────────────────────
    all_batch_students = [
        sid for sid, info in registry.items()
        if info["department"] == dept
        and int(info["semester"]) == sem
        and info["batch"] == batch
    ]

    if not all_batch_students:
        print(f"\n  [ERROR] No students found for {dept} Sem-{sem} Batch-{batch}.\n")
        return

    # ── Build set of ALL registered student IDs (any dept/sem/batch) ───────────
    # Used to detect wrong-class students
    all_registered_ids = set(int(sid) for sid in registry.keys())
    batch_student_ids  = set(int(sid) for sid in all_batch_students)

    # ── Load face detector ────────────────────────────────────────────────────
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        print(f"\n  [ERROR] Haar cascade not found: {CASCADE_PATH}\n")
        return

    # ── Open webcam ───────────────────────────────────────────────────────────
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("\n  [ERROR] Cannot open webcam.\n")
        return

    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  820)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    session_start = datetime.now()
    # present_ids: {student_id_str → datetime of marking}
    present_ids   = {}

    print(f"\n  ── Session Started ─────────────────────────────────")
    print(f"     {dept} | Semester {sem} | Batch {batch} | {subject}")
    print(f"     Students in batch : {len(all_batch_students)}")
    print(f"     Started at        : {session_start.strftime('%H:%M:%S')}")
    print(f"  ────────────────────────────────────────────────────")
    print("  [INFO] Students may walk past the camera.")
    print("  [INFO] Press 'q' to end session | 's' for mid-session summary\n")

    prev_time = datetime.now()
    fps       = 0.0

    # Per-face flash messages: sid → (message, colour, expire_frame)
    flash_msgs  = {}
    frame_count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        frame_count += 1
        now     = datetime.now()
        elapsed = (now - prev_time).total_seconds()
        fps     = 1.0 / elapsed if elapsed > 0 else fps
        prev_time = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = detector.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            sid_int, confidence = recognizer.predict(face_roi)
            sid_str = str(sid_int)

            if confidence >= CONFIDENCE_THRESHOLD:
                # ── Unknown face ───────────────────────────────────────────
                draw_face_label(frame, x, y, w, h,
                                "Unknown",
                                f"Conf: {confidence:.1f}",
                                None,
                                RED)
                continue

            # ── Recognised — check class membership ───────────────────────
            student_info = registry.get(sid_str, {})
            name = student_info.get("name", f"ID_{sid_int}")

            if sid_int not in batch_student_ids:
                # Student is registered but WRONG class
                s_dept = student_info.get("department", "?")
                s_sem  = student_info.get("semester",   "?")
                s_bat  = student_info.get("batch",      "?")
                draw_face_label(frame, x, y, w, h,
                                f"{name}",
                                f"NOT ENROLLED IN THIS CLASS",
                                f"{s_dept} Sem-{s_sem} Batch-{s_bat}",
                                ORANGE)
            else:
                # ── Correct class student ──────────────────────────────────
                if sid_str not in present_ids:
                    # First time seen → mark present
                    present_ids[sid_str] = now
                    ts_str = now.strftime("%H:%M:%S")
                    print(f"  [PRESENT] {name} (ID: {sid_int}) — {ts_str}")
                    flash_msgs[sid_str] = (
                        f"MARKED PRESENT  {ts_str}", GREEN, frame_count + 90
                    )

                # Badge text
                if sid_str in flash_msgs and frame_count < flash_msgs[sid_str][2]:
                    badge = flash_msgs[sid_str][0]
                    box_c = flash_msgs[sid_str][1]
                else:
                    badge = "ALREADY PRESENT"
                    box_c = (0, 150, 0)

                draw_face_label(frame, x, y, w, h,
                                name,
                                f"ID: {sid_int}  Conf: {confidence:.1f}",
                                badge,
                                box_c)

        # ── Top bar ───────────────────────────────────────────────────────
        bar_w = frame.shape[1] - 230
        cv2.rectangle(frame, (0, 0), (bar_w, 32), DARK, -1)
        cv2.putText(frame,
                    f"{dept} Sem-{sem} Batch-{batch}  |  {subject}"
                    f"  |  {now.strftime('%H:%M:%S')}",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.47, WHITE, 1, cv2.LINE_AA)

        # ── Side panel ────────────────────────────────────────────────────
        draw_side_panel(frame, dept, sem, batch, subject,
                        session_start, present_ids,
                        all_batch_students, registry, fps)

        cv2.imshow("Batch Attendance System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            fp = save_csv(dept, sem, batch, subject, session_start,
                          present_ids, all_batch_students, registry)
            print_summary(dept, sem, batch, subject,
                          present_ids, all_batch_students, registry, fp)

    cam.release()
    cv2.destroyAllWindows()

    # ── Final save ────────────────────────────────────────────────────────────
    filepath = save_csv(dept, sem, batch, subject, session_start,
                        present_ids, all_batch_students, registry)
    print_summary(dept, sem, batch, subject,
                  present_ids, all_batch_students, registry, filepath)


if __name__ == "__main__":
    main()