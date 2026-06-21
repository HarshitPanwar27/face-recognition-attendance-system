import cv2
import os
import json

from courses import COURSES, list_departments, list_semesters

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, "dataset")
REGISTRY_FILE = os.path.join(BASE_DIR, "student_registry.json")

SAMPLES_PER_STUDENT = 100

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
GREEN  = (0, 210, 0)
RED    = (0, 0, 210)
WHITE  = (255, 255, 255)
YELLOW = (0, 215, 255)
GRAY   = (160, 160, 160)

# ── Registry helpers ───────────────────────────────────────────────────────────

def load_registry():
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}


def save_registry(registry):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=4)


# ── Terminal menu helpers ──────────────────────────────────────────────────────

def pick_from_list(prompt, options):
    """Show a numbered list and return the chosen item."""
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


# ── Capture helpers ────────────────────────────────────────────────────────────

def draw_progress_bar(frame, count, total):
    h = frame.shape[0]
    cv2.rectangle(frame, (10, h - 40), (630, h - 15), (50, 50, 50), -1)
    fill = int(610 * count / total)
    cv2.rectangle(frame, (10, h - 40), (10 + fill, h - 15), GREEN, -1)
    cv2.putText(frame, f"{count}/{total}", (635, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*58)
    print("  STUDENT FACE CAPTURE — Face Attendance System v2")
    print("="*58)

    registry = load_registry()

    # ── Student ID ─────────────────────────────────────────────────────────────
    while True:
        try:
            sid = int(input("\n  Enter Student ID (numeric, e.g. 1): ").strip())
            break
        except ValueError:
            print("  [!] Please enter a valid numeric ID.")

    if str(sid) in registry:
        print(f"\n  [WARN] Student ID {sid} already exists in registry:")
        s = registry[str(sid)]
        print(f"         {s['name']} | {s['department']} Sem-{s['semester']} Batch-{s['batch']}")
        overwrite = input("  Re-capture this student? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("  Skipping. Run again for a different student ID.\n")
            return

    # ── Name ───────────────────────────────────────────────────────────────────
    name = input("  Enter Student Name         : ").strip()
    if not name:
        name = f"Student_{sid}"

    # ── Department ─────────────────────────────────────────────────────────────
    departments = list_departments()
    department  = pick_from_list("Select Department:", departments)

    # ── Semester ───────────────────────────────────────────────────────────────
    semesters = list_semesters()
    semester  = int(pick_from_list("Select Semester:", [str(s) for s in semesters]))

    # ── Batch ──────────────────────────────────────────────────────────────────
    batch = input("  Enter Batch (e.g. A / B / ECE6A): ").strip().upper()
    if not batch:
        batch = "A"

    # ── Confirm ────────────────────────────────────────────────────────────────
    print("\n  ── Student Details ─────────────────────────────────")
    print(f"     ID         : {sid}")
    print(f"     Name       : {name}")
    print(f"     Department : {department}")
    print(f"     Semester   : {semester}")
    print(f"     Batch      : {batch}")
    print("  ────────────────────────────────────────────────────")
    confirm = input("\n  Confirm and start capture? (y/n): ").strip().lower()
    if confirm != 'y':
        print("  Cancelled.\n")
        return

    # ── Save registry entry first ──────────────────────────────────────────────
    registry[str(sid)] = {
        "name":       name,
        "department": department,
        "semester":   semester,
        "batch":      batch,
    }
    save_registry(registry)

    # ── Prepare dataset folder ─────────────────────────────────────────────────
    # Structure: dataset/<DEPT>/<SEM>/<BATCH>/<student_id>/
    student_dir = os.path.join(
        DATASET_DIR, department, str(semester), batch, str(sid)
    )
    os.makedirs(student_dir, exist_ok=True)

    # ── Load face detector ─────────────────────────────────────────────────────
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        print(f"\n  [ERROR] Haar cascade not found at: {cascade_path}")
        return

    # ── Open webcam ────────────────────────────────────────────────────────────
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("\n  [ERROR] Cannot open webcam. Check camera connection.")
        return

    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    count    = 0
    frame_no = 0

    print(f"\n  [INFO] Webcam open. Look into the camera.")
    print(f"  [INFO] Capturing {SAMPLES_PER_STUDENT} images for '{name}'...")
    print(f"  [INFO] Press 'q' to quit early.\n")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        frame_no += 1
        display = frame.copy()
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            if count < SAMPLES_PER_STUDENT and frame_no % 2 == 0:
                count += 1
                face_img = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                cv2.imwrite(os.path.join(student_dir, f"{count}.jpg"), face_img)

            cv2.rectangle(display, (x, y), (x+w, y+h), GREEN, 2)
            cv2.putText(display, name, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1, cv2.LINE_AA)

        # HUD
        cv2.putText(display, f"Capturing: {name}  [{department} | Sem-{semester} | Batch-{batch}]",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, YELLOW, 1, cv2.LINE_AA)
        cv2.putText(display, f"Samples: {count}/{SAMPLES_PER_STUDENT}",
                    (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 1, cv2.LINE_AA)
        cv2.putText(display, "Press 'q' to quit",
                    (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1, cv2.LINE_AA)

        draw_progress_bar(display, count, SAMPLES_PER_STUDENT)

        if count >= SAMPLES_PER_STUDENT:
            cv2.putText(display, "  CAPTURE COMPLETE! Close this window.",
                        (60, display.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2, cv2.LINE_AA)

        cv2.imshow(f"Capturing — {name}", display)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= SAMPLES_PER_STUDENT:
            break

    cam.release()
    cv2.destroyAllWindows()

    print(f"\n  [OK] Captured {count} samples for '{name}'")
    print(f"       Saved to: {student_dir}")
    print(f"\n  Run this script again for the next student.")
    print("  After ALL students are captured → run  2_train_model.py\n")


if __name__ == "__main__":
    main()