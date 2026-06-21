import cv2
import numpy as np
import os
import sys
import json

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR   = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR   = os.path.join(BASE_DIR, "trainer")
REGISTRY_FILE = os.path.join(BASE_DIR, "student_registry.json")


def load_registry():
    if not os.path.exists(REGISTRY_FILE):
        print("\n  [ERROR] student_registry.json not found.")
        print("  Run 1_capture_dataset.py first.\n")
        sys.exit(1)
    with open(REGISTRY_FILE, "r") as f:
        return json.load(f)


def train_group(dept, sem, batch, student_ids, registry):
    """Train one LBPH model for a single Dept/Sem/Batch group."""
    faces  = []
    labels = []

    for sid in student_ids:
        folder = os.path.join(DATASET_DIR, dept, str(sem), batch, str(sid))
        if not os.path.isdir(folder):
            print(f"    [WARN] No dataset folder found for ID {sid}. Skipping.")
            continue

        img_files = [f for f in os.listdir(folder)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not img_files:
            print(f"    [WARN] No images for ID {sid}. Skipping.")
            continue

        loaded = 0
        for img_file in img_files:
            img = cv2.imread(os.path.join(folder, img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))
            faces.append(img)
            labels.append(int(sid))
            loaded += 1

        name = registry.get(str(sid), {}).get("name", f"ID_{sid}")
        print(f"    ID {sid:<5} {name:<22} — {loaded} images loaded")

    if not faces:
        print(f"    [SKIP] No images found for {dept} Sem-{sem} Batch-{batch}.")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    recognizer.train(
        [np.array(f, dtype=np.uint8) for f in faces],
        np.array(labels, dtype=np.int32)
    )

    save_dir = os.path.join(TRAINER_DIR, dept, str(sem), batch)
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "trainer.yml")
    recognizer.write(model_path)

    print(f"    Model saved → {model_path}")
    return True


def main():
    print("\n" + "="*58)
    print("  LBPH MODEL TRAINING — Face Attendance System v2")
    print("="*58)

    if not os.path.isdir(DATASET_DIR):
        print("\n  [ERROR] dataset/ folder not found.")
        print("  Run 1_capture_dataset.py first.\n")
        sys.exit(1)

    registry = load_registry()

    # ── Group students by (dept, semester, batch) ──────────────────────────────
    groups = {}   # key: (dept, sem, batch) → list of student IDs
    for sid, info in registry.items():
        key = (info["department"], str(info["semester"]), info["batch"])
        groups.setdefault(key, []).append(sid)

    if not groups:
        print("\n  [ERROR] Registry is empty. Run 1_capture_dataset.py first.\n")
        sys.exit(1)

    print(f"\n  Found {len(groups)} class group(s) to train.\n")

    trained = 0
    for (dept, sem, batch), sids in sorted(groups.items()):
        print(f"  ── Training: {dept} | Sem-{sem} | Batch-{batch} "
              f"({len(sids)} student(s)) ──")
        ok = train_group(dept, int(sem), batch, sids, registry)
        if ok:
            trained += 1
        print()

    print("="*58)
    print(f"  TRAINING COMPLETE — {trained}/{len(groups)} group(s) trained.")
    print("="*58)
    print("\n  Now run  3_take_attendance.py  to mark attendance.\n")


if __name__ == "__main__":
    main()