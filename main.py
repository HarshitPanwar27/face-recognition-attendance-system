"""
========================================================
  FACE RECOGNITION ATTENDANCE SYSTEM  v3
  MAIN GUI APPLICATION — PyQt5
  Author : Harshit Panwar
========================================================

HOW TO RUN:
  python main_app.py

DEPENDENCIES:
  pip install PyQt5 opencv-contrib-python numpy matplotlib
"""

import sys
import os
import cv2
import threading
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QTextEdit, QProgressBar, QSplitter, QFrame,
    QSizePolicy, QScrollArea, QMessageBox
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize
)
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPalette, QIcon,
    QPainter, QLinearGradient, QBrush
)
from PyQt5.QtWidgets import QGraphicsDropShadowEffect

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import core
import courses
from styles import (
    MAIN_STYLE, SECTION_HEADER_STYLE, CARD_STYLE,
    STATUS_OK_STYLE, STATUS_ERR_STYLE, STATUS_WARN_STYLE, STATUS_INFO_STYLE,
    JIIT_BLUE, JIIT_GOLD, JIIT_BLUE_MID, JIIT_BLUE_LIGHT, JIIT_BLUE_GLOW,
    WHITE, LIGHT_GRAY, OFF_WHITE, DARK_TEXT, SUBTLE_TEXT, BORDER, MID_GRAY,
    SUCCESS_GREEN, ERROR_RED, INFO_BLUE
)

# ══════════════════════════════════════════════════════════════════════════════
#  WORKER THREADS
# ══════════════════════════════════════════════════════════════════════════════

class CaptureWorker(QThread):
    progress  = pyqtSignal(int, int, object)   # count, total, bgr_frame
    finished  = pyqtSignal(int)                # count captured
    error     = pyqtSignal(str)

    def __init__(self, sid, name, department, semester, batch):
        super().__init__()
        self.sid = sid; self.name = name
        self.department = department
        self.semester = semester; self.batch = batch
        self._stop = False

    def stop(self): self._stop = True

    def run(self):
        import cv2, numpy as np
        CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(CASCADE)
        if detector.empty():
            self.error.emit("Haar cascade not found.")
            return
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            self.error.emit("Cannot open webcam.")
            return
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        save_dir = core.dataset_path(self.department, self.semester, self.batch, self.sid)
        os.makedirs(save_dir, exist_ok=True)

        count = 0; frame_no = 0
        while count < core.SAMPLES_PER_STUDENT and not self._stop:
            ret, frame = cam.read()
            if not ret: continue
            frame_no += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(60,60))
            display = frame.copy()
            for (x,y,w,h) in faces:
                if frame_no % 2 == 0 and count < core.SAMPLES_PER_STUDENT:
                    count += 1
                    face_img = cv2.resize(gray[y:y+h,x:x+w], (200,200))
                    cv2.imwrite(os.path.join(save_dir, f"{count}.jpg"), face_img)
                cv2.rectangle(display,(x,y),(x+w,y+h),(0,200,0),2)
                cv2.putText(display, self.name,(x,y-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,215,255),1,cv2.LINE_AA)
            # HUD
            fh,fw = display.shape[:2]
            cv2.rectangle(display,(0,0),(fw,30),(25,25,25),-1)
            cv2.putText(display,
                f"Capturing: {self.name} [{self.department}|Sem-{self.semester}|{self.batch}]",
                (8,20),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,215,255),1,cv2.LINE_AA)
            cv2.rectangle(display,(8,fh-34),(fw-8,fh-12),(50,50,50),-1)
            fill = int((fw-16)*count/core.SAMPLES_PER_STUDENT)
            cv2.rectangle(display,(8,fh-34),(8+fill,fh-12),(0,200,0),-1)
            cv2.putText(display,f"{count}/{core.SAMPLES_PER_STUDENT}",
                (fw//2-20,fh-16),cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1,cv2.LINE_AA)
            self.progress.emit(count, core.SAMPLES_PER_STUDENT, display)
            self.msleep(10)
        cam.release()
        self.finished.emit(count)


class TrainWorker(QThread):
    log      = pyqtSignal(str)
    finished = pyqtSignal(int)
    error    = pyqtSignal(str)

    def run(self):
        core.train_all(
            on_log=lambda m: self.log.emit(m),
            on_done=lambda n: self.finished.emit(n),
            on_error=lambda e: self.error.emit(e),
        )


class AttendanceWorker(QThread):
    frame_ready  = pyqtSignal(object)          # bgr frame
    marked       = pyqtSignal(str, str, object)# sid, name, datetime
    wrong_class  = pyqtSignal(str, dict)       # name, info
    finished     = pyqtSignal(dict, str)       # present_ids, filepath

    def __init__(self, dept, sem, batch, subject):
        super().__init__()
        self.dept = dept; self.sem = sem
        self.batch = batch; self.subject = subject
        self._stop = False

    def stop(self): self._stop = True

    def run(self):
        present, fp = core.run_attendance_session(
            self.dept, self.sem, self.batch, self.subject,
            on_frame=lambda f: self.frame_ready.emit(f),
            on_marked=lambda s,n,t: self.marked.emit(s,n,t),
            on_wrong_class=lambda n,i: self.wrong_class.emit(n,i),
            stop_flag=lambda: self._stop,
        )
        if present is None:
            self.finished.emit({}, fp)   # fp holds error msg
        else:
            self.finished.emit(present, fp)


# ══════════════════════════════════════════════════════════════════════════════
#  REUSABLE WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

def make_label(text, bold=False, size=13, color=DARK_TEXT, muted=False):
    lbl = QLabel(text)
    c = MID_GRAY if muted else color
    lbl.setStyleSheet(
        f"color:{c}; font-size:{size}px;"
        + (" font-weight:700;" if bold else " font-weight:500;")
    )
    return lbl


def make_section_header(text):
    container = QWidget()
    container.setFixedHeight(42)
    lay = QHBoxLayout(container)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    # Left accent bar
    accent = QFrame()
    accent.setFixedWidth(5)
    accent.setStyleSheet(f"background:{JIIT_GOLD}; border-radius:3px; border:none;")
    lay.addWidget(accent)
    lay.addSpacing(0)
    lbl = QLabel(text)
    lbl.setStyleSheet(SECTION_HEADER_STYLE + "border-top-left-radius:0px; border-bottom-left-radius:0px;")
    lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    lay.addWidget(lbl, 1)
    return container


def make_card(shadow=True):
    w = QFrame()
    w.setStyleSheet(f"QFrame{{ {CARD_STYLE} }}")
    if shadow:
        eff = QGraphicsDropShadowEffect(w)
        eff.setBlurRadius(18)
        eff.setOffset(0, 3)
        eff.setColor(QColor(15, 23, 42, 30))
        w.setGraphicsEffect(eff)
    return w


def separator():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFixedHeight(1)
    line.setStyleSheet(f"background:{LIGHT_GRAY}; border:none; margin:2px 0;")
    return line


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER BANNER
# ══════════════════════════════════════════════════════════════════════════════

class HeaderBanner(QWidget):
    """
    Header painted manually via paintEvent so the gradient
    is guaranteed to render regardless of platform/theme.
    """
    def __init__(self):
        super().__init__()
        self.setFixedHeight(96)
        # setAttribute ensures the widget paints its own background
        self.setAttribute(Qt.WA_StyledBackground, False)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(24, 0, 24, 0)
        lay.setSpacing(0)

        # ── Left: institution + title ──────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(4)
        left.setAlignment(Qt.AlignVCenter)

        inst = QLabel("JAYPEE INSTITUTE OF INFORMATION TECHNOLOGY, NOIDA")
        inst.setStyleSheet(
            f"color:{JIIT_GOLD}; font-size:10px; font-weight:800;"
            f"letter-spacing:1.2px; background:transparent;"
        )

        title = QLabel("Face Recognition Based Attendance Management System")
        title.setStyleSheet(
            f"color:{WHITE}; font-size:16px; font-weight:700;"
            f"letter-spacing:0.2px; background:transparent;"
        )

        meta = QLabel("Minor Project-2  •  Course: 15B19EC691  •  Even Semester 2026")
        meta.setStyleSheet(
            f"color:rgba(255,255,255,160); font-size:11px; background:transparent;"
        )

        left.addWidget(inst)
        left.addWidget(title)
        left.addWidget(meta)
        lay.addLayout(left, 1)

        # ── Centre divider ─────────────────────────────────────────────────
        div = QFrame()
        div.setFixedSize(1, 56)
        div.setStyleSheet("background:rgba(255,255,255,60); border:none;")
        lay.addSpacing(24)
        lay.addWidget(div)
        lay.addSpacing(24)

        # ── Right: student info ────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(5)
        right.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        name_badge = QLabel("  🎓  Harshit Panwar   |   23102202  ")
        name_badge.setStyleSheet(
            f"color:{JIIT_BLUE}; background:{JIIT_GOLD};"
            f"font-size:12px; font-weight:800; border-radius:6px;"
            f"padding:5px 12px; letter-spacing:0.3px;"
        )
        name_badge.setAlignment(Qt.AlignCenter)

        sup = QLabel("Supervisor: Prof. Shamim Akhter")
        sup.setStyleSheet(
            f"color:rgba(255,255,255,160); font-size:11px; background:transparent;"
        )
        sup.setAlignment(Qt.AlignRight)

        right.addWidget(name_badge)
        right.addWidget(sup)
        lay.addLayout(right)

    def paintEvent(self, event):
        """Paint navy-to-blue gradient + gold bottom stripe."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Main gradient background
        grad = QLinearGradient(0, 0, self.width(), 0)
        grad.setColorAt(0.0,  QColor("#1A3C6E"))
        grad.setColorAt(0.55, QColor("#1E4D8C"))
        grad.setColorAt(1.0,  QColor("#2557A7"))
        painter.fillRect(0, 0, self.width(), self.height(), QBrush(grad))

        # Gold bottom stripe (4 px)
        painter.fillRect(0, self.height() - 4, self.width(), 4, QColor("#E8A000"))

        painter.end()
        super().paintEvent(event)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — CAPTURE DATASET
# ══════════════════════════════════════════════════════════════════════════════

class CaptureTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10); root.setContentsMargins(16,14,16,14)

        root.addWidget(make_section_header("  📷  Student Registration & Face Capture"))

        content = QHBoxLayout()
        content.setSpacing(14)

        # ── Left: form ────────────────────────────────────────────────────────
        form_card = make_card()
        form_card.setFixedWidth(320)
        fl = QVBoxLayout(form_card)
        fl.setContentsMargins(18,18,18,18); fl.setSpacing(12)

        fl.addWidget(make_label("Student Details", bold=True, size=14, color=JIIT_BLUE))
        fl.addWidget(separator())

        self.sid_edit   = QLineEdit(); self.sid_edit.setPlaceholderText("e.g. 1")
        self.name_edit  = QLineEdit(); self.name_edit.setPlaceholderText("Full name")

        self.dept_combo = QComboBox()
        self.dept_combo.addItems(courses.DEPARTMENTS)
        self.dept_combo.currentTextChanged.connect(self._update_batches)

        self.sem_combo  = QComboBox()
        self.sem_combo.addItems([str(s) for s in courses.SEMESTERS])

        self.batch_combo = QComboBox()
        self._update_batches(self.dept_combo.currentText())

        form = QFormLayout(); form.setSpacing(10)
        for lbl, w in [
            ("Student ID *", self.sid_edit),
            ("Full Name *",  self.name_edit),
            ("Department",   self.dept_combo),
            ("Semester",     self.sem_combo),
            ("Batch",        self.batch_combo),
        ]:
            form.addRow(make_label(lbl), w)
        fl.addLayout(form)

        fl.addSpacing(8)
        self.capture_btn = QPushButton("▶  Start Capture")
        self.capture_btn.clicked.connect(self._start_capture)
        fl.addWidget(self.capture_btn)

        self.stop_btn = QPushButton("⏹  Stop Capture")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_capture)
        fl.addWidget(self.stop_btn)

        fl.addSpacing(4)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, core.SAMPLES_PER_STUDENT)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Samples: %v / %m")
        fl.addWidget(self.progress_bar)

        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet(STATUS_INFO_STYLE)
        self.status_lbl.setWordWrap(True)
        fl.addWidget(self.status_lbl)

        fl.addStretch()

        # Registry summary
        fl.addWidget(separator())
        fl.addWidget(make_label("Registered Students", bold=True, color=JIIT_BLUE))
        self.reg_table = QTableWidget(0, 4)
        self.reg_table.setHorizontalHeaderLabels(["ID", "Name", "Dept", "Batch"])
        self.reg_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.reg_table.setAlternatingRowColors(True)
        self.reg_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.reg_table.setFixedHeight(180)
        fl.addWidget(self.reg_table)
        self._refresh_registry()

        content.addWidget(form_card)

        # ── Right: webcam feed ────────────────────────────────────────────────
        cam_card = make_card()
        cl = QVBoxLayout(cam_card)
        cl.setContentsMargins(12,12,12,12)
        cl.addWidget(make_label("Live Camera Feed", bold=True, color=JIIT_BLUE))
        self.cam_label = QLabel()
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setMinimumSize(520, 390)
        self.cam_label.setStyleSheet(
            f"background:#0D1B2A; border-radius:12px; color:#4A6FA5; font-size:15px; font-weight:600; border: 2px solid #1A3C6E;")
        self.cam_label.setText("\n\n📷  Camera feed will appear here\n  after clicking  \u25b6  Start Capture")
        cl.addWidget(self.cam_label, 1)
        content.addWidget(cam_card, 1)

        root.addLayout(content, 1)

    def _update_batches(self, dept):
        self.batch_combo.clear()
        self.batch_combo.addItems(courses.get_batches(dept))

    def _refresh_registry(self):
        reg = core.load_registry()
        self.reg_table.setRowCount(0)
        for sid, info in sorted(reg.items(), key=lambda x: int(x[0])):
            r = self.reg_table.rowCount()
            self.reg_table.insertRow(r)
            for c, val in enumerate([sid, info["name"],
                                      info["department"],
                                      info["batch"]]):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                self.reg_table.setItem(r, c, item)

    def _start_capture(self):
        sid_txt = self.sid_edit.text().strip()
        name    = self.name_edit.text().strip()
        dept    = self.dept_combo.currentText()
        sem     = int(self.sem_combo.currentText())
        batch   = self.batch_combo.currentText()

        if not sid_txt or not name:
            self.status_lbl.setText("⚠ Student ID and Name are required.")
            self.status_lbl.setStyleSheet(STATUS_WARN_STYLE)
            return
        try:
            sid = int(sid_txt)
        except ValueError:
            self.status_lbl.setText("⚠ Student ID must be numeric.")
            self.status_lbl.setStyleSheet(STATUS_WARN_STYLE)
            return

        core.add_student(sid, name, dept, sem, batch)

        self.progress_bar.setValue(0)
        self.capture_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_lbl.setText(f"Capturing for {name}...")
        self.status_lbl.setStyleSheet(STATUS_INFO_STYLE)

        self.worker = CaptureWorker(sid, name, dept, sem, batch)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _stop_capture(self):
        if self.worker:
            self.worker.stop()

    def _on_progress(self, count, total, frame_bgr):
        self.progress_bar.setValue(count)
        rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img   = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix   = QPixmap.fromImage(img).scaled(
            self.cam_label.width(), self.cam_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.cam_label.setPixmap(pix)

    def _on_done(self, count):
        self.capture_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_lbl.setText(f"✓ Captured {count} samples successfully!")
        self.status_lbl.setStyleSheet(STATUS_OK_STYLE)
        self._refresh_registry()

    def _on_error(self, msg):
        self.capture_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_lbl.setText(f"✗ Error: {msg}")
        self.status_lbl.setStyleSheet(STATUS_ERR_STYLE)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════

class TrainTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10); root.setContentsMargins(16,14,16,14)
        root.addWidget(make_section_header("  🧠  Train LBPH Face Recognition Model"))

        content = QHBoxLayout(); content.setSpacing(14)

        # ── Left: controls ────────────────────────────────────────────────────
        ctrl_card = make_card(); ctrl_card.setFixedWidth(300)
        cl = QVBoxLayout(ctrl_card)
        cl.setContentsMargins(18,18,18,18); cl.setSpacing(14)

        cl.addWidget(make_label("Model Training", bold=True, size=14, color=JIIT_BLUE))
        cl.addWidget(separator())

        info = QLabel(
            "Training reads all captured face images and builds\n"
            "an LBPH recognition model per class group.\n\n"
            "• Run AFTER capturing faces for all students.\n"
            "• Re-run if a new student is added.\n"
            "• One model per (Dept × Semester × Batch)."
        )
        info.setStyleSheet(f"color:{DARK_TEXT}; font-size:12px; background:transparent;")
        info.setWordWrap(True)
        cl.addWidget(info)

        cl.addStretch()
        self.train_btn = QPushButton("▶  Start Training")
        self.train_btn.setObjectName("success")
        self.train_btn.clicked.connect(self._start_train)
        cl.addWidget(self.train_btn)

        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 0)   # indeterminate
        self.train_progress.setVisible(False)
        cl.addWidget(self.train_progress)

        self.train_status = QLabel("Ready to train.")
        self.train_status.setStyleSheet(STATUS_INFO_STYLE)
        self.train_status.setWordWrap(True)
        cl.addWidget(self.train_status)
        cl.addStretch()

        # Trained models list
        cl.addWidget(separator())
        cl.addWidget(make_label("Trained Models", bold=True, color=JIIT_BLUE))
        self.model_table = QTableWidget(0, 3)
        self.model_table.setHorizontalHeaderLabels(["Dept", "Sem", "Batch"])
        self.model_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.model_table.setAlternatingRowColors(True)
        self.model_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.model_table.setFixedHeight(200)
        cl.addWidget(self.model_table)
        self._refresh_models()

        content.addWidget(ctrl_card)

        # ── Right: log ────────────────────────────────────────────────────────
        log_card = make_card()
        ll = QVBoxLayout(log_card)
        ll.setContentsMargins(12,12,12,12)
        ll.addWidget(make_label("Training Log", bold=True, color=JIIT_BLUE))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Training output will appear here...")
        ll.addWidget(self.log_box, 1)
        content.addWidget(log_card, 1)

        root.addLayout(content, 1)

    def _refresh_models(self):
        self.model_table.setRowCount(0)
        trainer_dir = core.TRAINER_DIR
        if not os.path.isdir(trainer_dir):
            return
        for dept in os.listdir(trainer_dir):
            dp = os.path.join(trainer_dir, dept)
            if not os.path.isdir(dp): continue
            for sem in os.listdir(dp):
                sp = os.path.join(dp, sem)
                if not os.path.isdir(sp): continue
                for batch in os.listdir(sp):
                    mp = os.path.join(sp, batch, "trainer.yml")
                    if os.path.exists(mp):
                        r = self.model_table.rowCount()
                        self.model_table.insertRow(r)
                        for c, val in enumerate([dept, sem, batch]):
                            item = QTableWidgetItem(val)
                            item.setTextAlignment(Qt.AlignCenter)
                            self.model_table.setItem(r, c, item)

    def _start_train(self):
        self.train_btn.setEnabled(False)
        self.train_progress.setVisible(True)
        self.log_box.clear()
        self.train_status.setText("Training in progress...")
        self.train_status.setStyleSheet(STATUS_INFO_STYLE)

        self.worker = TrainWorker()
        self.worker.log.connect(lambda m: self.log_box.append(m))
        self.worker.finished.connect(self._on_train_done)
        self.worker.error.connect(self._on_train_error)
        self.worker.start()

    def _on_train_done(self, n):
        self.train_btn.setEnabled(True)
        self.train_progress.setVisible(False)
        self.train_status.setText(f"✓ Training complete — {n} model(s) saved.")
        self.train_status.setStyleSheet(STATUS_OK_STYLE)
        self._refresh_models()

    def _on_train_error(self, msg):
        self.train_btn.setEnabled(True)
        self.train_progress.setVisible(False)
        self.train_status.setText(f"✗ {msg}")
        self.train_status.setStyleSheet(STATUS_ERR_STYLE)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — TAKE ATTENDANCE
# ══════════════════════════════════════════════════════════════════════════════

class AttendanceTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker      = None
        self.present_ids = {}
        self.session_filepath = ""
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10); root.setContentsMargins(16,14,16,14)
        root.addWidget(make_section_header("  ✅  Real-Time Batch Attendance Marking"))

        content = QHBoxLayout(); content.setSpacing(14)

        # ── Left: session setup + live list ──────────────────────────────────
        left_card = make_card(); left_card.setFixedWidth(330)
        ll = QVBoxLayout(left_card)
        ll.setContentsMargins(18,18,18,18); ll.setSpacing(10)

        ll.addWidget(make_label("Session Setup", bold=True, size=14, color=JIIT_BLUE))
        ll.addWidget(separator())

        self.att_dept_combo = QComboBox()
        self.att_dept_combo.addItems(courses.DEPARTMENTS)
        self.att_dept_combo.currentTextChanged.connect(self._on_dept_changed)

        self.att_sem_combo = QComboBox()
        self.att_sem_combo.addItems([str(s) for s in courses.SEMESTERS])
        self.att_sem_combo.currentTextChanged.connect(self._update_att_subjects)

        self.att_batch_combo = QComboBox()
        self._update_att_batches(self.att_dept_combo.currentText())

        self.att_subj_combo = QComboBox()
        self._update_att_subjects()

        form = QFormLayout(); form.setSpacing(8)
        for lbl, w in [
            ("Department", self.att_dept_combo),
            ("Semester",   self.att_sem_combo),
            ("Batch",      self.att_batch_combo),
            ("Subject",    self.att_subj_combo),
        ]:
            form.addRow(make_label(lbl), w)
        ll.addLayout(form)

        ll.addSpacing(6)
        self.start_att_btn = QPushButton("▶  Start Session")
        self.start_att_btn.setObjectName("success")
        self.start_att_btn.clicked.connect(self._start_session)
        ll.addWidget(self.start_att_btn)

        self.stop_att_btn = QPushButton("⏹  End Session & Save")
        self.stop_att_btn.setObjectName("danger")
        self.stop_att_btn.setEnabled(False)
        self.stop_att_btn.clicked.connect(self._stop_session)
        ll.addWidget(self.stop_att_btn)

        self.att_status = QLabel("Set up a session and click Start.")
        self.att_status.setStyleSheet(STATUS_INFO_STYLE)
        self.att_status.setWordWrap(True)
        ll.addWidget(self.att_status)

        ll.addWidget(separator())

        # Stats bar
        stats_row = QHBoxLayout()
        self.stat_total   = self._stat_box("Total", "0", JIIT_BLUE)
        self.stat_present = self._stat_box("Present", "0", "#1B8A4E")
        self.stat_absent  = self._stat_box("Absent", "0", "#C0392B")
        stats_row.addWidget(self.stat_total)
        stats_row.addWidget(self.stat_present)
        stats_row.addWidget(self.stat_absent)
        ll.addLayout(stats_row)

        ll.addWidget(make_label("Marked Present", bold=True, color=JIIT_BLUE))
        self.present_table = QTableWidget(0, 3)
        self.present_table.setHorizontalHeaderLabels(["ID", "Name", "Time"])
        self.present_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.present_table.setAlternatingRowColors(True)
        self.present_table.setEditTriggers(QTableWidget.NoEditTriggers)
        ll.addWidget(self.present_table, 1)

        content.addWidget(left_card)

        # ── Right: camera feed ────────────────────────────────────────────────
        cam_card = make_card()
        cl = QVBoxLayout(cam_card)
        cl.setContentsMargins(12,12,12,12)
        cl.addWidget(make_label("Live Recognition Feed", bold=True, color=JIIT_BLUE))
        self.att_cam_lbl = QLabel()
        self.att_cam_lbl.setAlignment(Qt.AlignCenter)
        self.att_cam_lbl.setMinimumSize(500, 400)
        self.att_cam_lbl.setStyleSheet(
            "background:#0D1B2A; border-radius:12px; color:#4A6FA5; font-size:15px; font-weight:600; border: 2px solid #1A3C6E;")
        self.att_cam_lbl.setText("\n\n🎥  Live recognition feed will appear here\n  after clicking  \u25b6  Start Session")
        cl.addWidget(self.att_cam_lbl, 1)
        content.addWidget(cam_card, 1)

        root.addLayout(content, 1)

    def _stat_box(self, label, value, color):
        box = QFrame()
        box.setStyleSheet(
            f"background:{color};"
            f"border-radius:12px;"
            f"border:none;"
        )
        eff = QGraphicsDropShadowEffect(box)
        eff.setBlurRadius(14); eff.setOffset(0,3)
        eff.setColor(QColor(15,23,42,40))
        box.setGraphicsEffect(eff)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10,10,10,10); lay.setSpacing(2)
        v_lbl = QLabel(value)
        v_lbl.setStyleSheet(
            f"color:white; font-size:26px; font-weight:800; background:transparent; letter-spacing:-0.5px;"
        )
        v_lbl.setAlignment(Qt.AlignCenter)
        t_lbl = QLabel(label.upper())
        t_lbl.setStyleSheet(
            f"color:rgba(255,255,255,0.75); font-size:10px; font-weight:700;"
            f"letter-spacing:1.2px; background:transparent;"
        )
        t_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(v_lbl); lay.addWidget(t_lbl)
        box.setFixedHeight(76)
        box._val_lbl = v_lbl
        return box

    def _update_att_batches(self, dept):
        self.att_batch_combo.clear()
        self.att_batch_combo.addItems(courses.get_batches(dept))

    def _on_dept_changed(self, dept):
        self._update_att_batches(dept)
        self._update_att_subjects()

    def _update_att_subjects(self):
        dept = self.att_dept_combo.currentText()
        sem  = int(self.att_sem_combo.currentText())

        self.att_subj_combo.clear()

        subjects = courses.get_subjects(dept, sem)

        if subjects:
            self.att_subj_combo.addItems(subjects)

    def _start_session(self):
        dept    = self.att_dept_combo.currentText()
        sem     = int(self.att_sem_combo.currentText())
        batch   = self.att_batch_combo.currentText()
        subject = self.att_subj_combo.currentText()

        if not subject:
            self.att_status.setText("⚠ No subjects found for this selection.")
            self.att_status.setStyleSheet(STATUS_WARN_STYLE)
            return

        batch_students = core.get_batch_students(dept, sem, batch)
        if not batch_students:
            self.att_status.setText(f"⚠ No students registered for {dept} Sem-{sem} {batch}.")
            self.att_status.setStyleSheet(STATUS_WARN_STYLE)
            return

        mp = core.model_path(dept, sem, batch)
        if not os.path.exists(mp):
            self.att_status.setText(f"⚠ No trained model for {dept} Sem-{sem} {batch}. Run training first.")
            self.att_status.setStyleSheet(STATUS_WARN_STYLE)
            return

        self.present_ids = {}
        self.present_table.setRowCount(0)
        self._update_stats(dept, sem, batch)

        self.start_att_btn.setEnabled(False)
        self.stop_att_btn.setEnabled(True)
        self.att_status.setText(f"Session running: {dept} Sem-{sem} {batch} | {subject}")
        self.att_status.setStyleSheet(STATUS_OK_STYLE)

        self.worker = AttendanceWorker(dept, sem, batch, subject)
        self.worker.frame_ready.connect(self._show_att_frame)
        self.worker.marked.connect(self._on_marked)
        self.worker.wrong_class.connect(self._on_wrong_class)
        self.worker.finished.connect(self._on_session_done)
        self.worker.start()

    def _stop_session(self):
        if self.worker:
            self.worker.stop()
        self.stop_att_btn.setEnabled(False)

    def _show_att_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.att_cam_lbl.width(), self.att_cam_lbl.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.att_cam_lbl.setPixmap(pix)

    def _on_marked(self, sid, name, ts):
        self.present_ids[sid] = ts
        r = self.present_table.rowCount()
        self.present_table.insertRow(r)
        for c, val in enumerate([sid, name, ts.strftime("%H:%M:%S")]):
            item = QTableWidgetItem(str(val))
            item.setTextAlignment(Qt.AlignCenter)
            item.setForeground(QColor("#1B8A4E"))
            self.present_table.setItem(r, c, item)
        self.present_table.scrollToBottom()
        dept  = self.att_dept_combo.currentText()
        sem   = int(self.att_sem_combo.currentText())
        batch = self.att_batch_combo.currentText()
        self._update_stats(dept, sem, batch)

    def _on_wrong_class(self, name, info):
        self.att_status.setText(
            f"⚠ {name} [{info.get('department','?')} "
            f"Sem-{info.get('semester','?')} {info.get('batch','?')}]"
            f" — NOT enrolled in this class."
        )
        self.att_status.setStyleSheet(STATUS_WARN_STYLE)

    def _update_stats(self, dept, sem, batch):
        total   = len(core.get_batch_students(dept, sem, batch))
        present = len(self.present_ids)
        absent  = total - present
        self.stat_total._val_lbl.setText(str(total))
        self.stat_present._val_lbl.setText(str(present))
        self.stat_absent._val_lbl.setText(str(absent))

    def _on_session_done(self, present_ids, filepath):
        self.start_att_btn.setEnabled(True)
        self.stop_att_btn.setEnabled(False)
        self.session_filepath = filepath
        if present_ids is not None and os.path.exists(filepath):
            self.att_status.setText(f"✓ Session saved → {os.path.basename(filepath)}")
            self.att_status.setStyleSheet(STATUS_OK_STYLE)
        else:
            self.att_status.setText(f"✗ {filepath}")
            self.att_status.setStyleSheet(STATUS_ERR_STYLE)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — REPORTS
# ══════════════════════════════════════════════════════════════════════════════

class ReportsTab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10); root.setContentsMargins(16,14,16,14)
        root.addWidget(make_section_header("  📊  Attendance Reports"))

        # Filter bar
        filter_card = make_card()
        filter_card.setFixedHeight(80)
        fl = QHBoxLayout(filter_card)
        fl.setContentsMargins(16,12,16,12); fl.setSpacing(14)

        fl.addWidget(make_label("Filter by:", bold=True))

        self.f_dept = QComboBox(); self.f_dept.addItem("All Departments")
        self.f_dept.addItems(courses.DEPARTMENTS)
        fl.addWidget(make_label("Dept:")); fl.addWidget(self.f_dept)

        self.f_batch = QComboBox(); self.f_batch.addItem("All Batches")
        for d in courses.DEPARTMENTS:
            self.f_batch.addItems(courses.get_batches(d))
        fl.addWidget(make_label("Batch:")); fl.addWidget(self.f_batch)

        fl.addStretch()
        refresh_btn = QPushButton("🔄  Refresh")
        refresh_btn.clicked.connect(self._load_sessions)
        fl.addWidget(refresh_btn)
        root.addWidget(filter_card)

        # Main split: session list | detail (table + chart)
        splitter = QSplitter(Qt.Horizontal)

        # ── Session list ──────────────────────────────────────────────────────
        sess_card = make_card()
        sl = QVBoxLayout(sess_card)
        sl.setContentsMargins(10,10,10,10)
        sl.addWidget(make_label("Sessions", bold=True, color=JIIT_BLUE))
        self.session_table = QTableWidget(0, 4)
        self.session_table.setHorizontalHeaderLabels(["Dept", "Batch", "Subject", "Date"])
        self.session_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.session_table.setAlternatingRowColors(True)
        self.session_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.session_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.session_table.itemSelectionChanged.connect(self._on_session_selected)
        sl.addWidget(self.session_table, 1)
        splitter.addWidget(sess_card)

        # ── Detail: table + chart ─────────────────────────────────────────────
        detail_card = make_card()
        dl = QVBoxLayout(detail_card)
        dl.setContentsMargins(10,10,10,10)

        dl.addWidget(make_label("Session Detail", bold=True, color=JIIT_BLUE))

        detail_split = QSplitter(Qt.Vertical)

        # Attendance table
        self.detail_table = QTableWidget(0, 5)
        self.detail_table.setHorizontalHeaderLabels(
            ["ID", "Name", "Dept", "Time", "Status"])
        self.detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.detail_table.setAlternatingRowColors(True)
        self.detail_table.setEditTriggers(QTableWidget.NoEditTriggers)
        detail_split.addWidget(self.detail_table)

        # Pie chart
        self.fig, self.ax = plt.subplots(figsize=(5, 3.2), facecolor="#F5F7FA")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(220)
        detail_split.addWidget(self.canvas)

        dl.addWidget(detail_split, 1)
        splitter.addWidget(detail_card)
        splitter.setSizes([340, 560])

        root.addWidget(splitter, 1)

        self._sessions_data = []
        self._load_sessions()

    def _load_sessions(self):
        dept  = self.f_dept.currentText()
        batch = self.f_batch.currentText()
        d_filter = None if dept  == "All Departments" else dept
        b_filter = None if batch == "All Batches"     else batch

        self._sessions_data = core.get_all_sessions(d_filter, b_filter)
        self.session_table.setRowCount(0)

        for sess in self._sessions_data:
            r = self.session_table.rowCount()
            self.session_table.insertRow(r)
            date_part = sess["filename"].replace(".csv", "").split("_")[0]
            for c, val in enumerate([sess["department"], sess["batch"],
                                      sess["subject"], date_part]):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                self.session_table.setItem(r, c, item)

    def _on_session_selected(self):
        rows = self.session_table.selectedItems()
        if not rows:
            return
        row_idx = self.session_table.currentRow()
        if row_idx >= len(self._sessions_data):
            return
        sess = self._sessions_data[row_idx]
        records = core.read_csv_records(sess["filepath"])
        self._populate_detail(records)

    def _populate_detail(self, records):
        self.detail_table.setRowCount(0)
        present_count = 0; absent_count = 0

        for rec in records:
            r = self.detail_table.rowCount()
            self.detail_table.insertRow(r)
            status = rec.get("Status", "")
            if status == "Present":
                present_count += 1
                fg = QColor("#1B8A4E")
            else:
                absent_count += 1
                fg = QColor("#C0392B")

            for c, key in enumerate(["Student_ID", "Student_Name",
                                      "Department", "Time", "Status"]):
                item = QTableWidgetItem(rec.get(key, ""))
                item.setTextAlignment(Qt.AlignCenter)
                item.setForeground(fg)
                self.detail_table.setItem(r, c, item)

        # Pie chart
        self.ax.clear()
        total = present_count + absent_count
        if total == 0:
            self.ax.text(0.5, 0.5, "No data", ha="center", va="center",
                         transform=self.ax.transAxes, fontsize=13, color="#8C99AE")
        else:
            sizes  = [present_count, absent_count]
            labels = [f"Present\n{present_count}", f"Absent\n{absent_count}"]
            colors = ["#1B8A4E", "#C0392B"]
            explode = (0.04, 0)
            wedges, texts, autotexts = self.ax.pie(
                sizes, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90,
                explode=explode, textprops={"fontsize": 11}
            )
            for at in autotexts:
                at.set_color("white"); at.set_fontweight("bold")
            self.ax.set_title(
                f"Attendance Summary  |  Total: {total}",
                fontsize=12, color=JIIT_BLUE, fontweight="bold", pad=10
            )
            self.ax.set_facecolor("#F5F7FA")

        self.fig.tight_layout()
        self.canvas.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Face Recognition Attendance System — JIIT Noida")
        self.setMinimumSize(1100, 720)
        self.setStyleSheet(MAIN_STYLE)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        # Header
        root.addWidget(HeaderBanner())

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setContentsMargins(8,8,8,8)

        icons = ["📷", "🧠", "✅", "📊"]
        tab_widgets = [CaptureTab(), TrainTab(), AttendanceTab(), ReportsTab()]
        tab_names   = ["Capture Dataset", "Train Model",
                       "Take Attendance", "Reports"]

        for icon, name, widget in zip(icons, tab_names, tab_widgets):
            self.tabs.addTab(widget, f"  {icon}  {name}  ")

        root.addWidget(self.tabs, 1)

        # Status bar
        self.statusBar().setStyleSheet(
            f"background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {JIIT_BLUE},stop:1 {JIIT_BLUE_MID});"
            f"color:rgba(255,255,255,0.85); font-size:12px; padding:4px 16px;"
            f"border-top: 1px solid {JIIT_BLUE_GLOW};"
        )
        self.statusBar().showMessage(
            f"  •  JIIT Face Attendance System v3"
            f"   │   Harshit Panwar (23102202)"
            f"   │   {datetime.now().strftime('%A, %d %B %Y')}"
            f"   │   Ready"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("JIIT Attendance System")
    app.setStyle("Fusion")

    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()