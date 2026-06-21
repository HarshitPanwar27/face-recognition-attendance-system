"""
========================================================
  FACE RECOGNITION ATTENDANCE SYSTEM  
  Developed by Harshit Panwar
========================================================

"""

import sys
import os
import cv2
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QTextEdit, QProgressBar, QSplitter, QFrame,
    QScrollArea, QButtonGroup
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal
)
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor
)
from PyQt5.QtWidgets import QGraphicsDropShadowEffect

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import core
import courses
from styles import (
    MAIN_STYLE, SECTION_HEADER_STYLE, CARD_STYLE,
    STATUS_OK_STYLE, STATUS_ERR_STYLE, STATUS_WARN_STYLE, STATUS_INFO_STYLE,
    PRIMARY_BLUE, ACCENT_GOLD, PRIMARY_BLUE_MID, PRIMARY_BLUE_LIGHT, PRIMARY_BLUE_GLOW,
    PRIMARY_DARK, WHITE, LIGHT_GRAY, OFF_WHITE, DARK_TEXT, SUBTLE_TEXT, BORDER, MID_GRAY,
    SUCCESS_GREEN, SUCCESS_BG, ERROR_RED, ERROR_BG, WARN_ORANGE, WARN_BG, INFO_BLUE,
    SIDEBAR_BG, SIDEBAR_BG_HOVER, SIDEBAR_BG_ACTIVE, SIDEBAR_TEXT, SIDEBAR_TEXT_MUTED,
    SIDEBAR_TEXT_ACTIVE, SIDEBAR_DIVIDER, SIDEBAR_BTN_STYLE, SIDEBAR_BTN_ACTIVE_STYLE,
    STAT_BLUE_BG, STAT_GREEN_BG, STAT_RED_BG, STAT_AMBER_BG,
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
#  ICONS  
# ══════════════════════════════════════════════════════════════════════════════

ICON_DASHBOARD = "▦"
ICON_CAPTURE   = "◎"
ICON_TRAIN     = "◈"
ICON_ATTEND    = "✓"
ICON_REPORTS   = "▤"
ICON_PLAY      = "▶"
ICON_STOP      = "■"
ICON_REFRESH   = "↻"
ICON_CAMERA    = "◎"
ICON_USERS     = "◍"
ICON_MODEL     = "◈"
ICON_CALENDAR  = "▤"
ICON_CLOCK     = "◷"
ICON_CHEVRON   = "›"
ICON_DOT       = "•"


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


def make_field_label(text):
    """Small uppercase caption label used above form fields."""
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        f"color:{SUBTLE_TEXT}; font-size:10.5px; font-weight:700;"
        f"letter-spacing:0.6px; background:transparent;"
    )
    return lbl


def make_section_header(icon, text, subtitle=None):
    """Page-level header: icon chip + title (+ optional subtitle)."""
    container = QWidget()
    lay = QHBoxLayout(container)
    lay.setContentsMargins(0, 0, 0, 4)
    lay.setSpacing(12)

    chip = QLabel(icon)
    chip.setFixedSize(40, 40)
    chip.setAlignment(Qt.AlignCenter)
    chip.setStyleSheet(
        f"background:{PRIMARY_BLUE_LIGHT}; color:{PRIMARY_BLUE};"
        f"border-radius:10px; font-size:18px; font-weight:700;"
    )
    lay.addWidget(chip)

    text_box = QVBoxLayout()
    text_box.setSpacing(1)
    title = QLabel(text)
    title.setStyleSheet(
        f"color:{DARK_TEXT}; font-size:18px; font-weight:800; background:transparent;"
        f"letter-spacing:-0.2px;"
    )
    text_box.addWidget(title)
    if subtitle:
        sub = QLabel(subtitle)
        sub.setStyleSheet(f"color:{SUBTLE_TEXT}; font-size:12px; background:transparent;")
        text_box.addWidget(sub)
    lay.addLayout(text_box, 1)
    return container


def make_card(shadow=True, padded=False):
    w = QFrame()
    w.setStyleSheet(f"QFrame{{ {CARD_STYLE} }}")
    if padded:
        w.setContentsMargins(0, 0, 0, 0)
    if shadow:
        eff = QGraphicsDropShadowEffect(w)
        eff.setBlurRadius(22)
        eff.setOffset(0, 4)
        eff.setColor(QColor(15, 23, 42, 22))
        w.setGraphicsEffect(eff)
    return w


def separator():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFixedHeight(1)
    line.setStyleSheet(f"background:{LIGHT_GRAY}; border:none; margin:2px 0;")
    return line


def vfield(label_text, widget):
    """Label-above-field layout block (modern SaaS form pattern)."""
    box = QVBoxLayout()
    box.setSpacing(5)
    box.addWidget(make_field_label(label_text))
    box.addWidget(widget)
    return box


class StatCard(QFrame):
    """
    Reusable dashboard / summary stat card.
    Used both on the Dashboard page and inside the Attendance tab,
    so the same visual stat language appears everywhere in the app.
    """
    def __init__(self, label, value, icon, accent_color, bg_color, parent=None, compact=False):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame{{ background:{WHITE}; border:1px solid {LIGHT_GRAY}; border-radius:14px; }}"
        )
        eff = QGraphicsDropShadowEffect(self)
        eff.setBlurRadius(18); eff.setOffset(0, 3)
        eff.setColor(QColor(15, 23, 42, 18))
        self.setGraphicsEffect(eff)

        pad = 12 if compact else 18
        lay = QHBoxLayout(self)
        lay.setContentsMargins(pad, 14 if compact else 16, pad, 14 if compact else 16)
        lay.setSpacing(10 if compact else 14)

        icon_size = 36 if compact else 44
        icon_chip = QLabel(icon)
        icon_chip.setFixedSize(icon_size, icon_size)
        icon_chip.setAlignment(Qt.AlignCenter)
        icon_chip.setStyleSheet(
            f"background:{bg_color}; color:{accent_color};"
            f"border-radius:{10 if compact else 12}px; font-size:{15 if compact else 18}px; font-weight:700;"
        )
        lay.addWidget(icon_chip)

        text_box = QVBoxLayout()
        text_box.setSpacing(2)
        self.value_lbl = QLabel(str(value))
        self.value_lbl.setStyleSheet(
            f"color:{DARK_TEXT}; font-size:{18 if compact else 23}px; font-weight:800; background:transparent;"
            f"letter-spacing:-0.4px;"
        )
        self.label_lbl = QLabel(label.upper())
        self.label_lbl.setStyleSheet(
            f"color:{SUBTLE_TEXT}; font-size:{9 if compact else 10}px; font-weight:700;"
            f"letter-spacing:0.3px; background:transparent;"
        )
        self.label_lbl.setWordWrap(compact)
        text_box.addWidget(self.value_lbl)
        text_box.addWidget(self.label_lbl)
        lay.addLayout(text_box, 1)

        self.setMinimumHeight(70 if compact else 78)
        self.setMinimumWidth(140 if compact else 220)

    def set_value(self, value):
        self.value_lbl.setText(str(value))


class CompactStatCard(QFrame):
    """Smaller stat tile used inline in the Attendance live-session panel."""
    def __init__(self, label, value, accent_color, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame{{ background:{accent_color}; border-radius:12px; border:none; }}"
        )
        eff = QGraphicsDropShadowEffect(self)
        eff.setBlurRadius(14); eff.setOffset(0, 3)
        eff.setColor(QColor(15, 23, 42, 35))
        self.setGraphicsEffect(eff)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(2)
        self.value_lbl = QLabel(str(value))
        self.value_lbl.setStyleSheet(
            "color:white; font-size:24px; font-weight:800; background:transparent; letter-spacing:-0.5px;"
        )
        self.value_lbl.setAlignment(Qt.AlignCenter)
        t_lbl = QLabel(label.upper())
        t_lbl.setStyleSheet(
            "color:rgba(255,255,255,0.85); font-size:10px; font-weight:700;"
            "letter-spacing:1.1px; background:transparent;"
        )
        t_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.value_lbl)
        lay.addWidget(t_lbl)
        self.setFixedHeight(74)

    def set_value(self, value):
        self.value_lbl.setText(str(value))


def status_pill(text, kind="info"):
    """Returns a styled QLabel pill for table cells (Present/Absent/etc)."""
    from styles import PILL_GREEN, PILL_RED, PILL_AMBER
    styles = {"ok": PILL_GREEN, "err": PILL_RED, "warn": PILL_AMBER}
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(styles.get(kind, PILL_GREEN))
    return lbl


class ClickableRow(QFrame):
    """
    A QFrame that behaves like a button but supports a rich internal
    layout cleanly (QPushButton reserves extra text-baseline space when
    given a child layout, which left a visible gap above the content).
    Used for the Dashboard 'Quick Actions' list rows.
    """
    clicked = pyqtSignal(int)

    def __init__(self, payload, parent=None):
        super().__init__(parent)
        self._payload = payload
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._payload)
        super().mousePressEvent(event)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

class Sidebar(QWidget):
    """
    Persistent left navigation rail for the main window.
    Emits navigate(index) when a nav item is clicked; MainWindow
    listens and switches the QStackedWidget page accordingly.
    """
    navigate = pyqtSignal(int)

    NAV_ITEMS = [
        (ICON_DASHBOARD, "Dashboard"),
        (ICON_CAPTURE,   "Capture Dataset"),
        (ICON_TRAIN,     "Train Model"),
        (ICON_ATTEND,    "Take Attendance"),
        (ICON_REPORTS,   "Reports"),
    ]

    def __init__(self):
        super().__init__()
        self.setObjectName("sidebar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedWidth(248)
        self._buttons = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Brand block ───────────────────────────────────────────────────
        brand = QWidget()
        brand.setAttribute(Qt.WA_StyledBackground, True)
        brand.setStyleSheet("background: transparent;")
        brand.setFixedHeight(72)
        bl = QVBoxLayout(brand)
        bl.setContentsMargins(14, 0, 10, 0)
        bl.setSpacing(0)

        mark_row = QHBoxLayout()
        mark_row.setSpacing(9)
        mark = QLabel("HP")
        mark.setFixedSize(32, 32)
        mark.setAlignment(Qt.AlignCenter)
        mark.setStyleSheet(
            f"background:{SIDEBAR_BG_HOVER}; color:{SIDEBAR_TEXT_ACTIVE};"
            f"border-radius:6px; border:1px solid {SIDEBAR_DIVIDER};"
            f"font-size:11px; font-weight:700; letter-spacing:0.5px;"
        )
        mark_row.addWidget(mark)

        name_box = QVBoxLayout()
        name_box.setSpacing(0)
        title = QLabel("Face Attendance System")
        title.setStyleSheet(
            f"color:{SIDEBAR_TEXT_ACTIVE}; font-size:12px; font-weight:700;"
            f"background:transparent;"
        )
        ver = QLabel("Attendance Management")
        ver.setStyleSheet(
            f"color:{SIDEBAR_TEXT_MUTED}; font-size:9px; font-weight:500;"
            f"background:transparent;"
        )
        name_box.addWidget(title)
        name_box.addWidget(ver)
        mark_row.addLayout(name_box, 1)
        bl.addLayout(mark_row)
        root.addWidget(brand)

        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background:{SIDEBAR_DIVIDER}; border:none;")
        root.addWidget(div)

        # ── Nav items ─────────────────────────────────────────────────────
        nav_box = QVBoxLayout()
        nav_box.setContentsMargins(14, 16, 14, 16)
        nav_box.setSpacing(4)

        label = QLabel("NAVIGATION")
        label.setStyleSheet(
            f"color:{SIDEBAR_TEXT_MUTED}; font-size:10px; font-weight:700;"
            f"letter-spacing:1px; background:transparent; padding-left:6px;"
        )
        nav_box.addWidget(label)
        nav_box.addSpacing(4)

        for i, (icon, label_text) in enumerate(self.NAV_ITEMS):
            btn = QPushButton(f"  {icon}      {label_text}")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(SIDEBAR_BTN_STYLE)
            btn.clicked.connect(lambda _, idx=i: self._on_click(idx))
            nav_box.addWidget(btn)
            self._buttons.append(btn)

        nav_box.addStretch()
        root.addLayout(nav_box, 1)

        self.set_active(0)

    def _on_click(self, idx):
        self.set_active(idx)
        self.navigate.emit(idx)

    def set_active(self, idx):
        for i, btn in enumerate(self._buttons):
            label_text = self.NAV_ITEMS[i][1]
            icon = self.NAV_ITEMS[i][0]
            btn.setStyleSheet(SIDEBAR_BTN_ACTIVE_STYLE if i == idx else SIDEBAR_BTN_STYLE)


# ══════════════════════════════════════════════════════════════════════════════
#  TOP BAR  
# ══════════════════════════════════════════════════════════════════════════════

class TopBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedHeight(60)
        self.setStyleSheet(f"background:{WHITE}; border-bottom:1px solid {LIGHT_GRAY};")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(28, 0, 28, 0)

        self.page_title = QLabel("Dashboard")
        self.page_title.setStyleSheet(
            f"color:{DARK_TEXT}; font-size:15px; font-weight:700; background:transparent;"
        )
        lay.addWidget(self.page_title)
        lay.addStretch()

    def set_title(self, text):
        self.page_title.setText(text)


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD PAGE 
# ══════════════════════════════════════════════════════════════════════════════

class DashboardPage(QWidget):
    """
    System overview page — shows registered student counts, trained
    class groups, recent attendance sessions, and a department
    breakdown chart, all pulled from the registry and attendance
    records on disk.
    """
    go_to_page = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 22, 28, 22)
        outer.setSpacing(18)

        outer.addWidget(make_section_header(
            ICON_DASHBOARD, "Dashboard",
            "Overview of students, trained models and attendance activity"
        ))

        # ── Stat cards row ───────────────────────────────────────────────
        stats_row = QHBoxLayout()
        stats_row.setSpacing(16)
        self.card_students = StatCard("Total Students", "0", ICON_USERS, PRIMARY_BLUE, PRIMARY_BLUE_LIGHT)
        self.card_models   = StatCard("Trained Models", "0", ICON_MODEL, "#7C3AED", "#EDE9FE")
        self.card_sessions = StatCard("Sessions Logged", "0", ICON_CALENDAR, SUCCESS_GREEN, SUCCESS_BG)
        self.card_last     = StatCard("Last Session", "—", ICON_CLOCK, WARN_ORANGE, WARN_BG)
        for c in (self.card_students, self.card_models, self.card_sessions, self.card_last):
            stats_row.addWidget(c)
        outer.addLayout(stats_row)

        # ── Quick actions + recent sessions ─────────────────────────────
        mid_row = QHBoxLayout()
        mid_row.setSpacing(16)

        # Quick actions card — 2x2 grid of action cards
        qa_card = make_card()
        qa_lay = QVBoxLayout(qa_card)
        qa_lay.setContentsMargins(20, 18, 20, 18)
        qa_lay.setSpacing(12)
        qa_lay.addWidget(make_label("Quick Actions", bold=True, size=13.5, color=DARK_TEXT))
        qa_lay.addWidget(separator())

        actions = [
            (ICON_CAPTURE, "Capture Dataset", "Register a student & collect face samples", 1),
            (ICON_TRAIN,   "Train Model", "Build LBPH models for all class groups", 2),
            (ICON_ATTEND,  "Take Attendance", "Mark attendance for a live class", 3),
            (ICON_REPORTS, "View Reports", "Browse sessions & attendance analytics", 4),
        ]
        grid = QGridLayout()
        grid.setSpacing(12)
        for i, (icon, title, desc, page_idx) in enumerate(actions):
            card = ClickableRow(page_idx)
            card.clicked.connect(self.go_to_page.emit)
            card.setStyleSheet(
                f"QFrame#qaRow{{ background:{OFF_WHITE}; border:1px solid {LIGHT_GRAY};"
                f"border-radius:10px; }}"
                f"QFrame#qaRow:hover{{ background:{PRIMARY_BLUE_LIGHT}; border:1px solid {PRIMARY_BLUE_GLOW}; }}"
            )
            card.setObjectName("qaRow")
            card.setMinimumHeight(96)
            clay = QVBoxLayout(card)
            clay.setContentsMargins(14, 14, 14, 14)
            clay.setSpacing(8)

            top_row = QHBoxLayout()
            ic = QLabel(icon)
            ic.setFixedSize(32, 32)
            ic.setAlignment(Qt.AlignCenter)
            ic.setStyleSheet(
                f"background:{WHITE}; color:{PRIMARY_BLUE}; border-radius:8px; font-size:14px; font-weight:700;"
            )
            top_row.addWidget(ic)
            top_row.addStretch()
            chev = QLabel(ICON_CHEVRON)
            chev.setStyleSheet(f"color:{MID_GRAY}; font-size:16px; background:transparent;")
            top_row.addWidget(chev)
            clay.addLayout(top_row)

            t = QLabel(title)
            t.setStyleSheet(f"color:{DARK_TEXT}; font-size:12.5px; font-weight:700; background:transparent;")
            d = QLabel(desc)
            d.setStyleSheet(f"color:{SUBTLE_TEXT}; font-size:10px; background:transparent;")
            d.setWordWrap(True)
            clay.addWidget(t)
            clay.addWidget(d)
            clay.addStretch()

            grid.addWidget(card, i // 2, i % 2)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        qa_lay.addLayout(grid)
        qa_lay.addStretch()

        mid_row.addWidget(qa_card, 1)

        # Recent sessions card
        rs_card = make_card()
        rs_lay = QVBoxLayout(rs_card)
        rs_lay.setContentsMargins(20, 18, 20, 18)
        rs_lay.setSpacing(10)
        rs_lay.addWidget(make_label("Recent Attendance Sessions", bold=True, size=13.5, color=DARK_TEXT))
        rs_lay.addWidget(separator())

        self.recent_table = QTableWidget(0, 4)
        self.recent_table.setHorizontalHeaderLabels(["Department", "Batch", "Subject", "Date"])
        self.recent_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.recent_table.setAlternatingRowColors(True)
        self.recent_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.recent_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.recent_table.verticalHeader().setVisible(False)
        self.recent_table.setMinimumHeight(220)
        rs_lay.addWidget(self.recent_table, 1)

        mid_row.addWidget(rs_card, 1)
        outer.addLayout(mid_row, 1)

    def refresh(self):
        """Re-pull all summary numbers — safe to call anytime (read-only)."""
        # Students
        registry = core.load_registry()
        self.card_students.set_value(len(registry))

        # Trained models — scan TRAINER_DIR exactly like TrainTab already does
        model_count = 0
        trainer_dir = core.TRAINER_DIR
        if os.path.isdir(trainer_dir):
            for dept in os.listdir(trainer_dir):
                dp = os.path.join(trainer_dir, dept)
                if not os.path.isdir(dp):
                    continue
                for sem in os.listdir(dp):
                    sp = os.path.join(dp, sem)
                    if not os.path.isdir(sp):
                        continue
                    for batch in os.listdir(sp):
                        mp = os.path.join(sp, batch, "trainer.yml")
                        if os.path.exists(mp):
                            model_count += 1
        self.card_models.set_value(model_count)

        # Sessions
        sessions = core.get_all_sessions()
        self.card_sessions.set_value(len(sessions))
        if sessions:
            date_part = sessions[0]["filename"].replace(".csv", "").split("_")[0]
            self.card_last.set_value(date_part)
        else:
            self.card_last.set_value("—")

        # Recent sessions table (top 8)
        self.recent_table.setRowCount(0)
        for sess in sessions[:8]:
            r = self.recent_table.rowCount()
            self.recent_table.insertRow(r)
            date_part = sess["filename"].replace(".csv", "").split("_")[0]
            for c, val in enumerate([sess["department"], sess["batch"], sess["subject"], date_part]):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                self.recent_table.setItem(r, c, item)


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA VIEWPORT 
# ══════════════════════════════════════════════════════════════════════════════

def make_camera_viewport(placeholder_icon, placeholder_text):
    lbl = QLabel()
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setMinimumSize(480, 380)
    lbl.setStyleSheet(
        f"background:#0B1220; border-radius:14px; color:#5B6B95; font-size:14px;"
        f"font-weight:600; border: 1px solid #1E293B;"
    )
    lbl.setText(f"\n\n{placeholder_icon}\n\n{placeholder_text}")
    return lbl


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — CAPTURE DATASET
# ══════════════════════════════════════════════════════════════════════════════

class CaptureTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(18); root.setContentsMargins(28, 22, 28, 22)

        root.addWidget(make_section_header(
            ICON_CAPTURE, "Capture Dataset",
            "Register a student and collect face samples for training"
        ))

        content = QHBoxLayout()
        content.setSpacing(18)

        # ── Left: form ────────────────────────────────────────────────────────
        form_scroll = QScrollArea()
        form_scroll.setWidgetResizable(True)
        form_scroll.setFixedWidth(340)
        form_scroll.setFrameShape(QFrame.NoFrame)
        form_scroll.setStyleSheet("QScrollArea{ background:transparent; border:none; }")

        form_card = make_card()
        fl = QVBoxLayout(form_card)
        fl.setContentsMargins(22, 20, 22, 20); fl.setSpacing(14)

        fl.addWidget(make_label("Student Details", bold=True, size=14, color=DARK_TEXT))
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

        form = QVBoxLayout(); form.setSpacing(12)
        form.addLayout(vfield("Student ID *", self.sid_edit))
        form.addLayout(vfield("Full Name *", self.name_edit))
        form.addLayout(vfield("Department", self.dept_combo))
        form.addLayout(vfield("Semester", self.sem_combo))
        form.addLayout(vfield("Batch", self.batch_combo))
        fl.addLayout(form)

        fl.addSpacing(4)
        self.capture_btn = QPushButton(f"{ICON_PLAY}   Start Capture")
        self.capture_btn.setCursor(Qt.PointingHandCursor)
        self.capture_btn.clicked.connect(self._start_capture)
        fl.addWidget(self.capture_btn)

        self.stop_btn = QPushButton(f"{ICON_STOP}   Stop Capture")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_capture)
        fl.addWidget(self.stop_btn)

        fl.addSpacing(2)
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
        reg_header = QHBoxLayout()
        reg_header.addWidget(make_label("Registered Students", bold=True, color=DARK_TEXT))
        reg_header.addStretch()
        self.reg_count_lbl = make_label("0", bold=True, color=PRIMARY_BLUE)
        reg_header.addWidget(self.reg_count_lbl)
        fl.addLayout(reg_header)

        self.reg_table = QTableWidget(0, 4)
        self.reg_table.setHorizontalHeaderLabels(["ID", "Name", "Dept", "Batch"])
        self.reg_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.reg_table.setAlternatingRowColors(True)
        self.reg_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.reg_table.verticalHeader().setVisible(False)
        self.reg_table.setFixedHeight(190)
        fl.addWidget(self.reg_table)
        self._refresh_registry()

        form_scroll.setWidget(form_card)
        content.addWidget(form_scroll)

        # ── Right: webcam feed ────────────────────────────────────────────────
        cam_card = make_card()
        cl = QVBoxLayout(cam_card)
        cl.setContentsMargins(18, 18, 18, 18); cl.setSpacing(12)
        cam_header = QHBoxLayout()
        cam_header.addWidget(make_label("Live Camera Feed", bold=True, color=DARK_TEXT))
        cam_header.addStretch()
        self.cam_badge = QLabel("● Idle")
        self.cam_badge.setStyleSheet(
            f"color:{MID_GRAY}; background:{OFF_WHITE}; border-radius:8px;"
            f"font-size:11px; font-weight:700; padding:4px 10px;"
        )
        cam_header.addWidget(self.cam_badge)
        cl.addLayout(cam_header)

        self.cam_label = make_camera_viewport(ICON_CAMERA, "Camera feed will appear here\nafter clicking Start Capture")
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
        self.reg_count_lbl.setText(str(len(reg)))

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
        self.cam_badge.setText("● Recording")
        self.cam_badge.setStyleSheet(
            f"color:{ERROR_RED}; background:{ERROR_BG}; border-radius:8px;"
            f"font-size:11px; font-weight:700; padding:4px 10px;"
        )

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
        self.cam_badge.setText("● Idle")
        self.cam_badge.setStyleSheet(
            f"color:{MID_GRAY}; background:{OFF_WHITE}; border-radius:8px;"
            f"font-size:11px; font-weight:700; padding:4px 10px;"
        )
        self._refresh_registry()

    def _on_error(self, msg):
        self.capture_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_lbl.setText(f"✗ Error: {msg}")
        self.status_lbl.setStyleSheet(STATUS_ERR_STYLE)
        self.cam_badge.setText("● Idle")
        self.cam_badge.setStyleSheet(
            f"color:{MID_GRAY}; background:{OFF_WHITE}; border-radius:8px;"
            f"font-size:11px; font-weight:700; padding:4px 10px;"
        )

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════

class TrainTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(18); root.setContentsMargins(28, 22, 28, 22)
        root.addWidget(make_section_header(
            ICON_TRAIN, "Train Model",
            "Build LBPH face-recognition models for every class group"
        ))

        content = QHBoxLayout(); content.setSpacing(18)

        # ── Left: controls ────────────────────────────────────────────────────
        ctrl_card = make_card(); ctrl_card.setFixedWidth(320)
        cl = QVBoxLayout(ctrl_card)
        cl.setContentsMargins(22, 20, 22, 20); cl.setSpacing(14)

        cl.addWidget(make_label("Model Training", bold=True, size=14, color=DARK_TEXT))
        cl.addWidget(separator())

        info_card = QFrame()
        info_card.setStyleSheet(
            f"background:{PRIMARY_BLUE_LIGHT}; border-radius:10px; border:none;"
        )
        info_lay = QVBoxLayout(info_card)
        info_lay.setContentsMargins(14, 12, 14, 12)
        info = QLabel(
            "Training reads all captured face images and builds "
            "an LBPH recognition model per class group.\n\n"
            "•  Run after capturing faces for all students\n"
            "•  Re-run if a new student is added\n"
            "•  One model per Department × Semester × Batch"
        )
        info.setStyleSheet(f"color:{PRIMARY_DARK}; font-size:12px; background:transparent;")
        info.setWordWrap(True)
        info_lay.addWidget(info)
        cl.addWidget(info_card)

        cl.addStretch()
        self.train_btn = QPushButton(f"{ICON_PLAY}   Start Training")
        self.train_btn.setObjectName("success")
        self.train_btn.setCursor(Qt.PointingHandCursor)
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
        cl.addWidget(make_label("Trained Models", bold=True, color=DARK_TEXT))
        self.model_table = QTableWidget(0, 3)
        self.model_table.setHorizontalHeaderLabels(["Dept", "Sem", "Batch"])
        self.model_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.model_table.setAlternatingRowColors(True)
        self.model_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.model_table.verticalHeader().setVisible(False)
        self.model_table.setFixedHeight(190)
        cl.addWidget(self.model_table)
        self._refresh_models()

        content.addWidget(ctrl_card)

        # ── Right: log ────────────────────────────────────────────────────────
        log_card = make_card()
        ll = QVBoxLayout(log_card)
        ll.setContentsMargins(18, 18, 18, 18); ll.setSpacing(10)
        ll.addWidget(make_label("Training Log", bold=True, color=DARK_TEXT))
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
#  PAGE 3 — TAKE ATTENDANCE
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
        root.setSpacing(18); root.setContentsMargins(28, 22, 28, 22)
        root.addWidget(make_section_header(
            ICON_ATTEND, "Take Attendance",
            "Run a live recognition session to mark a class as present"
        ))

        content = QHBoxLayout(); content.setSpacing(18)

        # ── Left: session setup + live list ──────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(350)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setStyleSheet("QScrollArea{ background:transparent; border:none; }")

        left_card = make_card(shadow=False)
        ll = QVBoxLayout(left_card)
        ll.setContentsMargins(22, 20, 22, 20); ll.setSpacing(12)

        ll.addWidget(make_label("Session Setup", bold=True, size=14, color=DARK_TEXT))
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

        form = QVBoxLayout(); form.setSpacing(10)
        form.addLayout(vfield("Department", self.att_dept_combo))
        form.addLayout(vfield("Semester", self.att_sem_combo))
        form.addLayout(vfield("Batch", self.att_batch_combo))
        form.addLayout(vfield("Subject", self.att_subj_combo))
        ll.addLayout(form)

        ll.addSpacing(4)
        self.start_att_btn = QPushButton(f"{ICON_PLAY}   Start Session")
        self.start_att_btn.setObjectName("success")
        self.start_att_btn.setCursor(Qt.PointingHandCursor)
        self.start_att_btn.clicked.connect(self._start_session)
        ll.addWidget(self.start_att_btn)

        self.stop_att_btn = QPushButton(f"{ICON_STOP}   End Session && Save")
        self.stop_att_btn.setObjectName("danger")
        self.stop_att_btn.setCursor(Qt.PointingHandCursor)
        self.stop_att_btn.setEnabled(False)
        self.stop_att_btn.clicked.connect(self._stop_session)
        ll.addWidget(self.stop_att_btn)

        self.att_status = QLabel("Set up a session and click Start.")
        self.att_status.setStyleSheet(STATUS_INFO_STYLE)
        self.att_status.setWordWrap(True)
        ll.addWidget(self.att_status)

        ll.addWidget(separator())

        # Stats bar — now built from the shared CompactStatCard component
        stats_row = QHBoxLayout()
        stats_row.setSpacing(8)
        self.stat_total   = CompactStatCard("Total", "0", PRIMARY_BLUE)
        self.stat_present = CompactStatCard("Present", "0", SUCCESS_GREEN)
        self.stat_absent  = CompactStatCard("Absent", "0", ERROR_RED)
        stats_row.addWidget(self.stat_total)
        stats_row.addWidget(self.stat_present)
        stats_row.addWidget(self.stat_absent)
        ll.addLayout(stats_row)

        ll.addWidget(make_label("Marked Present", bold=True, color=DARK_TEXT))
        self.present_table = QTableWidget(0, 3)
        self.present_table.setHorizontalHeaderLabels(["ID", "Name", "Time"])
        self.present_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.present_table.setAlternatingRowColors(True)
        self.present_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.present_table.verticalHeader().setVisible(False)
        self.present_table.setMinimumHeight(220)
        ll.addWidget(self.present_table, 1)

        left_scroll.setWidget(left_card)
        content.addWidget(left_scroll)

        # ── Right: camera feed ────────────────────────────────────────────────
        cam_card = make_card()
        cl = QVBoxLayout(cam_card)
        cl.setContentsMargins(18, 18, 18, 18); cl.setSpacing(12)
        cam_header = QHBoxLayout()
        cam_header.addWidget(make_label("Live Recognition Feed", bold=True, color=DARK_TEXT))
        cam_header.addStretch()
        self.att_cam_badge = QLabel("● Idle")
        self.att_cam_badge.setStyleSheet(
            f"color:{MID_GRAY}; background:{OFF_WHITE}; border-radius:8px;"
            f"font-size:11px; font-weight:700; padding:4px 10px;"
        )
        cam_header.addWidget(self.att_cam_badge)
        cl.addLayout(cam_header)

        self.att_cam_lbl = make_camera_viewport(
            ICON_ATTEND, "Live recognition feed will appear here\nafter clicking Start Session"
        )
        cl.addWidget(self.att_cam_lbl, 1)
        content.addWidget(cam_card, 1)

        root.addLayout(content, 1)

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
        self.att_cam_badge.setText("● Live")
        self.att_cam_badge.setStyleSheet(
            f"color:{ERROR_RED}; background:{ERROR_BG}; border-radius:8px;"
            f"font-size:11px; font-weight:700; padding:4px 10px;"
        )

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
            item.setForeground(QColor(SUCCESS_GREEN))
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
        self.stat_total.set_value(total)
        self.stat_present.set_value(present)
        self.stat_absent.set_value(absent)

    def _on_session_done(self, present_ids, filepath):
        self.start_att_btn.setEnabled(True)
        self.stop_att_btn.setEnabled(False)
        self.session_filepath = filepath
        self.att_cam_badge.setText("● Idle")
        self.att_cam_badge.setStyleSheet(
            f"color:{MID_GRAY}; background:{OFF_WHITE}; border-radius:8px;"
            f"font-size:11px; font-weight:700; padding:4px 10px;"
        )
        if present_ids is not None and os.path.exists(filepath):
            self.att_status.setText(f"✓ Session saved → {os.path.basename(filepath)}")
            self.att_status.setStyleSheet(STATUS_OK_STYLE)
        else:
            self.att_status.setText(f"✗ {filepath}")
            self.att_status.setStyleSheet(STATUS_ERR_STYLE)

# ══════════════════════════════════════════════════════════════════════════════
#  ATTENDANCE ANALYTICS  
# ══════════════════════════════════════════════════════════════════════════════

ATTENDANCE_THRESHOLD = 75.0  # percent; below this a (student, subject) pair is a defaulter


def compute_attendance_analytics(department=None, batch=None):
    """
    Walks every saved attendance session CSV (through the existing
    core.get_all_sessions / core.read_csv_records helpers — the same
    ones ReportsTab already uses) and derives:

      - per (student, subject) totals, attended count, and percentage
      - per-subject rollups (total classes, average %, below-threshold count)
      - per-student overall percentage across all their subjects
      - the defaulter list (any student/subject pair below ATTENDANCE_THRESHOLD)
      - overall summary stats for the analytics dashboard cards

    Returns a dict; nothing is cached or written to disk.
    """
    sessions = core.get_all_sessions(department, batch)
    registry = core.load_registry()

    # student_subject[(sid, subject)] = {"total": int, "attended": int, "name": str}
    student_subject = {}
    # subject_totals[subject] = number of distinct sessions held for that subject
    subject_sessions = {}

    for sess in sessions:
        records = core.read_csv_records(sess["filepath"])
        subj = sess["subject"]
        subject_sessions[subj] = subject_sessions.get(subj, 0) + 1
        for rec in records:
            sid = rec.get("Student_ID", "")
            name = rec.get("Student_Name", registry.get(sid, {}).get("name", sid))
            status = rec.get("Status", "")
            key = (sid, subj)
            if key not in student_subject:
                student_subject[key] = {"total": 0, "attended": 0, "name": name}
            student_subject[key]["total"] += 1
            if status == "Present":
                student_subject[key]["attended"] += 1

    # ── per (student, subject) rows ──────────────────────────────────────────
    rows = []
    for (sid, subj), d in student_subject.items():
        pct = (d["attended"] / d["total"] * 100) if d["total"] else 0.0
        rows.append({
            "student_id": sid,
            "name": d["name"],
            "subject": subj,
            "total": d["total"],
            "attended": d["attended"],
            "pct": round(pct, 1),
        })
    rows.sort(key=lambda r: (r["name"], r["subject"]))

    # ── defaulters ────────────────────────────────────────────────────────────
    defaulters = [r for r in rows if r["pct"] < ATTENDANCE_THRESHOLD]
    defaulters.sort(key=lambda r: r["pct"])

    # ── subject-wise rollup ───────────────────────────────────────────────────
    subject_rows = []
    by_subject = {}
    for r in rows:
        by_subject.setdefault(r["subject"], []).append(r)
    for subj, items in by_subject.items():
        avg_pct = sum(i["pct"] for i in items) / len(items) if items else 0.0
        below = sum(1 for i in items if i["pct"] < ATTENDANCE_THRESHOLD)
        subject_rows.append({
            "subject": subj,
            "total_classes": subject_sessions.get(subj, 0),
            "avg_pct": round(avg_pct, 1),
            "below_threshold": below,
        })
    subject_rows.sort(key=lambda r: r["subject"])

    # ── student-wise overall rollup (average across that student's subjects) ──
    student_rows = []
    by_student = {}
    for r in rows:
        by_student.setdefault(r["student_id"], []).append(r)
    for sid, items in by_student.items():
        overall = sum(i["pct"] for i in items) / len(items) if items else 0.0
        student_rows.append({
            "student_id": sid,
            "name": items[0]["name"],
            "subjects_count": len(items),
            "overall_pct": round(overall, 1),
        })
    student_rows.sort(key=lambda r: r["name"])

    # ── dashboard summary numbers ─────────────────────────────────────────────
    all_pcts = [r["pct"] for r in rows]
    summary = {
        "total_students": len(by_student),
        "avg_pct": round(sum(all_pcts) / len(all_pcts), 1) if all_pcts else 0.0,
        "highest_pct": round(max(all_pcts), 1) if all_pcts else 0.0,
        "lowest_pct": round(min(all_pcts), 1) if all_pcts else 0.0,
        "below_threshold_count": len(set(r["student_id"] for r in defaulters)),
    }

    return {
        "rows": rows,                    # per student+subject
        "subject_rows": subject_rows,    # per subject
        "student_rows": student_rows,    # per student (overall)
        "defaulters": defaulters,
        "summary": summary,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — REPORTS
# ══════════════════════════════════════════════════════════════════════════════

class ReportsTab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    # ────────────────────────────────────────────────────────────────────────
    #  TOP-LEVEL LAYOUT: header + segmented switcher + stacked sub-views
    # ────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(18); root.setContentsMargins(28, 22, 28, 22)
        root.addWidget(make_section_header(
            ICON_REPORTS, "Reports",
            "Browse saved attendance sessions and view attendance analytics"
        ))

        # Segmented switcher: Sessions | Analytics
        switch_row = QHBoxLayout()
        switch_row.setSpacing(8)
        self.btn_sessions = QPushButton("Sessions")
        self.btn_analytics = QPushButton("Analytics")
        for b in (self.btn_sessions, self.btn_analytics):
            b.setCursor(Qt.PointingHandCursor)
            b.setCheckable(True)
            b.setMinimumHeight(34)
            b.setMinimumWidth(120)
        self._switch_group = QButtonGroup(self)
        self._switch_group.setExclusive(True)
        self._switch_group.addButton(self.btn_sessions, 0)
        self._switch_group.addButton(self.btn_analytics, 1)
        self.btn_sessions.setChecked(True)
        self._switch_group.idClicked.connect(self._on_switch_view)
        switch_row.addWidget(self.btn_sessions)
        switch_row.addWidget(self.btn_analytics)
        switch_row.addStretch()
        root.addLayout(switch_row)
        self._apply_switch_styles()

        self.view_stack = QStackedWidget()
        self.sessions_view = self._build_sessions_view()
        self.analytics_view = self._build_analytics_view()
        self.view_stack.addWidget(self.sessions_view)
        self.view_stack.addWidget(self.analytics_view)
        root.addWidget(self.view_stack, 1)

    def _apply_switch_styles(self):
        active = (
            f"QPushButton{{ background:{PRIMARY_BLUE}; color:white; border:none;"
            f"border-radius:8px; font-size:12.5px; font-weight:700; }}"
        )
        inactive = (
            f"QPushButton{{ background:{WHITE}; color:{SUBTLE_TEXT}; border:1px solid {LIGHT_GRAY};"
            f"border-radius:8px; font-size:12.5px; font-weight:600; }}"
            f"QPushButton:hover{{ background:{OFF_WHITE}; }}"
        )
        self.btn_sessions.setStyleSheet(active if self.btn_sessions.isChecked() else inactive)
        self.btn_analytics.setStyleSheet(active if self.btn_analytics.isChecked() else inactive)

    def _on_switch_view(self, idx):
        self._apply_switch_styles()
        self.view_stack.setCurrentIndex(idx)
        if idx == 1:
            self._refresh_analytics()

    # ────────────────────────────────────────────────────────────────────────
    #  SESSIONS VIEW  
    # ────────────────────────────────────────────────────────────────────────
    def _build_sessions_view(self):
        view = QWidget()
        vroot = QVBoxLayout(view)
        vroot.setContentsMargins(0, 0, 0, 0)
        vroot.setSpacing(18)

        # Filter bar
        filter_card = make_card()
        filter_card.setFixedHeight(92)
        fl = QHBoxLayout(filter_card)
        fl.setContentsMargins(20, 14, 20, 14); fl.setSpacing(16)

        fl.addWidget(make_label("Filter by", bold=True, color=DARK_TEXT))

        self.f_dept = QComboBox(); self.f_dept.addItem("All Departments")
        self.f_dept.addItems(courses.DEPARTMENTS)
        fl.addLayout(vfield("Department", self.f_dept))

        self.f_batch = QComboBox(); self.f_batch.addItem("All Batches")
        for d in courses.DEPARTMENTS:
            self.f_batch.addItems(courses.get_batches(d))
        fl.addLayout(vfield("Batch", self.f_batch))

        fl.addStretch()
        refresh_btn = QPushButton(f"{ICON_REFRESH}   Refresh")
        refresh_btn.setObjectName("secondary")
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.clicked.connect(self._load_sessions)
        fl.addWidget(refresh_btn)
        vroot.addWidget(filter_card)

        # Main split: session list | detail (table + chart)
        splitter = QSplitter(Qt.Horizontal)

        # ── Session list ──────────────────────────────────────────────────────
        sess_card = make_card()
        sl = QVBoxLayout(sess_card)
        sl.setContentsMargins(16, 16, 16, 16); sl.setSpacing(10)
        sl.addWidget(make_label("Sessions", bold=True, color=DARK_TEXT))
        self.session_table = QTableWidget(0, 4)
        self.session_table.setHorizontalHeaderLabels(["Dept", "Batch", "Subject", "Date"])
        self.session_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.session_table.setAlternatingRowColors(True)
        self.session_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.session_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.session_table.verticalHeader().setVisible(False)
        self.session_table.itemSelectionChanged.connect(self._on_session_selected)
        sl.addWidget(self.session_table, 1)
        splitter.addWidget(sess_card)

        # ── Detail: table + chart ─────────────────────────────────────────────
        detail_card = make_card()
        dl = QVBoxLayout(detail_card)
        dl.setContentsMargins(16, 16, 16, 16); dl.setSpacing(10)

        dl.addWidget(make_label("Session Detail", bold=True, color=DARK_TEXT))

        detail_split = QSplitter(Qt.Vertical)

        # Attendance table
        self.detail_table = QTableWidget(0, 5)
        self.detail_table.setHorizontalHeaderLabels(
            ["ID", "Name", "Dept", "Time", "Status"])
        self.detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.detail_table.setAlternatingRowColors(True)
        self.detail_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.detail_table.verticalHeader().setVisible(False)
        detail_split.addWidget(self.detail_table)

        # Pie chart
        self.fig, self.ax = plt.subplots(figsize=(5, 3.2), facecolor="#FFFFFF")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(220)
        detail_split.addWidget(self.canvas)

        dl.addWidget(detail_split, 1)
        splitter.addWidget(detail_card)
        splitter.setSizes([340, 560])

        vroot.addWidget(splitter, 1)

        self._sessions_data = []
        self._load_sessions()
        return view

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
                fg = QColor(SUCCESS_GREEN)
            else:
                absent_count += 1
                fg = QColor(ERROR_RED)

            for c, key in enumerate(["Student_ID", "Student_Name",
                                      "Department", "Time", "Status"]):
                val = rec.get(key, "")
                if key == "Status":
                    # Render status as a pill widget instead of plain colored text
                    pill = status_pill(val, "ok" if val == "Present" else "err")
                    self.detail_table.setCellWidget(r, c, pill)
                else:
                    item = QTableWidgetItem(val)
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setForeground(fg if key == "Student_Name" else QColor(DARK_TEXT))
                    self.detail_table.setItem(r, c, item)

        # Pie chart
        self.ax.clear()
        total = present_count + absent_count
        if total == 0:
            self.ax.text(0.5, 0.5, "No data", ha="center", va="center",
                         transform=self.ax.transAxes, fontsize=13, color=MID_GRAY)
        else:
            sizes  = [present_count, absent_count]
            labels = [f"Present\n{present_count}", f"Absent\n{absent_count}"]
            colors = [SUCCESS_GREEN, ERROR_RED]
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
                fontsize=12, color=PRIMARY_BLUE, fontweight="bold", pad=10
            )
            self.ax.set_facecolor("#FFFFFF")

        self.fig.tight_layout()
        self.canvas.draw()

    # ────────────────────────────────────────────────────────────────────────
    #  ANALYTICS VIEW  
    # ────────────────────────────────────────────────────────────────────────
    def _build_analytics_view(self):
        view = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea{ background:transparent; border:none; }")

        inner = QWidget()
        vroot = QVBoxLayout(inner)
        vroot.setContentsMargins(0, 0, 4, 4)
        vroot.setSpacing(18)

        # Filter bar (reuses same Department/Batch filters as Sessions view)
        afilter_card = make_card()
        afilter_card.setFixedHeight(92)
        afl = QHBoxLayout(afilter_card)
        afl.setContentsMargins(20, 14, 20, 14); afl.setSpacing(16)
        afl.addWidget(make_label("Filter by", bold=True, color=DARK_TEXT))

        self.a_dept = QComboBox(); self.a_dept.addItem("All Departments")
        self.a_dept.addItems(courses.DEPARTMENTS)
        afl.addLayout(vfield("Department", self.a_dept))

        self.a_batch = QComboBox(); self.a_batch.addItem("All Batches")
        for d in courses.DEPARTMENTS:
            self.a_batch.addItems(courses.get_batches(d))
        afl.addLayout(vfield("Batch", self.a_batch))

        afl.addStretch()
        a_refresh_btn = QPushButton(f"{ICON_REFRESH}   Refresh")
        a_refresh_btn.setObjectName("secondary")
        a_refresh_btn.setCursor(Qt.PointingHandCursor)
        a_refresh_btn.clicked.connect(self._refresh_analytics)
        afl.addWidget(a_refresh_btn)
        vroot.addWidget(afilter_card)

        # ── Summary stat cards ───────────────────────────────────────────────
        stats_row = QHBoxLayout()
        stats_row.setSpacing(14)
        self.a_card_students = StatCard("Total Students", "0", ICON_USERS, PRIMARY_BLUE, PRIMARY_BLUE_LIGHT, compact=True)
        self.a_card_avg      = StatCard("Average Attendance", "0%", ICON_REPORTS, "#7C3AED", "#EDE9FE", compact=True)
        self.a_card_high     = StatCard("Highest Attendance", "0%", ICON_TRAIN, SUCCESS_GREEN, SUCCESS_BG, compact=True)
        self.a_card_low      = StatCard("Lowest Attendance", "0%", ICON_CLOCK, WARN_ORANGE, WARN_BG, compact=True)
        self.a_card_below    = StatCard("Below 75%", "0", ICON_ATTEND, ERROR_RED, ERROR_BG, compact=True)
        for c in (self.a_card_students, self.a_card_avg, self.a_card_high,
                  self.a_card_low, self.a_card_below):
            stats_row.addWidget(c)
        vroot.addLayout(stats_row)

        # ── Charts row ────────────────────────────────────────────────────────
        charts_card = make_card()
        cl = QVBoxLayout(charts_card)
        cl.setContentsMargins(18, 16, 18, 16); cl.setSpacing(10)
        cl.addWidget(make_label("Attendance Charts", bold=True, color=DARK_TEXT))

        self.a_fig, (self.a_ax1, self.a_ax2, self.a_ax3) = plt.subplots(
            1, 3, figsize=(13, 3.4), facecolor="#FFFFFF"
        )
        self.a_canvas = FigureCanvas(self.a_fig)
        self.a_canvas.setMinimumHeight(260)
        cl.addWidget(self.a_canvas)
        vroot.addWidget(charts_card)

        # ── Student search ───────────────────────────────────────────────────
        search_card = make_card()
        ssl = QVBoxLayout(search_card)
        ssl.setContentsMargins(18, 16, 18, 16); ssl.setSpacing(10)
        ssl.addWidget(make_label("Student Search", bold=True, color=DARK_TEXT))
        self.student_search = QLineEdit()
        self.student_search.setPlaceholderText("Search by student name or ID...")
        self.student_search.textChanged.connect(self._on_student_search)
        ssl.addWidget(self.student_search)

        self.search_result_lbl = QLabel("")
        self.search_result_lbl.setStyleSheet(STATUS_INFO_STYLE)
        self.search_result_lbl.setWordWrap(True)
        self.search_result_lbl.setVisible(False)
        ssl.addWidget(self.search_result_lbl)

        self.search_table = QTableWidget(0, 4)
        self.search_table.setHorizontalHeaderLabels(["Subject", "Total Classes", "Attended", "Attendance %"])
        self.search_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.search_table.setAlternatingRowColors(True)
        self.search_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.search_table.verticalHeader().setVisible(False)
        self.search_table.setMinimumHeight(160)
        ssl.addWidget(self.search_table)
        vroot.addWidget(search_card)

        # ── Subject-wise statistics ──────────────────────────────────────────
        subj_card = make_card()
        sjl = QVBoxLayout(subj_card)
        sjl.setContentsMargins(18, 16, 18, 16); sjl.setSpacing(10)
        sjl.addWidget(make_label("Subject-wise Attendance Statistics", bold=True, color=DARK_TEXT))
        self.subject_table = QTableWidget(0, 4)
        self.subject_table.setHorizontalHeaderLabels(
            ["Subject", "Total Classes", "Average Attendance", "Students Below 75%"])
        self.subject_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.subject_table.setAlternatingRowColors(True)
        self.subject_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.subject_table.verticalHeader().setVisible(False)
        self.subject_table.setMinimumHeight(160)
        sjl.addWidget(self.subject_table)
        vroot.addWidget(subj_card)

        # ── Student Attendance Percentage (full breakdown) ───────────────────
        stu_card = make_card()
        stl = QVBoxLayout(stu_card)
        stl.setContentsMargins(18, 16, 18, 16); stl.setSpacing(10)
        stl.addWidget(make_label("Student Attendance Percentage", bold=True, color=DARK_TEXT))
        self.student_pct_table = QTableWidget(0, 5)
        self.student_pct_table.setHorizontalHeaderLabels(
            ["Student", "Subject", "Total Classes", "Classes Attended", "Attendance %"])
        self.student_pct_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.student_pct_table.setAlternatingRowColors(True)
        self.student_pct_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.student_pct_table.verticalHeader().setVisible(False)
        self.student_pct_table.setMinimumHeight(200)
        stl.addWidget(self.student_pct_table)
        vroot.addWidget(stu_card)

        # ── Defaulter list ────────────────────────────────────────────────────
        def_card = make_card()
        dfl = QVBoxLayout(def_card)
        dfl.setContentsMargins(18, 16, 18, 16); dfl.setSpacing(10)
        def_header = QHBoxLayout()
        def_header.addWidget(make_label(
            f"Defaulter List  (below {int(ATTENDANCE_THRESHOLD)}%)", bold=True, color=ERROR_RED))
        def_header.addStretch()
        self.defaulter_count_lbl = QLabel("0 students")
        self.defaulter_count_lbl.setStyleSheet(STATUS_ERR_STYLE)
        def_header.addWidget(self.defaulter_count_lbl)
        dfl.addLayout(def_header)

        self.defaulter_table = QTableWidget(0, 3)
        self.defaulter_table.setHorizontalHeaderLabels(["Student Name", "Subject", "Attendance %"])
        self.defaulter_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.defaulter_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.defaulter_table.verticalHeader().setVisible(False)
        self.defaulter_table.setMinimumHeight(160)
        dfl.addWidget(self.defaulter_table)
        vroot.addWidget(def_card)

        scroll.setWidget(inner)
        outer = QVBoxLayout(view)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._analytics_data = None
        return view

    def _refresh_analytics(self):
        dept  = self.a_dept.currentText()
        batch = self.a_batch.currentText()
        d_filter = None if dept  == "All Departments" else dept
        b_filter = None if batch == "All Batches"     else batch

        data = compute_attendance_analytics(d_filter, b_filter)
        self._analytics_data = data

        # Summary cards
        s = data["summary"]
        self.a_card_students.set_value(s["total_students"])
        self.a_card_avg.set_value(f"{s['avg_pct']}%")
        self.a_card_high.set_value(f"{s['highest_pct']}%")
        self.a_card_low.set_value(f"{s['lowest_pct']}%")
        self.a_card_below.set_value(s["below_threshold_count"])

        # Subject-wise table
        self.subject_table.setRowCount(0)
        for row in data["subject_rows"]:
            r = self.subject_table.rowCount()
            self.subject_table.insertRow(r)
            vals = [row["subject"], row["total_classes"],
                    f"{row['avg_pct']}%", row["below_threshold"]]
            for c, val in enumerate(vals):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                if c == 2 and row["avg_pct"] < ATTENDANCE_THRESHOLD:
                    item.setForeground(QColor(ERROR_RED))
                self.subject_table.setItem(r, c, item)

        # Student attendance percentage table (per student+subject)
        self.student_pct_table.setRowCount(0)
        for row in data["rows"]:
            r = self.student_pct_table.rowCount()
            self.student_pct_table.insertRow(r)
            vals = [row["name"], row["subject"], row["total"],
                    row["attended"], f"{row['pct']}%"]
            below = row["pct"] < ATTENDANCE_THRESHOLD
            for c, val in enumerate(vals):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                if below:
                    item.setForeground(QColor(ERROR_RED))
                    if c == 4:
                        font = item.font(); font.setBold(True); item.setFont(font)
                self.student_pct_table.setItem(r, c, item)

        # Defaulter list — visually highlighted rows
        self.defaulter_table.setRowCount(0)
        for row in data["defaulters"]:
            r = self.defaulter_table.rowCount()
            self.defaulter_table.insertRow(r)
            vals = [row["name"], row["subject"], f"{row['pct']}%"]
            for c, val in enumerate(vals):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                item.setBackground(QColor(ERROR_BG))
                item.setForeground(QColor(ERROR_RED))
                font = item.font(); font.setBold(True); item.setFont(font)
                self.defaulter_table.setItem(r, c, item)
        self.defaulter_count_lbl.setText(
            f"{len(set(d['student_id'] for d in data['defaulters']))} students"
        )

        # Re-apply any active search filter
        if self.student_search.text().strip():
            self._on_student_search(self.student_search.text())
        else:
            self.search_table.setRowCount(0)
            self.search_result_lbl.setVisible(False)

        self._draw_analytics_charts(data)

    def _on_student_search(self, text):
        text = text.strip().lower()
        self.search_table.setRowCount(0)
        if not text or not self._analytics_data:
            self.search_result_lbl.setVisible(False)
            return

        matches = [
            r for r in self._analytics_data["rows"]
            if text in r["name"].lower() or text in str(r["student_id"]).lower()
        ]
        if not matches:
            self.search_result_lbl.setText("No matching student found.")
            self.search_result_lbl.setStyleSheet(STATUS_WARN_STYLE)
            self.search_result_lbl.setVisible(True)
            return

        name = matches[0]["name"]
        sid = matches[0]["student_id"]
        overall = sum(m["pct"] for m in matches) / len(matches)
        self.search_result_lbl.setText(
            f"{name}  (ID: {sid})   —   Overall Attendance: {overall:.1f}%   "
            f"across {len(matches)} subject(s)"
        )
        self.search_result_lbl.setStyleSheet(
            STATUS_OK_STYLE if overall >= ATTENDANCE_THRESHOLD else STATUS_ERR_STYLE
        )
        self.search_result_lbl.setVisible(True)

        for m in matches:
            r = self.search_table.rowCount()
            self.search_table.insertRow(r)
            vals = [m["subject"], m["total"], m["attended"], f"{m['pct']}%"]
            below = m["pct"] < ATTENDANCE_THRESHOLD
            for c, val in enumerate(vals):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                if below:
                    item.setForeground(QColor(ERROR_RED))
                self.search_table.setItem(r, c, item)

    def _draw_analytics_charts(self, data):
        for ax in (self.a_ax1, self.a_ax2, self.a_ax3):
            ax.clear()

        # Chart 1: Attendance % by Student (overall)
        student_rows = sorted(data["student_rows"], key=lambda r: r["overall_pct"])
        if student_rows:
            names = [r["name"] for r in student_rows]
            pcts = [r["overall_pct"] for r in student_rows]
            colors = [ERROR_RED if p < ATTENDANCE_THRESHOLD else SUCCESS_GREEN for p in pcts]
            self.a_ax1.barh(names, pcts, color=colors)
            self.a_ax1.axvline(ATTENDANCE_THRESHOLD, color=WARN_ORANGE, linestyle="--", linewidth=1)
            self.a_ax1.set_xlim(0, 100)
            self.a_ax1.set_title("Attendance % by Student", fontsize=10, fontweight="bold")
            self.a_ax1.tick_params(axis="y", labelsize=8)
            self.a_ax1.tick_params(axis="x", labelsize=8)
        else:
            self.a_ax1.text(0.5, 0.5, "No data", ha="center", va="center",
                             transform=self.a_ax1.transAxes, color=MID_GRAY)

        # Chart 2: Attendance % by Subject
        subject_rows = data["subject_rows"]
        if subject_rows:
            subs = [r["subject"] for r in subject_rows]
            pcts2 = [r["avg_pct"] for r in subject_rows]
            colors2 = [ERROR_RED if p < ATTENDANCE_THRESHOLD else PRIMARY_BLUE for p in pcts2]
            self.a_ax2.bar(subs, pcts2, color=colors2)
            self.a_ax2.axhline(ATTENDANCE_THRESHOLD, color=WARN_ORANGE, linestyle="--", linewidth=1)
            self.a_ax2.set_ylim(0, 100)
            self.a_ax2.set_title("Attendance % by Subject", fontsize=10, fontweight="bold")
            self.a_ax2.tick_params(axis="x", labelsize=7, rotation=30)
            self.a_ax2.tick_params(axis="y", labelsize=8)
        else:
            self.a_ax2.text(0.5, 0.5, "No data", ha="center", va="center",
                             transform=self.a_ax2.transAxes, color=MID_GRAY)

        # Chart 3: Defaulter Distribution (defaulters vs OK, by unique student)
        defaulter_ids = set(d["student_id"] for d in data["defaulters"])
        total_ids = set(r["student_id"] for r in data["rows"])
        ok_count = len(total_ids) - len(defaulter_ids)
        def_count = len(defaulter_ids)
        if total_ids:
            sizes = [ok_count, def_count]
            labels = [f"Above {int(ATTENDANCE_THRESHOLD)}%\n{ok_count}", f"Below {int(ATTENDANCE_THRESHOLD)}%\n{def_count}"]
            colors3 = [SUCCESS_GREEN, ERROR_RED]
            self.a_ax3.pie(sizes, labels=labels, colors=colors3, autopct="%1.0f%%",
                            startangle=90, textprops={"fontsize": 8})
            self.a_ax3.set_title("Defaulter Distribution", fontsize=10, fontweight="bold")
        else:
            self.a_ax3.text(0.5, 0.5, "No data", ha="center", va="center",
                             transform=self.a_ax3.transAxes, color=MID_GRAY)

        self.a_fig.tight_layout()
        self.a_canvas.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Attendance System")
        self.setMinimumSize(1180, 740)
        self.setStyleSheet(MAIN_STYLE)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # ── Sidebar ───────────────────────────────────────────────────────
        self.sidebar = Sidebar()
        self.sidebar.navigate.connect(self._on_navigate)
        root.addWidget(self.sidebar)

        # ── Right side: top bar + stacked pages ──────────────────────────
        right = QWidget()
        right.setObjectName("contentArea")
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0); right_lay.setSpacing(0)

        self.topbar = TopBar()
        right_lay.addWidget(self.topbar)

        self.stack = QStackedWidget()

        self.dashboard_page = DashboardPage()
        self.dashboard_page.go_to_page.connect(self._on_navigate_and_select)
        self.capture_page    = CaptureTab()
        self.train_page      = TrainTab()
        self.attendance_page = AttendanceTab()
        self.reports_page    = ReportsTab()

        for page in [self.dashboard_page, self.capture_page, self.train_page,
                     self.attendance_page, self.reports_page]:
            self.stack.addWidget(page)

        right_lay.addWidget(self.stack, 1)
        root.addWidget(right, 1)

        # Status bar
        self.statusBar().setStyleSheet(
            f"background:{WHITE}; color:{SUBTLE_TEXT}; font-size:11.5px;"
            f"padding:4px 16px; border-top: 1px solid {LIGHT_GRAY};"
        )
        self.statusBar().showMessage(
            "Face Attendance System   │   Developed by Harshit Panwar"
        )

        self._page_titles = [
            "Dashboard", "Capture Dataset", "Train Model",
            "Take Attendance", "Reports",
        ]
        self.stack.setCurrentIndex(0)

    def _on_navigate(self, idx):
        self.stack.setCurrentIndex(idx)
        self.topbar.set_title(self._page_titles[idx])
        if idx == 0:
            self.dashboard_page.refresh()
        # Keep registry / model tables fresh when a user revisits a page
        if idx == 1:
            self.capture_page._refresh_registry()
        if idx == 2:
            self.train_page._refresh_models()

    def _on_navigate_and_select(self, idx):
        """Used by Dashboard 'quick action' rows to jump + highlight sidebar."""
        self.sidebar.set_active(idx)
        self._on_navigate(idx)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Face Attendance System")
    app.setStyle("Fusion")

    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()