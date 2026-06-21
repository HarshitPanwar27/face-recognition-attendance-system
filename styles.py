"""
========================================================
  DESIGN SYSTEM v4 — Modern SaaS-Inspired UI
  Face Recognition Attendance System
  Developed by Harshit Panwar
========================================================

"""

# ── Core Palette ───────────────────────────────────────
PRIMARY_BLUE        = "#4F46E5"   # Indigo 600 — primary brand / accents
PRIMARY_BLUE_MID    = "#6366F1"   # Indigo 500 — hover state
PRIMARY_BLUE_LIGHT  = "#EEF2FF"   # Indigo 50  — soft backgrounds
PRIMARY_BLUE_GLOW   = "#818CF8"   # Indigo 400 — highlights / glows
PRIMARY_DARK        = "#312E81"   # Indigo 900 — sidebar background

ACCENT_GOLD        = "#F59E0B"    # Amber 500 — secondary accent
ACCENT_GOLD_LIGHT  = "#FEF3C7"    # Amber 100

WHITE            = "#FFFFFF"
OFF_WHITE        = "#F8FAFC"      # App canvas background (Slate 50)
SURFACE          = "#FFFFFF"
LIGHT_GRAY       = "#E2E8F0"      # Slate 200 — borders / dividers
MID_GRAY         = "#94A3B8"      # Slate 400 — muted text / icons
DARK_TEXT        = "#0F172A"      # Slate 900 — primary text
SUBTLE_TEXT      = "#64748B"      # Slate 500 — secondary text
BORDER           = "#E2E8F0"      # Slate 200

SUCCESS_GREEN    = "#059669"      # Emerald 600
SUCCESS_BG       = "#D1FAE5"
ERROR_RED        = "#DC2626"      # Red 600
ERROR_BG         = "#FEE2E2"
WARN_ORANGE      = "#D97706"      # Amber 600
WARN_BG          = "#FEF3C7"
INFO_BLUE        = "#2563EB"      # Blue 600
INFO_BG          = "#DBEAFE"

# ── New tokens: sidebar / dashboard ───────────────────────────────────
SIDEBAR_BG         = "#1E1B4B"     # Indigo 950
SIDEBAR_BG_HOVER   = "#312E81"     # Indigo 900
SIDEBAR_BG_ACTIVE  = "#4F46E5"     # matches PRIMARY_BLUE
SIDEBAR_TEXT       = "#C7D2FE"     # Indigo 200
SIDEBAR_TEXT_MUTED = "#8280C2"
SIDEBAR_TEXT_ACTIVE = "#FFFFFF"
SIDEBAR_DIVIDER    = "rgba(255,255,255,0.08)"

CANVAS_BG          = OFF_WHITE
CARD_BORDER        = LIGHT_GRAY

STAT_BLUE_BG    = "#EEF2FF"
STAT_GREEN_BG   = "#D1FAE5"
STAT_RED_BG     = "#FEE2E2"
STAT_AMBER_BG   = "#FEF3C7"


MAIN_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {OFF_WHITE};
    font-family: "Segoe UI", "Inter", "Arial", sans-serif;
}}
QWidget#sidebar {{
    background-color: {SIDEBAR_BG};
}}
QWidget#contentArea {{
    background-color: {OFF_WHITE};
}}
QLabel {{
    color: {DARK_TEXT};
    font-size: 13px;
    background: transparent;
}}
QLineEdit {{
    background: {WHITE};
    border: 1.5px solid {BORDER};
    border-radius: 8px;
    padding: 9px 14px;
    font-size: 13px;
    color: {DARK_TEXT};
    min-height: 36px;
    selection-background-color: {PRIMARY_BLUE_LIGHT};
    selection-color: {PRIMARY_BLUE};
}}
QLineEdit:focus {{
    border: 1.5px solid {PRIMARY_BLUE};
    background: {WHITE};
}}
QLineEdit::placeholder {{
    color: {MID_GRAY};
}}
QComboBox {{
    background: {WHITE};
    border: 1.5px solid {BORDER};
    border-radius: 8px;
    padding: 7px 14px;
    font-size: 13px;
    color: {DARK_TEXT};
    min-height: 36px;
}}
QComboBox:focus, QComboBox:on {{
    border: 1.5px solid {PRIMARY_BLUE};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 30px;
    border-left: 1px solid {BORDER};
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
    background: {PRIMARY_BLUE_LIGHT};
}}
QComboBox::down-arrow {{
    width: 10px;
    height: 10px;
}}
QComboBox QAbstractItemView {{
    background: {WHITE};
    border: 1.5px solid {PRIMARY_BLUE_GLOW};
    border-radius: 8px;
    selection-background-color: {PRIMARY_BLUE_LIGHT};
    selection-color: {PRIMARY_BLUE};
    padding: 4px;
    font-size: 13px;
    outline: none;
}}
QComboBox QAbstractItemView::item {{
    padding: 8px 12px;
    border-radius: 5px;
    min-height: 28px;
}}
QPushButton {{
    background: {PRIMARY_BLUE};
    color: {WHITE};
    font-size: 13px;
    font-weight: 600;
    border: none;
    border-radius: 9px;
    padding: 10px 20px;
    min-height: 38px;
}}
QPushButton:hover {{
    background: {PRIMARY_BLUE_MID};
}}
QPushButton:pressed {{
    background: {PRIMARY_DARK};
}}
QPushButton:disabled {{
    background: {LIGHT_GRAY};
    color: {MID_GRAY};
}}
QPushButton#danger {{
    background: {ERROR_RED};
}}
QPushButton#danger:hover {{
    background: #EF4444;
}}
QPushButton#danger:disabled {{
    background: {LIGHT_GRAY};
    color: {MID_GRAY};
}}
QPushButton#success {{
    background: {SUCCESS_GREEN};
}}
QPushButton#success:hover {{
    background: #10B981;
}}
QPushButton#secondary {{
    background: {WHITE};
    color: {PRIMARY_BLUE};
    border: 1.5px solid {LIGHT_GRAY};
}}
QPushButton#secondary:hover {{
    background: {PRIMARY_BLUE_LIGHT};
    border: 1.5px solid {PRIMARY_BLUE_GLOW};
}}
QTableWidget {{
    background: {WHITE};
    border: 1px solid {LIGHT_GRAY};
    border-radius: 10px;
    gridline-color: {OFF_WHITE};
    font-size: 12.5px;
    color: {DARK_TEXT};
    selection-background-color: {PRIMARY_BLUE_LIGHT};
    selection-color: {PRIMARY_BLUE};
    outline: none;
}}
QTableWidget::item {{
    padding: 8px 8px;
    border: none;
    border-bottom: 1px solid {OFF_WHITE};
}}
QTableWidget::item:selected {{
    background: {PRIMARY_BLUE_LIGHT};
    color: {PRIMARY_BLUE};
}}
QHeaderView {{
    background: transparent;
    border: none;
}}
QHeaderView::section {{
    background: {OFF_WHITE};
    color: {SUBTLE_TEXT};
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 0.4px;
    padding: 10px 8px;
    border: none;
    border-bottom: 1.5px solid {LIGHT_GRAY};
}}
QHeaderView::section:first {{ border-top-left-radius: 10px; }}
QHeaderView::section:last  {{ border-top-right-radius: 10px; }}
QTableWidget::item:alternate {{ background: #FBFCFE; }}
QScrollBar:vertical {{
    background: transparent;
    width: 9px;
    border-radius: 4px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {LIGHT_GRAY};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {PRIMARY_BLUE_GLOW}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 9px;
    border-radius: 4px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {LIGHT_GRAY};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{ background: {PRIMARY_BLUE_GLOW}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}
QTextEdit {{
    background: #0F172A;
    color: #A5B4FC;
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
    border: 1px solid {LIGHT_GRAY};
    border-radius: 10px;
    padding: 12px;
}}
QProgressBar {{
    border: none;
    border-radius: 6px;
    background: {LIGHT_GRAY};
    text-align: center;
    color: {WHITE};
    font-weight: 700;
    font-size: 11px;
    height: 18px;
}}
QProgressBar::chunk {{
    background: {PRIMARY_BLUE};
    border-radius: 6px;
}}
QSplitter::handle {{
    background: {LIGHT_GRAY};
    width: 2px;
    height: 2px;
}}
QSplitter::handle:hover {{ background: {PRIMARY_BLUE_GLOW}; }}
QToolTip {{
    background: {DARK_TEXT};
    color: {WHITE};
    border: none;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 12px;
}}
"""

CARD_STYLE = f"""
    background: {WHITE};
    border: 1px solid {LIGHT_GRAY};
    border-radius: 14px;
"""

SECTION_HEADER_STYLE = f"""
    font-size: 13px;
    font-weight: 700;
    color: {DARK_TEXT};
    background: transparent;
    letter-spacing: 0.2px;
"""

STATUS_OK_STYLE = (
    f"color:{SUCCESS_GREEN}; font-weight:700; font-size:12px;"
    f"background:{SUCCESS_BG}; border:1px solid #A7F3D0;"
    f"border-radius:8px; padding:8px 12px;"
)
STATUS_ERR_STYLE = (
    f"color:{ERROR_RED}; font-weight:700; font-size:12px;"
    f"background:{ERROR_BG}; border:1px solid #FECACA;"
    f"border-radius:8px; padding:8px 12px;"
)
STATUS_WARN_STYLE = (
    f"color:{WARN_ORANGE}; font-weight:700; font-size:12px;"
    f"background:{WARN_BG}; border:1px solid #FDE68A;"
    f"border-radius:8px; padding:8px 12px;"
)
STATUS_INFO_STYLE = (
    f"color:{INFO_BLUE}; font-weight:600; font-size:12px;"
    f"background:{INFO_BG}; border:1px solid #BFDBFE;"
    f"border-radius:8px; padding:8px 12px;"
)

# ── New: sidebar nav button style ──
SIDEBAR_BTN_STYLE = f"""
QPushButton {{
    text-align: left;
    background: transparent;
    color: {SIDEBAR_TEXT};
    border: none;
    border-radius: 10px;
    padding: 11px 14px;
    font-size: 13px;
    font-weight: 600;
    min-height: 20px;
}}
QPushButton:hover {{
    background: {SIDEBAR_BG_HOVER};
    color: {SIDEBAR_TEXT_ACTIVE};
}}
"""

SIDEBAR_BTN_ACTIVE_STYLE = f"""
QPushButton {{
    text-align: left;
    background: {SIDEBAR_BG_ACTIVE};
    color: {SIDEBAR_TEXT_ACTIVE};
    border: none;
    border-radius: 10px;
    padding: 11px 14px;
    font-size: 13px;
    font-weight: 700;
    min-height: 20px;
}}
QPushButton:hover {{
    background: {PRIMARY_BLUE_MID};
}}
"""

# ── Status pill helper styles (used for table Present/Absent cells, etc) ──
PILL_GREEN = f"color:{SUCCESS_GREEN}; background:{SUCCESS_BG}; border-radius:6px; font-weight:700; font-size:11px; padding:3px 10px;"
PILL_RED   = f"color:{ERROR_RED}; background:{ERROR_BG}; border-radius:6px; font-weight:700; font-size:11px; padding:3px 10px;"
PILL_AMBER = f"color:{WARN_ORANGE}; background:{WARN_BG}; border-radius:6px; font-weight:700; font-size:11px; padding:3px 10px;"