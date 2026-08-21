"""SCGen7 design system — colours, typography, spacing, and Qt stylesheet.

Design language:
  - Warm dark palette (not cold slate)
  - Inter font family (fallback: Segoe UI, system)
  - Muted gold accent (authoritative, legal)
  - Generous whitespace, subtle borders
  - Information-dense when needed, spacious when not
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Colour tokens
# ---------------------------------------------------------------------------

class C:
    """Colour palette — warm dark theme with gold accent."""

    # Backgrounds (warm, not cold)
    BG_BASE = "#18181B"        # Deepest background
    BG_SURFACE = "#1E1E22"     # Cards, panels
    BG_RAISED = "#26262B"      # Raised elements (hover states, input fields)
    BG_OVERLAY = "#2C2C32"     # Overlays, dropdowns, tooltips

    # Borders
    BORDER = "#33333A"         # Subtle default border
    BORDER_STRONG = "#44444D"  # Strong border (focus, active)

    # Text
    TEXT_PRIMARY = "#EDEDEF"    # Primary text — warm white
    TEXT_SECONDARY = "#9898A0"  # Secondary text — muted
    TEXT_TERTIARY = "#6B6B73"   # Tertiary — hints, placeholders
    TEXT_INVERSE = "#18181B"    # Text on light backgrounds

    # Accent — muted gold (authoritative, legal tone)
    ACCENT = "#C9A96E"
    ACCENT_HOVER = "#D4B87E"
    ACCENT_MUTED = "#8B7A50"
    ACCENT_BG = "#2A2620"      # Subtle accent background tint

    # Semantic
    SUCCESS = "#5CB67A"
    SUCCESS_BG = "#1A2B1F"
    WARNING = "#D4A843"
    WARNING_BG = "#2A2518"
    ERROR = "#D45B5B"
    ERROR_BG = "#2A1A1A"
    INFO = "#5B8FD4"
    INFO_BG = "#1A2030"

    # Scrollbar
    SCROLLBAR_BG = "#1E1E22"
    SCROLLBAR_HANDLE = "#3A3A42"
    SCROLLBAR_HOVER = "#4A4A54"

    # Selection
    SELECTION_BG = "#3A3528"
    SELECTION_TEXT = "#EDEDEF"


# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

class T:
    """Typography tokens."""

    FAMILY = '"Inter", "Segoe UI", "SF Pro Display", -apple-system, sans-serif'
    FAMILY_MONO = '"JetBrains Mono", "Cascadia Code", "Consolas", monospace'

    SIZE_XS = "11px"
    SIZE_SM = "12px"
    SIZE_BASE = "13px"
    SIZE_MD = "14px"
    SIZE_LG = "16px"
    SIZE_XL = "20px"
    SIZE_2XL = "24px"

    WEIGHT_NORMAL = "400"
    WEIGHT_MEDIUM = "500"
    WEIGHT_SEMIBOLD = "600"
    WEIGHT_BOLD = "700"

    LINE_HEIGHT = "1.5"
    LINE_HEIGHT_TIGHT = "1.3"


# ---------------------------------------------------------------------------
# Spacing & Layout
# ---------------------------------------------------------------------------

class S:
    """Spacing and dimension tokens."""

    # Spacing scale (px)
    XS = 4
    SM = 8
    MD = 12
    LG = 16
    XL = 24
    XXL = 32

    # Layout dimensions
    SIDEBAR_WIDTH = 54         # Icon-only sidebar
    STATUS_BAR_HEIGHT = 40     # Top bar
    FOOTER_HEIGHT = 28         # Bottom status bar
    BORDER_RADIUS = 6          # Default radius
    BORDER_RADIUS_LG = 10      # Large radius (cards)
    INPUT_HEIGHT = 36          # Default input height


# ---------------------------------------------------------------------------
# Qt Stylesheet
# ---------------------------------------------------------------------------

def build_stylesheet() -> str:
    """Build the complete Qt stylesheet for SCGen7."""
    return f"""
    /* ===== Global ===== */
    * {{
        font-family: {T.FAMILY};
        font-size: {T.SIZE_BASE};
        color: {C.TEXT_PRIMARY};
        outline: none;
    }}

    QMainWindow {{
        background-color: {C.BG_BASE};
    }}

    QWidget {{
        background-color: transparent;
    }}

    /* ===== Scrollbars ===== */
    QScrollBar:vertical {{
        background: {C.SCROLLBAR_BG};
        width: 8px;
        margin: 0;
        border: none;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background: {C.SCROLLBAR_HANDLE};
        min-height: 30px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {C.SCROLLBAR_HOVER};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        background: {C.SCROLLBAR_BG};
        height: 8px;
        margin: 0;
        border: none;
        border-radius: 4px;
    }}
    QScrollBar::handle:horizontal {{
        background: {C.SCROLLBAR_HANDLE};
        min-width: 30px;
        border-radius: 4px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {C.SCROLLBAR_HOVER};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}

    /* ===== Labels ===== */
    QLabel {{
        color: {C.TEXT_PRIMARY};
        background: transparent;
        padding: 0;
    }}
    QLabel[class="secondary"] {{
        color: {C.TEXT_SECONDARY};
        font-size: {T.SIZE_SM};
    }}
    QLabel[class="heading"] {{
        font-size: {T.SIZE_LG};
        font-weight: {T.WEIGHT_SEMIBOLD};
    }}
    QLabel[class="title"] {{
        font-size: {T.SIZE_XL};
        font-weight: {T.WEIGHT_BOLD};
    }}

    /* ===== Buttons ===== */
    QPushButton {{
        background-color: {C.BG_RAISED};
        color: {C.TEXT_PRIMARY};
        border: 1px solid {C.BORDER};
        border-radius: {S.BORDER_RADIUS}px;
        padding: 6px 14px;
        font-size: {T.SIZE_BASE};
        font-weight: {T.WEIGHT_MEDIUM};
        min-height: 28px;
    }}
    QPushButton:hover {{
        background-color: {C.BG_OVERLAY};
        border-color: {C.BORDER_STRONG};
    }}
    QPushButton:pressed {{
        background-color: {C.BG_SURFACE};
    }}
    QPushButton:disabled {{
        color: {C.TEXT_TERTIARY};
        border-color: {C.BORDER};
    }}
    QPushButton[class="primary"] {{
        background-color: {C.ACCENT};
        color: {C.TEXT_INVERSE};
        border: none;
        font-weight: {T.WEIGHT_SEMIBOLD};
    }}
    QPushButton[class="primary"]:hover {{
        background-color: {C.ACCENT_HOVER};
    }}
    QPushButton[class="ghost"] {{
        background-color: transparent;
        border: none;
    }}
    QPushButton[class="ghost"]:hover {{
        background-color: {C.BG_RAISED};
    }}

    /* ===== Inputs ===== */
    QLineEdit, QTextEdit, QPlainTextEdit {{
        background-color: {C.BG_RAISED};
        color: {C.TEXT_PRIMARY};
        border: 1px solid {C.BORDER};
        border-radius: {S.BORDER_RADIUS}px;
        padding: 6px 10px;
        font-size: {T.SIZE_BASE};
        selection-background-color: {C.SELECTION_BG};
        selection-color: {C.SELECTION_TEXT};
    }}
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        border-color: {C.ACCENT};
    }}
    QLineEdit::placeholder {{
        color: {C.TEXT_TERTIARY};
    }}

    /* ===== ComboBox ===== */
    QComboBox {{
        background-color: {C.BG_RAISED};
        color: {C.TEXT_PRIMARY};
        border: 1px solid {C.BORDER};
        border-radius: {S.BORDER_RADIUS}px;
        padding: 5px 10px;
        min-height: 28px;
    }}
    QComboBox:hover {{
        border-color: {C.BORDER_STRONG};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {C.BG_OVERLAY};
        border: 1px solid {C.BORDER_STRONG};
        border-radius: {S.BORDER_RADIUS}px;
        selection-background-color: {C.ACCENT_BG};
        selection-color: {C.TEXT_PRIMARY};
        padding: 4px;
    }}

    /* ===== Tab Widget ===== */
    QTabWidget::pane {{
        border: none;
        background-color: {C.BG_SURFACE};
    }}
    QTabBar {{
        background: transparent;
    }}
    QTabBar::tab {{
        background: transparent;
        color: {C.TEXT_SECONDARY};
        border: none;
        border-bottom: 2px solid transparent;
        padding: 8px 16px;
        font-size: {T.SIZE_SM};
        font-weight: {T.WEIGHT_MEDIUM};
    }}
    QTabBar::tab:selected {{
        color: {C.TEXT_PRIMARY};
        border-bottom-color: {C.ACCENT};
    }}
    QTabBar::tab:hover:!selected {{
        color: {C.TEXT_PRIMARY};
    }}

    /* ===== Splitter ===== */
    QSplitter::handle {{
        background: {C.BORDER};
    }}
    QSplitter::handle:horizontal {{
        width: 1px;
    }}
    QSplitter::handle:vertical {{
        height: 1px;
    }}

    /* ===== List/Tree ===== */
    QListWidget, QTreeWidget {{
        background-color: {C.BG_SURFACE};
        border: none;
        outline: none;
    }}
    QListWidget::item {{
        padding: 6px 12px;
        border: none;
        border-radius: {S.BORDER_RADIUS}px;
        margin: 1px 4px;
    }}
    QListWidget::item:selected {{
        background-color: {C.ACCENT_BG};
        color: {C.TEXT_PRIMARY};
    }}
    QListWidget::item:hover:!selected {{
        background-color: {C.BG_RAISED};
    }}

    /* ===== Group Box ===== */
    QGroupBox {{
        font-size: {T.SIZE_SM};
        font-weight: {T.WEIGHT_SEMIBOLD};
        color: {C.TEXT_SECONDARY};
        border: 1px solid {C.BORDER};
        border-radius: {S.BORDER_RADIUS_LG}px;
        margin-top: 12px;
        padding: 16px 12px 12px 12px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        padding: 0 8px;
    }}

    /* ===== ToolTip ===== */
    QToolTip {{
        background-color: {C.BG_OVERLAY};
        color: {C.TEXT_PRIMARY};
        border: 1px solid {C.BORDER_STRONG};
        border-radius: 4px;
        padding: 6px 10px;
        font-size: {T.SIZE_SM};
    }}

    /* ===== Menu ===== */
    QMenu {{
        background-color: {C.BG_OVERLAY};
        border: 1px solid {C.BORDER_STRONG};
        border-radius: {S.BORDER_RADIUS}px;
        padding: 4px;
    }}
    QMenu::item {{
        padding: 6px 24px 6px 12px;
        border-radius: 4px;
    }}
    QMenu::item:selected {{
        background-color: {C.ACCENT_BG};
    }}
    QMenu::separator {{
        height: 1px;
        background: {C.BORDER};
        margin: 4px 8px;
    }}

    /* ===== Progress Bar ===== */
    QProgressBar {{
        background-color: {C.BG_RAISED};
        border: none;
        border-radius: 3px;
        height: 6px;
        text-align: center;
    }}
    QProgressBar::chunk {{
        background-color: {C.ACCENT};
        border-radius: 3px;
    }}

    /* ===== Checkbox / Radio ===== */
    QCheckBox {{
        spacing: 8px;
        color: {C.TEXT_PRIMARY};
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {C.BORDER_STRONG};
        border-radius: 4px;
        background: {C.BG_RAISED};
    }}
    QCheckBox::indicator:checked {{
        background: {C.ACCENT};
        border-color: {C.ACCENT};
    }}

    /* ===== Slider ===== */
    QSlider::groove:horizontal {{
        height: 4px;
        background: {C.BG_RAISED};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        width: 14px;
        height: 14px;
        margin: -5px 0;
        background: {C.ACCENT};
        border-radius: 7px;
    }}
    """


# Pre-built stylesheet instance
STYLESHEET = build_stylesheet()
