"""Modern stylesheet for SC Gen 6 UI - Slate theme with violet accent."""

# Modern slate color palette (Clean, sophisticated, professional)
COLORS = {
    # Primary accent - refined violet/indigo
    "primary": "#8b7cf6",           # Soft violet
    "primary_hover": "#a29bfa",     # Lighter violet on hover
    "primary_light": "#c4b5fd",     # Light violet for highlights
    "primary_muted": "#6366f1",     # Deeper indigo for secondary

    # Background colors - pure slate/charcoal
    "bg_dark": "#0f0f12",           # Near black with slight blue tint
    "bg_medium": "#16161a",         # Dark slate
    "bg_light": "#1e1e24",          # Lighter slate
    "bg_card": "#1a1a20",           # Card background
    "bg_hover": "#252530",          # Hover state
    "bg_input": "#121215",          # Input fields (slightly darker)

    # Text colors - clean whites
    "text_primary": "#f4f4f5",      # Pure cool white
    "text_secondary": "#a1a1aa",    # Medium gray
    "text_muted": "#71717a",        # Muted gray
    "text_highlight": "#ffffff",    # Pure white

    # Accent colors - refined and consistent
    "success": "#4ade80",           # Vibrant green
    "warning": "#fbbf24",           # Clean amber
    "error": "#f87171",             # Soft red
    "info": "#60a5fa",              # Sky blue

    # Border colors
    "border": "#27272a",            # Subtle zinc border
    "border_light": "#3f3f46",      # Lighter border
    "border_focus": "#8b7cf6",      # Violet focus ring
}

# Font configuration - narrow and crisp
FONT_FAMILY = "'Barlow Condensed', 'Roboto Condensed', 'Arial Narrow', 'Segoe UI', system-ui, sans-serif"
FONT_FAMILY_BODY = "'Barlow', 'Roboto', 'Segoe UI', system-ui, sans-serif"


def get_modern_stylesheet() -> str:
    """Get modern warm stylesheet for the application."""
    return f"""
    /* ==========================================================================
       MAIN WINDOW & BASE STYLES
       ========================================================================== */

    QMainWindow {{
        background-color: {COLORS['bg_dark']};
        color: {COLORS['text_primary']};
    }}

    QWidget {{
        background-color: {COLORS['bg_dark']};
        color: {COLORS['text_primary']};
        font-family: {FONT_FAMILY_BODY};
        font-size: 10pt;
    }}

    /* ==========================================================================
       SPLITTERS - Minimal and clean
       ========================================================================== */

    QSplitter::handle {{
        background-color: {COLORS['bg_dark']};
        border: none;
        width: 2px;
        height: 2px;
    }}

    QSplitter::handle:hover {{
        background-color: {COLORS['primary_muted']};
    }}

    /* ==========================================================================
       PANELS & CONTAINERS
       ========================================================================== */

    QGroupBox {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        margin-top: 1.5em;
        padding: 14px;
        font-weight: 500;
        color: {COLORS['text_primary']};
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 10px;
        color: {COLORS['text_highlight']};
        font-family: {FONT_FAMILY};
        font-weight: 600;
        font-size: 11pt;
        letter-spacing: 0.5px;
        background-color: {COLORS['bg_dark']};
    }}

    QFrame {{
        border: none;
    }}

    QFrame[frameShape="4"] {{ /* HLine */
        background-color: {COLORS['border']};
        max-height: 1px;
    }}
    
    QFrame[frameShape="5"] {{ /* VLine */
        background-color: {COLORS['border']};
        max-width: 1px;
    }}

    /* ==========================================================================
       BUTTONS - Clean and modern
       ========================================================================== */

    QPushButton {{
        background-color: {COLORS['primary']};
        color: {COLORS['bg_dark']};
        border: none;
        border-radius: 8px;
        padding: 10px 18px;
        font-family: {FONT_FAMILY};
        font-weight: 600;
        font-size: 10pt;
        letter-spacing: 0.3px;
        min-height: 20px;
    }}

    QPushButton:hover {{
        background-color: {COLORS['primary_hover']};
    }}

    QPushButton:pressed {{
        background-color: {COLORS['primary_muted']};
    }}

    QPushButton:disabled {{
        background-color: {COLORS['bg_light']};
        color: {COLORS['text_muted']};
    }}

    /* Secondary button style - outline */
    QPushButton[styleClass="secondary"] {{
        background-color: transparent;
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border_light']};
    }}

    QPushButton[styleClass="secondary"]:hover {{
        background-color: {COLORS['bg_hover']};
        border-color: {COLORS['text_secondary']};
    }}

    QPushButton[styleClass="secondary"]:checked {{
        background-color: {COLORS['bg_hover']};
        border-color: {COLORS['primary']};
        color: {COLORS['primary']};
    }}

    /* Danger button style */
    QPushButton[styleClass="danger"] {{
        background-color: transparent;
        color: {COLORS['error']};
        border: 1px solid {COLORS['border']};
    }}
    
    QPushButton[styleClass="danger"]:hover {{
        background-color: {COLORS['error']};
        color: white;
        border-color: {COLORS['error']};
    }}

    /* Ghost/Icon buttons */
    QPushButton[styleClass="ghost"] {{
        background-color: transparent;
        color: {COLORS['text_secondary']};
        border: none;
        padding: 8px 12px;
    }}

    QPushButton[styleClass="ghost"]:hover {{
        background-color: {COLORS['bg_light']};
        color: {COLORS['text_primary']};
    }}

    /* Navigation buttons */
    QPushButton[styleClass="nav"] {{
        background-color: transparent;
        color: {COLORS['text_secondary']};
        border: none;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: left;
        font-weight: 500;
    }}

    QPushButton[styleClass="nav"]:hover {{
        background-color: {COLORS['bg_hover']};
        color: {COLORS['text_primary']};
    }}

    QPushButton[styleClass="nav"]:checked {{
        background-color: {COLORS['bg_light']};
        color: {COLORS['primary']};
        border-left: 3px solid {COLORS['primary']};
        border-radius: 0 8px 8px 0;
    }}

    /* ==========================================================================
       TEXT INPUTS - Refined and subtle
       ========================================================================== */

    QTextEdit, QPlainTextEdit, QLineEdit {{
        background-color: {COLORS['bg_input']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 12px;
        color: {COLORS['text_primary']};
        selection-background-color: {COLORS['primary_muted']};
        selection-color: {COLORS['text_highlight']};
    }}

    QTextEdit:focus, QPlainTextEdit:focus, QLineEdit:focus {{
        border: 1px solid {COLORS['border_focus']};
        background-color: {COLORS['bg_input']};
    }}

    QLineEdit::placeholder {{
        color: {COLORS['text_muted']};
    }}

    /* ==========================================================================
       COMBO BOXES
       ========================================================================== */

    QComboBox {{
        background-color: {COLORS['bg_input']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 8px 14px;
        color: {COLORS['text_primary']};
        min-height: 24px;
        font-family: {FONT_FAMILY};
    }}

    QComboBox:hover {{
        border-color: {COLORS['border_light']};
    }}

    QComboBox:focus {{
        border-color: {COLORS['border_focus']};
    }}

    QComboBox::drop-down {{
        border: none;
        padding-right: 10px;
    }}

    QComboBox::down-arrow {{
        image: url(none);
        border: none;
        width: 0;
        height: 0;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid {COLORS['text_secondary']};
        margin-right: 8px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        color: {COLORS['text_primary']};
        selection-background-color: {COLORS['primary_muted']};
        padding: 6px;
        outline: none;
    }}

    /* ==========================================================================
       LISTS & TREES
       ========================================================================== */

    QListWidget, QTreeWidget {{
        background-color: {COLORS['bg_medium']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 6px;
        color: {COLORS['text_primary']};
        outline: none;
    }}

    QListWidget::item, QTreeWidget::item {{
        padding: 10px 12px;
        border-radius: 6px;
        margin: 2px 0;
    }}

    QListWidget::item:hover, QTreeWidget::item:hover {{
        background-color: {COLORS['bg_hover']};
    }}

    QListWidget::item:selected, QTreeWidget::item:selected {{
        background-color: {COLORS['primary_muted']};
        color: {COLORS['text_highlight']};
    }}
    
    QHeaderView::section {{
        background-color: {COLORS['bg_light']};
        color: {COLORS['text_secondary']};
        padding: 8px 12px;
        border: none;
        border-bottom: 1px solid {COLORS['border']};
        font-family: {FONT_FAMILY};
        font-weight: 600;
        letter-spacing: 0.5px;
    }}

    /* ==========================================================================
       CHECKBOXES - Clean custom style
       ========================================================================== */

    QCheckBox {{
        spacing: 10px;
        color: {COLORS['text_primary']};
    }}

    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 5px;
        border: 2px solid {COLORS['border_light']};
        background-color: {COLORS['bg_input']};
    }}

    QCheckBox::indicator:hover {{
        border-color: {COLORS['primary_muted']};
    }}

    QCheckBox::indicator:checked {{
        background-color: {COLORS['primary']};
        border-color: {COLORS['primary']};
    }}

    /* ==========================================================================
       LABELS
       ========================================================================== */

    QLabel {{
        color: {COLORS['text_primary']};
        background-color: transparent;
    }}

    QLabel[styleClass="title"] {{
        font-family: {FONT_FAMILY};
        font-size: 20pt;
        font-weight: 600;
        letter-spacing: 1px;
        color: {COLORS['text_highlight']};
        padding: 4px 0;
    }}

    QLabel[styleClass="subtitle"] {{
        font-family: {FONT_FAMILY};
        font-size: 13pt;
        font-weight: 500;
        color: {COLORS['text_primary']};
        padding: 4px 0;
    }}

    QLabel[styleClass="section"] {{
        font-family: {FONT_FAMILY};
        font-size: 9pt;
        font-weight: 600;
        color: {COLORS['text_muted']};
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding: 10px 0 6px 0;
    }}

    QLabel[styleClass="muted"] {{
        color: {COLORS['text_muted']};
        font-size: 9pt;
    }}
    
    QLabel[styleClass="status"] {{
        font-family: {FONT_FAMILY};
        font-weight: 600;
        padding: 6px 10px;
        border-radius: 6px;
        background-color: {COLORS['bg_light']};
    }}

    /* ==========================================================================
       SLIDERS - Refined
       ========================================================================== */

    QSlider::groove:horizontal {{
        border: none;
        height: 4px;
        background: {COLORS['bg_light']};
        border-radius: 2px;
    }}

    QSlider::handle:horizontal {{
        background: {COLORS['primary']};
        border: none;
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}

    QSlider::handle:horizontal:hover {{
        background: {COLORS['primary_hover']};
    }}

    QSlider::sub-page:horizontal {{
        background: {COLORS['primary_muted']};
        border-radius: 2px;
    }}

    /* ==========================================================================
       SCROLLBARS - Minimal and elegant
       ========================================================================== */

    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 0;
    }}

    QScrollBar::handle:vertical {{
        background: {COLORS['border_light']};
        border-radius: 4px;
        min-height: 40px;
        margin: 2px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: {COLORS['text_muted']};
    }}

    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        background: none;
        border: none;
    }}

    QScrollBar:horizontal {{
        background: transparent;
        height: 8px;
        margin: 0;
    }}

    QScrollBar::handle:horizontal {{
        background: {COLORS['border_light']};
        border-radius: 4px;
        min-width: 40px;
        margin: 2px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background: {COLORS['text_muted']};
    }}

    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal,
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        background: none;
        border: none;
    }}

    /* ==========================================================================
       TAB WIDGET
       ========================================================================== */

    QTabWidget::pane {{
        border: 1px solid {COLORS['border']};
        background-color: {COLORS['bg_card']};
        border-radius: 8px;
        top: -1px;
    }}

    QTabBar::tab {{
        background-color: transparent;
        color: {COLORS['text_muted']};
        padding: 12px 24px;
        border: none;
        border-bottom: 2px solid transparent;
        font-family: {FONT_FAMILY};
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-right: 4px;
    }}

    QTabBar::tab:selected {{
        color: {COLORS['primary']};
        border-bottom: 2px solid {COLORS['primary']};
    }}

    QTabBar::tab:hover:!selected {{
        color: {COLORS['text_primary']};
        background-color: {COLORS['bg_hover']};
        border-radius: 6px 6px 0 0;
    }}

    /* ==========================================================================
       MENU BAR
       ========================================================================== */

    QMenuBar {{
        background-color: {COLORS['bg_dark']};
        color: {COLORS['text_primary']};
        padding: 4px;
        border-bottom: 1px solid {COLORS['border']};
    }}

    QMenuBar::item {{
        padding: 8px 14px;
        border-radius: 6px;
        background: transparent;
    }}

    QMenuBar::item:selected {{
        background-color: {COLORS['bg_light']};
    }}

    QMenu {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 6px;
    }}

    QMenu::item {{
        padding: 10px 24px;
        border-radius: 6px;
        color: {COLORS['text_primary']};
    }}

    QMenu::item:selected {{
        background-color: {COLORS['primary_muted']};
        color: {COLORS['text_highlight']};
    }}

    /* ==========================================================================
       STATUS BAR
       ========================================================================== */

    QStatusBar {{
        background-color: {COLORS['bg_dark']};
        color: {COLORS['text_muted']};
        border-top: 1px solid {COLORS['border']};
        font-size: 9pt;
    }}

    /* ==========================================================================
       PROGRESS BARS
       ========================================================================== */

    QProgressBar {{
        background-color: {COLORS['bg_light']};
        border-radius: 4px;
        text-align: center;
        color: {COLORS['text_secondary']};
        font-size: 9pt;
    }}

    QProgressBar::chunk {{
        background-color: {COLORS['primary']};
        border-radius: 4px;
    }}

    /* ==========================================================================
       SPIN BOXES
       ========================================================================== */

    QSpinBox, QDoubleSpinBox {{
        background-color: {COLORS['bg_input']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 6px 10px;
        color: {COLORS['text_primary']};
    }}

    QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {COLORS['border_focus']};
    }}

    QSpinBox::up-button, QSpinBox::down-button,
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        background-color: {COLORS['bg_light']};
        border: none;
        border-radius: 3px;
        width: 16px;
        margin: 2px;
    }}

    QSpinBox::up-button:hover, QSpinBox::down-button:hover,
    QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: {COLORS['bg_hover']};
    }}

    /* ==========================================================================
       DATE/TIME EDITS
       ========================================================================== */

    QDateEdit, QTimeEdit, QDateTimeEdit {{
        background-color: {COLORS['bg_input']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 6px 10px;
        color: {COLORS['text_primary']};
    }}

    QDateEdit:focus, QTimeEdit:focus, QDateTimeEdit:focus {{
        border-color: {COLORS['border_focus']};
    }}

    /* ==========================================================================
       TOOLTIPS
       ========================================================================== */

    QToolTip {{
        background-color: {COLORS['bg_card']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 9pt;
    }}

    /* ==========================================================================
       DIALOGS
       ========================================================================== */

    QDialog {{
        background-color: {COLORS['bg_dark']};
    }}

    QMessageBox {{
        background-color: {COLORS['bg_dark']};
    }}

    QMessageBox QLabel {{
        color: {COLORS['text_primary']};
    }}

    /* ==========================================================================
       LOADING STATES
       ========================================================================== */

    QProgressBar {{
        background-color: {COLORS['bg_input']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        text-align: center;
        color: {COLORS['text_primary']};
        min-height: 20px;
    }}

    QProgressBar::chunk {{
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {COLORS['primary_muted']}, 
            stop:1 {COLORS['primary']});
        border-radius: 5px;
    }}

    /* Pulsing style for indeterminate progress */
    QProgressBar[styleClass="pulsing"]::chunk {{
        background-color: {COLORS['primary']};
    }}

    /* ==========================================================================
       STATUS BADGES
       ========================================================================== */

    QLabel[styleClass="badge"] {{
        background-color: {COLORS['bg_card']};
        color: {COLORS['text_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 4px 10px;
        font-size: 9pt;
        font-weight: 500;
    }}

    QLabel[styleClass="badge-success"] {{
        background-color: rgba(125, 184, 125, 0.15);
        color: {COLORS['success']};
        border-color: {COLORS['success']};
    }}

    QLabel[styleClass="badge-warning"] {{
        background-color: rgba(212, 165, 74, 0.15);
        color: {COLORS['warning']};
        border-color: {COLORS['warning']};
    }}

    QLabel[styleClass="badge-error"] {{
        background-color: rgba(196, 90, 90, 0.15);
        color: {COLORS['error']};
        border-color: {COLORS['error']};
    }}

    /* ==========================================================================
       GHOST BUTTONS (Minimal styling)
       ========================================================================== */

    QPushButton[styleClass="ghost"] {{
        background-color: transparent;
        color: {COLORS['text_secondary']};
        border: none;
        padding: 6px 12px;
    }}

    QPushButton[styleClass="ghost"]:hover {{
        background-color: {COLORS['bg_hover']};
        color: {COLORS['text_primary']};
    }}

    /* ==========================================================================
       LOADING SKELETON
       ========================================================================== */

    QFrame[styleClass="skeleton"] {{
        background-color: {COLORS['bg_card']};
        border-radius: 4px;
    }}
    """
