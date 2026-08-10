"""The editor's skin: the viewport's palette, worn by the chrome.

The app has had a visual identity since the raymarcher first drew a frame —
a dark field, ivory chunk dots, link blue, selection orange. This module
does not invent a second one; every constant below is sourced from a value
the render already owns, and ``tests/test_theme.py`` holds them to their
sources so the chrome cannot drift away from the viewport it frames
(``docs/CHROME.md``).

Fusion is the base style because it is the one Qt style that obeys a dark
palette everywhere — indicators, sliders, menus — where the platform styles
paint half their widgets from OS resources a palette never reaches. The
stylesheet then covers what palettes cannot say: group-box framing, dock
titles, slim scrollbars, accent focus rings.

Applied only by :func:`apply`, which only ``main()`` calls: tests and
embedders get stock widgets unless they opt in, so no assertion elsewhere
quietly starts depending on a stylesheet.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

__all__ = ["ACCENT", "FIELD", "INK", "INFO", "WINDOW", "apply"]

# ----------------------------------------------------------------------
# the palette, with each colour's source named
# ----------------------------------------------------------------------

#: The raymarcher's background — MetaballSettings.background_colour
#: (0.11, 0.12, 0.14) — as chrome sees it. Deepest tone in the app; the
#: viewport keeps it, and every panel surface is a legible step above it.
FIELD = "#1c1f24"

#: Panel background: one step off FIELD, so docks read as furniture
#: standing in front of the render rather than holes cut into it.
WINDOW = "#23262d"

#: Raised controls (buttons, headers), and the hover a step above that.
RAISED = "#2c3039"
HOVER = "#363b46"

#: Borders and separators — visible against WINDOW, quiet against RAISED.
BORDER = "#3d434f"

#: Text: the rig overlay's chunk-dot ivory, viewport.CHUNK_COLOUR.
INK = "#e8eef8"

#: Secondary text — INK stepped down, for notes and disabled-adjacent copy.
DIM = "#9aa3b4"

#: Disabled text: dim enough to read as off, light enough to read at all.
QUIET = "#5d6474"

#: The accent: viewport.SELECTED_COLOUR, the orange the app has always
#: meant "chosen" by. Selection highlights, focus rings, checked states.
ACCENT = "#ffba48"

#: Ink for text sitting on the accent.
ON_ACCENT = "#1f1607"

#: Informational blue: viewport.LINK_COLOUR, the overlay's link stroke.
INFO = "#78beeb"


def _palette() -> QPalette:
    palette = QPalette()
    groups = (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive)
    roles = {
        QPalette.ColorRole.Window: WINDOW,
        QPalette.ColorRole.WindowText: INK,
        QPalette.ColorRole.Base: FIELD,
        QPalette.ColorRole.AlternateBase: WINDOW,
        QPalette.ColorRole.Text: INK,
        QPalette.ColorRole.Button: RAISED,
        QPalette.ColorRole.ButtonText: INK,
        QPalette.ColorRole.ToolTipBase: RAISED,
        QPalette.ColorRole.ToolTipText: INK,
        QPalette.ColorRole.Highlight: ACCENT,
        QPalette.ColorRole.HighlightedText: ON_ACCENT,
        QPalette.ColorRole.Link: INFO,
        QPalette.ColorRole.PlaceholderText: QUIET,
        QPalette.ColorRole.Light: HOVER,
        QPalette.ColorRole.Midlight: BORDER,
        QPalette.ColorRole.Mid: BORDER,
        QPalette.ColorRole.Dark: FIELD,
        QPalette.ColorRole.Shadow: "#101216",
    }
    for group in groups:
        for role, colour in roles.items():
            palette.setColor(group, role, QColor(colour))
    disabled = QPalette.ColorGroup.Disabled
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(disabled, role, QColor(QUIET))
    palette.setColor(disabled, QPalette.ColorRole.Highlight, QColor(BORDER))
    palette.setColor(disabled, QPalette.ColorRole.HighlightedText, QColor(DIM))
    return palette


QSS = f"""
QWidget {{
    color: {INK};
}}
QMainWindow, QDialog {{
    background: {WINDOW};
}}

QMenuBar {{
    background: {WINDOW};
    border-bottom: 1px solid {BORDER};
}}
QMenuBar::item {{
    padding: 4px 10px;
    background: transparent;
}}
QMenuBar::item:selected {{
    background: {HOVER};
    border-radius: 4px;
}}
QMenu {{
    background: {RAISED};
    border: 1px solid {BORDER};
    padding: 4px;
}}
QMenu::item {{
    padding: 5px 24px 5px 12px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background: {ACCENT};
    color: {ON_ACCENT};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 8px;
}}

QDockWidget {{
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}
QDockWidget::title {{
    background: {WINDOW};
    padding: 6px 10px;
    border-bottom: 1px solid {BORDER};
    text-transform: uppercase;
    font-size: 8pt;
    letter-spacing: 1px;
    color: {DIM};
}}
QTabBar::tab {{
    background: {WINDOW};
    color: {DIM};
    padding: 5px 12px;
    border: 1px solid {BORDER};
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}}
QTabBar::tab:selected {{
    background: {RAISED};
    color: {INK};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover:!selected {{
    background: {HOVER};
}}

QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: {DIM};
    font-size: 8pt;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

QPushButton {{
    background: {RAISED};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 5px 14px;
}}
QPushButton:hover {{
    background: {HOVER};
    border-color: {DIM};
}}
QPushButton:pressed {{
    background: {FIELD};
}}
QPushButton:checked {{
    background: {ACCENT};
    color: {ON_ACCENT};
    border-color: {ACCENT};
}}
QPushButton:disabled {{
    color: {QUIET};
    background: {WINDOW};
}}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {FIELD};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 3px 8px;
    selection-background-color: {ACCENT};
    selection-color: {ON_ACCENT};
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
    border-color: {ACCENT};
}}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QLineEdit:disabled {{
    color: {QUIET};
    background: {WINDOW};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background: {RAISED};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    selection-color: {ON_ACCENT};
}}

QTableWidget {{
    background: {FIELD};
    gridline-color: {WINDOW};
    border: 1px solid {BORDER};
    border-radius: 5px;
}}
QHeaderView::section {{
    background: {WINDOW};
    color: {DIM};
    border: none;
    border-bottom: 1px solid {BORDER};
    border-right: 1px solid {BORDER};
    padding: 4px 8px;
    font-size: 8pt;
}}
/* No uppercase or letter-spacing on table headers, deliberately: Qt
   measures section widths from the *untransformed* text, so styling that
   widens text after measurement truncates every narrow column — the
   Rooms and Doors tables shipped reading "AN|" and "ROC". Dock titles
   and group-box titles keep the treatment; nothing measures them. */
QTableWidget::item:selected {{
    background: {ACCENT};
    color: {ON_ACCENT};
}}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {DIM};
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: 5px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {DIM};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}

QToolBar {{
    background: {WINDOW};
    border-bottom: 1px solid {BORDER};
    padding: 3px 6px;
    spacing: 2px;
}}
QToolBar::separator {{
    width: 1px;
    background: {BORDER};
    margin: 5px 6px;
}}
QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 4px 12px;
}}
QToolButton:hover {{
    background: {HOVER};
    border-color: {BORDER};
}}
QToolButton:pressed {{
    background: {FIELD};
}}
QToolButton:checked {{
    background: {ACCENT};
    color: {ON_ACCENT};
}}
QToolButton:disabled {{
    color: {QUIET};
}}

QToolTip {{
    background: {RAISED};
    color: {INK};
    border: 1px solid {BORDER};
    padding: 5px 8px;
}}

QStatusBar {{
    background: {WINDOW};
    border-top: 1px solid {BORDER};
    color: {DIM};
}}

QScrollArea, QAbstractScrollArea {{
    background: transparent;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
"""


def apply(app: QApplication) -> None:
    """Dress the application: Fusion, the palette, the stylesheet.

    Idempotent and cheap; call once from the entrypoint. Widgets created
    before or after both take the style — Qt repolishes on set.
    """
    app.setStyle("Fusion")
    app.setPalette(_palette())
    app.setStyleSheet(QSS)
