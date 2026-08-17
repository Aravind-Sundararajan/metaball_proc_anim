"""The inspector: every panel one always-visible icon, none of them hidden.

The editor grew eleven panels and put eight of them in a tabified dock, which
Qt renders as a row of tabs at the bottom of the stack. At any dock width worth
giving the panels, that row does not fit: Qt elides the labels until they read
"C… C… S… L…", and turning on scroll buttons — the obvious fix — is worse,
because it takes two panels off screen entirely. Navigation that hides itself
is not navigation.

So the panels move into a stack with a **rail**: one icon per panel, all of
them visible at once, the active one marked in the accent, the name of the
active one spelled out in a header above the panel so the icon never has to
carry the meaning alone. Eleven icons cost about four hundred vertical pixels
in a column that has nine hundred, which is the whole reason a vertical strip
beats a horizontal one here — the window has height to spare and no width.

The rail is on the outer edge, so the reading order is render, then panel, then
the switch that chose it — the arrangement every editor with a side inspector
converges on.
"""

from __future__ import annotations

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRect,
    QSize,
    Signal,
)
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mesh_xml.editor import icons, theme

__all__ = ["Inspector"]

#: Rail button edge in pixels, and the glyph inside it. Big enough to hit
#: without aiming, small enough that eleven of them do not fill the column.
BUTTON_PX = 36
GLYPH_PX = 19

#: How long the marker takes to travel between two buttons, in milliseconds.
#: Shorter than the viewport's camera glide on purpose: that one is carrying a
#: rotation an author has to read, while this is carrying "the mark is now
#: here" over a few dozen pixels, and a slow answer to a click reads as lag
#: rather than as motion.
MARKER_MS = 140

#: Driven by ``QPropertyAnimation`` rather than by the frame loop that eases
#: the camera. The two look alike and are not: the viewport's loop exists to
#: *stop* repainting a raymarched shader when nothing changes, and everything
#: it drives has to answer to that gate. A widget marker has no such gate — Qt
#: repaints it when it moves — so borrowing the loop would couple a dock's
#: chrome to the GL surface's idle policy for nothing.


class Inspector(QWidget):
    """A stack of panels, a rail that switches them, and a header that names one."""

    #: The visible panel changed; the window updates nothing but may want to
    #: know. Carries the panel's title.
    shown = Signal(str)

    #: The author asked for the panels' prose to fold or unfold. The window owns
    #: the answer, because the Scene dock's hints are outside this widget.
    hints_requested = Signal(bool)

    def __init__(
        self, panels: dict[str, QWidget], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)

        self._titles: list[str] = []
        #: Two glyphs per panel, swapped on selection. Qt's checked state does
        #: not reach QIcon's mode, so a checked button keeps drawing the
        #: normal icon — the active panel would wear the accent everywhere
        #: except the one mark an author actually looks at.
        self._glyphs: list[tuple[QIcon, QIcon] | None] = []
        self._stack = QStackedWidget()
        self._heading = QLabel()
        self._heading.setObjectName("inspectorHeading")
        self._buttons = QButtonGroup(self)
        self._buttons.setExclusive(True)

        rail = QWidget()
        rail.setObjectName("rail")
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(4, 6, 4, 6)
        rail_layout.setSpacing(2)

        # Built before the buttons so it sits under them in the sibling stack:
        # the buttons paint transparently, so the marker shows through, and a
        # marker painted *over* a glyph would hide the thing it is marking.
        self._marker = QWidget(rail)
        self._marker.setObjectName("railMarker")
        self._marker.hide()
        self._slide = QPropertyAnimation(self._marker, b"geometry", self)
        self._slide.setDuration(MARKER_MS)
        self._slide.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._rail = rail
        rail.installEventFilter(self)

        for index, (title, widget) in enumerate(panels.items()):
            self._titles.append(title)
            self._stack.addWidget(widget)
            button = QToolButton()
            button.setObjectName("railButton")
            button.setCheckable(True)
            button.setAutoRaise(True)
            button.setFixedSize(QSize(BUTTON_PX, BUTTON_PX))
            button.setIconSize(QSize(GLYPH_PX, GLYPH_PX))
            glyph = icons.panel_icon(title)
            if glyph is not None:
                self._glyphs.append((glyph, icons.panel_icon(title, theme.ACCENT)))
                button.setIcon(glyph)
            else:
                # No glyph is a naming gap, not a crash: show the initial so
                # the panel stays reachable while someone adds the icon.
                self._glyphs.append(None)
                button.setText(title[:1])
            button.setToolTip(title)
            self._buttons.addButton(button, index)
            rail_layout.addWidget(button)

        rail_layout.addStretch(1)

        # The hint toggle rides the heading rather than the toolbar or the menu
        # alone: folded prose that can only be found under &View is prose an
        # author never learns exists, and this is the one strip that is always
        # above the panel whose prose it unfolds.
        self._hints = QToolButton()
        self._hints.setObjectName("hintButton")
        self._hints.setCheckable(True)
        self._hints.setAutoRaise(True)
        self._hints.setIcon(icons.icon("hint"))
        self._hints.setIconSize(QSize(13, 13))
        self._hints.setToolTip("Show the explanatory prose under these controls (F1)")
        self._hints.toggled.connect(self.hints_requested.emit)

        heading_row = QWidget()
        heading_row.setObjectName("inspectorHeader")
        heading_layout = QHBoxLayout(heading_row)
        heading_layout.setContentsMargins(0, 0, 6, 0)
        heading_layout.setSpacing(0)
        heading_layout.addWidget(self._heading, 1)
        heading_layout.addWidget(self._hints)

        panel_side = QWidget()
        panel_layout = QVBoxLayout(panel_side)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        panel_layout.addWidget(heading_row)
        panel_layout.addWidget(self._stack, 1)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(panel_side, 1)
        layout.addWidget(rail)

        self._buttons.idClicked.connect(self.show_index)
        if self._titles:
            self.show_index(0)

    def show_index(self, index: int) -> None:
        """Raise panel ``index`` and mark its rail button."""
        if not (0 <= index < len(self._titles)):
            return
        self._stack.setCurrentIndex(index)
        button = self._buttons.button(index)
        if button is not None and not button.isChecked():
            button.setChecked(True)
        for slot, glyphs in enumerate(self._glyphs):
            other = self._buttons.button(slot)
            if other is not None and glyphs is not None:
                other.setIcon(glyphs[1] if slot == index else glyphs[0])
        self._heading.setText(self._titles[index].upper())
        self._place_marker(animate=True)
        self.shown.emit(self._titles[index])

    def eventFilter(self, watched: QWidget, event) -> bool:
        """Keep the marker on its button when the rail is laid out or resized.

        The marker's geometry is set by hand rather than by the layout, so it
        has no idea the buttons have moved. It also cannot be placed in
        ``__init__``: nothing has been laid out yet there, every button reports
        a null rectangle, and the marker would open life as a stripe in the
        corner. Show and resize are the two moments the truth changes.
        """
        if watched is self._rail and event.type() in (
            event.Type.Show,
            event.Type.Resize,
        ):
            self._place_marker(animate=False)
        return super().eventFilter(watched, event)

    def _place_marker(self, animate: bool) -> None:
        """Move the mark to the checked button, sliding or snapping."""
        button = self._buttons.button(self._stack.currentIndex())
        if button is None or not button.width():
            # Pre-layout: no geometry to move to yet. The show event that
            # follows will place it, so leaving it hidden is the honest state.
            return
        self._slide.stop()
        target = button.geometry()
        if animate and self._marker.isVisible():
            self._slide.setStartValue(self._marker.geometry())
            self._slide.setEndValue(target)
            self._slide.start()
        else:
            # Nothing to slide *from* on the first placement, and a marker that
            # flew in from the corner on open would announce a change that had
            # not happened.
            self._marker.setGeometry(QRect(target))
        self._marker.show()

    def show_panel(self, title: str) -> None:
        """Raise the panel with this dock title, if there is one."""
        if title in self._titles:
            self.show_index(self._titles.index(title))

    def set_hints_shown(self, shown: bool) -> None:
        """Mark the toggle without re-asking the window that just answered."""
        if self._hints.isChecked() != shown:
            self._hints.blockSignals(True)
            self._hints.setChecked(shown)
            self._hints.blockSignals(False)

    @property
    def current(self) -> str:
        """The visible panel's title."""
        return self._titles[self._stack.currentIndex()] if self._titles else ""
