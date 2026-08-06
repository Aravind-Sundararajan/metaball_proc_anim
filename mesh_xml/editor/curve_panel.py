"""The radius curve: authored in curve space, applied to a chain as one edit.

The retired application had a radius-curve editor, and deleting it took the
workflow with it. :class:`~mesh_xml.core.curve.Curve1D` was written in Phase 1 to
replace the sampling half of that widget, has been fully tested since, and until
now was reachable from nothing at all. This panel is the other half.

The division of labour is the entire point, and it is the thing the old widget
got wrong. ``Curve1D`` owns the Bezier maths, the key ordering and the codomain
clamp, and it knows nothing about pixels. This panel holds a ``Curve1D`` as its
model and drives it in curve space. Exactly one object — :class:`_CurveView` —
knows that a curve-space point has a pixel address, and it is an affine map and
its algebraic inverse, nothing more. The old editor had a ``2.0 - (y / 180) *
1.9`` pixel-to-radius conversion pasted into eight separate methods, which is
precisely why its sampler and its renderer disagreed about the curve the author
had drawn; there is one such conversion here and it converts nothing but
coordinates.

**The curve is drawn from its own samples, not from ``cubicTo``.** A ``QPainter``
path built out of :meth:`Curve1D.segment_control_points` would be the obvious
renderer, and the control points would be right. It would still be able to lie:
``y_range`` clamps at read time, so a key dragged toward zero draws as a dipping
cubic while :meth:`Curve1D.sample_many` returns the flat clamped floor that Apply
would actually write. Drawing the polyline straight from ``sample_many`` makes a
renderer/sampler disagreement structurally impossible rather than merely
tested-for, and it is the only way the clamp appears in the picture. The control
points are still read from ``segment_control_points`` for the handles, because
that method is the curve's single source of truth for which handle belongs to
which side of a segment.

**``x`` is normalised position along the chain, never arc length.** Chunk *i* of
*n* samples at ``i / (n - 1)``. Arc length is the more tempting domain for a
taper and it is wrong here for a blunt reason: the rig moves. Arc length changes
every tick the solver runs, so the same curve applied twice would produce two
different rigs and a curve edit would stop being reproducible. Index position is
a property of the topology, which is what the author is actually addressing.

**Applying is one undo entry.** Every :class:`~mesh_xml.editor.document.SetRadius`
goes into a single :class:`~mesh_xml.editor.document.Macro`, so a curve edit
undoes as a curve edit. Twenty separate commands on the stack would make undo
useless for exactly the operation that produces the most of them.

Nothing here is stored on the document. There is no curve in the rig format and
no field for one on :class:`~mesh_xml.editor.document.Document`, so the curve is
panel state announced through :attr:`CurvePanel.curve_changed` rather than
smuggled onto the document behind its own contract — the same reasoning that
keeps ``SurfacePanel.show_rig_overlay`` where it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QMouseEvent, QPainter, QPalette, QPen, QPolygonF
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from mesh_xml.core.curve import Curve1D
from mesh_xml.editor.document import Document, Macro, SetRadius

# The latch, the form helpers and the note style are deliberately shared with the
# five original panels rather than re-implemented: a second copy of the
# reentrancy guard is exactly the scatter of ``updating_position`` booleans that
# module exists to have deleted.
from mesh_xml.editor.panels import _form, _note, _Panel, _spin, _user_edit

__all__ = ["CurvePanel"]

#: Radius bounds the curve is allowed to report. ``Body.set_radius`` refuses
#: anything non-positive, so a curve feeding it needs a codomain that cannot
#: reach zero however far a handle is dragged.
MIN_RADIUS = 0.01
MAX_RADIUS = 4.0
Y_RANGE = (MIN_RADIUS, MAX_RADIUS)

#: Where a curve starts when there is no rig to read a level off.
DEFAULT_RADIUS = 0.3

#: How far outside the chain a key may be pushed. ``x`` beyond ``[0, 1]`` is
#: perfectly legal — ``Curve1D`` holds the end value out there rather than
#: extrapolating — but no chunk can ever land on it, so the editor bounds it
#: rather than letting the numeric field and the plot disagree about a key the
#: spin box's own range cannot express.
X_LIMIT = 2.0

#: A curve with one key is a constant and has no segment to draw a handle on, so
#: the editor never lets the author remove its way down to one.
MIN_KEYS = 2

_MARGIN = 12.0
_MIN_WIDTH, _MIN_HEIGHT = 260, 170
_HIT_RADIUS = 8.0
_KEY_RADIUS = 4.0
_HANDLE_RADIUS = 3.0
_CHUNK_RADIUS = 2.5
_GRID_LINES = 4
_VIEW_PAD = 0.10
_MIN_SPAN = 1e-6
_READOUT = ".4f"
_SPIN_STEP = 0.01


class _Part(Enum):
    """Which piece of a key a drag has hold of."""

    KEY = "key"
    IN = "in"
    OUT = "out"


@dataclass(frozen=True)
class _Grab:
    """A drag in flight: what is held, and where it sits relative to the cursor.

    The offset is kept in pixels because the view window is fixed for the
    duration of a drag, and it is what stops a key from snapping its centre onto
    the cursor the instant it is picked up.
    """

    part: _Part
    index: int
    dx: float
    dy: float


@dataclass(frozen=True)
class _CurveView:
    """The one place curve space meets pixels.

    An affine map and its algebraic inverse, so a point pushed out to the screen
    and dragged back in lands where it started instead of creeping by a fraction
    of a unit per drag. ``y`` flips, because pixels grow downward and radii do
    not.

    Every span is floored: Qt really does deliver zero-height widgets, and a
    view that divides by the span would surface as ZeroDivisionError from inside
    a paint callback.
    """

    width: float
    height: float
    x0: float
    x1: float
    y0: float
    y1: float
    margin: float = _MARGIN

    @property
    def span_x(self) -> float:
        return max(self.x1 - self.x0, _MIN_SPAN)

    @property
    def span_y(self) -> float:
        return max(self.y1 - self.y0, _MIN_SPAN)

    def plot(self) -> QRectF:
        """The drawable rectangle, inset by the margin."""
        return QRectF(
            self.margin,
            self.margin,
            max(self.width - 2.0 * self.margin, 1.0),
            max(self.height - 2.0 * self.margin, 1.0),
        )

    def to_pixel(self, x: float, y: float) -> QPointF:
        plot = self.plot()
        return QPointF(
            plot.left() + (x - self.x0) / self.span_x * plot.width(),
            plot.top() + (self.y1 - y) / self.span_y * plot.height(),
        )

    def to_curve(self, px: float, py: float) -> tuple[float, float]:
        plot = self.plot()
        return (
            self.x0 + (px - plot.left()) / plot.width() * self.span_x,
            self.y1 - (py - plot.top()) / plot.height() * self.span_y,
        )


def _padded(lo: float, hi: float) -> tuple[float, float]:
    """Widen ``[lo, hi]`` so its contents do not sit on the frame."""
    span = hi - lo
    pad = span * _VIEW_PAD if span > _MIN_SPAN else max(abs(hi), 1.0) * _VIEW_PAD
    return (lo - pad, hi + pad)


def _distance(a: QPointF, b: QPointF) -> float:
    return math.hypot(a.x() - b.x(), a.y() - b.y())


class _CurveCanvas(QWidget):
    """The curve drawn, and the only widget that turns a mouse into curve space.

    Handles are shown and draggable for the selected key alone. Every key's
    handles at once turns a four-key taper into twelve targets inside a
    300-pixel box, where they overlap each other and their own keys and the hit
    tests start fighting; selecting first is what every curve editor asks for and
    it costs one click.
    """

    #: The curve was mutated by a drag.
    edited = Signal()
    #: The user picked a different key. Only ever emitted from a mouse
    #: interaction — :meth:`select` is silent, so a panel writing the selection
    #: back cannot bounce.
    key_selected = Signal(int)

    def __init__(self, curve: Curve1D, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._curve = curve
        self._selected = 0
        self._grab: _Grab | None = None
        self._chunks: np.ndarray = np.zeros(0, dtype=np.float64)
        self._marker: float | None = None
        self._window: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)

        self.setMinimumSize(_MIN_WIDTH, _MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.fit()

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    @property
    def curve(self) -> Curve1D:
        return self._curve

    @property
    def selected(self) -> int:
        return self._selected

    @property
    def window(self) -> tuple[float, float, float, float]:
        """The visible curve-space rectangle, ``(x0, x1, y0, y1)``."""
        return self._window

    def set_curve(self, curve: Curve1D) -> None:
        self._curve = curve
        self._grab = None
        self.select(0)

    def select(self, index: int) -> None:
        """Move the selection without announcing it."""
        count = len(self._curve)
        self._selected = min(max(index, 0), count - 1) if count else 0
        self.update()

    def set_chunk_positions(self, positions: np.ndarray) -> None:
        """The curve-space x each chunk of the chain will be sampled at."""
        self._chunks = np.asarray(positions, dtype=np.float64).ravel()
        self.update()

    def set_marker(self, x: float | None) -> None:
        """Where the numeric readout is reading, drawn so the two agree visibly."""
        self._marker = None if x is None else float(x)
        self.update()

    def view(self) -> _CurveView:
        return _CurveView(float(self.width()), float(self.height()), *self._window)

    def fit(self) -> None:
        """Frame the whole curve. Explicit, never automatic.

        Re-fitting on every mutation would rescale the plot underneath a key the
        author is still holding, so the key would chase the cursor and never
        arrive.
        """
        xs = [0.0, 1.0]
        ys: list[float] = []
        for index, key in enumerate(self._curve.keys):
            xs.append(key.x)
            ys.append(key.y)
            # Every key's handles, not just the selected one's: otherwise
            # selecting a key can pop a handle outside the frame.
            for point in self.handles(index).values():
                xs.append(point[0])
                ys.append(point[1])
        if not ys:
            ys = [0.0, 1.0]
        self._window = (*_padded(min(xs), max(xs)), *_padded(min(ys), max(ys)))
        self.update()

    def handles(self, index: int) -> dict[_Part, tuple[float, float]]:
        """Resolved handle positions for key ``index``, auto handles included.

        Sourced from :meth:`Curve1D.segment_control_points`, which is the curve's
        own single source of truth for which handle faces which segment. The
        outer handle of each terminal key is absent by design: an auto handle
        there collapses onto the key itself, where it has no effect on the shape
        and no hit target distinguishable from the key.
        """
        handles: dict[_Part, tuple[float, float]] = {}
        count = len(self._curve)
        if count < 2 or not 0 <= index < count:
            return handles
        if index > 0:
            handles[_Part.IN] = self._curve.segment_control_points(index - 1)[2]
        if index < count - 1:
            handles[_Part.OUT] = self._curve.segment_control_points(index)[1]
        return handles

    # ------------------------------------------------------------------
    # painting
    # ------------------------------------------------------------------

    def paintEvent(self, event: object) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        view = self.view()
        palette = self.palette()
        self._paint_frame(painter, view, palette)
        self._paint_marker(painter, view, palette)
        self._paint_curve(painter, view, palette)
        self._paint_chunks(painter, view, palette)
        self._paint_keys(painter, view, palette)
        painter.end()

    def _paint_frame(
        self, painter: QPainter, view: _CurveView, palette: QPalette
    ) -> None:
        plot = view.plot()
        painter.fillRect(plot, palette.base())
        grid = QPen(palette.color(QPalette.ColorRole.Mid))
        grid.setWidthF(1.0)
        painter.setPen(grid)
        for step in range(1, _GRID_LINES):
            fraction = step / _GRID_LINES
            x = plot.left() + plot.width() * fraction
            y = plot.top() + plot.height() * fraction
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        painter.setPen(QPen(palette.color(QPalette.ColorRole.Shadow)))
        painter.drawRect(plot)

    def _paint_marker(
        self, painter: QPainter, view: _CurveView, palette: QPalette
    ) -> None:
        if self._marker is None:
            return
        plot = view.plot()
        pen = QPen(palette.color(QPalette.ColorRole.Highlight))
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        x = view.to_pixel(self._marker, 0.0).x()
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))

    def _paint_curve(
        self, painter: QPainter, view: _CurveView, palette: QPalette
    ) -> None:
        plot = view.plot()
        count = max(int(plot.width()), 2)
        xs = np.linspace(view.x0, view.x1, count)
        ys = self._curve.sample_many(xs)
        pen = QPen(palette.color(QPalette.ColorRole.Text))
        pen.setWidthF(2.0)
        painter.setPen(pen)
        painter.drawPolyline(
            QPolygonF(
                [view.to_pixel(float(x), float(y)) for x, y in zip(xs, ys)]
            )
        )

    def _paint_chunks(
        self, painter: QPainter, view: _CurveView, palette: QPalette
    ) -> None:
        """A dot per chunk, at the exact value Apply would write to it."""
        if not len(self._chunks):
            return
        ys = self._curve.sample_many(self._chunks)
        colour = palette.color(QPalette.ColorRole.Highlight)
        painter.setPen(QPen(colour))
        painter.setBrush(colour)
        for x, y in zip(self._chunks, ys):
            painter.drawEllipse(
                view.to_pixel(float(x), float(y)), _CHUNK_RADIUS, _CHUNK_RADIUS
            )
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_keys(
        self, painter: QPainter, view: _CurveView, palette: QPalette
    ) -> None:
        text = palette.color(QPalette.ColorRole.Text)
        accent = palette.color(QPalette.ColorRole.Highlight)
        for index, key in enumerate(self._curve.keys):
            centre = view.to_pixel(key.x, key.y)
            if index == self._selected:
                painter.setPen(QPen(accent))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                for point in self.handles(index).values():
                    pixel = view.to_pixel(*point)
                    painter.drawLine(centre, pixel)
                    painter.drawEllipse(pixel, _HANDLE_RADIUS, _HANDLE_RADIUS)
            painter.setPen(QPen(text))
            painter.setBrush(accent if index == self._selected else palette.base())
            painter.drawEllipse(centre, _KEY_RADIUS, _KEY_RADIUS)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    # ------------------------------------------------------------------
    # dragging
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() is not Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        grab = self._hit(event.position())
        if grab is None:
            return
        if grab.part is _Part.KEY and grab.index != self._selected:
            self._selected = grab.index
            self.key_selected.emit(grab.index)
        self._grab = grab
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        grab = self._grab
        if grab is None:
            return
        position = event.position()
        x, y = self._clamped(
            *self.view().to_curve(position.x() + grab.dx, position.y() + grab.dy)
        )

        if grab.part is _Part.KEY:
            # A key dragged past a neighbour is re-sorted, so the index the drag
            # is holding stops being the key it picked up unless it is refreshed
            # from what move_key reports.
            index = self._curve.move_key(grab.index, x, y)
            if index != grab.index:
                self._grab = replace(grab, index=index)
            if index != self._selected:
                self._selected = index
                self.key_selected.emit(index)
        elif grab.part is _Part.IN:
            self._curve.set_handles(grab.index, in_handle=(x, y))
        else:
            self._curve.set_handles(grab.index, out_handle=(x, y))

        self.update()
        self.edited.emit()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() is Qt.MouseButton.LeftButton:
            self._grab = None
        else:
            super().mouseReleaseEvent(event)

    def _hit(self, position: QPointF) -> _Grab | None:
        view = self.view()
        keys = self._curve.keys

        # Handles first. One parked on top of a neighbouring key must still be
        # grabbable, and only the selected key has any.
        if 0 <= self._selected < len(keys):
            for part, point in self.handles(self._selected).items():
                pixel = view.to_pixel(*point)
                if _distance(pixel, position) <= _HIT_RADIUS:
                    return self._grab_at(part, self._selected, pixel, position)

        best: int | None = None
        best_distance = _HIT_RADIUS
        for index, key in enumerate(keys):
            distance = _distance(view.to_pixel(key.x, key.y), position)
            if distance <= best_distance:
                best, best_distance = index, distance
        if best is None:
            return None
        pixel = view.to_pixel(keys[best].x, keys[best].y)
        return self._grab_at(_Part.KEY, best, pixel, position)

    @staticmethod
    def _grab_at(
        part: _Part, index: int, pixel: QPointF, position: QPointF
    ) -> _Grab:
        return _Grab(part, index, pixel.x() - position.x(), pixel.y() - position.y())

    def _clamped(self, x: float, y: float) -> tuple[float, float]:
        """Hold a dragged point inside the bounds the numeric fields can express.

        ``y_range`` clamps at read time, so a key dragged below it would still
        sample legally — but a spin box cannot display a value its own range
        forbids, and the number and the dot would then disagree about where the
        key is. Clamping the drag leaves one answer.
        """
        y_range = self._curve.y_range
        if y_range is not None:
            y = min(max(y, y_range[0]), y_range[1])
        return (min(max(x, -X_LIMIT), X_LIMIT), y)


class CurvePanel(_Panel):
    """A radius curve over a chain, and the one button that makes it a rig edit.

    The curve is the panel's own state — see the module docstring for why it is
    not the document's — so every control here refreshes from the curve, and only
    :meth:`apply_to_chain` ever writes the body.

    The chain is the selection in table order, falling back to the whole rig when
    nothing is selected. That is the same rule ``ChunkPanel``'s Connect button
    already uses, and reusing it means an author does not have to hold two
    different ideas of what "the chain" means in one window.
    """

    #: The curve changed. Panel state with no home on the document and none in
    #: the rig format, announced rather than smuggled onto either.
    curve_changed = Signal()

    def __init__(self, document: Document, parent: QWidget | None = None) -> None:
        super().__init__(document, parent)

        self._curve = self._level_curve()
        self._canvas = _CurveCanvas(self._curve)
        self._key_x = _spin(-X_LIMIT, X_LIMIT, _SPIN_STEP)
        self._key_y = _spin(MIN_RADIUS, MAX_RADIUS, _SPIN_STEP)
        self._add = QPushButton("Add Key")
        self._remove = QPushButton("Remove Key")
        self._fit = QPushButton("Fit View")
        self._reset = QPushButton("Reset")
        self._apply = QPushButton("Apply to Chain")
        self._sample_x = _spin(0.0, 1.0, 0.05)
        self._sample_y = QLabel()
        self._target = QLabel()

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_canvas(), 1)
        layout.addWidget(self._build_keys())
        layout.addWidget(self._build_readout())
        layout.addWidget(self._build_apply())

        self._canvas.edited.connect(self._on_canvas_edited)
        self._canvas.key_selected.connect(self._on_key_selected)
        self._key_x.valueChanged.connect(self._on_key_x_changed)
        self._key_y.valueChanged.connect(self._on_key_y_changed)
        self._add.clicked.connect(self._on_add)
        self._remove.clicked.connect(self._on_remove)
        self._fit.clicked.connect(self._on_fit)
        self._reset.clicked.connect(self._on_reset)
        self._apply.clicked.connect(self._on_apply)
        self._sample_x.valueChanged.connect(self._on_sample_changed)

        self._follow(document.topology_changed, document.selection_changed)
        # A body replacement drops the undo stack and the selection already; a
        # curve levelled at the old rig's radii would be the one thing left
        # describing a rig that is gone.
        document.body_replaced.connect(self._on_body_replaced)

        self.refresh()

    # ------------------------------------------------------------------
    # model
    # ------------------------------------------------------------------

    @property
    def curve(self) -> Curve1D:
        return self._curve

    @property
    def canvas(self) -> _CurveCanvas:
        """The plot. Exposed so a test can drive a drag without an event loop."""
        return self._canvas

    def target_ids(self) -> list[int]:
        """The chain Apply writes: the selection in table order, or the whole rig."""
        ids = self._document.body.chunk_ids()
        selection = self._document.selection
        chain = [chunk_id for chunk_id in ids if chunk_id in selection]
        return chain or ids

    def sample_positions(self) -> np.ndarray:
        """Curve ``x`` for each chunk of the chain: normalised index position.

        See the module docstring for why this is not arc length. A one-chunk
        chain samples at ``0.0``; an empty one samples nowhere.
        """
        return np.linspace(0.0, 1.0, len(self.target_ids()))

    def sampled_radii(self) -> np.ndarray:
        """What Apply would write, in chain order.

        One :meth:`Curve1D.sample_many` call, so what the panel reports, what the
        canvas draws and what the body receives are the same arithmetic rather
        than three implementations of it.
        """
        return self._curve.sample_many(self.sample_positions())

    def apply_to_chain(self) -> bool:
        """Write the curve into the chain's radii as a single undo entry.

        Returns whether anything was submitted. Radii that already match are
        dropped, so re-applying an unchanged curve does not bury the stack in
        entries that undo to themselves — the same guard ``ChunkPanel._submit``
        makes for an empty selection.
        """
        ids = self.target_ids()
        radii = self.sampled_radii()
        body = self._document.body
        commands = [
            SetRadius(chunk_id, float(radius))
            for chunk_id, radius in zip(ids, radii)
            # body.radius is a float32 array, so the number that comes back out
            # is not the float64 that went in. Comparing without that round trip
            # makes every single apply look like a change.
            if _stored(radius) != body.chunk(chunk_id).radius
        ]
        if not commands:
            self._document.status.emit("Radius curve already matches the chain")
            return False
        self._document.do(Macro(commands, text="Radius Curve"))
        self._document.status.emit(
            f"Radius curve applied to {len(commands)} chunk(s)"
        )
        return True

    def set_curve(self, curve: Curve1D) -> None:
        """Replace the model, re-frame the plot, and announce it."""
        self._curve = curve
        self._canvas.set_curve(curve)
        self._canvas.fit()
        self.refresh()
        self.curve_changed.emit()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_canvas(self) -> QWidget:
        box = QGroupBox("Radius along the chain")
        layout = QVBoxLayout(box)
        layout.addWidget(self._canvas, 1)
        layout.addWidget(
            _note(
                "x is normalised position along the chain — chunk i of n sits "
                "at i/(n-1) — and y is radius in world units. Not arc length: "
                "arc length moves while the rig does, so the same curve would "
                "not apply twice the same way."
            )
        )
        self._canvas.setToolTip(
            "Drag a key to move it; its handles move with it. Select a key to "
            "reach its handles. The dots are the chunks Apply will write."
        )
        return box

    def _build_keys(self) -> QWidget:
        box = QGroupBox("Selected key")
        layout = _form()
        box.setLayout(layout)
        layout.addRow("x", self._key_x)
        layout.addRow("Radius", self._key_y)
        layout.addRow(self._build_buttons())

        self._key_y.setToolTip(
            f"Clamped to [{MIN_RADIUS:g}, {MAX_RADIUS:g}]: a radius curve has to "
            "stay positive however far a handle is dragged, because the body "
            "refuses a non-positive radius outright."
        )
        self._remove.setToolTip(
            f"Disabled at {MIN_KEYS} keys. Below that the curve is a constant "
            "with no segment, and there is nothing left to shape."
        )
        self._add.setToolTip(
            "Splits the widest segment exactly, so the shape does not move and "
            "the new key is a handle onto what is already there."
        )
        return box

    def _build_buttons(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for button in (self._add, self._remove, self._fit, self._reset):
            layout.addWidget(button)
        return row

    def _build_readout(self) -> QWidget:
        box = QGroupBox("Readout")
        layout = _form()
        box.setLayout(layout)
        layout.addRow("Sample at x", self._sample_x)
        layout.addRow("Radius", self._sample_y)
        self._sample_y.setToolTip(
            "Straight from Curve1D.sample, which is the same call Apply makes."
        )
        return box

    def _build_apply(self) -> QWidget:
        box = QGroupBox("Apply")
        layout = QVBoxLayout(box)
        layout.addWidget(self._apply)
        layout.addWidget(self._target)
        layout.addWidget(
            _note(
                "One undo entry per apply, whatever the chain length. The curve "
                "itself is view state and is not saved with the rig — only the "
                "radii it wrote are."
            )
        )
        self._apply.setToolTip(
            "Set every chunk in the chain to the curve's value at its position. "
            "Selection in table order, or the whole rig if nothing is selected."
        )
        return box

    # ------------------------------------------------------------------
    # handlers
    # ------------------------------------------------------------------

    @_user_edit
    def _on_canvas_edited(self) -> None:
        self.refresh()
        self.curve_changed.emit()

    @_user_edit
    def _on_key_selected(self, _index: int) -> None:
        self.refresh()

    @_user_edit
    def _on_key_x_changed(self, value: float) -> None:
        self._move_selected(x=value)

    @_user_edit
    def _on_key_y_changed(self, value: float) -> None:
        self._move_selected(y=value)

    @_user_edit
    def _on_add(self) -> None:
        self._canvas.select(self._curve.add_key())
        self.refresh()
        self.curve_changed.emit()

    @_user_edit
    def _on_remove(self) -> None:
        index = self._canvas.selected
        if len(self._curve) <= MIN_KEYS or not 0 <= index < len(self._curve):
            return
        self._curve.remove_key(index)
        self._canvas.select(min(index, len(self._curve) - 1))
        self.refresh()
        self.curve_changed.emit()

    @_user_edit
    def _on_fit(self) -> None:
        self._canvas.fit()

    @_user_edit
    def _on_reset(self) -> None:
        self.set_curve(self._level_curve())

    @_user_edit
    def _on_apply(self) -> None:
        self.apply_to_chain()

    @_user_edit
    def _on_sample_changed(self, _value: float) -> None:
        self.refresh()

    def _on_body_replaced(self) -> None:
        self.set_curve(self._level_curve())

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _move_selected(self, x: float | None = None, y: float | None = None) -> None:
        """Move the selected key, taking the untouched axis from the key itself.

        Not from that axis' own spin box: a spin box shows three decimals and a
        key does not, so reading one back to fill in the other half of the move
        would quietly round an axis the author never touched.
        """
        index = self._canvas.selected
        keys = self._curve.keys
        if not 0 <= index < len(keys):
            return
        key = keys[index]
        moved = self._curve.move_key(
            index, key.x if x is None else x, key.y if y is None else y
        )
        self._canvas.select(moved)
        self.refresh()
        self.curve_changed.emit()

    def _level_curve(self) -> Curve1D:
        """A flat curve at the chain's mean radius.

        Opening the panel on a rig should not propose flattening it to a number
        nobody chose. Starting level with what is already there means the first
        Apply changes as little as a curve can, and every later one is a
        deliberate edit rather than a correction of the panel's opening guess.
        """
        body = self._document.body
        radii = [
            body.chunk(chunk_id).radius
            for chunk_id in self.target_ids()
            if body.has_chunk(chunk_id)
        ]
        level = float(np.mean(radii)) if radii else DEFAULT_RADIUS
        level = min(max(level, MIN_RADIUS), MAX_RADIUS)
        return Curve1D.linear(0.0, level, 1.0, level, y_range=Y_RANGE)

    def _refresh(self) -> None:
        keys = self._curve.keys
        index = self._canvas.selected

        self._canvas.set_chunk_positions(self.sample_positions())
        self._canvas.set_marker(self._sample_x.value())

        selected = 0 <= index < len(keys)
        self._key_x.setEnabled(selected)
        self._key_y.setEnabled(selected)
        if selected:
            self._key_x.setValue(keys[index].x)
            self._key_y.setValue(keys[index].y)
        self._remove.setEnabled(len(keys) > MIN_KEYS)

        self._sample_y.setText(
            format(self._curve.sample(self._sample_x.value()), _READOUT)
        )

        ids = self.target_ids()
        self._apply.setEnabled(bool(ids))
        self._target.setText(self._target_text(ids))

    def _target_text(self, ids: list[int]) -> str:
        if not ids:
            return "Nothing to apply to — the rig is empty."
        scope = "selected" if self._document.selection else "whole rig"
        return f"{len(ids)} chunk(s), {scope}."


def _stored(radius: float) -> float:
    """The value a float64 radius becomes once the body has kept it."""
    return float(np.float32(radius))
