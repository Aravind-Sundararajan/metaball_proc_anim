"""Dock panels: every knob the editor exposes, and no state hiding behind them.

A panel here owns nothing. Each control is rendered *from* the document on the
way in and turned into a document write on the way out, so the only way for two
panels to disagree is for the document to disagree with itself. The old app did
the opposite — each panel held its own copy of what it displayed and wrote
straight into the manager — which is why a value edited in one place could sit
stale in another indefinitely.

That round trip creates the one hazard this module has to solve. Writing a
document value into a control makes Qt emit that control's own change signal,
which a panel reads as a fresh user edit and writes back, which makes the
document emit, which refreshes the control again. Sometimes that spins; more
often it does something quieter and worse, pushing a command onto the undo stack
for an edit nobody made, or wiping the selection every time the topology
changes. The old app fought this with a scatter of ``updating_position``
booleans, one per call site, each of which had to be remembered separately.

There is one latch here instead. :meth:`_Panel._sync` is held for the whole of a
refresh, and :func:`_user_edit` drops any control signal that arrives while it is
held. Every handler carries the decorator, so a missing guard is visible at the
point of declaration rather than at the point of failure.

Three pieces of state are deliberately *not* the document's, and all three are
marked as such at their definition: :attr:`SurfacePanel.show_rig_overlay`, which
:class:`~mesh_xml.render.gl.metaball.MetaballSettings` has no field for,
:attr:`SurfacePanel.orthographic`, which belongs to the viewport's
:class:`~mesh_xml.render.gl.camera.OrbitCamera` and not to the rig at all, and
:meth:`PlaybackPanel.single_step`, which has no counterpart on
:class:`~mesh_xml.editor.document.Document`. All three are exposed as panel
signals so the main window can wire them to the viewport rather than any of them
being smuggled onto the document behind its own contract. The radius curve in
:mod:`mesh_xml.editor.curve_panel` is a fourth, for the same reason.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from functools import wraps
from itertools import combinations
from typing import Callable, Iterator

import numpy as np
from PySide6.QtCore import (
    QItemSelectionModel,
    Qt,
    QTimer,
    Signal,
    SignalInstance,
)
from PySide6.QtGui import QColor, QFontDatabase
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mesh_xml.anim import serialize
from mesh_xml.core import solver
from mesh_xml.render import sdf as sdf_module
from mesh_xml.scene.door import Door
from mesh_xml.scene.room import Room
from mesh_xml.editor.document import (
    AddLink,
    Command,
    Creature,
    Document,
    Macro,
    RemoveChunk,
    RemoveController,
    RemoveLink,
    Rename,
    SetControllerField,
    SetPinned,
    SetRadius,
)

__all__ = [
    "PlaybackPanel",
    "SurfacePanel",
    "SolverPanel",
    "SmoothingPanel",
    "ChunkPanel",
    "CreaturePanel",
    "ControllerPanel",
    "RoomPanel",
    "build_panels",
]

#: Slider steps across the blend range. The slider is an int widget; the spin box
#: is the authority and the slider is a coarse handle onto the same value.
_BLEND_STEPS = 1000
_BLEND_MAX = 1.0

_DEBUG_MODES = (
    "0 - shaded",
    "1 - normals",
    "2 - packed depth",
    "3 - step count",
    "4 - albedo",
)

_AXES = ("X", "Y", "Z")

_NAME, _RADIUS, _PINNED = 0, 1, 2
_COLUMNS = ("Name", "Radius", "Pinned")
_RADIUS_FORMAT = ".3f"

_R_NAME, _R_CREATURES, _R_ABSTRACT, _R_FILE = 0, 1, 2, 3
_ROOM_COLUMNS = ("Name", "Creatures", "Abstract", "File")

_D_NAME, _D_AT, _D_TO_ROOM, _D_TO_DOOR, _D_RADIUS, _D_STATUS = range(6)
_DOOR_COLUMNS = ("Name", "At", "Room", "Door", "Radius", "")


def _user_edit(handler: Callable[..., None]) -> Callable[..., None]:
    """Drop a control signal that the panel's own refresh caused.

    Qt cannot distinguish "the user moved this" from "we just wrote to this",
    and both arrive as the same signal. The panel can: it knows whether it is
    mid-refresh. Decorating every handler rather than checking inside each one
    means the guard is part of the handler's declaration and cannot be half
    applied.
    """

    @wraps(handler)
    def guarded(self: _Panel, *args: object) -> None:
        if self.syncing:
            return
        handler(self, *args)

    return guarded


class _Panel(QWidget):
    """A widget bound to a document, never holding a value the document does not.

    Subclasses build their controls, connect each handler through
    :func:`_user_edit`, subscribe with :meth:`_follow`, and implement
    :meth:`_refresh` to render the document into those controls.
    """

    def __init__(self, document: Document, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._document = document
        self._syncing = 0

    @property
    def document(self) -> Document:
        return self._document

    @property
    def syncing(self) -> bool:
        """Whether a refresh is in flight, i.e. control signals are our own echo."""
        return self._syncing > 0

    def refresh(self) -> None:
        """Re-render every control from the document."""
        with self._sync():
            self._refresh()

    @contextmanager
    def _sync(self) -> Iterator[None]:
        """Hold the latch for the duration of a write into the controls.

        A counter rather than a flag, so a partial refresh nested inside a full
        one cannot release the latch early and let the tail of the outer refresh
        through as user edits.
        """
        self._syncing += 1
        try:
            yield
        finally:
            self._syncing -= 1

    def _follow(self, *signals: SignalInstance) -> None:
        """Refresh whenever any of these fire, whatever payload they carry."""
        for signal in signals:
            signal.connect(self._on_document_changed)

    def _on_document_changed(self, *_: object) -> None:
        self.refresh()

    def _refresh(self) -> None:
        raise NotImplementedError


def _form() -> QFormLayout:
    layout = QFormLayout()
    layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    layout.setLabelAlignment(
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    )
    return layout


def _spin(
    minimum: float,
    maximum: float,
    step: float,
    decimals: int = 3,
    suffix: str = "",
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    box.setSuffix(suffix)
    box.setKeyboardTracking(False)
    return box


def _admit(box, value) -> None:
    """Widen a spin box's range to include a loaded value before showing it.

    A panel's range is an authoring suggestion; the format's domain is wider
    (a damping of 1.02 rings, and the reader admits it — docs/SETTINGS.md
    §4). Clamping the display would show physics the room is not running,
    and the next nudge would silently commit the clamp to the file.
    """
    if value < box.minimum():
        box.setMinimum(value)
    if value > box.maximum():
        box.setMaximum(value)


def _note(text: str) -> QLabel:
    """Small print that belongs next to a control, not in a manual nobody opens."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    font = label.font()
    font.setItalic(True)
    font.setPointSizeF(max(font.pointSizeF() - 1.0, 6.0))
    label.setFont(font)
    return label


def _rgb(colour: QColor) -> tuple[float, float, float]:
    return (colour.redF(), colour.greenF(), colour.blueF())


def _colour(rgb: tuple[float, float, float]) -> QColor:
    return QColor.fromRgbF(*(min(max(float(v), 0.0), 1.0) for v in rgb))


def _swatch(button: QPushButton, colour: QColor) -> None:
    ink = "#000000" if colour.lightnessF() > 0.5 else "#ffffff"
    button.setText(colour.name())
    button.setStyleSheet(f"background-color: {colour.name()}; color: {ink};")


class PlaybackPanel(_Panel):
    """Transport, and an honest readout of what the clock is actually doing.

    Dropped ticks are the number worth watching. A simulation that cannot keep
    up runs slow rather than falling further behind, which is the right
    behaviour and is completely invisible unless something says so.
    """

    #: A single tick has been run outside the clock. The main window connects
    #: this to the viewport, because the document has no per-tick signal to
    #: repaint on and stepping while paused would otherwise draw nothing.
    stepped = Signal()

    READOUT_INTERVAL_MS = 100

    def __init__(self, document: Document, parent: QWidget | None = None) -> None:
        super().__init__(document, parent)

        self._play = QPushButton("Play")
        self._play.setCheckable(True)
        self._step = QPushButton("Step")
        self._rest = QPushButton("Rest Pose")
        self._ticks = QLabel()
        self._dropped = QLabel()
        self._alpha = QLabel()

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_transport())
        layout.addWidget(self._build_readout())
        layout.addStretch(1)

        self._play.toggled.connect(self._on_play_toggled)
        self._step.clicked.connect(self._on_step_clicked)
        self._rest.clicked.connect(self._on_rest_clicked)
        self._follow(document.running_changed)

        # The clock advances every frame and emits nothing, so the readout is
        # polled. Cheap, and it keeps Document free of a per-tick signal that
        # every other view would have to learn to ignore.
        self._timer = QTimer(self)
        self._timer.setInterval(self.READOUT_INTERVAL_MS)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

        self.refresh()

    def single_step(self) -> None:
        """Run exactly one simulation tick while paused.

        Delegates to :meth:`Document.step_once` rather than reproducing the tick
        body, so anything added to a tick later — controllers in Phase 4 — runs
        here too instead of silently applying only during playback.
        """
        document = self._document
        if document.running:
            return
        document.step_once()
        self.stepped.emit()
        self.refresh()

    def _build_transport(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for button in (self._play, self._step, self._rest):
            layout.addWidget(button)

        self._step.setToolTip(
            "Advance one fixed tick. The clock is not consumed, so the tick "
            "count below does not move."
        )
        self._rest.setToolTip(
            "Return every chunk to the pose captured when playback last stopped, "
            "and drop all motion."
        )
        return row

    def _build_readout(self) -> QWidget:
        box = QGroupBox(f"Clock ({solver.TICK_HZ:g} Hz)")
        layout = _form()
        box.setLayout(layout)

        fixed = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        for label, value in (
            ("Ticks", self._ticks),
            ("Dropped", self._dropped),
            ("Alpha", self._alpha),
        ):
            value.setFont(fixed)
            layout.addRow(label, value)

        self._dropped.setToolTip(
            "Ticks the clock refused to run to avoid a catch-up spiral. Anything "
            "above zero means the simulation is running slow."
        )
        return box

    @_user_edit
    def _on_play_toggled(self, playing: bool) -> None:
        self._document.set_running(playing)

    @_user_edit
    def _on_step_clicked(self) -> None:
        self.single_step()

    @_user_edit
    def _on_rest_clicked(self) -> None:
        self._document.reset_pose()
        self.refresh()

    def _refresh(self) -> None:
        running = self._document.running
        self._play.setChecked(running)
        self._play.setText("Pause" if running else "Play")
        self._step.setEnabled(not running)

        clock = self._document.clock
        self._ticks.setText(f"{clock.ticks}")
        self._dropped.setText(f"{clock.dropped}")
        self._alpha.setText(f"{clock.alpha:.3f}")


class SurfacePanel(_Panel):
    """The metaball look: blend, colour, and the debug views of the raymarch.

    Blend is the knob, so it gets the top of the panel and two ways to reach it.
    The slider and the spin box are not wired to each other; both write the
    document and both are re-read from it, which is the same path an external
    change takes and therefore the only path that has to be correct.
    """

    #: Whether to draw chunks and links over the surface. View state with no
    #: home in ``MetaballSettings`` and no business on the document, so the main
    #: window carries it from here to the viewport.
    overlay_changed = Signal(bool)

    #: Whether the viewport projects orthographically. The camera owns the
    #: projection and the panel does not own the camera, so this is panel state
    #: plus a signal for exactly the reason ``overlay_changed`` is: the
    #: alternative is a panel reaching through the main window into a widget it
    #: does not have, which is how the old app's settings ended up in three
    #: places at once.
    orthographic_changed = Signal(bool)

    def __init__(self, document: Document, parent: QWidget | None = None) -> None:
        super().__init__(document, parent)

        self._blend = _spin(0.0, _BLEND_MAX, 0.01)
        self._blend_slider = QSlider(Qt.Orientation.Horizontal)
        self._blend_slider.setRange(0, _BLEND_STEPS)
        self._surface_colour = QPushButton()
        self._background_colour = QPushButton()
        self._debug = QComboBox()
        self._debug.addItems(_DEBUG_MODES)
        self._max_steps = QSpinBox()
        self._max_steps.setRange(16, 1024)
        self._max_steps.setSingleStep(16)
        self._overlay = QCheckBox("Show rig overlay")
        self._show_rig_overlay = False
        self._ortho = QCheckBox("Orthographic")
        self._orthographic = False
        self._floor = QCheckBox("Show floor")
        self._creature_colour = QPushButton("Creature colour…")
        self._chunk_colour = QPushButton("Paint selection…")
        self._pattern = QComboBox()
        self._pattern.addItems(["none", *sdf_module.PATTERN_IDS])
        self._pattern_param = _spin(0.0, 64.0, 0.5, decimals=2)
        #: Why the pattern row is grey, told where the greyness is. A
        #: disabled combo reads as an empty one, and the reason lives in
        #: the selection — a fact this panel cannot show any other way.
        self._pattern_scope = _note("")

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_blend())
        layout.addWidget(self._build_appearance())
        layout.addWidget(self._build_tracing())
        layout.addStretch(1)

        self._blend.valueChanged.connect(self._on_blend_changed)
        self._blend_slider.valueChanged.connect(self._on_blend_slider_changed)
        self._surface_colour.clicked.connect(self._on_pick_surface_colour)
        self._background_colour.clicked.connect(self._on_pick_background_colour)
        self._debug.currentIndexChanged.connect(self._on_debug_changed)
        self._max_steps.valueChanged.connect(self._on_max_steps_changed)
        self._floor.toggled.connect(self._on_floor_toggled)
        self._creature_colour.clicked.connect(self._on_pick_creature_colour)
        self._chunk_colour.clicked.connect(self._on_pick_chunk_colour)
        self._pattern.currentIndexChanged.connect(self._on_pattern_changed)
        self._pattern_param.valueChanged.connect(self._on_pattern_param_changed)
        self._overlay.toggled.connect(self._on_overlay_toggled)
        self._ortho.toggled.connect(self._on_ortho_toggled)
        # Patterns are creature meta, so the panel re-reads on the signals
        # meta edits and selection moves arrive by — not just settings.
        self._follow(
            document.settings_changed,
            document.body_replaced,
            document.selection_changed,
            document.topology_changed,
        )

        self.refresh()

    @property
    def show_rig_overlay(self) -> bool:
        return self._show_rig_overlay

    def set_show_rig_overlay(self, show: bool) -> None:
        if show == self._show_rig_overlay:
            return
        self._show_rig_overlay = show
        with self._sync():
            self._overlay.setChecked(show)
        self.overlay_changed.emit(show)

    @property
    def orthographic(self) -> bool:
        return self._orthographic

    def set_orthographic(self, orthographic: bool) -> None:
        """Set the projection from outside, e.g. to adopt the camera's own state.

        Writing the checkbox under the latch is what makes this usable as the
        *initial* sync as well as an external change: without it the write would
        come straight back as a user edit and the signal would fire during
        construction, before the main window had finished connecting anything.
        """
        orthographic = bool(orthographic)
        if orthographic == self._orthographic:
            return
        self._orthographic = orthographic
        with self._sync():
            self._ortho.setChecked(orthographic)
        self.orthographic_changed.emit(orthographic)

    def set_surface_colour(self, rgb: tuple[float, float, float]) -> None:
        self._document.metaball.surface_colour = tuple(float(v) for v in rgb)
        self._document.settings_changed.emit()

    def set_background_colour(self, rgb: tuple[float, float, float]) -> None:
        self._document.metaball.background_colour = tuple(float(v) for v in rgb)
        self._document.settings_changed.emit()

    def _build_blend(self) -> QWidget:
        box = QGroupBox("Blend (k)")
        layout = QVBoxLayout(box)
        layout.addWidget(self._blend)
        layout.addWidget(self._blend_slider)
        layout.addWidget(
            _note(
                "Blend fattens as well as smooths. A chunk with two or more "
                "links sits inside both of its cones, so it reads about k/4 "
                "wider than its authored radius and the body beads slightly at "
                "its joints; degree-one tips are unaffected. Author radii "
                "against the blend you intend to keep."
            )
        )
        self._blend.setToolTip(
            "Smooth-minimum radius. 0 is a string of beads; the whole difference "
            "between a rig and a creature lives in this number."
        )
        return box

    def _build_appearance(self) -> QWidget:
        box = QGroupBox("Appearance")
        layout = _form()
        box.setLayout(layout)
        layout.addRow("Surface", self._surface_colour)
        layout.addRow("Background", self._background_colour)
        layout.addRow(self._creature_colour)
        layout.addRow(self._chunk_colour)
        layout.addRow("Pattern", self._pattern)
        layout.addRow("Pattern param", self._pattern_param)
        layout.addRow(self._pattern_scope)
        layout.addRow(self._overlay)
        layout.addRow(self._ortho)
        layout.addRow(self._floor)
        layout.addRow(
            _note(
                "Orthographic keeps the subject about the same size on screen: "
                "the camera derives its ortho height from the perspective "
                "framing, so the toggle is a comparison rather than a "
                "re-framing exercise. A view setting; it is not saved with the "
                "rig."
            )
        )
        self._overlay.setToolTip(
            "Draw chunks and links over the surface. A view setting; it is not "
            "saved with the rig."
        )
        self._ortho.setToolTip(
            "Drop perspective. Parallel edges stay parallel, so relative chunk "
            "sizes down a chain can be compared by eye instead of being "
            "confounded by depth."
        )
        self._floor.setToolTip(
            "Draw the reference grid at the solver's floor height, with the "
            "creature's shadow on it. Stands down on its own when a terrain "
            "is loaded — the terrain is the world, and two floors would "
            "contradict each other."
        )
        self._creature_colour.setToolTip(
            "Paint the whole creature the selection touches — every chunk of "
            "its connected component, one undo step, saved with the rig. "
            "Unpainted creatures keep their automatic palette hue."
        )
        self._chunk_colour.setToolTip(
            "Paint exactly the selected chunks, one undo step, saved with "
            "the rig. The surface grades between painted and unpainted "
            "neighbours at the blend — a bright head on a dull body."
        )
        self._pattern.setToolTip(
            "The selected creature's surface pattern, painted across its "
            "chunks like colour. Stripes band along each link (rings "
            "around a limb); countershade darkens the back and pales the "
            "belly. One undo step, saved with the rig."
        )
        self._pattern_param.setToolTip(
            "Stripes: bands per link. Countershade: strength, 0 to 1. "
            "Applied with the pattern, one undo step."
        )
        return box

    def _build_tracing(self) -> QWidget:
        box = QGroupBox("Raymarch")
        layout = _form()
        box.setLayout(layout)
        layout.addRow("Debug mode", self._debug)
        layout.addRow("Max steps", self._max_steps)
        self._debug.setToolTip(
            "Shaded is the render. The others expose the trace itself: normals, "
            "packed depth, and how many steps each ray needed."
        )
        self._max_steps.setToolTip(
            "Step budget per ray. Too low and thin geometry dissolves at grazing "
            "angles; mode 3 shows where the budget is going."
        )
        return box

    @_user_edit
    def _on_blend_changed(self, value: float) -> None:
        self._write_blend(value)

    @_user_edit
    def _on_blend_slider_changed(self, value: int) -> None:
        self._write_blend(value * _BLEND_MAX / _BLEND_STEPS)

    @_user_edit
    def _on_debug_changed(self, index: int) -> None:
        self._document.metaball.debug_mode = int(index)
        self._document.settings_changed.emit()

    @_user_edit
    def _on_max_steps_changed(self, value: int) -> None:
        self._document.metaball.max_steps = int(value)
        self._document.settings_changed.emit()

    @_user_edit
    def _on_overlay_toggled(self, show: bool) -> None:
        self.set_show_rig_overlay(show)

    @_user_edit
    def _on_ortho_toggled(self, orthographic: bool) -> None:
        self.set_orthographic(orthographic)

    @_user_edit
    def _on_floor_toggled(self, show: bool) -> None:
        self._document.metaball.show_floor = bool(show)
        self._document.settings_changed.emit()

    @_user_edit
    def _on_pick_surface_colour(self) -> None:
        picked = self._pick(self._document.metaball.surface_colour, "Surface colour")
        if picked is not None:
            self.set_surface_colour(picked)

    @_user_edit
    def _on_pick_background_colour(self) -> None:
        picked = self._pick(
            self._document.metaball.background_colour, "Background colour"
        )
        if picked is not None:
            self.set_background_colour(picked)

    @_user_edit
    def _on_pick_creature_colour(self) -> None:
        picked = self._pick(
            self._document.metaball.surface_colour, "Creature colour"
        )
        if picked is not None:
            self._document.set_creature_colour(picked)

    @_user_edit
    def _on_pick_chunk_colour(self) -> None:
        picked = self._pick(
            self._document.metaball.surface_colour, "Paint selection"
        )
        if picked is not None:
            self._document.set_chunk_colour(picked)

    @_user_edit
    def _on_pattern_changed(self, _index: int) -> None:
        name = self._pattern.currentText()
        # Switching patterns adopts the newcomer's default parameter — a
        # stripe count means nothing to a countershade strength.
        default = sdf_module.PATTERN_DEFAULTS.get(name)
        if default is not None:
            with self._sync():
                self._pattern_param.setValue(float(default))
        self._document.set_creature_pattern(
            name, float(self._pattern_param.value())
        )

    @_user_edit
    def _on_pattern_param_changed(self, value: float) -> None:
        name = self._pattern.currentText()
        if name in sdf_module.PATTERN_IDS:
            self._document.set_creature_pattern(name, float(value))

    def _pick(
        self, current: tuple[float, float, float], title: str
    ) -> tuple[float, float, float] | None:
        chosen = QColorDialog.getColor(_colour(current), self, title)
        return _rgb(chosen) if chosen.isValid() else None

    def _write_blend(self, value: float) -> None:
        self._document.metaball.blend = float(value)
        self._document.settings_edited()

    def _refresh(self) -> None:
        settings = self._document.metaball
        _admit(self._blend, float(settings.blend))
        self._blend.setValue(float(settings.blend))
        self._blend_slider.setValue(
            round(float(settings.blend) * _BLEND_STEPS / _BLEND_MAX)
        )
        _swatch(self._surface_colour, _colour(settings.surface_colour))
        _swatch(self._background_colour, _colour(settings.background_colour))
        self._debug.setCurrentIndex(
            min(max(int(settings.debug_mode), 0), len(_DEBUG_MODES) - 1)
        )
        self._max_steps.setValue(int(settings.max_steps))
        self._overlay.setChecked(self._show_rig_overlay)
        self._ortho.setChecked(self._orthographic)
        self._floor.setChecked(bool(settings.show_floor))

        name, param = self._document.creature_pattern()
        index = self._pattern.findText(name)
        self._pattern.setCurrentIndex(index if index >= 0 else 0)
        if name in sdf_module.PATTERN_IDS:
            self._pattern_param.setValue(float(param))
        selected = bool(self._document.selection)
        self._pattern.setEnabled(selected)
        self._pattern_param.setEnabled(selected)
        self._pattern_scope.setText(
            ""
            if selected
            else "Patterns apply to the selected creature — click one in "
            "the viewport or pick it in the Creatures roster."
        )


class SolverPanel(_Panel):
    """Tuning for one simulation tick.

    Zero-g remembers the gravity it switched off, because the alternative is
    that turning it on and back off leaves a rig weightless with three zeroes
    showing and no way back to what was authored.
    """

    def __init__(self, document: Document, parent: QWidget | None = None) -> None:
        super().__init__(document, parent)

        self._iterations = QSpinBox()
        self._iterations.setRange(1, 64)
        self._damping = _spin(0.0, 1.0, 0.01)
        self._max_velocity = _spin(0.1, 10_000.0, 1.0, decimals=1, suffix=" u/s")
        self._friction = _spin(0.0, 5.0, 0.05, decimals=2)
        self._self_contact = QCheckBox("Self-contact")
        self._contact_elasticity = _spin(0.0, 1.0, 0.05, decimals=2)
        self._self_friction = _spin(0.0, 5.0, 0.05, decimals=2)
        self._gravity = tuple(_spin(-1000.0, 1000.0, 0.1, decimals=2) for _ in _AXES)
        self._zero_g = QCheckBox("Zero gravity")

        authored = self._authored_gravity()
        self._gravity_memo = authored if any(authored) else solver.SolverConfig().gravity

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_stiffness())
        layout.addWidget(self._build_gravity())
        layout.addStretch(1)

        self._iterations.valueChanged.connect(self._on_iterations_changed)
        self._damping.valueChanged.connect(self._on_damping_changed)
        self._max_velocity.valueChanged.connect(self._on_max_velocity_changed)
        self._friction.valueChanged.connect(self._on_friction_changed)
        self._self_contact.toggled.connect(self._on_self_contact_toggled)
        self._contact_elasticity.valueChanged.connect(
            self._on_contact_elasticity_changed
        )
        self._self_friction.valueChanged.connect(self._on_self_friction_changed)
        for spin in self._gravity:
            spin.valueChanged.connect(self._on_gravity_changed)
        self._zero_g.toggled.connect(self._on_zero_g_toggled)
        self._follow(document.settings_changed, document.body_replaced)

        self.refresh()

    def _build_stiffness(self) -> QWidget:
        box = QGroupBox("Response")
        layout = _form()
        box.setLayout(layout)
        layout.addRow("Iterations", self._iterations)
        layout.addRow("Damping", self._damping)
        layout.addRow("Max velocity", self._max_velocity)
        layout.addRow("Friction (mu)", self._friction)
        layout.addRow(self._self_contact)
        layout.addRow("Contact elasticity", self._contact_elasticity)
        layout.addRow("Self-friction (mu)", self._self_friction)

        self._iterations.setToolTip(
            "Relaxation passes per tick. This is a per-creature softness knob, "
            "not a correctness setting: at 4 a hanging chain gives about 8% "
            "under load, and the give halves with each doubling. For a soft "
            "body that give is the point."
        )
        self._damping.setToolTip(
            "The fraction of the implicit Verlet velocity carried into the next "
            "tick. 1.0 conserves energy and rings forever, which reads as a wire "
            "sculpture rather than flesh."
        )
        self._max_velocity.setToolTip(
            "A backstop in world units per second, not a tuning knob. It exists "
            "so a degenerate rest length cannot fling a chunk out past where "
            "float32 positions resolve."
        )
        self._friction.setToolTip(
            "Coulomb coefficient for the ground plane. Grip scales with the "
            "load a foot carries: held outright below mu times the normal "
            "load, sliding above it. Zero is a perfectly slick floor. Values "
            "above 1 are legal and simply very grippy."
        )
        self._self_contact.setToolTip(
            "Keep the flesh from passing through itself: chunks and rendered "
            "spans collide when they belong to strangers in the graph — "
            "three or more links apart. Neighbours are exempt: within a "
            "limb, overlap is what makes the surface read as one volume. "
            "Off by default because rigs overlap on purpose."
        )
        self._contact_elasticity.setToolTip(
            "Fraction of an overlap closed per tick. 1.0 separates a lone "
            "pair in one tick and reads as billiard balls; lower values let "
            "limbs squash into each other and recover, which reads as flesh."
        )
        self._self_friction.setToolTip(
            "Coulomb coefficient between a body's own parts. Grip scales "
            "with the load actually crossing the contact, so a coiled body "
            "holds its wrap instead of sliding off itself. Zero separates "
            "without grip. Choosing to hold — and to let go — stays a "
            "controller feature."
        )
        return box

    def _build_gravity(self) -> QWidget:
        box = QGroupBox("Gravity")
        layout = _form()
        box.setLayout(layout)
        for axis, spin in zip(_AXES, self._gravity):
            layout.addRow(axis, spin)
        layout.addRow(self._zero_g)
        layout.addRow(
            _note(
                "Applied to a 2-D rig by trimming, so a 3-D gravity survives "
                "being handed to one unchanged."
            )
        )
        return box

    @_user_edit
    def _on_iterations_changed(self, value: int) -> None:
        self._document.solver_config.iterations = int(value)
        self._document.settings_edited()

    @_user_edit
    def _on_damping_changed(self, value: float) -> None:
        self._document.solver_config.damping = float(value)
        self._document.settings_edited()

    @_user_edit
    def _on_max_velocity_changed(self, value: float) -> None:
        self._document.solver_config.max_velocity = float(value)
        self._document.settings_edited()

    @_user_edit
    def _on_friction_changed(self, value: float) -> None:
        self._document.solver_config.ground_friction = float(value)
        self._document.settings_edited()

    @_user_edit
    def _on_self_contact_toggled(self, on: bool) -> None:
        self._document.solver_config.self_contact = bool(on)
        self._document.settings_edited()

    @_user_edit
    def _on_contact_elasticity_changed(self, value: float) -> None:
        self._document.solver_config.self_contact_elasticity = float(value)
        self._document.settings_edited()

    @_user_edit
    def _on_self_friction_changed(self, value: float) -> None:
        self._document.solver_config.self_friction = float(value)
        self._document.settings_edited()

    @_user_edit
    def _on_gravity_changed(self, _value: float) -> None:
        gravity = tuple(spin.value() for spin in self._gravity)
        if any(gravity):
            self._gravity_memo = gravity
        self._document.solver_config.gravity = gravity
        self._document.settings_edited()

    @_user_edit
    def _on_zero_g_toggled(self, zero: bool) -> None:
        if zero:
            authored = self._authored_gravity()
            if any(authored):
                self._gravity_memo = authored
            self._document.solver_config.gravity = (0.0, 0.0, 0.0)
        else:
            self._document.solver_config.gravity = self._gravity_memo
        self._document.settings_edited()

    def _authored_gravity(self) -> tuple[float, ...]:
        gravity = tuple(float(v) for v in self._document.solver_config.gravity)
        return (gravity + (0.0, 0.0, 0.0))[: len(_AXES)]

    def _refresh(self) -> None:
        config = self._document.solver_config
        gravity = self._authored_gravity()
        zero = not any(gravity)

        _admit(self._iterations, int(config.iterations))
        self._iterations.setValue(int(config.iterations))
        _admit(self._damping, float(config.damping))
        self._damping.setValue(float(config.damping))
        _admit(self._max_velocity, float(config.max_velocity))
        self._max_velocity.setValue(float(config.max_velocity))
        _admit(self._friction, float(config.ground_friction))
        self._friction.setValue(float(config.ground_friction))
        self._self_contact.setChecked(bool(config.self_contact))
        _admit(self._contact_elasticity, float(config.self_contact_elasticity))
        self._contact_elasticity.setValue(float(config.self_contact_elasticity))
        self._contact_elasticity.setEnabled(bool(config.self_contact))
        _admit(self._self_friction, float(config.self_friction))
        self._self_friction.setValue(float(config.self_friction))
        self._self_friction.setEnabled(bool(config.self_contact))
        for spin, value in zip(self._gravity, gravity):
            _admit(spin, value)
            spin.setValue(value)
            spin.setEnabled(not zero)
        self._zero_g.setChecked(zero)


class SmoothingPanel(_Panel):
    """The draw-point springs — the lag that makes a stiff rig read as flesh.

    :class:`~mesh_xml.render.draw_points.DrawPoints` stores stiffness and damping
    per point, and a tight head over a loose gut is the whole reason it does. The
    spin boxes address the **selected chunks**, falling back to every point when
    nothing is selected — the same rule ``ChunkPanel``'s Connect button uses, so
    "the selection" means one thing across the window.

    That scope is what makes the latch load bearing here rather than merely
    tidy. Changing the selection refreshes the boxes from the newly scoped
    points; unguarded, that write comes straight back as a user edit and is
    applied to the *new* scope, so clicking a second chunk silently flattens
    everything the author had just given the first one.

    ``DrawPoints.sync`` hands a freshly added point the *constructor* defaults,
    which the retired uniform version fixed by re-applying its one value to every
    point on every ``topology_changed``. That cure is now worse than the disease:
    it would wipe per-chunk authoring on any structural edit, including the
    :class:`Macro` a radius-curve apply submits. So the panel instead remembers
    the last **rig-wide** value it wrote and gives it to genuinely new ids only,
    tracked in :attr:`_known`. A per-chunk edit deliberately does not become the
    default; it is a statement about one chunk, not about the rig.

    Squash stays rig-wide because ``DrawPoints`` stores one scalar for it, and
    inventing a per-point squash here would be a field the renderer cannot read.
    """

    def __init__(self, document: Document, parent: QWidget | None = None) -> None:
        super().__init__(document, parent)

        self._stiffness = _spin(0.0, 1.0, 0.01)
        self._damping = _spin(0.0, 1.0, 0.01)
        self._squash = _spin(0.0, 5.0, 0.05)
        self._scope = QLabel()
        self._mixed = _note("")

        #: The last rig-wide pair, handed to chunks added later. ``None`` means
        #: the author has not made a rig-wide statement, so new points keep the
        #: ``DrawPoints`` constructor defaults — which is exactly what the boxes
        #: are already showing.
        self._default: tuple[float, float] | None = None
        self._known: set[int] = set(document.body.chunk_ids())

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_springs())
        layout.addStretch(1)

        self._stiffness.valueChanged.connect(self._on_spring_changed)
        self._damping.valueChanged.connect(self._on_spring_changed)
        self._squash.valueChanged.connect(self._on_squash_changed)
        self._follow(document.settings_changed, document.selection_changed)
        document.topology_changed.connect(self._on_topology_changed)
        document.body_replaced.connect(self._on_body_replaced)

        self.refresh()

    def scope_ids(self) -> list[int]:
        """The chunks the spin boxes write: the selection, or the whole rig."""
        ids = self._document.body.chunk_ids()
        selection = self._document.selection
        scoped = [chunk_id for chunk_id in ids if chunk_id in selection]
        return scoped or ids

    def _build_springs(self) -> QWidget:
        box = QGroupBox("Draw points")
        layout = _form()
        box.setLayout(layout)
        layout.addRow("Stiffness", self._stiffness)
        layout.addRow("Damping", self._damping)
        layout.addRow("Squash", self._squash)
        layout.addRow(self._scope)
        layout.addRow(self._mixed)
        layout.addRow(
            _note(
                "Per-tick coefficients, not per-second: they are applied once "
                f"per {solver.TICK_HZ:g} Hz tick and are only meaningful against "
                "that fixed rate. Do not read them as rates and do not scale "
                "them by frame time."
            )
        )
        layout.addRow(
            _note(
                "Stiffness and damping are stored per point, and these boxes "
                "write the selected chunks only — a tight head over a loose "
                "gut. Select nothing to write every point, which also sets what "
                "a chunk added later inherits. Squash is one number for the "
                "whole rig; the model has nowhere to put a per-point one."
            )
        )

        self._stiffness.setToolTip(
            "Fraction of the remaining gap to the chunk closed each tick. "
            "Stiffness 1 with damping 0 follows the chunks exactly, which is the "
            "baseline the lag is measured against."
        )
        self._damping.setToolTip(
            "Momentum carried in from the previous tick. This is what makes a "
            "point overshoot and settle rather than ease in."
        )
        self._squash.setToolTip(
            "Thins radii with speed. Isotropic, so it is the cheap honest half "
            "of squash-and-stretch; 0 disables it. Visual only — the solver "
            "never sees it."
        )
        return box

    @_user_edit
    def _on_spring_changed(self, _value: float) -> None:
        self._push_springs()
        self._document.settings_changed.emit()

    @_user_edit
    def _on_squash_changed(self, value: float) -> None:
        self._document.draw.squash = float(value)
        self._document.settings_changed.emit()

    def _on_topology_changed(self) -> None:
        self._adopt_new_points()

    def _on_body_replaced(self) -> None:
        """A new rig arrives with new draw points, so the old default is void."""
        self._default = None
        self._known = set(self._document.body.chunk_ids())
        self.refresh()

    def _push_springs(self) -> None:
        draw = self._document.draw
        stiffness = float(self._stiffness.value())
        damping = float(self._damping.value())
        mask = self._mask(self._document.selection or None)
        draw.stiffness[mask] = stiffness
        draw.damping[mask] = damping
        if not self._document.selection:
            self._default = (stiffness, damping)

    def _adopt_new_points(self) -> None:
        """Give ids the panel has never seen the rig-wide default, and only them."""
        ids = self._document.body.chunk_ids()
        fresh = {chunk_id for chunk_id in ids if chunk_id not in self._known}
        self._known = set(ids)
        if self._default is None or not fresh:
            return
        draw = self._document.draw
        mask = self._mask(fresh)
        draw.stiffness[mask], draw.damping[mask] = self._default

    def _mask(self, wanted: set[int] | None) -> np.ndarray:
        """Draw-point slots for ``wanted`` ids; ``None`` means every point.

        Built by walking the body's own id order rather than indexing by id,
        because ``DrawPoints`` is slot-indexed and a chunk's slot is not its id:
        ``Body.remove_chunk`` swaps the last chunk into the freed slot. Sizing
        from the draw array rather than the chunk count is what keeps a stale
        sync from raising IndexError out of a signal handler.
        """
        count = len(self._document.draw.stiffness)
        if wanted is None:
            return np.ones(count, dtype=bool)
        mask = np.zeros(count, dtype=bool)
        for slot, chunk_id in enumerate(self._document.body.chunk_ids()[:count]):
            mask[slot] = chunk_id in wanted
        return mask

    def _refresh(self) -> None:
        draw = self._document.draw
        mask = self._mask(self._document.selection or None)
        selected = len(self._document.selection)

        if mask.any():
            stiffness, damping = draw.stiffness[mask], draw.damping[mask]
            self._stiffness.setValue(float(stiffness[0]))
            self._damping.setValue(float(damping[0]))
            mixed = bool(
                (stiffness != stiffness[0]).any() or (damping != damping[0]).any()
            )
        else:
            mixed = False

        self._squash.setValue(float(draw.squash))
        self._scope.setText(
            f"Writing to {selected} selected chunk(s)."
            if selected
            else f"Writing to every draw point ({len(draw.stiffness)})."
        )
        self._mixed.setText(
            "These chunks differ; the boxes show the first and an edit levels "
            "them."
            if mixed
            else ""
        )


class CreaturePanel(_Panel):
    """The roster: pick a creature, name it, herd it.

    The combo is a *view* of connected components — the document recomputes
    it like a blend group, and row *n* here is surface group *n* out there.
    Picking a row selects the creature's chunks, which is what arms every
    per-creature edit (colour, tempo, duplicate): the roster does not act on
    creatures directly, it aims the selection the existing gestures already
    fire from.
    """

    def __init__(self, document: Document, parent: QWidget | None = None) -> None:
        super().__init__(document, parent)

        self._roster = QComboBox()
        self._name = QLineEdit()
        self._name.setPlaceholderText("unnamed")
        self._duplicate = QPushButton("Duplicate")
        self._empty = _note(
            "No creatures yet. Add (A) grows chunks; linked chunks are one "
            "creature, and every separate cluster is its own."
        )

        form = _form()
        form.addRow("Creature", self._roster)
        form.addRow("Name", self._name)
        form.addRow(self._duplicate)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._empty)
        layout.addStretch(1)

        self._roster.setToolTip(
            "Every creature in the document, in the surface's own group "
            "order. Picking one selects its chunks."
        )
        self._name.setToolTip(
            "The selected creature's name, painted across its chunks like "
            "its colour. Clear it to fall back to the generated label."
        )
        self._duplicate.setToolTip(
            "Copy every creature the selection touches — chunks, links, "
            "meta, and its controllers with ids remapped and world points "
            "carried along. One undo step; the copy takes the selection, so "
            "duplicating again builds a herd."
        )

        self._roster.activated.connect(self._on_pick)
        self._name.editingFinished.connect(self._on_rename)
        self._duplicate.clicked.connect(self._on_duplicate)
        self._follow(
            document.topology_changed,
            document.selection_changed,
            document.body_replaced,
            document.chunk_changed,
        )

        self.refresh()

    @_user_edit
    def _on_pick(self, index: int) -> None:
        creatures = self._document.creatures()
        if 0 <= index < len(creatures):
            self._document.set_selection(creatures[index].chunks)

    @_user_edit
    def _on_rename(self) -> None:
        creature = self._current_creature()
        if creature is None:
            return
        text = self._name.text().strip()
        # editingFinished also fires on plain focus-out; only a changed name
        # is an edit, or every glance at the panel would push a command.
        if (creature.name or "") == text:
            return
        self._document.rename_creature(text)

    @_user_edit
    def _on_duplicate(self) -> None:
        self._document.duplicate_creature()

    def _current_creature(self) -> Creature | None:
        selection = self._document.selection
        if not selection:
            return None
        for creature in self._document.creatures():
            if selection <= set(creature.chunks):
                return creature
        return None

    def _refresh(self) -> None:
        creatures = self._document.creatures()
        self._empty.setVisible(not creatures)
        self._roster.setEnabled(bool(creatures))

        self._roster.clear()
        for index, creature in enumerate(creatures):
            self._roster.addItem(
                f"{creature.label(index)} — {len(creature.chunks)} chunk(s)"
            )

        current = self._current_creature()
        if current is not None:
            self._roster.setCurrentIndex(
                next(
                    i
                    for i, c in enumerate(creatures)
                    if c.chunks == current.chunks
                )
            )
            self._name.setText(current.name or "")
        else:
            self._roster.setCurrentIndex(-1)
            self._name.clear()
        self._name.setEnabled(current is not None)
        self._duplicate.setEnabled(bool(self._document.selection))


class ControllerPanel(_Panel):
    """The gait, editable: every controller on the stack, its knobs, its switch.

    Widget rows are generated from the serialization registry's field list
    (:func:`mesh_xml.anim.serialize.fields_of`), so the panel edits exactly
    what the file can carry: a controller registered for v3 grows a UI here
    without this module changing, and a knob the format cannot save never
    appears. Numbers and booleans are editable; id lists and vectors show as
    read-only summaries, because editing those is rig surgery the viewport
    and the autorig do better.
    """

    def __init__(self, document: Document, parent: QWidget | None = None) -> None:
        super().__init__(document, parent)

        self._autorig = QPushButton("Fit autorig")
        self._rig_legs = QPushButton("Rig legs")
        self._tempo = _spin(0.05, 20.0, 0.25, decimals=2, suffix="×")
        self._empty = _note(
            "No controllers. Fit autorig to give this rig a spine, an idle "
            "breath, and a wave driver on its anchor — the same recipe the "
            "shipped examples carry. Rig legs turns leg-shaped appendages "
            "into a self-propelled walk."
        )

        #: One entry per built group: the controller type it was built for,
        #: its enabled checkbox, and the value widget per field. What lets a
        #: value edit refresh in place — see :meth:`_refresh`.
        self._groups: list[tuple[type, QCheckBox, dict[str, QWidget]]] = []

        self._list_host = QWidget()
        self._list_layout = QVBoxLayout(self._list_host)
        self._list_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self._list_host)

        fitters = QWidget()
        fitters_layout = QHBoxLayout(fitters)
        fitters_layout.setContentsMargins(0, 0, 0, 0)
        fitters_layout.addWidget(self._autorig)
        fitters_layout.addWidget(self._rig_legs)

        tempo_row = QWidget()
        tempo_layout = QHBoxLayout(tempo_row)
        tempo_layout.setContentsMargins(0, 0, 0, 0)
        tempo_layout.addWidget(QLabel("Tempo"))
        tempo_layout.addWidget(self._tempo, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(fitters)
        layout.addWidget(tempo_row)
        layout.addWidget(self._empty)
        layout.addWidget(scroll, 1)

        self._autorig.setToolTip(
            "Replace the stack with controllers fitted to the body's shape, "
            "plus a wave driver leading its anchor. One undo step, pin "
            "included."
        )
        self._rig_legs.setToolTip(
            "Compile the measured walker onto every leg-shaped appendage — "
            "three chunks grown from the spine, ending near the floor: "
            "limb IK with paired reaction, trot-phased footholds, posture "
            "pressing on the feet, a hip truss, and no pins. One undo step. "
            "The creature walks the way its spine points."
        )
        self._tempo.setToolTip(
            "The selected creature's gait clock. Rates and speeds multiply "
            "by this, periods and durations divide, so ×2 is the same "
            "strides twice as often. Applies to every creature the selection "
            "touches, one undo step; saved with the file."
        )
        self._autorig.clicked.connect(self._on_autorig)
        self._rig_legs.clicked.connect(self._on_rig_legs)
        self._tempo.valueChanged.connect(self._on_tempo_changed)
        self._follow(
            document.controllers_changed,
            document.body_replaced,
            document.selection_changed,
            document.topology_changed,
        )

        self.refresh()

    @_user_edit
    def _on_autorig(self) -> None:
        self._document.fit_autorig()

    @_user_edit
    def _on_rig_legs(self) -> None:
        self._document.rig_legs()

    @_user_edit
    def _on_tempo_changed(self, value: float) -> None:
        if abs(float(value) - self._document.creature_tempo()) < 1e-9:
            return
        self._document.set_creature_tempo(float(value))

    def _submit_field(self, index: int, field: str, value: object) -> None:
        if self.syncing:
            return
        stack = self._document.controllers.controllers
        if not 0 <= index < len(stack):
            return
        if getattr(stack[index], field) == value:
            return
        self._document.do(SetControllerField(index, field, value))

    def _remove(self, index: int) -> None:
        if self.syncing:
            return
        if 0 <= index < len(self._document.controllers.controllers):
            self._document.do(RemoveController(index))

    def _refresh(self) -> None:
        """Update in place when the stack's shape is unchanged; rebuild otherwise.

        The split is the same one :class:`ChunkPanel` documents for its table:
        a value edit arrives back as ``controllers_changed`` while Qt is still
        inside the spin's own change signal, and tearing the widget out from
        under that is a use-after-free with extra steps. So a
        :class:`SetControllerField` refreshes values into the widgets that
        already exist, and only a change in the stack's composition — remove,
        replace, open — rebuilds them.
        """
        self._tempo.setEnabled(bool(self._document.selection))
        self._tempo.setValue(self._document.creature_tempo())

        controllers = self._document.controllers.controllers
        self._empty.setVisible(not controllers)
        if [type(c) for c in controllers] != [t for t, _, _ in self._groups]:
            self._rebuild(controllers)
            return

        for (_cls, enabled, widgets), controller in zip(self._groups, controllers):
            enabled.setChecked(bool(getattr(controller, "enabled", True)))
            for field, widget in widgets.items():
                value = getattr(controller, field)
                if isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setValue(value)
                elif isinstance(widget, QLabel):
                    widget.setText(_summarise(value))

    def _rebuild(self, controllers: list[object]) -> None:
        self._groups.clear()
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Unparented immediately, deleted later. deleteLater alone
                # leaves the widget parented until the event loop collects it,
                # and in that window the old group renders on top of the new
                # one — visible any time a rebuild lands between two paints.
                widget.setParent(None)
                widget.deleteLater()
        for index, controller in enumerate(controllers):
            self._list_layout.addWidget(self._build_group(index, controller))
        self._list_layout.addStretch(1)

    def _build_group(self, index: int, controller: object) -> QGroupBox:
        box = QGroupBox(type(controller).__name__)
        form = _form()
        box.setLayout(form)

        enabled = QCheckBox("Enabled")
        enabled.setChecked(bool(getattr(controller, "enabled", True)))
        enabled.toggled.connect(
            lambda on, i=index: self._submit_field(i, "enabled", bool(on))
        )
        remove = QPushButton("Remove")
        remove.clicked.connect(lambda _=False, i=index: self._remove(i))
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(enabled)
        header_layout.addStretch(1)
        header_layout.addWidget(remove)
        form.addRow(header)

        widgets: dict[str, QWidget] = {}
        try:
            fields = serialize.fields_of(type(controller))
        except serialize.ControllerFormatError:
            form.addRow(_note("Not registered for saving; enabled only."))
            self._groups.append((type(controller), enabled, widgets))
            return box

        for field in fields:
            if field == "enabled":
                continue
            widget = self._field_widget(index, field, getattr(controller, field))
            widgets[field] = widget
            form.addRow(field, widget)
        self._groups.append((type(controller), enabled, widgets))
        return box

    def _field_widget(self, index: int, field: str, value: object) -> QWidget:
        if isinstance(value, bool):
            check = QCheckBox()
            check.setChecked(value)
            check.toggled.connect(
                lambda on, i=index, f=field: self._submit_field(i, f, bool(on))
            )
            return check
        if isinstance(value, int):
            spin = QSpinBox()
            spin.setRange(-1_000_000, 1_000_000)
            spin.setValue(int(value))
            spin.valueChanged.connect(
                lambda v, i=index, f=field: self._submit_field(i, f, int(v))
            )
            return spin
        if isinstance(value, float):
            spin = _spin(-100_000.0, 100_000.0, 0.1, decimals=3)
            spin.setValue(float(value))
            spin.valueChanged.connect(
                lambda v, i=index, f=field: self._submit_field(i, f, float(v))
            )
            return spin
        return QLabel(_summarise(value))


def _summarise(value: object) -> str:
    """A compact read-only rendering for non-scalar controller fields."""
    if value is None:
        return "—"
    if isinstance(value, (list, tuple)):
        if len(value) > 4:
            return f"{len(value)} value(s)"
        parts = [
            f"{v:.2f}" if isinstance(v, float) else str(v) for v in value
        ]
        return "(" + ", ".join(parts) + ")"
    return str(value)


class RoomPanel(_Panel):
    """The scene's rooms: pick one and the editor points at it — the cut.

    The selected row *is* the active room, and selecting another activates
    it, which every other panel experiences as File > Open. Activation is
    view state like the camera — no command, no dirty flag. The Name cell
    renames room and body as one value (file-grade, not undoable, and doors
    elsewhere are not chased — the document's status line counts the ones
    left addressing the old name). The Abstract checkbox moves a room
    between its tiers, deliberately per room rather than wholesale: an
    abstracted far side is a closed door, so parking rooms is an authoring
    statement, never a side effect of looking away (SCENE_BINDING.md §3).
    The File column is read-only truth about where each room saves,
    starred while it holds unsaved changes.

    Dialogs stay the window's: Open Into Scene and Remove ask for things a
    dock does not own (a file dialog, a discard confirmation), so they
    leave as signals the way the Surface panel's viewport toggles do.
    """

    #: The cut landed somewhere new; the window re-frames the viewport.
    activated = Signal()
    #: Open Into Scene… — the window owns the file dialog.
    open_requested = Signal()
    #: Remove the active room — the window owns the confirmation.
    remove_requested = Signal()

    def __init__(self, document: Document, parent: QWidget | None = None) -> None:
        super().__init__(document, parent)

        self._rows: list[Room] = []
        self._table = self._build_table()
        self._new = QPushButton("New")
        self._open = QPushButton("Open…")
        self._remove = QPushButton("Remove")
        self._doors = self._build_doors_table()
        self._add_door = QPushButton("Add Door")
        self._remove_door = QPushButton("Remove Door")

        layout = QVBoxLayout(self)
        layout.addWidget(self._table, 1)
        layout.addWidget(self._build_actions())
        layout.addWidget(
            _note(
                "Click a room to edit it; every room keeps simulating "
                "either way. Abstract parks a room at its cheap tier — "
                "anchors and brains only — and a door into a parked room "
                "is closed until it realizes."
            )
        )
        layout.addWidget(self._build_doors(), 1)

        self._new.setToolTip(
            "Add an empty room to the scene and switch to it. Its name is "
            "the scene's address for doors, editable in the table."
        )
        self._open.setToolTip(
            "Open a rig file as a new room beside the others, named from "
            "the file's own body."
        )
        self._remove.setToolTip(
            "Remove the active room from the scene. Doors elsewhere that "
            "lead to it become closed doors. The last room stays — a "
            "document is at least one room."
        )

        self._table.itemChanged.connect(self._on_item_changed)
        self._table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self._new.clicked.connect(self._on_new)
        self._open.clicked.connect(self._on_open)
        self._remove.clicked.connect(self._on_remove)
        self._doors.itemChanged.connect(self._on_door_changed)
        self._add_door.clicked.connect(self._on_add_door)
        self._remove_door.clicked.connect(self._on_remove_door)

        self._follow(
            document.scene_changed,
            document.dirty_changed,
            document.body_replaced,
            document.topology_changed,
        )

        self.refresh()

    def _build_table(self) -> QTableWidget:
        table = QTableWidget(0, len(_ROOM_COLUMNS))
        table.setHorizontalHeaderLabels(_ROOM_COLUMNS)
        table.verticalHeader().setVisible(False)
        table.setSortingEnabled(False)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        header = table.horizontalHeader()
        header.setSectionResizeMode(_R_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(
            _R_CREATURES, QHeaderView.ResizeMode.ResizeToContents
        )
        header.setSectionResizeMode(
            _R_ABSTRACT, QHeaderView.ResizeMode.ResizeToContents
        )
        header.setSectionResizeMode(_R_FILE, QHeaderView.ResizeMode.ResizeToContents)
        return table

    def _build_actions(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for button in (self._new, self._open, self._remove):
            layout.addWidget(button)
        return row

    def _build_doors_table(self) -> QTableWidget:
        table = QTableWidget(0, len(_DOOR_COLUMNS))
        table.setHorizontalHeaderLabels(_DOOR_COLUMNS)
        table.verticalHeader().setVisible(False)
        table.setSortingEnabled(False)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        header = table.horizontalHeader()
        for column in (_D_NAME, _D_TO_ROOM, _D_TO_DOOR):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        for column in (_D_AT, _D_RADIUS, _D_STATUS):
            header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        return table

    def _build_doors(self) -> QWidget:
        box = QGroupBox("Doors — the active room's exits")
        layout = QVBoxLayout(box)
        layout.addWidget(self._doors, 1)
        buttons = QWidget()
        buttons_layout = QHBoxLayout(buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.addWidget(self._add_door)
        buttons_layout.addWidget(self._remove_door)
        layout.addWidget(buttons)
        layout.addWidget(
            _note(
                "A door is a place plus a promise: brains walk to it, the "
                "scene carries whoever arrives to the far room's paired "
                "door. The last column says when a promise cannot be kept "
                "— a closed door is waited at, never an error. Saved with "
                "the room's file."
            )
        )
        self._add_door.setToolTip(
            "Add a door at the selection's centroid (the origin when "
            "nothing is selected), aimed at the first other room, arriving "
            "at a far door of the same name."
        )
        self._remove_door.setToolTip(
            "Remove the selected door. Far doors that paired with it "
            "become closed doors."
        )
        return box

    @_user_edit
    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        row, column = item.row(), item.column()
        if not 0 <= row < len(self._rows):
            return
        room = self._rows[row]
        if column == _R_NAME:
            if item.text().strip() != room.name:
                # rename_room refuses collisions and emits scene_changed
                # either way, which snaps a refused cell back to the truth.
                self._document.rename_room(room, item.text())
        elif column == _R_ABSTRACT:
            abstracted = item.checkState() == Qt.CheckState.Checked
            if abstracted != room.abstracted:
                self._document.set_room_abstracted(room, abstracted)

    @_user_edit
    def _on_table_selection_changed(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        if not 0 <= row < len(self._rows):
            return
        room = self._rows[row]
        if room is self._document.room:
            return
        self._document.set_active_room(room)
        self.activated.emit()

    @_user_edit
    def _on_new(self) -> None:
        document = self._document
        room = document.add_room(Room(name=document.unique_room_name()))
        document.set_active_room(room)
        self.activated.emit()

    @_user_edit
    def _on_open(self) -> None:
        self.open_requested.emit()

    @_user_edit
    def _on_remove(self) -> None:
        self.remove_requested.emit()

    @_user_edit
    def _on_door_changed(self, item: QTableWidgetItem) -> None:
        room = self._document.room
        index, column = item.row(), item.column()
        if not 0 <= index < len(room.doors):
            return
        door = room.doors[index]
        text = item.text().strip()
        try:
            if column == _D_NAME:
                updated = replace(door, name=text)
            elif column == _D_AT:
                updated = replace(
                    door, at=tuple(float(v) for v in text.split())
                )
            elif column == _D_TO_ROOM:
                updated = replace(door, to_room=text)
            elif column == _D_TO_DOOR:
                updated = replace(door, to_door=text)
            elif column == _D_RADIUS:
                updated = replace(door, radius=float(text))
            else:
                return
        except ValueError as exc:
            # Door validates itself on construction; a bad edit reads its
            # own refusal and the refresh snaps the cell back to the truth.
            self._document.status.emit(str(exc))
            self.refresh()
            return
        if updated != door:
            self._document.set_door(room, index, updated)

    @_user_edit
    def _on_add_door(self) -> None:
        self._document.add_door(self._document.room)

    @_user_edit
    def _on_remove_door(self) -> None:
        index = self._doors.currentRow()
        if 0 <= index < len(self._document.room.doors):
            self._document.remove_door(self._document.room, index)

    def _cell(self, row: int, column: int) -> QTableWidgetItem:
        item = self._table.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            if column == _R_ABSTRACT:
                item.setFlags(
                    Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                )
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            elif column != _R_NAME:
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
            self._table.setItem(row, column, item)
        return item

    def _write_row(self, row: int, room: Room) -> None:
        self._cell(row, _R_NAME).setText(room.name)
        self._cell(row, _R_CREATURES).setText(str(len(room.body.components())))
        self._cell(row, _R_ABSTRACT).setCheckState(
            Qt.CheckState.Checked if room.abstracted else Qt.CheckState.Unchecked
        )
        path = self._document.room_path(room)
        star = "*" if self._document.room_dirty(room) else ""
        if path is not None:
            text = f"{star}{path.name}"
        else:
            text = f"{star}unsaved" if star else "—"
        self._cell(row, _R_FILE).setText(text)

    def _door_cell(self, row: int, column: int) -> QTableWidgetItem:
        item = self._doors.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            if column == _D_STATUS:
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
            self._doors.setItem(row, column, item)
        return item

    def _write_door_row(self, row: int, door: Door) -> None:
        self._door_cell(row, _D_NAME).setText(door.name)
        self._door_cell(row, _D_AT).setText(
            " ".join(f"{float(v):g}" for v in door.at)
        )
        self._door_cell(row, _D_TO_ROOM).setText(door.to_room)
        self._door_cell(row, _D_TO_DOOR).setText(door.to_door)
        self._door_cell(row, _D_RADIUS).setText(f"{float(door.radius):g}")
        status = self._document.door_status(self._document.room, door)
        cell = self._door_cell(row, _D_STATUS)
        cell.setText(status)
        cell.setToolTip(status)

    def _apply_active(self) -> None:
        model = self._table.selectionModel()
        model.clearSelection()
        try:
            row = self._rows.index(self._document.room)
        except ValueError:
            return
        model.select(
            self._table.model().index(row, _R_NAME),
            QItemSelectionModel.SelectionFlag.Select
            | QItemSelectionModel.SelectionFlag.Rows,
        )

    def _refresh(self) -> None:
        self._rows = list(self._document.scene.rooms)
        self._table.setRowCount(len(self._rows))
        for row, room in enumerate(self._rows):
            self._write_row(row, room)
        self._apply_active()
        self._remove.setEnabled(len(self._rows) > 1)

        # Doors are room data abstraction never parks, so a parked room's
        # doors stay editable — unlike its chunks, they are not elsewhere.
        doors = self._document.room.doors
        self._doors.setRowCount(len(doors))
        for row, door in enumerate(doors):
            self._write_door_row(row, door)
        self._remove_door.setEnabled(bool(doors))


class ChunkPanel(_Panel):
    """The chunk table, bound to the selection.

    Rows map to chunk ids positionally through :attr:`_rows`, so the table is
    never sorted: a sortable header would silently redirect every edit to a
    different chunk. Ids are also not stable *positions* —
    :meth:`Body.remove_chunk` swaps the last chunk into the freed slot — which is
    why a structural change rebuilds rather than patching rows.

    Value edits do not rebuild. A rename arrives back as ``chunk_changed`` while
    Qt is still inside the ``itemChanged`` emission that caused it, and tearing
    the row out from under that would leave Qt holding a deleted item.
    """

    def __init__(self, document: Document, parent: QWidget | None = None) -> None:
        super().__init__(document, parent)

        self._rows: list[int] = []
        self._table = self._build_table()
        self._add = QPushButton("Add")
        self._connect = QPushButton("Connect")
        self._disconnect = QPushButton("Disconnect")
        self._delete = QPushButton("Delete")

        layout = QVBoxLayout(self)
        layout.addWidget(self._table, 1)
        layout.addWidget(self._build_actions())
        layout.addWidget(
            _note(
                "Add grows a linked chunk out of the selection, or a free one "
                "when nothing is selected. Connect chains the selection in "
                "table order; disconnect drops every link between selected "
                "chunks."
            )
        )

        self._add.setToolTip(
            "One selected chunk: a linked chunk continuing its chain, radius "
            "inherited. Several: a chunk linked to each, placed outward — how "
            "a girdle closes. None: a free chunk past the body's +x edge. The "
            "new chunk becomes the selection, so repeated Add extrudes a limb."
        )
        self._table.itemChanged.connect(self._on_item_changed)
        self._table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self._add.clicked.connect(self._on_add)
        self._connect.clicked.connect(self._on_connect)
        self._disconnect.clicked.connect(self._on_disconnect)
        self._delete.clicked.connect(self._on_delete)

        self._follow(document.topology_changed, document.body_replaced)
        document.chunk_changed.connect(self._on_chunk_changed)
        document.selection_changed.connect(self._on_selection_changed)

        self.refresh()

    @property
    def table(self) -> QTableWidget:
        return self._table

    def _build_table(self) -> QTableWidget:
        table = QTableWidget(0, len(_COLUMNS))
        table.setHorizontalHeaderLabels(_COLUMNS)
        table.verticalHeader().setVisible(False)
        table.setSortingEnabled(False)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        header = table.horizontalHeader()
        header.setSectionResizeMode(_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_RADIUS, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_PINNED, QHeaderView.ResizeMode.ResizeToContents)
        return table

    def _build_actions(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for button in (self._add, self._connect, self._disconnect, self._delete):
            layout.addWidget(button)
        return row

    @_user_edit
    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        row, column = item.row(), item.column()
        if not 0 <= row < len(self._rows):
            return
        chunk_id = self._rows[row]
        if not self._document.body.has_chunk(chunk_id):
            return
        chunk = self._document.body.chunk(chunk_id)

        if column == _NAME:
            if item.text() != chunk.name:
                self._document.do(Rename(chunk_id, item.text()))
        elif column == _RADIUS:
            self._commit_radius(row, chunk_id, item.text(), chunk.radius)
        elif column == _PINNED:
            pinned = item.checkState() == Qt.CheckState.Checked
            if pinned != chunk.pinned:
                self._document.do(SetPinned(chunk_id, pinned))

    def _commit_radius(
        self, row: int, chunk_id: int, text: str, current: float
    ) -> None:
        try:
            radius = float(text)
        except ValueError:
            radius = 0.0
        if radius <= 0.0:
            with self._sync():
                self._write_row(row, chunk_id)
            self._document.status.emit(f"Radius must be positive, not {text!r}")
            return
        if radius != current:
            self._document.do(SetRadius(chunk_id, radius))

    @_user_edit
    def _on_table_selection_changed(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        self._document.set_selection(
            self._rows[index.row()]
            for index in rows
            if 0 <= index.row() < len(self._rows)
        )

    @_user_edit
    def _on_add(self) -> None:
        self._document.create_chunk()

    @_user_edit
    def _on_connect(self) -> None:
        ids = self._selected_ids()
        body = self._document.body
        self._submit(
            [AddLink(a, b) for a, b in zip(ids, ids[1:]) if not body.has_link(a, b)],
            "Connect",
        )

    @_user_edit
    def _on_disconnect(self) -> None:
        ids = self._selected_ids()
        body = self._document.body
        self._submit(
            [RemoveLink(a, b) for a, b in combinations(ids, 2) if body.has_link(a, b)],
            "Disconnect",
        )

    @_user_edit
    def _on_delete(self) -> None:
        self._submit(
            [RemoveChunk(chunk_id) for chunk_id in self._selected_ids()],
            "Delete Chunks",
        )

    def _on_chunk_changed(self, chunk_id: int) -> None:
        if not self._document.body.has_chunk(chunk_id):
            return
        try:
            row = self._rows.index(chunk_id)
        except ValueError:
            return
        with self._sync():
            self._write_row(row, chunk_id)

    def _on_selection_changed(self) -> None:
        with self._sync():
            self._apply_selection()
            self._update_buttons()

    def _submit(self, commands: list[Command], text: str) -> None:
        if not commands:
            return
        if len(commands) == 1:
            self._document.do(commands[0])
        else:
            self._document.do(Macro(commands, text=text))

    def _selected_ids(self) -> list[int]:
        selection = self._document.selection
        return [chunk_id for chunk_id in self._rows if chunk_id in selection]

    def _cell(self, row: int, column: int) -> QTableWidgetItem:
        item = self._table.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            if column == _PINNED:
                item.setFlags(
                    Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                )
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, column, item)
        return item

    def _write_row(self, row: int, chunk_id: int) -> None:
        chunk = self._document.body.chunk(chunk_id)
        self._cell(row, _NAME).setText(chunk.name)
        self._cell(row, _RADIUS).setText(format(chunk.radius, _RADIUS_FORMAT))
        self._cell(row, _PINNED).setCheckState(
            Qt.CheckState.Checked if chunk.pinned else Qt.CheckState.Unchecked
        )

    def _apply_selection(self) -> None:
        model = self._table.selectionModel()
        model.clearSelection()
        flags = (
            QItemSelectionModel.SelectionFlag.Select
            | QItemSelectionModel.SelectionFlag.Rows
        )
        selection = self._document.selection
        for row, chunk_id in enumerate(self._rows):
            if chunk_id in selection:
                model.select(self._table.model().index(row, _NAME), flags)

    def _update_buttons(self) -> None:
        count = len(self._document.selection)
        self._connect.setEnabled(count >= 2)
        self._disconnect.setEnabled(count >= 2)
        self._delete.setEnabled(count >= 1)

    def _refresh(self) -> None:
        self._rows = self._document.body.chunk_ids()
        self._table.setRowCount(len(self._rows))
        for row, chunk_id in enumerate(self._rows):
            self._write_row(row, chunk_id)
        self._apply_selection()
        self._update_buttons()


def build_panels(document: Document) -> dict[str, QWidget]:
    """Every dock panel, keyed by the title its dock should carry.

    The main window assembles docks out of this without naming a single panel
    class, so adding one is an edit to this function alone.
    """
    # Imported here rather than at module scope: curve_panel builds on _Panel and
    # the shared form helpers from this module, so a top-level import in both
    # directions would be a cycle. This is the only edge that closes it.
    from mesh_xml.editor.curve_panel import CurvePanel

    return {
        "Playback": PlaybackPanel(document),
        "Rooms": RoomPanel(document),
        "Chunks": ChunkPanel(document),
        "Creatures": CreaturePanel(document),
        "Controllers": ControllerPanel(document),
        "Surface": SurfacePanel(document),
        "Solver": SolverPanel(document),
        "Smoothing": SmoothingPanel(document),
        "Curve": CurvePanel(document),
    }
