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
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mesh_xml.core import solver
from mesh_xml.editor.document import (
    AddLink,
    Command,
    Document,
    Macro,
    RemoveChunk,
    RemoveLink,
    Rename,
    SetPinned,
    SetRadius,
)

__all__ = [
    "PlaybackPanel",
    "SurfacePanel",
    "SolverPanel",
    "SmoothingPanel",
    "ChunkPanel",
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
)

_AXES = ("X", "Y", "Z")

_NAME, _RADIUS, _PINNED = 0, 1, 2
_COLUMNS = ("Name", "Radius", "Pinned")
_RADIUS_FORMAT = ".3f"


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
        self._overlay.toggled.connect(self._on_overlay_toggled)
        self._ortho.toggled.connect(self._on_ortho_toggled)
        self._follow(document.settings_changed, document.body_replaced)

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
        layout.addRow(self._overlay)
        layout.addRow(self._ortho)
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

    def _pick(
        self, current: tuple[float, float, float], title: str
    ) -> tuple[float, float, float] | None:
        chosen = QColorDialog.getColor(_colour(current), self, title)
        return _rgb(chosen) if chosen.isValid() else None

    def _write_blend(self, value: float) -> None:
        self._document.metaball.blend = float(value)
        self._document.settings_changed.emit()

    def _refresh(self) -> None:
        settings = self._document.metaball
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
            "Push overlapping chunks apart when they are strangers in the "
            "graph — three or more links from each other. Neighbours are "
            "exempt: within a limb, overlap is what makes the surface read as "
            "one volume. Off by default because rigs overlap on purpose."
        )
        self._contact_elasticity.setToolTip(
            "Fraction of an overlap closed per tick. 1.0 separates a lone "
            "pair in one tick and reads as billiard balls; lower values let "
            "limbs squash into each other and recover, which reads as flesh."
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
        self._document.settings_changed.emit()

    @_user_edit
    def _on_damping_changed(self, value: float) -> None:
        self._document.solver_config.damping = float(value)
        self._document.settings_changed.emit()

    @_user_edit
    def _on_max_velocity_changed(self, value: float) -> None:
        self._document.solver_config.max_velocity = float(value)
        self._document.settings_changed.emit()

    @_user_edit
    def _on_friction_changed(self, value: float) -> None:
        self._document.solver_config.ground_friction = float(value)
        self._document.settings_changed.emit()

    @_user_edit
    def _on_self_contact_toggled(self, on: bool) -> None:
        self._document.solver_config.self_contact = bool(on)
        self._document.settings_changed.emit()

    @_user_edit
    def _on_contact_elasticity_changed(self, value: float) -> None:
        self._document.solver_config.self_contact_elasticity = float(value)
        self._document.settings_changed.emit()

    @_user_edit
    def _on_gravity_changed(self, _value: float) -> None:
        gravity = tuple(spin.value() for spin in self._gravity)
        if any(gravity):
            self._gravity_memo = gravity
        self._document.solver_config.gravity = gravity
        self._document.settings_changed.emit()

    @_user_edit
    def _on_zero_g_toggled(self, zero: bool) -> None:
        if zero:
            authored = self._authored_gravity()
            if any(authored):
                self._gravity_memo = authored
            self._document.solver_config.gravity = (0.0, 0.0, 0.0)
        else:
            self._document.solver_config.gravity = self._gravity_memo
        self._document.settings_changed.emit()

    def _authored_gravity(self) -> tuple[float, ...]:
        gravity = tuple(float(v) for v in self._document.solver_config.gravity)
        return (gravity + (0.0, 0.0, 0.0))[: len(_AXES)]

    def _refresh(self) -> None:
        config = self._document.solver_config
        gravity = self._authored_gravity()
        zero = not any(gravity)

        self._iterations.setValue(int(config.iterations))
        self._damping.setValue(float(config.damping))
        self._max_velocity.setValue(float(config.max_velocity))
        self._friction.setValue(float(config.ground_friction))
        self._self_contact.setChecked(bool(config.self_contact))
        self._contact_elasticity.setValue(float(config.self_contact_elasticity))
        self._contact_elasticity.setEnabled(bool(config.self_contact))
        for spin, value in zip(self._gravity, gravity):
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
        "Chunks": ChunkPanel(document),
        "Surface": SurfacePanel(document),
        "Solver": SolverPanel(document),
        "Smoothing": SmoothingPanel(document),
        "Curve": CurvePanel(document),
    }
