"""The editor's single source of truth.

Every defect in §5 of the refactor plan traced back to the same shape: the old
app had three widgets each mutating the model directly and then each remembering
to refresh the others. Miss one refresh and the UI silently disagreed with the
data. So there is exactly one owner of state here, it emits when state changes,
and views subscribe. Views never write to the body directly — they submit
commands.

That indirection is also what makes undo possible at all. The old app had Undo
and Redo wired to empty methods for the whole of its life, because with edits
scattered across three widgets there was nothing central to record.

**Editing while the simulation runs** is allowed and behaves the way it reads:
commands act on the body, and the solver keeps acting on it too. Undoing a move
restores the position the chunk had before the edit, and the next tick carries
on from there. The alternative — locking edits during playback — makes the live
tweaking this tool exists for impossible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from PySide6.QtCore import QObject, Signal

from mesh_xml.anim import Clock, ControllerStack, autorig
from mesh_xml.anim.controllers.drive import WaveAnchor
from mesh_xml.core import Body, LinkType
from mesh_xml.core.solver import SolverConfig
from mesh_xml.core.terrain import Terrain
from mesh_xml.io import xml_legacy, xml_v2
from mesh_xml.render import DrawPoints, sdf
from mesh_xml.render.gl.metaball import MetaballSettings

__all__ = [
    "Command",
    "Document",
    "AddChunk",
    "RemoveChunk",
    "MoveChunk",
    "SetRadius",
    "SetPinned",
    "Rename",
    "AddLink",
    "RemoveLink",
    "SetControllerField",
    "RemoveController",
    "ReplaceControllers",
    "Macro",
]

#: Spacing of a grown chunk, in radii. Under 2.0 on purpose: neighbours must
#: overlap for the smooth-min surface to read as one volume rather than beads.
_GROW_SPACING = 1.6


class Command(Protocol):
    """A reversible edit. ``text`` is what the Undo menu item says."""

    text: str

    def apply(self, document: "Document") -> None: ...

    def revert(self, document: "Document") -> None: ...


@dataclass
class AddChunk:
    name: str
    pos: tuple[float, ...]
    radius: float = 0.25
    chunk_id: int | None = None
    text: str = "Add Chunk"

    def apply(self, document: Document) -> None:
        # Reuses the id on redo so any link recorded against it still resolves.
        self.chunk_id = document.body.add_chunk(
            self.name, self.pos, radius=self.radius, id=self.chunk_id
        )

    def revert(self, document: Document) -> None:
        document.body.remove_chunk(self.chunk_id)


@dataclass
class RemoveChunk:
    chunk_id: int
    text: str = "Delete Chunk"
    _name: str = ""
    _pos: tuple[float, ...] = ()
    _radius: float = 0.25
    _inv_mass: float = 1.0
    _meta: dict = field(default_factory=dict)
    _links: list[tuple[int, float, int, float]] = field(default_factory=list)

    def apply(self, document: Document) -> None:
        body = document.body
        chunk = body.chunk(self.chunk_id)
        self._name, self._pos = chunk.name, chunk.pos
        self._radius, self._inv_mass = chunk.radius, chunk.inv_mass
        self._meta = dict(chunk.meta)
        # Incident links die with the chunk, so they have to be recorded here or
        # undo silently returns a chunk with no connections.
        self._links = [
            (link.a if link.b == self.chunk_id else link.b, link.rest, int(link.type), link.elasticity)
            for link in body.links()
            if self.chunk_id in (link.a, link.b)
        ]
        body.remove_chunk(self.chunk_id)

    def revert(self, document: Document) -> None:
        body = document.body
        mass = 1.0 if self._inv_mass == 0.0 else 1.0 / self._inv_mass
        body.add_chunk(
            self._name,
            self._pos,
            radius=self._radius,
            mass=mass,
            pinned=self._inv_mass == 0.0,
            id=self.chunk_id,
            meta=self._meta,
        )
        for other, rest, link_type, elasticity in self._links:
            if body.has_chunk(other):
                body.add_link(
                    self.chunk_id, other, rest=rest, type=LinkType(link_type), elasticity=elasticity
                )


@dataclass
class MoveChunk:
    chunk_id: int
    pos: tuple[float, ...]
    text: str = "Move Chunk"
    _previous: tuple[float, ...] = ()

    def apply(self, document: Document) -> None:
        self._previous = document.body.chunk(self.chunk_id).pos
        document.body.set_pos(self.chunk_id, self.pos)

    def revert(self, document: Document) -> None:
        document.body.set_pos(self.chunk_id, self._previous)


@dataclass
class SetRadius:
    chunk_id: int
    radius: float
    text: str = "Set Radius"
    _previous: float = 0.0

    def apply(self, document: Document) -> None:
        self._previous = document.body.chunk(self.chunk_id).radius
        document.body.set_radius(self.chunk_id, self.radius)

    def revert(self, document: Document) -> None:
        document.body.set_radius(self.chunk_id, self._previous)


@dataclass
class SetPinned:
    chunk_id: int
    pinned: bool
    text: str = "Pin Chunk"
    _previous: float = 1.0

    def apply(self, document: Document) -> None:
        self._previous = document.body.chunk(self.chunk_id).inv_mass
        document.body.set_pinned(self.chunk_id, self.pinned)

    def revert(self, document: Document) -> None:
        if self._previous == 0.0:
            document.body.set_pinned(self.chunk_id, True)
        else:
            document.body.set_pinned(self.chunk_id, False, mass=1.0 / self._previous)


@dataclass
class Rename:
    chunk_id: int
    name: str
    text: str = "Rename Chunk"
    _previous: str = ""

    def apply(self, document: Document) -> None:
        self._previous = document.body.name_of(self.chunk_id)
        document.body.rename(self.chunk_id, self.name)

    def revert(self, document: Document) -> None:
        document.body.rename(self.chunk_id, self._previous)


@dataclass
class AddLink:
    a: int
    b: int
    rest: float | None = None
    type: LinkType = LinkType.NORMAL
    render: bool = True
    text: str = "Connect"
    _added: bool = False

    def apply(self, document: Document) -> None:
        self._added = document.body.add_link(
            self.a, self.b, rest=self.rest, type=self.type, render=self.render
        )

    def revert(self, document: Document) -> None:
        if self._added:
            document.body.remove_link(self.a, self.b)


@dataclass
class RemoveLink:
    a: int
    b: int
    text: str = "Disconnect"
    _rest: float = 0.0
    _type: int = 0
    _elasticity: float = 1.0

    def apply(self, document: Document) -> None:
        link = document.body.link(self.a, self.b)
        self._rest, self._type, self._elasticity = link.rest, int(link.type), link.elasticity
        document.body.remove_link(self.a, self.b)

    def revert(self, document: Document) -> None:
        document.body.add_link(
            self.a, self.b, rest=self._rest, type=LinkType(self._type), elasticity=self._elasticity
        )


@dataclass
class SetControllerField:
    """Assign one field on one controller of the document's stack.

    Indexed by stack position rather than holding the controller object, so a
    redo after the stack was replaced and restored still lands on the right
    controller — the same reason :class:`AddChunk` re-applies its id.
    """

    index: int
    field: str
    value: object
    text: str = "Edit Controller"
    _previous: object = None

    def apply(self, document: Document) -> None:
        controller = document.controllers.controllers[self.index]
        self._previous = getattr(controller, self.field)
        setattr(controller, self.field, self.value)

    def revert(self, document: Document) -> None:
        controller = document.controllers.controllers[self.index]
        setattr(controller, self.field, self._previous)


@dataclass
class RemoveController:
    index: int
    text: str = "Remove Controller"
    _controller: object = None

    def apply(self, document: Document) -> None:
        self._controller = document.controllers.controllers.pop(self.index)

    def revert(self, document: Document) -> None:
        document.controllers.controllers.insert(self.index, self._controller)


@dataclass
class ReplaceControllers:
    """Swap the whole stack, keeping the old one for undo."""

    stack: ControllerStack
    text: str = "Replace Controllers"
    _previous: ControllerStack | None = None

    def apply(self, document: Document) -> None:
        self._previous = document.controllers
        document.controllers = self.stack

    def revert(self, document: Document) -> None:
        document.controllers = self._previous


@dataclass
class Macro:
    """Several commands undone and redone as one. Reverts in reverse order."""

    commands: list[Command]
    text: str = "Edit"

    def apply(self, document: Document) -> None:
        for command in self.commands:
            command.apply(document)

    def revert(self, document: Document) -> None:
        for command in reversed(self.commands):
            command.revert(document)


#: Commands that change the chunk or link set, as opposed to only their values.
_STRUCTURAL = (AddChunk, RemoveChunk, AddLink, RemoveLink, Macro)

#: Commands after which the controller panel must re-read the stack. Macro is
#: included because a compound edit may carry controller commands inside it;
#: a spurious refresh is cheap and a missed one is a stale panel.
_CONTROLLER = (SetControllerField, RemoveController, ReplaceControllers, Macro)


class Document(QObject):
    """Owns the rig, the simulation state, and the undo stack."""

    body_replaced = Signal()
    topology_changed = Signal()
    chunk_changed = Signal(int)
    selection_changed = Signal()
    settings_changed = Signal()
    terrain_changed = Signal()
    controllers_changed = Signal()
    running_changed = Signal(bool)
    dirty_changed = Signal(bool)
    history_changed = Signal()
    status = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.body = Body()
        # The library defaults to no floor, which for an editor means pressing
        # play drops the rig out of frame and never brings it back. An app is
        # the right layer to decide there is a world.
        self.solver_config = SolverConfig(ground_y=0.0)
        self.metaball = MetaballSettings()
        self.clock = Clock()
        self.draw = DrawPoints(self.body)
        self.controllers = ControllerStack(self.body)
        self.selection: set[int] = set()
        self.path: Path | None = None

        self._running = False
        self._dirty = False
        self._undo: list[Command] = []
        self._redo: list[Command] = []
        self._rest_pose: dict[int, tuple[float, ...]] = {}

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    def set_running(self, running: bool) -> None:
        if running == self._running:
            return
        self._running = running
        if running:
            # Banked time from while we were paused would be spent as a burst of
            # catch-up ticks on the first frame, which looks like a lurch.
            self.clock.reset()
        else:
            self.capture_rest_pose()
        self.running_changed.emit(running)

    def capture_rest_pose(self) -> None:
        """Remember where things are, so :meth:`reset_pose` can come back here."""
        self._rest_pose = {cid: self.body.chunk(cid).pos for cid in self.body.chunk_ids()}

    def reset_pose(self) -> None:
        """Return every chunk to the captured rest pose and drop all motion."""
        for chunk_id, pos in self._rest_pose.items():
            if self.body.has_chunk(chunk_id):
                self.body.set_pos(chunk_id, pos)
        self.draw.snap(self.body)
        self.controllers.reset()
        self.clock.reset()
        self.topology_changed.emit()

    def tick(self) -> None:
        """One fixed simulation tick.

        The single place that knows what a tick consists of. Both :meth:`advance`
        and :meth:`step_once` route through it, so the controllers run on both
        paths. Two call sites each remembering the recipe is the exact shape this
        module exists to delete, and the failure mode is quiet: single-stepping
        would simply stop running them.
        """
        self.controllers.tick(self.body, self.clock.dt, self.solver_config)
        self.draw.update(self.body)

    def step_once(self) -> None:
        """Advance one tick whatever the play state, without disturbing the pose.

        Toggling ``running`` on and off to steal a tick would look equivalent and
        is not: pausing captures a new rest pose, so it would quietly overwrite
        the pose Reset restores to.
        """
        self.tick()

    def advance(self, elapsed: float) -> float:
        """Run the simulation for one frame. Returns the render blend factor.

        Draw points update even while paused so a chunk dragged with the mouse
        still eases into place rather than snapping.
        """
        if self._running:
            for _ in self.clock.advance(elapsed):
                self.tick()
            return self.clock.alpha

        self.draw.update(self.body)
        return 1.0

    def primitives(self) -> np.ndarray:
        """Packed SDF primitives at the current draw positions."""
        alpha = self.clock.alpha if self._running else 1.0
        return sdf.pack(
            self.body,
            positions=self.draw.render_pos(alpha),
            radii=self.draw.render_radius(self.body),
        )

    def group_rows(self) -> np.ndarray:
        """The body's blend groups, coloured from the authored surface colour.

        One creature per connected component: smooth-min flesh within it, a
        hard skin against every other. Group 0 wears the Surface panel's
        colour exactly, so a single-creature document looks precisely as
        authored; a herd spreads itself around the hue wheel from that base.
        """
        return sdf.group_table(self.body, self.metaball.surface_colour)

    # ------------------------------------------------------------------
    # the world
    # ------------------------------------------------------------------

    @property
    def terrain(self) -> Terrain | None:
        """The world the creature stands in, or ``None`` for the flat default.

        Read from :attr:`solver_config` rather than stored again: the solver
        collides with the terrain and the viewport draws it, and two copies of
        a world is exactly how the picture and the physics would drift apart.
        One world (TERRAIN.md §2) — this property is a view of it, not a
        second home for it.
        """
        return self.solver_config.terrain

    def set_terrain(self, terrain: Terrain | None) -> None:
        """Install a world, replacing the object wholesale.

        Identity is the change signal: terrain does not move per tick, so
        consumers (the viewport's upload) may treat a new object as "the world
        changed" and an unchanged one as nothing to do. Mutating a ``Terrain``
        in place therefore goes unnoticed on purpose — build a new one.

        Not a :class:`Command` and not a dirty-flag edit: terrain authoring
        stays in Python this round (TERRAIN.md §4), and :meth:`adopt` routes
        through here after :meth:`_replace` has already declared the document
        clean — a loaded world must not reopen as an unsaved change.
        """
        if terrain is self.solver_config.terrain:
            return
        self.solver_config.terrain = terrain
        self.terrain_changed.emit()

    # ------------------------------------------------------------------
    # editing
    # ------------------------------------------------------------------

    def do(self, command: Command) -> None:
        """Apply a command and push it onto the undo stack."""
        command.apply(self)
        self._undo.append(command)
        self._redo.clear()
        self._after_edit(command)

    def undo(self) -> None:
        if not self._undo:
            return
        command = self._undo.pop()
        command.revert(self)
        self._redo.append(command)
        self._after_edit(command)
        self.status.emit(f"Undo {command.text}")

    def redo(self) -> None:
        if not self._redo:
            return
        command = self._redo.pop()
        command.apply(self)
        self._undo.append(command)
        self._after_edit(command)
        self.status.emit(f"Redo {command.text}")

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_text(self) -> str:
        return self._undo[-1].text if self._undo else ""

    @property
    def redo_text(self) -> str:
        return self._redo[-1].text if self._redo else ""

    def _after_edit(self, command: Command) -> None:
        self.set_dirty(True)
        if isinstance(command, _STRUCTURAL):
            self.selection = {cid for cid in self.selection if self.body.has_chunk(cid)}
            self.draw.sync(self.body)
            self.topology_changed.emit()
            self.selection_changed.emit()
        else:
            chunk_id = getattr(command, "chunk_id", -1)
            self.chunk_changed.emit(chunk_id)
        if isinstance(command, _CONTROLLER):
            self.controllers_changed.emit()
        self.history_changed.emit()

    # ------------------------------------------------------------------
    # selection
    # ------------------------------------------------------------------

    def set_selection(self, chunk_ids) -> None:
        selection = {cid for cid in chunk_ids if self.body.has_chunk(cid)}
        if selection != self.selection:
            self.selection = selection
            self.selection_changed.emit()

    def toggle_selection(self, chunk_id: int) -> None:
        if chunk_id in self.selection:
            self.selection = self.selection - {chunk_id}
        else:
            self.selection = self.selection | {chunk_id}
        self.selection_changed.emit()

    # ------------------------------------------------------------------
    # authoring
    # ------------------------------------------------------------------

    def create_chunk(self) -> int:
        """Add a chunk, growing from the selection. Returns the new chunk's id.

        The rule, in three sentences. With exactly one chunk selected the new
        chunk is linked to it, inherits its radius, and continues the chain —
        along selected-minus-neighbour at the same spacing, or outward from a
        hub's neighbours (a lone chunk grows along +x). With several selected
        it sits outward from their centroid and links to each, which is how a
        girdle or a palm closes. With nothing selected it is a free chunk just
        past the body's +x extent — origin on an empty body — because a new
        component is what an empty selection says you meant.

        One call is one undo step: the chunk and its links arrive and leave
        together, via a :class:`Macro`. The new chunk becomes the selection, so
        repeated calls extrude a limb — the legacy editor's create-at-origin
        button never composed like this, which is why the plan asked for
        placement relative to the selection instead of reproducing it.

        The id is chosen here rather than by :meth:`Body.add_chunk` because the
        link commands must name their endpoint before it exists; ``AddChunk``
        re-applies the same id on redo, so the links stay resolvable.
        """
        body = self.body
        dim = body.dim
        selected = sorted(self.selection)
        along_x = np.zeros(dim)
        along_x[0] = 1.0

        def outward(origin: np.ndarray, away_from: np.ndarray) -> np.ndarray:
            direction = origin - away_from
            length = float(np.linalg.norm(direction))
            return direction / length if length > 1e-6 else along_x

        if not selected:
            radius = 0.5
            if body.n_chunks == 0:
                pos = np.zeros(dim)
            else:
                pos = body.pos.mean(axis=0).astype(np.float64)
                pos[0] = float(body.pos[:, 0].max()) + _GROW_SPACING * radius
        elif len(selected) == 1:
            anchor = np.asarray(body.chunk(selected[0]).pos)
            radius = body.chunk(selected[0]).radius
            neighbours = body.neighbors(selected[0])
            if len(neighbours) == 1:
                prev = np.asarray(body.chunk(neighbours[0]).pos)
                step = anchor - prev
                if float(np.linalg.norm(step)) <= 1e-6:
                    step = along_x * _GROW_SPACING * radius
                pos = anchor + step
            else:
                centre = (
                    np.mean([body.chunk(n).pos for n in neighbours], axis=0)
                    if neighbours
                    else anchor
                )
                pos = anchor + outward(anchor, centre) * _GROW_SPACING * radius
        else:
            radius = float(np.mean([body.chunk(s).radius for s in selected]))
            centre = np.mean([body.chunk(s).pos for s in selected], axis=0)
            pos = centre + outward(centre, body.pos.mean(axis=0)) * (
                _GROW_SPACING * radius
            )

        new_id = max(body.chunk_ids(), default=-1) + 1
        added = AddChunk(
            f"chunk{new_id}",
            tuple(float(v) for v in pos),
            radius=radius,
            chunk_id=new_id,
        )
        if selected:
            links: list[Command] = [AddLink(new_id, s) for s in selected]
            self.do(Macro([added, *links], text="Add Chunk"))
        else:
            self.do(added)
        self.set_selection((new_id,))
        return new_id

    def commit_drag(self, chunk_id: int, final: tuple[float, ...]) -> None:
        """Record a finished viewport drag as the edit the gesture meant.

        For an ordinary chunk that is a move. For a chunk a driver leads — any
        controller claiming it as ``anchor`` with an authored ``origin`` — the
        gesture means "the motion's home is now here": the origin is
        re-authored, minus the swing the next tick will add so the chunk stays
        exactly under the hand, and the move rides in the same undo step for
        the paused case, where no driver is running to make it moot. Without
        this, dragging the worm's head was a fight the hand lost one tick
        after letting go.

        The caller has already rewound its preview; every position this
        records arrives through the command stack.
        """
        commands: list[Command] = [MoveChunk(chunk_id, tuple(final))]
        for index, controller in enumerate(self.controllers.controllers):
            if getattr(controller, "anchor", None) != chunk_id:
                continue
            if not hasattr(controller, "origin"):
                continue
            origin = np.asarray(final, dtype=np.float64)
            swing = getattr(controller, "swing", None)
            if callable(swing):
                origin = origin - np.asarray(swing(self.controllers.time))[: origin.size]
            commands.append(
                SetControllerField(
                    index, "origin", tuple(float(v) for v in origin)
                )
            )
        if len(commands) == 1:
            self.do(commands[0])
        else:
            self.do(Macro(commands, text="Move Anchor"))

    def fit_autorig(self) -> None:
        """Fit an autorig plus wave driver to the body, as one undo step.

        The same recipe the shipped examples were upgraded with:
        :func:`~mesh_xml.anim.autorig.auto_controllers` for the shape, a
        :class:`~mesh_xml.anim.controllers.drive.WaveAnchor` at the front so
        the anchor leads. ``auto_controllers`` pins the anchor directly while
        fitting; that pin is rewound here and re-applied through a command, so
        undo takes the whole thing back out — stack, pin, and all.
        """
        body = self.body
        if body.n_chunks == 0:
            return
        inv_before = body.inv_mass.copy()
        fitted = autorig.auto_controllers(
            body, ground_y=self.solver_config.ground_y, seed=5
        )
        commands: list[Command] = []
        if fitted.anchor is not None:
            origin = body.chunk(fitted.anchor).pos
            fitted.stack.controllers.insert(0, WaveAnchor(fitted.anchor, origin))
            previous = float(inv_before[body.slot(fitted.anchor)])
            if previous > 0.0:
                # Rewind the fitter's direct pin to the exact prior mass, so
                # the SetPinned command both records it and re-applies the pin.
                body.set_pinned(fitted.anchor, False, mass=1.0 / previous)
                commands.append(SetPinned(fitted.anchor, True))
        commands.append(ReplaceControllers(fitted.stack))
        if len(commands) == 1:
            self.do(commands[0])
        else:
            self.do(Macro(commands, text="Fit Autorig"))

    def rig_legs(self) -> None:
        """Fit the walking stack to every leg-shaped appendage, as one undo step.

        The counterpart of :meth:`fit_autorig` for the decision it declines:
        :func:`~mesh_xml.anim.autorig.fit_legs` compiles the measured
        self-propelled walker recipe onto the body's leg candidates, and this
        method turns its plan into commands — the hip truss as mechanism
        links, every pin removed (a walker's weight must reach its feet; a
        driver's pin would absorb the posture holding it up), and the stack
        replaced whole. Undo puts back the pins, the old stack, and takes the
        truss out again.
        """
        ground = self.solver_config.ground_y
        fitted = autorig.fit_legs(
            self.body, ground_y=ground if ground is not None else 0.0
        )
        if fitted is None:
            self.status.emit(
                "No leg-shaped appendages: a leg is three chunks grown from "
                "the spine, ending near the floor"
            )
            return

        stack = ControllerStack(self.body, seed=self.controllers.seed)
        for controller in fitted.controllers:
            stack.add(controller)

        commands: list[Command] = [
            AddLink(a, b, render=False, text="Truss")
            for a, b in fitted.truss
            if not self.body.has_link(a, b)
        ]
        commands += [SetPinned(cid, False) for cid in fitted.unpin]
        commands.append(ReplaceControllers(stack))
        self.do(Macro(commands, text="Rig Legs"))
        self.status.emit(f"Rigged {fitted.description}")

    # ------------------------------------------------------------------
    # files
    # ------------------------------------------------------------------

    @property
    def dirty(self) -> bool:
        return self._dirty

    def set_dirty(self, dirty: bool) -> None:
        if dirty != self._dirty:
            self._dirty = dirty
            self.dirty_changed.emit(dirty)

    def new(self) -> None:
        self._replace(Body(), None)
        self.set_terrain(None)
        self.status.emit("New rig")

    def open(self, path) -> None:
        """Load a rig and its gait, falling back to the legacy reader for v1 files."""
        path = Path(path)
        controllers: ControllerStack | None = None
        terrain: Terrain | None = None
        try:
            rig = xml_v2.read(path)
            controllers = rig.controllers
            terrain = rig.terrain
            body = rig.body
            driven = len(controllers.controllers)
            note = f"Opened {path.name}"
            if driven:
                note += f" ({driven} controller(s))"
        except Exception:
            result = xml_legacy.read_legacy(path)
            body = result.body
            note = f"Opened {path.name} (legacy, {result.coords} coordinates)"
            if result.missing_mesh:
                note += f"; {len(result.missing_mesh)} bone(s) had no mesh"
        self.adopt(body, controllers=controllers, path=path, terrain=terrain)
        self.status.emit(note)

    def adopt(
        self,
        body: Body,
        controllers: ControllerStack | None = None,
        path=None,
        terrain: Terrain | None = None,
    ) -> None:
        """Take over a rig, optionally with the gait driving it and its world.

        The supported door for installing a body the document did not load
        itself, rather than callers reaching into the private replace path and
        guessing which signals to emit. Used both by :meth:`open` and by code
        that builds a creature directly.

        ``terrain=None`` *clears* any previous world rather than keeping it,
        for the same reason :meth:`_replace` drops the old controllers: what a
        previous document stood on means nothing to this one, and a stale
        floor under a fresh rig would be drawn and collided with silently.
        """
        self._replace(body, Path(path) if path else None)
        if controllers is not None:
            self.controllers = controllers
            self.controllers_changed.emit()
        self.set_terrain(terrain)

    def save(self, path) -> None:
        """Write the body, its gait, *and* its world.

        Passing the controllers is the whole point of the v3 format: without it
        a save keeps the skeleton and silently discards everything that made it
        move. The terrain rides along on the same reasoning — a creature saved
        without its world reloads standing on nothing.
        """
        path = Path(path)
        xml_v2.save(self.body, path, self.controllers, terrain=self.terrain)
        self.path = path
        self.set_dirty(False)
        self.status.emit(f"Saved {path.name}")

    def _replace(self, body: Body, path: Path | None) -> None:
        self.body = body
        self.path = path
        self.draw = DrawPoints(body)
        # Controllers hold chunk ids, and ids from the previous rig mean nothing
        # in this one; carrying them over would silently drive the wrong chunks.
        self.controllers = ControllerStack(body)
        self.clock.reset()
        self.selection.clear()
        self._undo.clear()
        self._redo.clear()
        self.capture_rest_pose()
        self.set_dirty(False)
        self.body_replaced.emit()
        self.topology_changed.emit()
        self.selection_changed.emit()
        self.controllers_changed.emit()
        self.history_changed.emit()

    @property
    def title(self) -> str:
        name = self.path.name if self.path else "Untitled"
        return f"{'*' if self._dirty else ''}{name} — mesh_xml"
