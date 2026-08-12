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

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol
import xml.etree.ElementTree as ET

import numpy as np
from PySide6.QtCore import QObject, Signal

from mesh_xml.anim import Clock, ControllerStack, autorig, place, serialize, tempo
from mesh_xml.anim.controllers.drive import WaveAnchor
from mesh_xml.core import Body, LinkType
from mesh_xml.core.solver import SolverConfig
from mesh_xml.core.terrain import Terrain
from mesh_xml.io import scene_xml, xml_legacy, xml_v2
from mesh_xml.render import DrawPoints, sdf
from mesh_xml.render.gl.metaball import MetaballSettings
from mesh_xml.scene.door import Door
from mesh_xml.scene.room import Room
from mesh_xml.scene.scene import Scene

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
    "SetMeta",
    "SetControllerField",
    "AddController",
    "RemoveController",
    "ReplaceControllers",
    "Macro",
    "Creature",
]

#: Spacing of a grown chunk, in radii. Under 2.0 on purpose: neighbours must
#: overlap for the smooth-min surface to read as one volume rather than beads.
_GROW_SPACING = 1.6

#: Chunk-meta key carrying the creature's authored name. Painted across the
#: component like colour and tempo: a creature has no object of its own, so
#: its name rides its chunks and dies with them.
CREATURE_META = "creature"


@dataclass(frozen=True)
class Creature:
    """One connected component, as the roster sees it.

    A view, not a noun: recomputed from the graph on demand, in the same
    first-seen-slot order the renderer numbers its blend groups — the
    roster's creature *n* wears surface group *n*. ``name`` is the first
    :data:`CREATURE_META` found on the component's chunks (id order), or
    ``None`` for the unnamed.
    """

    chunks: tuple[int, ...]
    name: str | None

    def label(self, index: int) -> str:
        """What a roster row says: the authored name, or an honest default."""
        return self.name if self.name else f"Creature {index}"


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
    mass: float = 1.0
    pinned: bool = False
    meta: dict | None = None
    chunk_id: int | None = None
    text: str = "Add Chunk"

    def apply(self, document: Document) -> None:
        # Reuses the id on redo so any link recorded against it still resolves.
        self.chunk_id = document.body.add_chunk(
            self.name,
            self.pos,
            radius=self.radius,
            mass=self.mass,
            pinned=self.pinned,
            meta=self.meta,
            id=self.chunk_id,
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
    elasticity: float = 1.0
    render: bool = True
    text: str = "Connect"
    _added: bool = False

    def apply(self, document: Document) -> None:
        self._added = document.body.add_link(
            self.a,
            self.b,
            rest=self.rest,
            type=self.type,
            elasticity=self.elasticity,
            render=self.render,
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
class SetMeta:
    """Assign (or, with ``None``, remove) one key of a chunk's opaque meta.

    Meta is the simulation-invisible store — import leftovers, creature
    colour — so this is a value command: the surface may need repainting but
    no topology moved.
    """

    chunk_id: int
    key: str
    value: str | None
    text: str = "Edit Meta"
    _previous: str | None = None
    _had: bool = False

    def apply(self, document: Document) -> None:
        meta = document.body.meta_of(self.chunk_id)
        self._had = self.key in meta
        self._previous = meta.get(self.key)
        if self.value is None:
            meta.pop(self.key, None)
        else:
            meta[self.key] = self.value

    def revert(self, document: Document) -> None:
        meta = document.body.meta_of(self.chunk_id)
        if self._had:
            meta[self.key] = self._previous
        else:
            meta.pop(self.key, None)


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
class AddController:
    """Append one controller to the stack.

    Holds the instance itself rather than a recipe, so redo restores the
    exact object any later :class:`SetControllerField` in the same history
    still points at through its stack index.
    """

    controller: object
    text: str = "Add Controller"

    def apply(self, document: Document) -> None:
        document.controllers.add(self.controller)

    def revert(self, document: Document) -> None:
        document.controllers.remove(self.controller)


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
_CONTROLLER = (
    SetControllerField,
    AddController,
    RemoveController,
    ReplaceControllers,
    Macro,
)


@dataclass
class _RoomState:
    """What the editor keeps per room that the room must not carry itself.

    A :class:`~mesh_xml.scene.room.Room` is Qt-free and editor-free — the
    package-2 firewall a test stands on — so everything the *editor* holds
    per room lives here instead, keyed by room on the document. One record
    is one room's editing session: the easing springs with their per-point
    tuning, the pose Reset returns to, the selection, an undo history that
    structurally cannot contain another room's commands, the file this room
    is, and the surface look (its blend saves with the room's file; the
    rest — colours, march quality — stays session-only).
    """

    draw: DrawPoints
    metaball: MetaballSettings
    rest_pose: dict[int, tuple[float, ...]] = field(default_factory=dict)
    selection: set[int] = field(default_factory=set)
    undo: list[Command] = field(default_factory=list)
    redo: list[Command] = field(default_factory=list)
    path: Path | None = None
    dirty: bool = False


class Document(QObject):
    """Owns the scene, the simulation state, and the per-room undo stacks."""

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
    #: The scene's shape changed — a room added, removed, renamed, moved
    #: between tiers, or saved. The rooms dock re-reads on this; the
    #: per-room signals above stay about the *active* room's contents.
    scene_changed = Signal()
    status = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        #: The scene this document edits: rooms in the plural, one clock
        #: between them (``docs/ROOMS.md`` package 6). A fresh document is a
        #: scene of one room — the null case that must stay bit-for-bit the
        #: old editor. A document *is still* one room at a time: ``room`` is
        #: the active one, drawn by the viewport and addressed by every
        #: panel, and the properties below keep the existing vocabulary
        #: (``document.body``, ``document.selection``) pointing at its
        #: state — so to a panel, a room switch is indistinguishable from
        #: opening a file.
        self.scene = Scene()
        self._room = self.scene.add_room(Room(name="document"))
        #: Editor state per room, keyed by the room it describes. The room
        #: stays Qt-free and editor-free — the package-2 firewall — so undo
        #: histories, draw springs, rest poses and file paths live here, on
        #: the document's side of the wall. Rooms enter through
        #: :meth:`add_room`, which is what keeps this table total.
        self._states: dict[Room, _RoomState] = {
            self._room: _RoomState(
                draw=DrawPoints(self._room.body), metaball=MetaballSettings()
            )
        }
        self._running = False

    # ------------------------------------------------------------------
    # the active room's state, in the document's vocabulary
    # ------------------------------------------------------------------

    @property
    def room(self) -> Room:
        """The active room: what the viewport draws and every panel edits."""
        return self._room

    @property
    def _state(self) -> _RoomState:
        return self._states[self._room]

    @property
    def body(self) -> Body:
        return self.room.body

    @body.setter
    def body(self, body: Body) -> None:
        self.room.body = body

    @property
    def controllers(self) -> ControllerStack:
        return self.room.controllers

    @controllers.setter
    def controllers(self, controllers: ControllerStack) -> None:
        self.room.controllers = controllers

    @property
    def solver_config(self) -> SolverConfig:
        return self.room.solver_config

    @solver_config.setter
    def solver_config(self, config: SolverConfig) -> None:
        self.room.solver_config = config

    @property
    def clock(self) -> Clock:
        """The clock — the scene's, whichever room is active.

        :meth:`Scene.add_room` aliases one clock into every room, so this
        property cannot show a per-room clock: no such object exists. The
        transport's ticks, dropped and alpha are scene-level truths by
        construction.
        """
        return self.room.clock

    @property
    def draw(self) -> DrawPoints:
        return self._state.draw

    @draw.setter
    def draw(self, draw: DrawPoints) -> None:
        self._state.draw = draw

    @property
    def metaball(self) -> MetaballSettings:
        return self._state.metaball

    @property
    def selection(self) -> set[int]:
        return self._state.selection

    @selection.setter
    def selection(self, chunk_ids) -> None:
        self._state.selection = set(chunk_ids)

    @property
    def path(self) -> Path | None:
        return self._state.path

    @path.setter
    def path(self, path: Path | None) -> None:
        self._state.path = path

    @property
    def _undo(self) -> list[Command]:
        return self._state.undo

    @property
    def _redo(self) -> list[Command]:
        return self._state.redo

    # ------------------------------------------------------------------
    # rooms
    # ------------------------------------------------------------------

    def add_room(self, room: Room) -> Room:
        """Adopt a room into the scene, with fresh editor state beside it.

        The one way in. :meth:`Scene.add_room` wires the shared clock and
        refuses duplicate names; this wraps it so every room the scene holds
        has an entry in the editor's table — switching, pose capture and
        save-all iterate that table and never guess.
        """
        self.scene.add_room(room)
        self._states[room] = _RoomState(
            draw=DrawPoints(room.body),
            metaball=MetaballSettings(),
            rest_pose={
                cid: room.body.chunk(cid).pos for cid in room.body.chunk_ids()
            },
        )
        self.scene_changed.emit()
        return room

    def remove_room(self, room: Room) -> Room:
        """Release a room and its editor state.

        The last room is refused — a document is at least one room, the
        null case the whole binding stands on. Removing the active room
        activates a neighbour first, so the viewport is never pointed at
        nothing. Doors elsewhere that named the departed room become
        closed doors, which is the scene's own answer, not an error.
        """
        if room not in self._states:
            raise ValueError(
                f"room {room.name!r} is not in this document's scene"
            )
        if len(self.scene.rooms) == 1:
            raise ValueError("a document is at least one room")
        if room is self._room:
            index = self.scene.rooms.index(room)
            neighbour = (
                self.scene.rooms[index - 1] if index else self.scene.rooms[1]
            )
            self.set_active_room(neighbour)
        self.scene.remove_room(room)
        del self._states[room]
        self.scene_changed.emit()
        self.status.emit(f"Removed room {room.name!r}")
        return room

    def rename_room(self, room: Room, name: str) -> None:
        """Rename a room — and its body: one value, one owner, two write sites.

        File-grade like the room list itself, not undoable this round.
        Doors address rooms by name and a rename does not chase their
        references — a door into the old name becomes a closed door, which
        the status message counts rather than hides. The body's name is
        what saves, so the room dirties.
        """
        if room not in self._states:
            raise ValueError(
                f"room {room.name!r} is not in this document's scene"
            )
        name = name.strip()
        if not name or name == room.name:
            return
        if any(other.name == name for other in self.scene.rooms if other is not room):
            self.status.emit(f"A room named {name!r} already exists")
            self.scene_changed.emit()
            return
        old = room.name
        room.name = name
        room.body.name = name
        self._dirty_room(room)
        stale = sum(
            1
            for other in self.scene.rooms
            for door in other.doors
            if door.to_room == old
        )
        if stale:
            self.status.emit(
                f"Renamed {old!r} to {name!r}; {stale} door(s) still "
                f"address {old!r} and are closed until re-aimed"
            )
        else:
            self.status.emit(f"Renamed {old!r} to {name!r}")
        self.scene_changed.emit()

    def set_room_abstracted(self, room: Room, abstracted: bool) -> None:
        """Move a room between its tiers — an explicit authoring act.

        The editor's cut deliberately is not :meth:`Scene.activate`: rooms
        stay realized unless an author parks one here, because an
        abstracted far side is a closed door and a scene-wide policy would
        show every traveller waiting in a doorway (SCENE_BINDING.md §3).
        Not an edit: no command, no dirty flag — a simulation state, like
        running. Realizing recaptures the rest pose, because the rebuilt
        creature standing wherever its brain walked *is* the pose Reset
        should return to.
        """
        if room not in self._states:
            raise ValueError(
                f"room {room.name!r} is not in this document's scene"
            )
        if abstracted == room.abstracted:
            return
        state = self._states[room]
        if abstracted:
            room.abstract()
        else:
            room.realize()
            state.rest_pose = {
                cid: room.body.chunk(cid).pos for cid in room.body.chunk_ids()
            }
        if room is self._room:
            state.selection = {
                cid for cid in state.selection if room.body.has_chunk(cid)
            }
            state.draw.snap(room.body)
            self.topology_changed.emit()
            self.selection_changed.emit()
            self.controllers_changed.emit()
            self.history_changed.emit()
        self.scene_changed.emit()

    def _dirty_room(self, room: Room) -> None:
        """Mark any room's file as differing from disk, active or not."""
        state = self._states[room]
        if not state.dirty:
            state.dirty = True
            if room is self._room:
                self.dirty_changed.emit(True)

    # ------------------------------------------------------------------
    # doors
    # ------------------------------------------------------------------
    #
    # Door edits are file-grade like the room list and the terrain: an
    # explicit authoring act, immediate, on no undo stack — but they DO
    # dirty the room, because doors save with its file and a clean flag
    # over an unsaved doorway is a lie Save All would act on.

    def add_door(self, room: Room) -> Door:
        """Author a door where the author is looking.

        Placed at the selection's centroid — the Add rule one noun up — or
        the origin when nothing is selected or the room is not on screen.
        Named uniquely within the room, aimed at the first *other* room
        (itself when alone, which the scene treats as closed — an honest
        placeholder), arriving at a far door of its own name, which is the
        pairing convention the shipped tests use: den's "east" meets
        meadow's "west" only because an author re-aims it, but "door"
        meeting "door" works out of the box.
        """
        if room not in self._states:
            raise ValueError(
                f"room {room.name!r} is not in this document's scene"
            )
        taken = {door.name for door in room.doors}
        name, suffix = "door", 2
        while name in taken:
            name = f"door {suffix}"
            suffix += 1

        state = self._states[room]
        if room is self._room and state.selection:
            centroid = np.mean(
                [room.body.chunk(cid).pos for cid in sorted(state.selection)],
                axis=0,
            )
            at = tuple(float(v) for v in centroid)
        else:
            at = (0.0,) * room.body.dim
        others = [r.name for r in self.scene.rooms if r is not room]
        door = Door(
            name, at, to_room=others[0] if others else room.name, to_door=name
        )
        room.doors = [*room.doors, door]
        self._dirty_room(room)
        self.scene_changed.emit()
        self.status.emit(f"Door {name!r} added to {room.name!r}")
        return door

    def set_door(self, room: Room, index: int, door: Door) -> None:
        """Replace one door record — how the dock edits a field.

        A duplicate name is refused: doors are the room's addressing the
        way rooms are the scene's, and ``ctx.doors`` is a dict that would
        silently keep one of the pair.
        """
        if room not in self._states:
            raise ValueError(
                f"room {room.name!r} is not in this document's scene"
            )
        if not 0 <= index < len(room.doors):
            raise IndexError(f"room {room.name!r} has no door {index}")
        if any(
            i != index and other.name == door.name
            for i, other in enumerate(room.doors)
        ):
            self.status.emit(
                f"A door named {door.name!r} already exists in {room.name!r}"
            )
            self.scene_changed.emit()
            return
        doors = list(room.doors)
        doors[index] = door
        room.doors = doors
        self._dirty_room(room)
        self.scene_changed.emit()

    def remove_door(self, room: Room, index: int) -> None:
        """Take a door out. Far sides that paired with it become closed doors."""
        if room not in self._states:
            raise ValueError(
                f"room {room.name!r} is not in this document's scene"
            )
        if not 0 <= index < len(room.doors):
            raise IndexError(f"room {room.name!r} has no door {index}")
        doors = list(room.doors)
        removed = doors.pop(index)
        room.doors = doors
        self._dirty_room(room)
        self.scene_changed.emit()
        self.status.emit(f"Removed door {removed.name!r} from {room.name!r}")

    def door_status(self, room: Room, door: Door) -> str:
        """Why a door is closed, or an empty string for one that would carry.

        The dock's surfacing of :meth:`Scene._cross`'s own refusals — the
        same conditions, answered before a creature walks there: an
        unknown room, the door's own room, a parked far side, a missing
        far door. Never an error; a closed door is a state, not a fault.
        """
        try:
            target = self.scene.room(door.to_room)
        except KeyError:
            return f"closed: no room {door.to_room!r}"
        if target is room:
            return "closed: leads to its own room"
        if target.abstracted:
            return f"closed: {target.name!r} is parked"
        if not any(far.name == door.to_door for far in target.doors):
            return f"closed: no door {door.to_door!r} in {target.name!r}"
        return ""

    def unique_room_name(self, base: str = "room") -> str:
        """A name no room in the scene holds: ``base``, then ``base 2``, on up."""
        names = {room.name for room in self.scene.rooms}
        if base not in names:
            return base
        index = 2
        while f"{base} {index}" in names:
            index += 1
        return f"{base} {index}"

    def room_path(self, room: Room) -> Path | None:
        """The file a room is, or ``None`` for one never saved."""
        return self._states[room].path

    def room_dirty(self, room: Room) -> bool:
        """Whether a room has unsaved changes, active or not."""
        return self._states[room].dirty

    def dirty_rooms(self) -> list[Room]:
        """Every room with unsaved changes, in scene order — the close prompt's list."""
        return [
            room for room in self.scene.rooms if self._states[room].dirty
        ]

    def set_active_room(self, room: Room) -> None:
        """Point the editor at ``room`` — the cut.

        To every panel this is File > Open: body, controllers, solver
        config, terrain, selection, history and dirty flag all change at
        once, so the emission suite is :meth:`_replace`'s plus the signals
        open never needed because it resets what a switch restores
        (``settings_changed``, ``terrain_changed``, ``dirty_changed``).

        Rooms tick while inactive — creatures cross, abstraction parks
        chunks — so the stashed selection is pruned against the body as it
        stands now, and the draw points snap to it rather than easing:
        animating across a cut would be a camera transition the cut exists
        to not have.
        """
        if room is self._room:
            return
        if room not in self._states:
            raise ValueError(
                f"room {room.name!r} is not in this document's scene; "
                "add it with add_room first"
            )
        self._room = room
        state = self._states[room]
        state.selection = {
            cid for cid in state.selection if room.body.has_chunk(cid)
        }
        state.draw.snap(room.body)
        self.body_replaced.emit()
        self.topology_changed.emit()
        self.selection_changed.emit()
        self.controllers_changed.emit()
        self.history_changed.emit()
        self.settings_changed.emit()
        self.terrain_changed.emit()
        self.dirty_changed.emit(state.dirty)
        self.status.emit(f"Room {room.name}")

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
        """Remember where things are, so :meth:`reset_pose` can come back here.

        Captured for every realized room, not only the active one: pause is
        scene-wide, and a room visited *after* the pause still owes Reset an
        answer. An abstracted room has nothing worth remembering — one
        parked anchor per creature — and recaptures when it realizes.
        """
        for room, state in self._states.items():
            if room.abstracted:
                continue
            state.rest_pose = {
                cid: room.body.chunk(cid).pos for cid in room.body.chunk_ids()
            }

    def reset_pose(self) -> None:
        """Return the active room to the captured rest pose and drop all motion.

        The active room only: Reset is a statement about what is on screen.
        What it deliberately cannot do is un-cross a creature that changed
        rooms during play — a door is a serialization boundary, and the
        ``has_chunk`` guard below makes the miss silent and honest.
        """
        for chunk_id, pos in self._state.rest_pose.items():
            if self.body.has_chunk(chunk_id):
                self.body.set_pos(chunk_id, pos)
        self.draw.snap(self.body)
        self.controllers.reset()
        self.clock.reset()
        self.topology_changed.emit()

    def tick(self) -> None:
        """One fixed step of the whole scene, plus the editor's per-tick work.

        What a tick *is* lives in :meth:`Scene.tick` — every room, looked at
        or not, then the door crossings the step earned. What belongs here
        is only what a window adds on top: easing the draw points that
        trail the *active* room's physics. Both :meth:`advance` and
        :meth:`step_once` route through this, so the controllers run on
        both paths — and a paused author single-stepping a crossing gets to
        watch it happen.
        """
        self.scene.tick()
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

    def attribute_rows(self) -> np.ndarray:
        """Per-primitive end colours — the weighted skin's table.

        Empty when nothing varies, which is
        :func:`~mesh_xml.render.sdf.pack_attributes`' contract: an
        unpainted document uploads no attributes and shades by the flat
        group colour on the code path it always ran, bit for bit.
        """
        return sdf.pack_attributes(self.body, self.metaball.surface_colour)

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
    # settings
    # ------------------------------------------------------------------

    def settings_edited(self) -> None:
        """A finished solver or blend edit — the doors rule for settings.

        File-grade like a door edit: an explicit authoring act, immediate,
        on no undo stack — but it DOES dirty the room, because solver and
        blend save with the room's file and a clean flag over an unsaved
        tuning is a lie Save All would act on. View-only settings (debug
        modes, colours, march quality) stay off this path: panels emit
        :attr:`settings_changed` raw for those and the file stays clean.
        """
        self._dirty_room(self._room)
        self.settings_changed.emit()

    # ------------------------------------------------------------------
    # editing
    # ------------------------------------------------------------------

    def do(self, command: Command) -> bool:
        """Apply a command and push it onto the active room's history.

        Refused while the room is abstracted: its creatures are parked
        data, and a command landing on the anchors-only body could orphan
        what :meth:`Room.realize` expects to find. Abstraction is a
        simulation state, not an edit state. Returns whether the command
        applied, so a caller with follow-up state of its own — a selection
        to move, an id to hand out — can decline along with it.
        """
        if self.room.abstracted:
            self.status.emit(f"Realize {self.room.name!r} to edit it")
            return False
        command.apply(self)
        self._undo.append(command)
        self._redo.clear()
        self._after_edit(command)
        return True

    def undo(self) -> None:
        if not self.can_undo:
            return
        command = self._undo.pop()
        command.revert(self)
        self._redo.append(command)
        self._after_edit(command)
        self.status.emit(f"Undo {command.text}")

    def redo(self) -> None:
        if not self.can_redo:
            return
        command = self._redo.pop()
        command.apply(self)
        self._undo.append(command)
        self._after_edit(command)
        self.status.emit(f"Redo {command.text}")

    @property
    def can_undo(self) -> bool:
        """Whether Undo has anywhere to go.

        Histories sleep while their room is abstracted: ids survive the
        abstract/realize round trip, but a dormant command's targets are
        parked data until then, and reverting into them would raise.
        """
        return bool(self._undo) and not self.room.abstracted

    @property
    def can_redo(self) -> bool:
        return bool(self._redo) and not self.room.abstracted

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

    def create_chunk(
        self,
        *,
        at: tuple[float, ...] | None = None,
        attach_to: int | None = None,
        radius: float | None = None,
    ) -> int | None:
        """Add a chunk, growing from the selection or at ``at``. Returns its id,
        or ``None`` when the room declined the edit (:meth:`do`'s abstracted
        refusal) — the selection stands, and no id is invented for a chunk
        that never existed.

        The rule, in three sentences. With exactly one chunk selected the new
        chunk is linked to it, inherits its radius, and continues the chain —
        along selected-minus-neighbour at the same spacing, or outward from a
        hub's neighbours (a lone chunk grows along +x). With several selected
        it sits outward from their centroid and links to each, which is how a
        girdle or a palm closes. With nothing selected it is a free chunk just
        past the body's +x extent — origin on an empty body — because a new
        component is what an empty selection says you meant.

        ``at`` is the same rule's second position source (``docs/PICK.md``
        §3): a gesture that already knows where — the surface pick, its
        offset along the normal already taken — states the position verbatim,
        and the selection stands aside. The hit states the attachment:
        ``attach_to`` earns exactly one link and, absent an explicit
        ``radius``, lends its own — grown anatomy matches local anatomy, the
        single-selection reasoning relocated to the cursor — while no
        ``attach_to`` means the free chunk's 0.5. Without ``at`` the
        overrides are refused rather than quietly dropped: the selection rule
        chooses links and radius itself. An ``attach_to`` naming no chunk is
        refused too, before any command applies — a live gesture hands over
        a live hit, so a stale id is a caller bug better named here than
        inside a half-applied :class:`Macro`.

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
        if at is None:
            if attach_to is not None or radius is not None:
                raise ValueError(
                    "attach_to and radius state a placement; "
                    "without at, the selection rule chooses both"
                )
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
            anchors = selected
        else:
            if attach_to is not None and not body.has_chunk(attach_to):
                raise ValueError(f"attach_to names no chunk {attach_to}")
            if len(at) != body.dim:
                raise ValueError(
                    f"at has {len(at)} components, expected {body.dim}; a "
                    "placement is stated in the body's own dimensionality"
                )
            pos = at
            if radius is None:
                radius = (
                    body.chunk(attach_to).radius if attach_to is not None else 0.5
                )
            anchors = [attach_to] if attach_to is not None else []

        new_id = max(body.chunk_ids(), default=-1) + 1
        added = AddChunk(
            f"chunk{new_id}",
            tuple(float(v) for v in pos),
            radius=radius,
            chunk_id=new_id,
        )
        if anchors:
            links: list[Command] = [AddLink(new_id, s) for s in anchors]
            applied = self.do(Macro([added, *links], text="Add Chunk"))
        else:
            applied = self.do(added)
        if not applied:
            # do() declined — an abstracted room — and nothing was built:
            # there is no chunk to select and no id to hand out.
            return None
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

    def set_creature_colour(self, rgb: tuple[float, float, float]) -> None:
        """Paint every creature the selection touches, as one undo step.

        The colour is written to *every* chunk of each selected chunk's
        connected component, not a representative: a creature has no object
        of its own, and a colour on one chunk would vanish the moment that
        chunk was deleted or the creature split. Written to meta under
        :data:`~mesh_xml.render.sdf.COLOUR_META`, which now round-trips
        through v3 saves, so a painted herd stays painted.
        """
        if not self.selection:
            self.status.emit("Select a chunk of the creature to colour first")
            return
        value = " ".join(f"{min(max(float(v), 0.0), 1.0):.4f}" for v in rgb)

        chunks: set[int] = set()
        for component in self._selected_creatures():
            chunks |= component

        commands: list[Command] = [
            SetMeta(cid, sdf.COLOUR_META, value) for cid in sorted(chunks)
        ]
        if len(commands) == 1:
            self.do(commands[0])
        else:
            self.do(Macro(commands, text="Creature Colour"))
        self.status.emit(f"Coloured {len(chunks)} chunk(s)")

    def creature_pattern(self) -> tuple[str, float]:
        """The first selected creature's pattern — what the panel shows."""
        creatures = self._selected_creatures()
        if not creatures:
            return "none", 0.0
        return sdf.pattern_of(self.body, sorted(creatures[0]))

    def set_creature_pattern(self, name: str, param: float | None = None) -> None:
        """Pattern every creature the selection touches, as one undo step.

        The tempo mechanism for the skin (``docs/TEXTURES.md`` package 3):
        the pattern name and its parameter ride every chunk of each
        component as meta, so they survive any one chunk's deletion and
        every format era that carries meta. A name outside the vocabulary —
        including the honest ``"none"`` — clears both keys and the skin
        returns to plain colours. ``param=None`` takes the pattern's
        default.
        """
        if not self.selection:
            self.status.emit("Select a chunk of the creature to pattern first")
            return
        name = name.strip().lower()
        known = name in sdf.PATTERN_IDS
        pattern_value = name if known else None
        param_value = None
        if known:
            chosen = param if param is not None else sdf.PATTERN_DEFAULTS[name]
            if not np.isfinite(float(chosen)):
                self.status.emit("Pattern parameter must be a finite number")
                return
            param_value = repr(float(chosen))

        chunks: set[int] = set()
        for component in self._selected_creatures():
            chunks |= component
        commands: list[Command] = []
        for cid in sorted(chunks):
            commands.append(
                SetMeta(cid, sdf.PATTERN_META, pattern_value, text="Creature Pattern")
            )
            commands.append(
                SetMeta(
                    cid, sdf.PATTERN_PARAM_META, param_value, text="Creature Pattern"
                )
            )
        self.do(Macro(commands, text="Creature Pattern"))
        self.status.emit(
            f"Pattern {name!r} on {len(chunks)} chunk(s)"
            if known
            else f"Pattern cleared on {len(chunks)} chunk(s)"
        )

    def set_chunk_colour(self, rgb: tuple[float, float, float]) -> None:
        """Paint exactly the selected chunks, as one undo step.

        The per-part gesture (``docs/TEXTURES.md`` package 2), beside
        :meth:`set_creature_colour` and deliberately narrower: a statement
        about the chunks in hand, not their component. The weighted skin is
        what makes the difference visible — unpainted neighbours keep their
        palette colour, and the surface grades between them at the blend.
        Same meta key, so the two gestures overwrite each other cleanly and
        both round-trip through every format era that carries meta.
        """
        if not self.selection:
            self.status.emit("Select the chunks to paint first")
            return
        value = " ".join(f"{min(max(float(v), 0.0), 1.0):.4f}" for v in rgb)
        commands: list[Command] = [
            SetMeta(cid, sdf.COLOUR_META, value, text="Paint Chunks")
            for cid in sorted(self.selection)
        ]
        if len(commands) == 1:
            self.do(commands[0])
        else:
            self.do(Macro(commands, text="Paint Chunks"))
        self.status.emit(f"Painted {len(commands)} chunk(s)")

    def _selected_creatures(self) -> list[set[int]]:
        """The connected component of each selected chunk, deduplicated.

        A creature has no object of its own — per-creature edits (colour,
        tempo) resolve the selection to whole components and act on those.
        Sorted seeds keep the component order stable for command building.
        """
        components: list[set[int]] = []
        seen: set[int] = set()
        for selected in sorted(self.selection):
            if selected in seen:
                continue
            component: set[int] = set()
            frontier = [selected]
            while frontier:
                current = frontier.pop()
                if current in component:
                    continue
                component.add(current)
                frontier.extend(self.body.neighbors(current))
            seen |= component
            components.append(component)
        return components

    def _tempo_of(self, chunks: set[int]) -> float:
        """The authored tempo riding a creature's chunks, 1.0 when unset.

        Read from the first chunk carrying a parseable value rather than the
        selection itself, because a chunk grown onto the creature after the
        tempo was set inherits no meta. Garbage falls back rather than
        breaking the edit, the same stance the colour reader takes.
        """
        for cid in sorted(chunks):
            raw = self.body.meta_of(cid).get(tempo.TEMPO_META)
            if raw is None:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if np.isfinite(value) and value > 0.0:
                return value
        return 1.0

    def creature_tempo(self) -> float:
        """The tempo of the first selected creature — what the panel shows."""
        creatures = self._selected_creatures()
        return self._tempo_of(creatures[0]) if creatures else 1.0

    def set_creature_tempo(self, value: float) -> None:
        """Retime every creature the selection touches, as one undo step.

        Two writes travel together: the authored value onto every chunk of
        each component (meta, like colour — so it reopens with the file), and
        the *ratio* from each creature's current tempo onto every tempo field
        of the controllers that reference its chunks
        (:func:`mesh_xml.anim.tempo.retime`). The ratio is per creature, so
        selecting a herd with mixed tempos lands them all at ``value`` rather
        than smearing one creature's ratio across the rest.
        """
        if not self.selection:
            self.status.emit("Select a chunk of the creature to retime first")
            return
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            self.status.emit("Tempo must be a positive number")
            return

        commands: list[Command] = []
        touched = 0
        for component in self._selected_creatures():
            current = self._tempo_of(component)
            factor = value / current
            if abs(factor - 1.0) < 1e-9:
                continue
            touched += len(component)
            commands += [
                SetMeta(cid, tempo.TEMPO_META, repr(value), text="Creature Tempo")
                for cid in sorted(component)
            ]
            for index, controller in enumerate(self.controllers.controllers):
                if not (tempo.chunk_refs(controller) & component):
                    continue
                changes = tempo.retime(controller, factor)
                commands += [
                    SetControllerField(index, field, new, text="Creature Tempo")
                    for field, new in sorted(changes.items())
                ]
        if not commands:
            return
        if len(commands) == 1:
            self.do(commands[0])
        else:
            self.do(Macro(commands, text="Creature Tempo"))
        self.status.emit(f"Tempo ×{value:g} on {touched} chunk(s)")

    def creatures(self) -> list[Creature]:
        """The roster: every connected component, named from its chunks.

        Component order is the renderer's blend-group order (first-seen
        slot), so row *n* here is the creature wearing group *n*'s surface.
        """
        roster: list[Creature] = []
        for component in self.body.components():
            name = None
            for cid in sorted(component):
                raw = self.body.meta_of(cid).get(CREATURE_META)
                if raw:
                    name = raw
                    break
            roster.append(Creature(chunks=tuple(sorted(component)), name=name))
        return roster

    def rename_creature(self, name: str) -> None:
        """Name every creature the selection touches, as one undo step.

        The name is painted across each component the way colour is, so it
        survives any one chunk's deletion; an empty name clears it, and the
        roster falls back to its honest generated label.
        """
        if not self.selection:
            self.status.emit("Select a chunk of the creature to name first")
            return
        value = name.strip() or None

        chunks: set[int] = set()
        for component in self._selected_creatures():
            chunks |= component
        commands: list[Command] = [
            SetMeta(cid, CREATURE_META, value, text="Rename Creature")
            for cid in sorted(chunks)
        ]
        if len(commands) == 1:
            self.do(commands[0])
        else:
            self.do(Macro(commands, text="Rename Creature"))
        self.status.emit(
            f"Named {len(chunks)} chunk(s) {value!r}" if value else "Name cleared"
        )

    def duplicate_creature(self) -> None:
        """Copy every creature the selection touches — flesh, gait and all.

        The herd gesture. Each copy is whole: chunks with their masses, pins
        and meta (colour, tempo, name), links with their rests, types and
        render flags, and the creature's own controllers rebuilt with ids
        remapped (:func:`mesh_xml.anim.serialize.remap_ids`) and their world
        points carried sideways (:func:`mesh_xml.anim.place.translate`) — a
        copy left pointing at its ancestor's ``origin`` would fight it for
        one soul. Placement is the Add rule one creature up: past the body's
        +x extent, clear of everything, copies of a multi-selection laid out
        in a row. One undo step; the copies take the selection, so
        duplicating again herds.
        """
        if not self.selection:
            self.status.emit("Select a chunk of the creature to duplicate first")
            return
        body = self.body
        everything = [body.chunk(cid) for cid in body.chunk_ids()]
        frontier = max(c.pos[0] + c.radius for c in everything)
        next_id = max(body.chunk_ids(), default=-1) + 1

        commands: list[Command] = []
        copies: list[int] = []
        taken: set[int] = set()
        components = self._selected_creatures()
        for component in components:
            chunks = {cid: body.chunk(cid) for cid in sorted(component)}
            spacing = _GROW_SPACING * max(c.radius for c in chunks.values())
            left = min(c.pos[0] - c.radius for c in chunks.values())
            right = max(c.pos[0] + c.radius for c in chunks.values())
            dx = frontier + spacing - left
            frontier = right + dx
            offset = np.zeros(body.dim)
            offset[0] = dx

            mapping: dict[int, int] = {}
            for cid in chunks:
                mapping[cid] = next_id
                next_id += 1
            for cid, chunk in chunks.items():
                commands.append(
                    AddChunk(
                        chunk.name,
                        tuple(float(p) + float(o) for p, o in zip(chunk.pos, offset)),
                        radius=chunk.radius,
                        mass=1.0 if chunk.pinned else chunk.mass,
                        pinned=chunk.pinned,
                        meta=dict(chunk.meta),
                        chunk_id=mapping[cid],
                        text="Duplicate Creature",
                    )
                )
            for link in body.links():
                if link.a in mapping and link.b in mapping:
                    commands.append(
                        AddLink(
                            mapping[link.a],
                            mapping[link.b],
                            rest=link.rest,
                            type=link.type,
                            elasticity=link.elasticity,
                            render=link.render,
                            text="Duplicate Creature",
                        )
                    )
            for index, controller in enumerate(self.controllers.controllers):
                if index in taken or not (tempo.chunk_refs(controller) & component):
                    continue
                taken.add(index)
                try:
                    copy = serialize.remap_ids(controller, mapping)
                except serialize.ControllerFormatError:
                    # An unregistered controller cannot be rebuilt; the
                    # original keeps running, the copy goes without.
                    continue
                moved = place.translate(copy, offset)
                if moved:
                    copy = replace(copy, **moved)
                commands.append(AddController(copy, text="Duplicate Creature"))
            copies.extend(mapping.values())

        self.do(Macro(commands, text="Duplicate Creature"))
        self.set_selection(copies)
        self.status.emit(f"Duplicated {len(components)} creature(s)")

    def export_mesh(self, path, resolution: int = 96) -> Path:
        """Polygonise the surface as drawn and write it beside the document.

        The mesh honours the blend groups — smooth flesh within a creature, a
        hard skin between them — so an exported herd arrives in Blender as
        the separate animals the viewport shows, not as one weld. Positions
        are the draw positions, matching what the viewer sees. A painted or
        patterned skin rides as OBJ vertex colours, evaluated by the same
        albedo fold the viewport shades by (``docs/TEXTURES.md`` package 4);
        an unpainted body exports the file it always exported, byte for
        byte, because its attribute table is empty.
        """
        from mesh_xml.io.mesh_export import write_mesh
        from mesh_xml.render import field
        from mesh_xml.render.meshing import polygonise

        primitives = self.primitives()
        groups = sdf.group_table(self.body)[:, :2]
        mesh = polygonise(
            primitives,
            blend=self.metaball.blend,
            resolution=resolution,
            groups=groups,
        )
        attributes = self.attribute_rows()
        colours = None
        if len(attributes):
            colours = field.albedo(
                mesh.vertices,
                primitives,
                attributes,
                self.metaball.blend,
                groups=groups,
            )
        out = write_mesh(
            Path(path), mesh.vertices, mesh.faces, mesh.normals, colours=colours
        )
        note = f"Exported {len(mesh.vertices)} vertices to {out.name}"
        if colours is not None:
            note += " (painted)"
        if mesh.open_edges:
            note += f" ({mesh.open_edges} open edge(s) — raise the resolution)"
        self.status.emit(note)
        return out

    # ------------------------------------------------------------------
    # files
    # ------------------------------------------------------------------

    @property
    def dirty(self) -> bool:
        """Whether the *active room* has unsaved changes. A room is a file."""
        return self._state.dirty

    def set_dirty(self, dirty: bool) -> None:
        if dirty != self._state.dirty:
            self._state.dirty = dirty
            self.dirty_changed.emit(dirty)

    @property
    def any_dirty(self) -> bool:
        """Whether any room in the scene has unsaved changes — the close prompt's question."""
        return any(state.dirty for state in self._states.values())

    def new(self) -> None:
        """A fresh rig on today's defaults — :meth:`adopt`'s null case, exactly.

        Routed through adopt so the solver and blend reset with everything
        else: New after a tuned file must not carry that file's physics into
        a rig that never stated it (docs/SETTINGS.md §5).
        """
        self.adopt(Body())
        self.status.emit("New rig")

    def open(self, path) -> None:
        """Load a rig and its gait into the *active room*, replacing its contents."""
        path = Path(path)
        body, controllers, terrain, doors, solver, blend, note = _read_rig(path)
        self.adopt(
            body,
            controllers=controllers,
            path=path,
            terrain=terrain,
            doors=doors,
            solver=solver,
            blend=blend,
        )
        self._adopt_room_name(path)
        self.status.emit(note)

    def _adopt_room_name(self, path: Path) -> None:
        """The active room takes the opened file's name.

        Names are the scene's addressing, and doors in sibling files were
        authored against them: a den whose file says "den" must answer to
        it, or every door to the den is closed on a technicality. The
        file's own body name wins, the stem stands in for a default-named
        body, and a collision with a sibling room suffixes — same ladder as
        :meth:`open_as_room`. Not :meth:`rename_room`: a load is not an
        edit, so no dirty flag and no stale-door warning.
        """
        body = self.room.body
        base = body.name if body.name and body.name != "body" else path.stem
        if base == self.room.name:
            return
        taken = {room.name for room in self.scene.rooms if room is not self.room}
        name, suffix = base, 2
        while name in taken:
            name = f"{base} {suffix}"
            suffix += 1
        self.room.name = name
        body.name = name
        self.scene_changed.emit()

    def open_as_room(self, path) -> Room:
        """Open a file as a new room of the scene, and point the editor at it.

        The multi-room counterpart of :meth:`open`, which replaces the
        active room. The room takes the file's own body name — names are
        the scene's addressing, and doors in other files were authored
        against them — falling back to the file stem for a default-named
        body, and suffixing on collision rather than refusing: two copies
        of one file are a legitimate scene, and the second answers to
        "meadow 2". The suffix writes back to the body so room and body
        stay one value.
        """
        path = Path(path)
        body, controllers, terrain, doors, solver, blend, note = _read_rig(path)
        base = body.name if body.name and body.name != "body" else path.stem
        name = self.unique_room_name(base)

        room = Room(name=name)
        room.body = body
        body.name = name
        room.controllers = (
            controllers if controllers is not None else ControllerStack(body)
        )
        # A fresh Room is already founded; only a file that states its
        # physics overrides that, so the opinion stays spelled once, in Room.
        if solver is not None:
            room.solver_config = solver
        room.solver_config.terrain = terrain
        room.doors = list(doors)
        self.add_room(room)
        self._states[room].path = path
        if blend is not None:
            self._states[room].metaball.blend = blend
        self.set_active_room(room)
        if name != base:
            note += f" as room {name!r}"
        self.status.emit(note)
        return room

    def adopt(
        self,
        body: Body,
        controllers: ControllerStack | None = None,
        path=None,
        terrain: Terrain | None = None,
        doors: Sequence[Door] = (),
        solver: SolverConfig | None = None,
        blend: float | None = None,
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
        ``doors`` defaults to none on the same terms — an exit into the old
        file's world is a promise the new one never made. ``solver=None`` and
        ``blend=None`` reset on the same terms again, to the room's founding
        floor-at-zero config and the dataclass blend: the file is the truth
        about both records, and the previous session's tuning under a fresh
        rig would be a physics it never stated.
        """
        self._replace(body, Path(path) if path else None)
        if controllers is not None:
            self.controllers = controllers
            self.controllers_changed.emit()
        config = solver if solver is not None else Room.founding_config()
        # The incoming config carries the *old* world across the swap, so
        # :meth:`set_terrain` still judges — and announces — the world change
        # itself, including a terrain cleared by a file that has none.
        config.terrain = self.solver_config.terrain
        self.solver_config = config
        self.set_terrain(terrain)
        self.room.doors = list(doors)
        self._state.metaball.blend = (
            blend if blend is not None else MetaballSettings().blend
        )
        # _replace stops short of settings on SCENE_BINDING.md §2.1's terms —
        # open used to reset them, and a reset needed no announcement. Solver
        # and blend restore from the file, so the panels must re-read.
        self.settings_changed.emit()

    def save(self, path) -> None:
        """Write the active room to ``path``: body, gait, world, and doors.

        Passing the controllers is the whole point of the v3 format: without it
        a save keeps the skeleton and silently discards everything that made it
        move. The terrain rides along on the same reasoning — a creature saved
        without its world reloads standing on nothing — and the doors on the
        same again: a room saved without its exits reloads as a dead end.
        """
        self.save_room(self.room, path)

    def open_scene(self, path) -> None:
        """Replace the whole scene with a manifest's assembly.

        Every room file is read *before* the live scene is touched, and
        the new scene is built complete off to the side — so a missing
        file, a duplicate name, or a broken rig leaves the current scene
        standing, and the error names the room rather than the line
        number. Refusal is wholesale, the transfer module's stance. The
        manifest's names win over the files' own body names: the manifest
        is the addressing the doors compiled against.
        """
        path = Path(path)
        scene_name, refs = scene_xml.read_scene(path)
        if not refs:
            raise ValueError(f"{path.name} lists no rooms")

        scene = Scene(name=scene_name)
        states: dict[Room, _RoomState] = {}
        for ref in refs:
            src = path.parent / ref.src
            try:
                body, controllers, terrain, doors, solver, blend, _note = (
                    _read_rig(src)
                )
            except Exception as exc:
                raise ValueError(
                    f"room {ref.name!r}: {src.name}: {exc}"
                ) from exc
            room = Room(name=ref.name)
            room.body = body
            body.name = ref.name
            room.controllers = (
                controllers if controllers is not None else ControllerStack(body)
            )
            if solver is not None:
                room.solver_config = solver
            room.solver_config.terrain = terrain
            room.doors = list(doors)
            scene.add_room(room)
            metaball = MetaballSettings()
            if blend is not None:
                metaball.blend = blend
            states[room] = _RoomState(
                draw=DrawPoints(body),
                metaball=metaball,
                rest_pose={
                    cid: body.chunk(cid).pos for cid in body.chunk_ids()
                },
                path=src,
            )

        self.scene = scene
        self._states = states
        self._room = scene.rooms[0]
        self.body_replaced.emit()
        self.topology_changed.emit()
        self.selection_changed.emit()
        self.controllers_changed.emit()
        self.history_changed.emit()
        self.settings_changed.emit()
        self.terrain_changed.emit()
        self.dirty_changed.emit(False)
        self.scene_changed.emit()
        self.status.emit(f"Opened scene {path.name} ({len(refs)} room(s))")

    def save_scene_as(self, path) -> None:
        """Write the manifest, and every room that owes the disk something.

        A room that never had a file gets one beside the manifest, named
        for itself; dirty rooms save to their own files; clean rooms with
        files are only *listed* — the manifest records assembly, it does
        not rewrite what already agrees with disk. The scene takes the
        manifest's stem as its name, the same adoption a room performs at
        open.
        """
        path = Path(path)
        self.scene.name = path.stem
        rooms: list[tuple[str, Path]] = []
        for room in self.scene.rooms:
            state = self._states[room]
            target = state.path
            if target is None:
                target = path.parent / f"{room.name}.xml"
                self.save_room(room, target)
            elif state.dirty:
                self.save_room(room, target)
            rooms.append((room.name, target))
        scene_xml.save_scene(path, self.scene.name, rooms)
        self.status.emit(f"Saved scene {path.name} ({len(rooms)} room(s))")

    def save_room(self, room: Room, path=None) -> Path:
        """Write any room of the scene to its file, active or not.

        What Save All walks. ``path=None`` means the room's own remembered
        file; a room that never had one is refused by name rather than
        guessed at — that is Save As's question to ask.
        """
        state = self._states[room]
        target = Path(path) if path is not None else state.path
        if target is None:
            raise ValueError(f"room {room.name!r} has no file; use Save As")
        xml_v2.save(
            room.body,
            target,
            room.controllers,
            terrain=room.solver_config.terrain,
            doors=tuple(room.doors),
            solver=room.solver_config,
            blend=state.metaball.blend,
        )
        state.path = target
        if state.dirty:
            state.dirty = False
            if room is self._room:
                self.dirty_changed.emit(False)
        self.scene_changed.emit()
        self.status.emit(f"Saved {target.name}")
        return target

    def _replace(self, body: Body, path: Path | None) -> None:
        state = self._state
        self.body = body
        state.path = path
        state.draw = DrawPoints(body)
        # Controllers hold chunk ids, and ids from the previous rig mean nothing
        # in this one; carrying them over would silently drive the wrong chunks.
        self.controllers = ControllerStack(body)
        self.clock.reset()
        state.selection.clear()
        state.undo.clear()
        state.redo.clear()
        # The active room's pose only: opening a file into this room says
        # nothing about where the others should return to.
        state.rest_pose = {cid: body.chunk(cid).pos for cid in body.chunk_ids()}
        self.set_dirty(False)
        self.body_replaced.emit()
        self.topology_changed.emit()
        self.selection_changed.emit()
        self.controllers_changed.emit()
        self.history_changed.emit()

    @property
    def title(self) -> str:
        name = self.path.name if self.path else "Untitled"
        return f"{'*' if self.dirty else ''}{name} — mesh_xml"


def _read_rig(
    path: Path,
) -> tuple[
    Body,
    ControllerStack | None,
    Terrain | None,
    Sequence[Door],
    SolverConfig | None,
    float | None,
    str,
]:
    """Read any supported rig file, v3 down to the legacy eras.

    The one reader both :meth:`Document.open` and :meth:`Document.open_as_room`
    share, so the fallback ladder cannot fork between them. Returns the body,
    the gait (``None`` for files that cannot describe one), the world, the
    doors, the solver tuning and the blend (``None`` each for eras that could
    not describe them), and the status line the read earned.
    """
    try:
        rig = xml_v2.read(path)
        note = f"Opened {path.name}"
        driven = len(rig.controllers.controllers)
        if driven:
            note += f" ({driven} controller(s))"
        return (
            rig.body,
            rig.controllers,
            rig.terrain,
            rig.doors,
            rig.solver,
            rig.blend,
            note,
        )
    except Exception:
        # A scene manifest is not an older rig era; falling through to the
        # legacy reader would bury the one error message that says what to
        # do instead. A versioned <body> is not one either: it failed as the
        # format it names, and the refusal that says how — a solver arity, a
        # NaN damping — is the message worth surfacing.
        if scene_xml.is_manifest(path) or _is_versioned_body(path):
            raise
        result = xml_legacy.read_legacy(path)
        note = f"Opened {path.name} (legacy, {result.coords} coordinates)"
        if result.missing_mesh:
            note += f"; {len(result.missing_mesh)} bone(s) had no mesh"
        return result.body, None, None, (), None, None, note


def _is_versioned_body(path: Path) -> bool:
    """Whether the file's root tag says ``<body>`` — ``is_manifest``'s sniff.

    Decides only "did the v3/v2 reader already own this file"; an
    unparseable file is not a versioned body, and the legacy ladder that
    then runs produces the real error.
    """
    try:
        for _event, element in ET.iterparse(path, events=("start",)):
            return element.tag == "body"
    except (OSError, ET.ParseError):
        return False
    return False
