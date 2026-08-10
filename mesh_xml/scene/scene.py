"""The scene: rooms in the plural, one clock between them.

``docs/ROOMS.md`` takes this decision in advance and this module makes it
literal: adding a room to a scene *aliases* the scene's own clock into it,
so "one clock per scene" is not a convention that two accumulators could
quietly drift away from — there is no second accumulator. Rooms then tick
from the scene's :meth:`advance`, and cross-room time cannot skew because
there is exactly one supply of ticks to skew against. A room outside any
scene keeps the private clock it was born with and works alone, which is
what the package-2 smoke tests already drive; a room inside a scene is
driven by the scene, and calling ``Room.advance`` on it directly would
spend the shared accumulator behind the scene's back.

What a scene deliberately is not, yet: a file. The v3 format stays a
per-body document; doors and their persistence are package 5's contract.
"""

from __future__ import annotations

import numpy as np

from mesh_xml.anim.clock import Clock
from mesh_xml.scene.room import Room
from mesh_xml.scene.transfer import move_creature

__all__ = ["Scene"]


class Scene:
    """An ordered set of rooms sharing one simulation clock."""

    def __init__(self, name: str = "scene") -> None:
        self.name = name
        self.clock = Clock()
        self.rooms: list[Room] = []

    def add_room(self, room: Room) -> Room:
        """Adopt a room: its clock becomes the scene's, its name must be new.

        Names are the scene's addressing scheme — a door will say "to the
        meadow", not "to room index 2" — so a duplicate is refused at the
        door rather than discovered by one.
        """
        if any(existing.name == room.name for existing in self.rooms):
            raise ValueError(f"the scene already has a room named {room.name!r}")
        room.clock = self.clock
        self.rooms.append(room)
        return room

    def remove_room(self, room: Room) -> Room:
        """Release a room: it leaves the roster and owns its time again.

        The clock un-aliases — a lone room ticking the scene's accumulator
        from outside would spend ticks behind the scene's back, the exact
        hazard :meth:`add_room` wired away. Doors elsewhere that name the
        departed room become closed doors, the scene's standing answer to a
        promise that cannot be kept: nothing raises at crossing time, the
        traveller waits.
        """
        if room not in self.rooms:
            raise ValueError(f"no room {room.name!r} in this scene")
        self.rooms.remove(room)
        room.clock = Clock()
        return room

    def room(self, name: str) -> Room:
        for room in self.rooms:
            if room.name == name:
                return room
        known = ", ".join(sorted(room.name for room in self.rooms)) or "none"
        raise KeyError(f"no room named {name!r}; rooms are {known}")

    def activate(self, name: str) -> Room:
        """The camera enters ``name``: Rain World's realization boundary.

        The named room realizes if it was abstracted; every other room not
        already at the cheap tier drops to it. Nothing stops simulating
        anywhere — :meth:`tick` still ticks every room, and the abstracted
        ones keep thinking — activation only moves the one expensive tier
        to where the viewport is pointed.
        """
        active = self.room(name)
        for room in self.rooms:
            if room is active:
                if room.abstracted:
                    room.realize()
            elif not room.abstracted:
                room.abstract()
        return active

    def tick(self) -> None:
        """One fixed step everywhere, then the crossings the step earned.

        Every room ticks, looked at or not; afterwards the scene carries
        every creature that reached its named exit. Rooms first, doors
        second, so a crossing sees the world the tick actually produced.
        """
        for room in self.rooms:
            room.tick()
        self._cross()

    def _cross(self) -> None:
        """Carry every creature standing in a door it asked to leave by.

        The brain's half is walking the anchor to the door (``exit`` on the
        brain, the door's point through ``ctx.doors``); this is the scene's
        half — the only place a creature changes rooms. Arrival is the
        anchor inside the door's radius; the crossing is
        :func:`move_creature` with the paired door's offset, so a creature
        steps out of the far door having kept its stride. The goal is spent
        by crossing: ``exit`` clears on arrival, because one door is one
        journey — chained travel is a brain re-deciding, not a scene
        assuming.

        Doors keep their promises only as far as they can: an address to a
        room or door that does not exist, an abstracted room on either
        side, is a closed door — the creature waits in the doorway, and
        nothing raises mid-tick. Abstract-space migration is the transfer
        module's named future work, not a silent behaviour here.
        """
        crossings: list[tuple[Room, Room, int, np.ndarray]] = []
        for room in self.rooms:
            if room.abstracted or not room.doors:
                continue
            doors = {door.name: door for door in room.doors}
            for brain in room.controllers.controllers:
                wanted = getattr(brain, "exit", "")
                door = doors.get(wanted)
                anchor = getattr(brain, "anchor", None)
                if door is None or anchor is None or not room.body.has_chunk(anchor):
                    continue
                pos = np.asarray(room.body.chunk(anchor).pos, dtype=np.float64)
                at = np.asarray(door.at, dtype=np.float64)[: pos.size]
                if float(np.linalg.norm(pos - at)) > door.radius:
                    continue
                try:
                    target = self.room(door.to_room)
                except KeyError:
                    continue
                if target is room or target.abstracted:
                    continue
                arrival = next(
                    (far for far in target.doors if far.name == door.to_door), None
                )
                if arrival is None:
                    continue
                offset = np.asarray(arrival.at, dtype=np.float64)[: pos.size] - at
                crossings.append((room, target, int(anchor), offset))

        for room, target, anchor, offset in crossings:
            if not room.body.has_chunk(anchor):
                continue  # a second brain of a creature that already crossed
            mapping = move_creature(room, target, anchor, offset=tuple(offset))
            arrived = set(mapping.values())
            for controller in target.controllers.controllers:
                if getattr(controller, "anchor", None) in arrived and getattr(
                    controller, "exit", ""
                ):
                    controller.exit = ""

    def advance(self, elapsed: float) -> float:
        """Bank wall time and spend it on whole ticks across every room."""
        for _ in self.clock.advance(elapsed):
            self.tick()
        return self.clock.alpha
