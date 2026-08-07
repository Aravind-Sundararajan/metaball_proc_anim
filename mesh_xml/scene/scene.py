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

from mesh_xml.anim.clock import Clock
from mesh_xml.scene.room import Room

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

    def room(self, name: str) -> Room:
        for room in self.rooms:
            if room.name == name:
                return room
        known = ", ".join(sorted(room.name for room in self.rooms)) or "none"
        raise KeyError(f"no room named {name!r}; rooms are {known}")

    def tick(self) -> None:
        """One fixed step everywhere. Every room ticks, looked at or not."""
        for room in self.rooms:
            room.tick()

    def advance(self, elapsed: float) -> float:
        """Bank wall time and spend it on whole ticks across every room."""
        for _ in self.clock.advance(elapsed):
            self.tick()
        return self.clock.alpha
