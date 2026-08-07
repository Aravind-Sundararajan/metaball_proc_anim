"""The door: a named place paired with a promise.

Rooms made the world plural and, in the same stroke, made it disconnected:
``ctx.peers`` stops at the room boundary by construction, and so does
everything else. A door is the one sanctioned way through — a named world
point in this room, and a promise about another: "east goes to the meadow,
and arrives at its west". Pairing is by name on both ends because names are
already the scene's addressing scheme (:meth:`~mesh_xml.scene.scene.Scene`
refuses duplicate room names at the door rather than letting one be
discovered by a crossing), so a door says "to the meadow", never "to room
index 2".

The door itself is inert data, and staying inert is its whole design. A
brain that wants to leave reads the door's point out of ``ctx.doors`` and
walks its own anchor there; the *scene* watches for arrival within
``radius`` and performs the crossing, because only the scene can see both
sides. That keeps the tier's law intact — brains write goals, controllers
write forces, and nobody reaches across a room — even for the one object
whose entire purpose is the other side.

``radius`` is the arrival threshold in world units, chosen at creature
scale: a bit over two default chunk radii, so a doorway is a place to stand
in rather than a point to hit, and a brain that stops at its own arrival
threshold (with its sway still ringing) is already inside.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Door"]


@dataclass(frozen=True)
class Door:
    """A named point in this room, and the name of where it leads.

    ``at`` is the world point brains walk to; ``to_room`` and ``to_door``
    are scene addressing — the destination room's name and the name of the
    door to arrive at inside it. Whether either name answers to anything is
    the scene's business at crossing time, not this record's: a door is
    authored before its far side exists, the same way a controller may
    honestly hold a chunk id the body no longer has.
    """

    name: str
    at: tuple[float, ...]
    to_room: str
    to_door: str
    radius: float = 0.9

    def __post_init__(self) -> None:
        # Stored as the plain tuple of floats the annotation promises,
        # whatever sequence authored it — frozen means hashable, and a list
        # slipped through here would quietly break that.
        object.__setattr__(self, "at", tuple(float(v) for v in self.at))
        if not self.name:
            raise ValueError(
                "a door needs a name; the scene addresses it by nothing else"
            )
        if not self.to_room:
            raise ValueError(f"door {self.name!r} needs a destination room")
        if not self.to_door:
            raise ValueError(f"door {self.name!r} needs a door to arrive at")
        if self.radius <= 0.0:
            raise ValueError(
                f"door {self.name!r}: radius must be positive, got {self.radius}"
            )
