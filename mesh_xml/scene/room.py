"""The room: the unit of simulation, with no editor in sight.

Rain World's world is made of rooms — each one a space, a population, and a
simulation that runs whether or not a camera is pointed at it. This module is
that noun for this codebase, extracted from the editor's ``Document`` once
the scene tier needed a plural: a body, the
stack that drives it, the solver settings that are its physics, and the
clock that ticks it. Nothing here imports Qt, and a test proves it stays
that way — a game loop owns rooms directly, and the editor's ``Document`` is
exactly one room wearing undo, signals, and a file path.

The one opinion a room holds is that there is a ground: ``SolverConfig``
alone defaults to no floor, which for a *place* would mean everything in it
falls out of frame forever. The library keeps that neutrality; a room is
where a world begins, so it starts with a floor at zero and terrain replaces
it the moment one is authored.

Abstraction is the same room at its cheap
tier, and the argument was sitting in the brain contract all along: a brain
steers only its own pinned anchor, so a creature reduced to that anchor is
one its brain cannot tell from the whole. :meth:`Room.abstract` keeps one
pinned chunk per creature and parks the rest — chunks, links, the gait
controllers — as data on the room, while the brains stay on the stack at
full authority, thinking at their own cadence and walking their anchors
through the cheap room exactly as they would through the expensive one. The
tick keeps its shape and only its cost changes: the same controller pass,
the same solver, over one chunk per creature and no links at all.
:meth:`Room.realize` rebuilds each creature from its parked form around
wherever the brain has since walked the anchor — Rain World's
``AbstractCreature`` arriving at the camera with a body owed to it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from mesh_xml.anim import place, serialize, tempo
from mesh_xml.anim.clock import Clock
from mesh_xml.anim.controller import Controller, ControllerStack
from mesh_xml.core.body import Body, Chunk, Link
from mesh_xml.core.solver import SolverConfig

__all__ = ["Room"]


@dataclass(frozen=True)
class _ParkedCreature:
    """One creature as data: everything :meth:`Room.abstract` took away.

    ``controllers`` keeps each parked controller with the stack index it
    held, because stack order is rng draw order and has to survive the round
    trip. The anchor's pre-park pin state and park-time position are what
    :meth:`Room.realize` needs to hand the pin back and to measure how far
    the brain walked in the meantime.
    """

    anchor: int
    anchor_was_pinned: bool
    anchor_pos: tuple[float, ...]
    chunks: tuple[Chunk, ...]
    links: tuple[Link, ...]
    controllers: tuple[tuple[int, Controller], ...]


def _anchor_of(members: set[int], stack: list[Controller]) -> int:
    """The one chunk a parked creature keeps.

    A brain's ``anchor`` wins — ``think_period`` is the brain contract, the
    same duck-typing :mod:`mesh_xml.anim.behavior` defines the tier by —
    because that is the chunk the brain will keep steering while the rest is
    data. A creature nothing thinks for keeps its minimum id: arbitrary, but
    stable across the round trip, which is all an anchor owes anyone.
    """
    for controller in stack:
        if not hasattr(controller, "think_period"):
            continue
        anchor = getattr(controller, "anchor", None)
        if anchor in members:
            return int(anchor)
    return min(members)


class Room:
    """A space, its population, and the simulation that runs it."""

    @staticmethod
    def founding_config() -> SolverConfig:
        """The room's founding opinion as a value: a floor at zero.

        Spelled once so the editor's open paths can restate it when a file
        carries no physics — several drifting copies of one opinion is the
        "rarely-exercised spelling rots first" hazard one layer up.
        """
        return SolverConfig(ground_y=0.0)

    def __init__(self, name: str = "room") -> None:
        self.name = name
        self.body = Body()
        self.controllers = ControllerStack(self.body)
        self.solver_config = self.founding_config()
        self.clock = Clock()
        #: The room's exits (:class:`~mesh_xml.scene.door.Door`), authored
        #: like terrain is authored: data the room owns and the tick
        #: republishes to the stack, so a brain's ``exit`` goal can find
        #: where its named door stands this tick, however the list was
        #: edited since the last one.
        self.doors: list = []
        self._abstracted = False
        self._parked: list[_ParkedCreature] = []

    def tick(self) -> None:
        """One fixed step: controllers, integration, projection, collision.

        The single place that knows what a room's tick consists of — the
        same sentence :meth:`Document.tick` used to carry, moved down to
        where a game loop can reach it without a window.
        """
        self.controllers.doors = {door.name: door.at for door in self.doors}
        self.controllers.tick(self.body, self.clock.dt, self.solver_config)

    def advance(self, elapsed: float) -> float:
        """Run ``elapsed`` seconds of wall time; return the render blend.

        The fixed-timestep accumulator loop, for callers that own nothing
        per-tick: a headless game loop, an abstracted room. The editor keeps
        its own loop because its draw points ease once per tick, not once
        per frame.
        """
        for _ in self.clock.advance(elapsed):
            self.tick()
        return self.clock.alpha

    @property
    def abstracted(self) -> bool:
        """Whether the room is at its cheap tier.

        An explicit flag rather than "is anything parked", because an empty
        room abstracts to an empty parked list and is abstracted all the
        same — the flag, not the list, is the witness.
        """
        return self._abstracted

    def abstract(self) -> None:
        """Drop to anchors-only: one pinned chunk per creature, the rest parked.

        Each connected component keeps exactly one chunk — its brain's
        anchor, or the minimum id where nothing thinks — pinned, so gravity
        cannot drop it now that no links hold it up and so the brain keeps
        the same kinematic authority it always had. Chunks, links and the
        non-brain controllers park as data, each controller with the stack
        index it held. Brains stay on the stack and keep thinking and
        steering; that is the whole point.
        """
        if self._abstracted:
            raise ValueError(
                f"room {self.name!r} is already abstracted; a creature "
                "cannot be parked twice"
            )
        stack = list(self.controllers.controllers)
        claimed: set[int] = set()
        parked: list[_ParkedCreature] = []
        for component in self.body.components():
            members = set(component)
            anchor = self.body.chunk(_anchor_of(members, stack))
            held: list[tuple[int, Controller]] = []
            for index, controller in enumerate(stack):
                if id(controller) in claimed:
                    continue
                if hasattr(controller, "think_period"):
                    continue  # brains stay: an abstract creature still thinks
                if not (tempo.chunk_refs(controller) & members):
                    continue
                claimed.add(id(controller))
                held.append((index, controller))
            parked.append(
                _ParkedCreature(
                    anchor=anchor.id,
                    anchor_was_pinned=anchor.pinned,
                    anchor_pos=anchor.pos,
                    chunks=tuple(self.body.chunk(cid) for cid in component),
                    links=tuple(
                        link
                        for link in self.body.links()
                        if link.a in members and link.b in members
                    ),
                    controllers=tuple(held),
                )
            )

        for record in parked:
            for chunk in record.chunks:
                if chunk.id != record.anchor:
                    self.body.remove_chunk(chunk.id)
            self.body.set_pinned(record.anchor, True)
            for _, controller in record.controllers:
                self.controllers.remove(controller)
        self._parked = parked
        self._abstracted = True

    def realize(self) -> None:
        """Rebuild every parked creature around wherever its anchor now stands.

        The gap between the anchor's position now and at park time is how far
        the brain walked while the room was cheap; chunks, links and
        controllers come back displaced by exactly that. Parked controllers
        rebuild fresh through the serializer's identity remap — the door
        ruling applied to abstraction: a parked picker must not wake holding
        a foothold from a world that moved — and reinsert at their remembered
        stack indexes, so the rng draw order that stack order *is* survives
        the round trip.
        """
        if not self._abstracted:
            raise ValueError(
                f"room {self.name!r} is already realized; there is nothing "
                "parked to wake"
            )
        arrivals: list[tuple[int, Controller]] = []
        for record in self._parked:
            landed = np.asarray(
                self.body.chunk(record.anchor).pos, dtype=np.float64
            )
            offset = landed - np.asarray(record.anchor_pos, dtype=np.float64)
            for chunk in record.chunks:
                if chunk.id == record.anchor:
                    continue
                self.body.add_chunk(
                    chunk.name,
                    tuple(float(p) + float(o) for p, o in zip(chunk.pos, offset)),
                    radius=chunk.radius,
                    mass=1.0 if chunk.pinned else chunk.mass,
                    pinned=chunk.pinned,
                    id=chunk.id,
                    meta=dict(chunk.meta),
                )
            for link in record.links:
                self.body.add_link(
                    link.a,
                    link.b,
                    rest=link.rest,
                    type=link.type,
                    elasticity=link.elasticity,
                    render=link.render,
                )
            if not record.anchor_was_pinned:
                snapshot = next(
                    chunk for chunk in record.chunks if chunk.id == record.anchor
                )
                self.body.set_pinned(record.anchor, False, mass=snapshot.mass)
            for index, controller in record.controllers:
                rebuilt = serialize.remap_ids(controller, {})
                changes = place.translate(rebuilt, offset)
                if changes:
                    rebuilt = replace(rebuilt, **changes)
                arrivals.append((index, rebuilt))

        stack = self.controllers.controllers
        for index, controller in sorted(arrivals, key=lambda pair: pair[0]):
            stack.insert(min(index, len(stack)), controller)
        self._parked = []
        self._abstracted = False
