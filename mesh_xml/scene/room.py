"""The room: the unit of simulation, with no editor in sight.

Rain World's world is made of rooms — each one a space, a population, and a
simulation that runs whether or not a camera is pointed at it. This module is
that noun for this codebase, extracted from the editor's ``Document`` once
the scene tier needed a plural (``docs/ROOMS.md`` package 2): a body, the
stack that drives it, the solver settings that are its physics, and the
clock that ticks it. Nothing here imports Qt, and a test proves it stays
that way — a game loop owns rooms directly, and the editor's ``Document`` is
exactly one room wearing undo, signals, and a file path.

The one opinion a room holds is that there is a ground: ``SolverConfig``
alone defaults to no floor, which for a *place* would mean everything in it
falls out of frame forever. The library keeps that neutrality; a room is
where a world begins, so it starts with a floor at zero and terrain replaces
it the moment one is authored.
"""

from __future__ import annotations

from mesh_xml.anim.clock import Clock
from mesh_xml.anim.controller import ControllerStack
from mesh_xml.core.body import Body
from mesh_xml.core.solver import SolverConfig

__all__ = ["Room"]


class Room:
    """A space, its population, and the simulation that runs it."""

    def __init__(self, name: str = "room") -> None:
        self.name = name
        self.body = Body()
        self.controllers = ControllerStack(self.body)
        self.solver_config = SolverConfig(ground_y=0.0)
        self.clock = Clock()

    def tick(self) -> None:
        """One fixed step: controllers, integration, projection, collision.

        The single place that knows what a room's tick consists of — the
        same sentence :meth:`Document.tick` used to carry, moved down to
        where a game loop can reach it without a window.
        """
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
