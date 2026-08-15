"""The controller contract, and the one rule that keeps it stable.

**Controllers contribute forces. They never write positions.**

That is the whole architecture, and it is not a style preference. Constraint
projection runs immediately after integration, every tick, and it will happily
undo whatever a controller just placed. A controller that assigns positions is
therefore in a fight with the solver that it loses a few times per tick, and the
symptom is not an obvious failure — it is a rig that buzzes, or one whose limbs
work until a link tightens and then stop, or one that is stable at four solver
iterations and explodes at sixteen. Contributing force instead makes the solver
an ally: the controller says where it wants to go and the constraints decide
what the body can actually do about it.

Rain World arrives at the same place from the other direction. Its creature AI
sets *velocities* on body chunks and lets `BodyChunkConnection` relax afterward,
and its limbs live in the graphics module precisely so that reaching for a
foothold cannot destabilise the physics.

The tick order this implies, and which :class:`ControllerStack` enforces:

1. zero the force accumulator
2. run every controller, each adding force
3. integrate, consuming those forces
4. project constraints
5. collide with the world

Per-rig randomness is seeded from a stable id rather than the global generator,
so two creatures built from one template differ from each other and each one is
the same across runs. Reproducibility matters more than it sounds: an
intermittent instability that cannot be replayed cannot be debugged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from mesh_xml.core.body import Body
from mesh_xml.core.solver import SolverConfig, step

__all__ = ["ControlContext", "Controller", "ControllerStack"]


@dataclass
class ControlContext:
    """What a controller is given for one tick.

    ``forces`` is indexed by *slot*, matching :attr:`Body.pos`, not by chunk id.
    Use :meth:`slot` to convert; ids are what survive editing, slots are what
    index the arrays.
    """

    body: Body
    forces: np.ndarray
    dt: float
    time: float
    rng: np.random.Generator
    config: SolverConfig | None = None
    held: frozenset[int] = frozenset()
    """Chunks a hand currently owns — the editor's drag, a game's grab.

    Authority, not force. A pinned anchor has infinite mass, so no force a
    grab could apply would ever move it; Rain World resolves the same standoff
    by re-balancing the grasp's authority over the chunk, and this is that
    mechanism's smallest form. Kinematic drivers must stand down for held
    chunks; force controllers may keep contributing, because forces reconcile
    and writes fight."""

    peers: tuple = ()
    """Every controller on the stack this tick, in stack order, self included.

    Perception, not command: this is Rain World's room object list — the
    shared context a creature senses others through. The convention that
    keeps it safe is the same one the whole tier runs on: read your peers,
    write only what you own. A controller that mutates a peer mid-tick has
    made tick order load-bearing, which is the bug the force accumulator
    exists to prevent."""

    doors: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    """The room's exits — door name to world point — as the stack's owner
    declares them each tick.

    Perception, not command, exactly like ``peers``: a brain whose goal is
    to leave reads where its named door stands and walks its own anchor
    there, and the crossing itself stays the scene's, since only the scene
    can see both sides. Empty means a room with no way out, which is every
    room from before doors existed."""

    def slot(self, chunk_id: int) -> int:
        return self.body.slot(chunk_id)

    def has(self, chunk_id: int) -> bool:
        return self.body.has_chunk(chunk_id)

    def add_force(self, chunk_id: int, vector) -> None:
        """Add force to a chunk. A chunk that no longer exists is ignored.

        Silence rather than a raise because deleting a chunk is a routine edit,
        and a controller still holding that id is the expected state for the rest
        of the tick, not a programming error. Raising here would take the whole
        simulation down in response to pressing Delete.
        """
        if not self.body.has_chunk(chunk_id):
            return
        self.forces[self.body.slot(chunk_id)] += np.asarray(vector, dtype=np.float32)

    def position(self, chunk_id: int) -> np.ndarray:
        return self.body.pos[self.body.slot(chunk_id)].copy()

    def velocity(self, chunk_id: int) -> np.ndarray:
        """Implicit Verlet velocity in world units per second."""
        slot = self.body.slot(chunk_id)
        return (self.body.pos[slot] - self.body.prev_pos[slot]) / np.float32(self.dt)

    @property
    def terrain(self):
        """The world's SDF, or ``None``. Passed through from the solver config.

        Controllers that place things in the world — a foothold picker — take
        their surface at construction today; this exists so a controller *can*
        read the world the solver will actually collide against, rather than
        being handed a second copy that can disagree with it.
        """
        return self.config.terrain if self.config is not None else None

    def seek(
        self,
        chunk_id: int,
        target,
        strength: float,
        damping: float = 0.0,
        max_force: float | None = None,
    ) -> np.ndarray:
        """Add a damped spring force pulling a chunk toward a world point.

        This is the primitive nearly every controller wants, and having it in one
        place is what keeps "write a target, not a position" from being merely
        advice. ``damping`` opposes current velocity; without some of it a pure
        spring overshoots and rings forever, which reads as a twitch.

        ``max_force`` clamps the result. A chunk far from its target otherwise
        receives a force proportional to that distance, and one badly placed
        target can inject enough energy in a single tick to throw the rig.

        .. warning::
           This is **one-sided**: the chunk is pulled and nothing is pushed back.
           So seeking a chunk toward another chunk is not a spring between them,
           it is a thruster, and a chain of them injects net momentum that
           accumulates. A free chain whose every link seeks the one ahead will
           slowly fly. Where a mutual attraction is what you want, apply equal
           and opposite force to both ends yourself, or lean on the link
           constraint that is already there.
        """
        if not self.body.has_chunk(chunk_id):
            return np.zeros(self.body.dim, dtype=np.float32)
        slot = self.body.slot(chunk_id)
        offset = np.asarray(target, dtype=np.float32) - self.body.pos[slot]
        force = offset * np.float32(strength)
        if damping:
            force -= self.velocity(chunk_id) * np.float32(damping)
        if max_force is not None:
            magnitude = float(np.linalg.norm(force))
            if magnitude > max_force > 0.0:
                force *= np.float32(max_force / magnitude)
        self.forces[slot] += force
        return force


@runtime_checkable
class Controller(Protocol):
    """Something that contributes force to a body once per tick."""

    enabled: bool

    def update(self, ctx: ControlContext) -> None: ...


class ControllerStack:
    """Owns the controllers, the force accumulator, and the tick order.

    Held by whatever owns the body — the editor document, or a game's creature
    object. The accumulator is reallocated when the chunk count changes rather
    than every tick, since a rig's topology changes on an edit and its positions
    change forty times a second.
    """

    def __init__(self, body: Body, seed: int = 0) -> None:
        self.controllers: list[Controller] = []
        self.seed = seed
        self.time = 0.0
        self.rng = np.random.default_rng(seed)
        #: Chunks a hand owns right now; see :attr:`ControlContext.held`.
        #: Plain mutable state rather than a tick argument because a grab
        #: outlives any one tick and belongs to whoever owns the stack.
        self.held: set[int] = set()
        #: The room's exits by name; see :attr:`ControlContext.doors`.
        #: Plain mutable state on ``held``'s terms: a doorway outlives any
        #: one tick and belongs to whoever owns the stack — the scene
        #: declares it from the room's authored doors, a test writes it
        #: directly.
        self.doors: dict[str, tuple[float, ...]] = {}
        self._forces = np.zeros((body.n_chunks, body.dim), dtype=np.float32)

    @property
    def forces(self) -> np.ndarray:
        """Last tick's accumulated force. Useful for debug overlays."""
        return self._forces

    def add(self, controller: Controller) -> Controller:
        self.controllers.append(controller)
        return controller

    def remove(self, controller: Controller) -> None:
        if controller in self.controllers:
            self.controllers.remove(controller)

    def clear(self) -> None:
        self.controllers.clear()

    def reset(self) -> None:
        """Rewind time, randomness, and any controller state.

        Controllers that derive everything from ``ctx.time`` rewind for free, but
        some genuinely cannot — a foothold and a gait mode are not functions of a
        clock. Any controller exposing ``reset()`` gets called, so the caller does
        not have to know which kind each one is.
        """
        self.time = 0.0
        self.rng = np.random.default_rng(self.seed)
        self._forces[:] = 0.0
        for controller in self.controllers:
            reset = getattr(controller, "reset", None)
            if callable(reset):
                reset()

    def accumulate(
        self, body: Body, dt: float, config: SolverConfig | None = None
    ) -> np.ndarray:
        """Run every enabled controller and return the force accumulator.

        Separated from :meth:`tick` so a caller can inspect the forces before
        they are integrated, which is what the debug overlay and the tests use.
        """
        if self._forces.shape != (body.n_chunks, body.dim):
            self._forces = np.zeros((body.n_chunks, body.dim), dtype=np.float32)
        self._forces[:] = 0.0

        if not self.controllers:
            return self._forces

        ctx = ControlContext(
            body=body,
            forces=self._forces,
            dt=dt,
            time=self.time,
            rng=self.rng,
            config=config,
            held=frozenset(self.held),
            peers=tuple(self.controllers),
            doors=dict(self.doors),
        )
        for controller in self.controllers:
            if getattr(controller, "enabled", True):
                controller.update(ctx)
        return self._forces

    def tick(self, body: Body, dt: float, config: SolverConfig) -> None:
        """One full tick: controllers, then integration, projection, collision."""
        forces = self.accumulate(body, dt, config)
        step(body, dt, config, forces)
        self.time += dt
