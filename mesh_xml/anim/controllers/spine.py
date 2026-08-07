"""Spine, head and ambient controllers — what makes a chain read as a creature.

Everything here contributes force and nothing writes a position, for the reason
:mod:`mesh_xml.anim.controller` sets out: projection runs immediately after
integration and overwrites whatever a controller placed, so a controller that
assigns positions loses that argument several times per tick. The failure is not
loud — it is a rig that buzzes, or one that is stable at four solver iterations
and diverges at sixteen — which is why the rule is a rule rather than a habit.

:class:`SpineFollow` is two mechanisms, and they are separate on purpose.

The **follow** term aims each chunk at a point one rest separation *behind* the
chunk ahead of it, where "behind" is read off the leader's own direction of
travel rather than off the segment's current axis. Seeking the leader's position
instead cannot express this: a force toward a single point is collinear with the
segment, so the follower cuts every corner, and the chain is left carrying a
permanent inward load of ``strength * separation`` that the links must cancel
forever. That load is external — it acts on the follower and not on the leader —
so it accumulates down the chain and the whole body creeps after its own head.
Aiming behind the leader is instead exactly zero when the chain is already
trailing correctly and grows only when the leader turns or pulls ahead, which is
what makes a tail trace the head's path rather than swing like rope.
``heading_speed`` is the speed above which the leader's motion, rather than the
chain's current axis, decides where "behind" is; below it the term degrades
smoothly into a rest-length spring instead of chasing the direction of a
near-zero velocity, which is all noise.

The **bend limit** is what stops a body folding back through itself. The angle
at an interior joint is fixed by the triangle its two segments make, and the
segment lengths are already held by links, so the one free variable is the
distance between the joint's two *neighbours*. The limit is therefore applied as
an equal and opposite pair pushing those neighbours apart. Clamping the joint
chunk's position would be both the forbidden position write and useless, since
the next projection pass would put it back. Because the pair sums to zero it
injects no momentum, so a chain pressed hard against its limit does not walk
across the world. The magnitude scales with how far past the limit the joint has
gone: a joint inside the limit costs nothing, and one driven hard against it
pushes back hard, which is a spring rather than a wall and so cannot inject a
step change in energy.

Every controller filters its configured ids against the live body each tick. The
editor deletes chunks while the simulation is running, and a controller holding
an id that no longer resolves has to skip it rather than raise mid-tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mesh_xml.anim.controller import ControlContext

__all__ = ["SpineFollow", "LookAt", "Breathe"]

_EPS = 1e-6
_TAU = 2.0 * math.pi


def _live(ctx: ControlContext, chunk_ids: list[int]) -> list[int]:
    """The configured ids that still resolve, in configured order.

    A deleted chunk closes the gap rather than splitting the chain: its links
    went with it, so the separation used for the newly adjacent pair falls back
    to their current distance and nothing is yanked together.
    """
    return [cid for cid in chunk_ids if ctx.body.has_chunk(cid)]


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < _EPS:
        return None
    return vector / np.float32(norm)


def _separation(ctx: ControlContext, a: int, b: int, override: float | None) -> float:
    """How far apart two chained chunks want to be.

    The link's rest length when there is one, because that is the distance the
    solver is going to enforce anyway and disagreeing with it means the follow
    term and the projection pass pull against each other every tick.
    """
    if override is not None:
        return override
    if ctx.body.has_link(a, b):
        return ctx.body.link(a, b).rest
    return float(np.linalg.norm(ctx.position(a) - ctx.position(b)))


@dataclass
class SpineFollow:
    """An ordered chain where each chunk trails the one ahead, up to a bend limit.

    ``chain`` runs head first. ``damping`` is the lag: it opposes the follower's
    current velocity, so the chain settles into its leader's path instead of
    ringing around it.

    ``max_bend_degrees`` is measured from straight — ``0`` allows no bend at all
    and ``180`` allows the joint to close completely. :attr:`min_joint_degrees`
    is the same limit stated as the interior angle at the joint, which is what an
    observer measures.

    ``lead`` is an optional world point the head seeks, which is how a caller
    steers the whole body by moving one point.
    """

    chain: list[int]
    strength: float = 24.0
    damping: float = 3.0
    max_bend_degrees: float = 50.0
    bend_strength: float = 800.0
    lead: tuple[float, ...] | None = None
    lead_strength: float | None = None
    spacing: float | None = None
    heading_speed: float = 2.0
    max_force: float | None = 200.0
    enabled: bool = True

    #: Tempo dimensions (:mod:`mesh_xml.anim.tempo`): a faster creature turns
    #: its heading faster; the gains are muscle, not clock, and stay authored.
    TEMPO = {"heading_speed": 1}

    @property
    def min_joint_degrees(self) -> float:
        """The tightest interior angle the joints are allowed to settle at."""
        return 180.0 - self.max_bend_degrees

    def update(self, ctx: ControlContext) -> None:
        chain = _live(ctx, self.chain)
        if not chain:
            return
        if self.lead is not None:
            pull = self.strength if self.lead_strength is None else self.lead_strength
            ctx.seek(chain[0], self.lead, pull, self.damping, self.max_force)
        self._follow(ctx, chain)
        self._limit_bend(ctx, chain)

    def _follow(self, ctx: ControlContext, chain: list[int]) -> None:
        if self.strength <= 0.0:
            return
        for leader, follower in zip(chain, chain[1:]):
            axis = _unit(ctx.position(leader) - ctx.position(follower))
            if axis is None:
                continue
            drift = axis * np.float32(self.heading_speed) + ctx.velocity(leader)
            heading = _unit(drift)
            if heading is None:
                continue
            gap = np.float32(_separation(ctx, leader, follower, self.spacing))
            target = ctx.position(leader) - heading * gap
            ctx.seek(follower, target, self.strength, self.damping, self.max_force)

    def _limit_bend(self, ctx: ControlContext, chain: list[int]) -> None:
        if len(chain) < 3 or self.bend_strength <= 0.0:
            return

        slots = np.array([ctx.slot(cid) for cid in chain], dtype=np.intp)
        pos = ctx.body.pos[slots]
        incoming = pos[1:-1] - pos[:-2]
        outgoing = pos[2:] - pos[1:-1]
        span = pos[:-2] - pos[2:]

        lengths = np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1)
        reach = np.linalg.norm(span, axis=1)
        measurable = (lengths > _EPS) & (reach > _EPS)

        cosine = np.einsum("ij,ij->i", incoming, outgoing) / np.where(
            measurable, lengths, np.float32(1.0)
        )
        bend = np.arccos(np.clip(cosine, np.float32(-1.0), np.float32(1.0)))
        excess = bend - np.float32(math.radians(self.max_bend_degrees))
        past_limit = measurable & (excess > 0.0)
        if not past_limit.any():
            return

        magnitude = np.where(past_limit, excess * np.float32(self.bend_strength), 0.0)
        if self.max_force is not None:
            magnitude = np.minimum(magnitude, np.float32(self.max_force))
        apart = span / np.where(measurable, reach, np.float32(1.0))[:, None]
        push = (apart * magnitude[:, None]).astype(np.float32, copy=False)

        np.add.at(ctx.forces, slots[:-2], push)
        np.add.at(ctx.forces, slots[2:], -push)


@dataclass
class LookAt:
    """Point a short sub-chain at a world target, pivoting about its base.

    ``chain`` runs base first. Only the chunks past the base are pulled: the base
    is the neck the rest of the body already holds, and seeking it too would turn
    a head turn into a lunge.
    """

    chain: list[int]
    target: tuple[float, ...] | None = None
    strength: float = 20.0
    damping: float = 2.0
    spacing: float | None = None
    max_force: float | None = 200.0
    enabled: bool = True

    def update(self, ctx: ControlContext) -> None:
        if self.target is None:
            return
        chain = _live(ctx, self.chain)
        if len(chain) < 2:
            return

        base = ctx.position(chain[0])
        aim = _unit(np.asarray(self.target, dtype=np.float32) - base)
        if aim is None:
            return

        reach = 0.0
        for previous, chunk in zip(chain, chain[1:]):
            reach += _separation(ctx, previous, chunk, self.spacing)
            ctx.seek(
                chunk,
                base + aim * np.float32(reach),
                self.strength,
                self.damping,
                self.max_force,
            )


@dataclass
class Breathe:
    """A slow ambient swell, so a resting rig is not a still image.

    Phases are drawn once from ``ctx.rng`` and kept, which is what a phase
    offset is: a property of the individual, not of the moment. Redrawing them
    every tick would be noise, and deriving them from a private counter would
    survive :meth:`ControllerStack.reset` and quietly break replay. Time comes
    from ``ctx.time`` for the same reason — resetting the stack genuinely
    rewinds the oscillation instead of restarting it from wherever it was.

    With no ``axis`` the group swells and shrinks about its own centroid, which
    is what a ribcage does. An ``axis`` oscillates every chunk along one fixed
    world direction instead, and must match ``body.dim``.
    """

    chunks: list[int]
    amplitude: float = 3.0
    frequency: float = 0.25
    axis: tuple[float, ...] | None = None
    enabled: bool = True
    _phase: np.ndarray | None = field(default=None, init=False, repr=False)

    #: Tempo dimensions (:mod:`mesh_xml.anim.tempo`): breath keeps time with
    #: the gait; its depth (``amplitude``) is a force and stays.
    TEMPO = {"frequency": 1}

    def update(self, ctx: ControlContext) -> None:
        if self._phase is None or self._phase.size != len(self.chunks):
            # Drawn for the whole configured list rather than the live subset, so
            # what the generator is asked for depends on the rig and not on which
            # chunks happen to exist on the tick this first runs. Re-editing the
            # list redraws: a stale phase array shorter than its ids would index
            # out of range mid-tick, which is the one failure mode a controller
            # is not allowed to have.
            self._phase = ctx.rng.uniform(0.0, _TAU, size=len(self.chunks))

        live = [i for i, cid in enumerate(self.chunks) if ctx.body.has_chunk(cid)]
        if not live:
            return

        index = np.array(live, dtype=np.intp)
        slots = np.array([ctx.slot(self.chunks[i]) for i in live], dtype=np.intp)
        phase = np.float32(_TAU * self.frequency * ctx.time) + self._phase[index]
        wave = np.sin(phase).astype(np.float32) * np.float32(self.amplitude)

        direction = self._directions(ctx, slots)
        if direction is None:
            return
        np.add.at(ctx.forces, slots, (direction * wave[:, None]).astype(np.float32))

    def _directions(self, ctx: ControlContext, slots: np.ndarray) -> np.ndarray | None:
        if self.axis is not None:
            unit = _unit(np.asarray(self.axis, dtype=np.float32))
            if unit is None:
                return None
            return np.broadcast_to(unit, (slots.size, unit.size))

        pos = ctx.body.pos[slots]
        outward = pos - pos.mean(axis=0)
        spread = np.linalg.norm(outward, axis=1)
        usable = spread > _EPS
        outward = outward / np.where(usable, spread, np.float32(1.0))[:, None]
        outward[~usable] = 0.0
        return outward
