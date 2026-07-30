"""Verlet integration plus position-based projection of a :class:`Body`'s links.

This is Rain World's two-phase tick. ``BodyChunk.Update()`` integrates every
chunk independently, then ``BodyChunkConnection.Update()`` runs a handful of
relaxation passes that drag chunks back onto their rest distances. Nothing here
solves a linear system: constraints are *projected*, repeatedly and
approximately, which is why an over-constrained rig stays stable and why cycles
cost no more than chains.

Velocity is never stored. It is implied by ``pos - prev_pos``, so a constraint
that displaces a chunk also changes its velocity for free. That coupling is the
whole reason the motion reads as flesh rather than as a rigid snap.

The projection pass is Jacobi rather than Gauss-Seidel: every link is measured
against the same positions and all the corrections are scattered together. A
chunk sitting on ``k`` links would then receive ``k`` full corrections and
overshoot, so its accumulated correction is divided by the number of links that
actually acted on it in that pass. That division makes each pass a convex
combination of the constraints, which cannot expand. Without it, a chunk
carrying four or more links — a hip girdle, a palm — overshoots hard enough to
reach NaN within a few hundred ticks at the default elasticity of ``1.0``,
which would defeat the entire reason :class:`Body` permits cycles. A chunk on a
single link divides by one, so its correction is exactly the inverse-mass
weighted share and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mesh_xml.core.body import Body, LinkType

__all__ = [
    "TICK_HZ",
    "TICK_DT",
    "SolverConfig",
    "step",
    "fixed_step",
    "render_pos",
]

TICK_HZ = 40.0
"""Rain World's simulation rate. Tuning is expressed per tick, not per second."""

TICK_DT = 1.0 / TICK_HZ

_MAX_CATCHUP_TICKS = 8
_COINCIDENT_EPS = 1e-6


@dataclass
class SolverConfig:
    """Tuning for one simulation tick. Mutable, because an editor drives it live.

    ``damping`` is Rain World's ``airFriction``: the fraction of the implicit
    Verlet velocity carried into the next tick. ``1.0`` conserves energy and
    rings forever, which reads as a wire sculpture rather than a creature.

    ``iterations`` buys stiffness. Projection converges geometrically, so four
    passes make a spine that reads as bone while one pass reads as rubber.

    ``max_velocity`` is in world units per second and is a backstop, not a
    tuning knob. A rig authored with a degenerate rest length, or one teleported
    across the world, otherwise gains enough speed in a single tick to land
    outside the range where float32 positions resolve at all.
    """

    gravity: tuple[float, ...] = (0.0, -9.81, 0.0)
    damping: float = 0.98
    iterations: int = 4
    max_velocity: float = 100.0


def step(body: Body, dt: float, config: SolverConfig) -> None:
    """Advance ``body`` one fixed tick in place.

    ``dt`` must be constant across calls: a Verlet position update infers
    velocity from the previous step's displacement and silently rescales that
    velocity when the step length changes. :func:`fixed_step` supplies it.
    """
    if body.n_chunks == 0:
        return
    _integrate(body, dt, config)
    _project(body, config)


def fixed_step(
    body: Body,
    elapsed: float,
    config: SolverConfig,
    accumulator: float = 0.0,
    *,
    dt: float = TICK_DT,
    max_ticks: int = _MAX_CATCHUP_TICKS,
) -> tuple[float, float]:
    """Consume whole ticks out of a wall-clock ``elapsed`` and report the rest.

    Returns ``(accumulator, alpha)``. Feed ``accumulator`` back in next frame
    and hand ``alpha`` to :func:`render_pos`: it is the fraction of a tick the
    renderer is ahead of the simulation, and interpolating by it is what stops a
    40 Hz simulation from looking like 40 Hz on a 144 Hz display.

    At most ``max_ticks`` run per call. A frame that cannot keep up discards the
    overflow rather than carrying it forward, because carrying it forward means
    the next frame owes even more ticks — the spiral where catching up costs
    more than it recovers.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    accumulator = max(accumulator + elapsed, 0.0)
    for _ in range(max_ticks):
        if accumulator < dt:
            break
        step(body, dt, config)
        accumulator -= dt
    if accumulator >= dt:
        accumulator %= dt

    return accumulator, min(accumulator / dt, 1.0)


def render_pos(body: Body, alpha: float) -> np.ndarray:
    """Draw positions: ``prev_pos`` blended toward ``pos`` by ``alpha``.

    A fresh ``(N, dim)`` array rather than a view, so the simulation may tick
    while a frame is still being drawn from the result.
    """
    prev = body.prev_pos
    return prev + (body.pos - prev) * np.float32(alpha)


def _integrate(body: Body, dt: float, config: SolverConfig) -> None:
    pos = body.pos
    prev = body.prev_pos
    free = body.inv_mass != 0.0

    velocity = (pos - prev) * np.float32(config.damping)
    limit = np.float32(config.max_velocity * dt)
    speed = np.linalg.norm(velocity, axis=1)
    over = speed > limit
    if over.any():
        velocity[over] *= (limit / speed[over])[:, None]

    accel = _gravity_vector(config.gravity, body.dim) * np.float32(dt * dt)

    saved = pos.copy()
    pos[free] += velocity[free] + accel
    prev[:] = saved


def _project(body: Body, config: SolverConfig) -> None:
    """Relax every link toward its rest distance, ``config.iterations`` times.

    Two chunks at the same position have no axis to correct along. Rather than
    skip the link and let a pair sit welded together forever, they are separated
    along the first basis vector: an arbitrary direction is a rig that recovers,
    while no direction is a rig that is permanently wrong. Links with both ends
    pinned are skipped outright, since they have no one to move.
    """
    if body.n_links == 0:
        return

    pos = body.pos
    free = body.inv_mass != 0.0
    a = body.link_a
    b = body.link_b
    rest = body.link_rest
    elasticity = body.link_elasticity

    inv_a = body.inv_mass[a]
    inv_b = body.inv_mass[b]
    inv_sum = inv_a + inv_b
    solvable = inv_sum > 0.0
    inv_sum = np.where(solvable, inv_sum, np.float32(1.0))

    is_pull = body.link_type == LinkType.PULL
    is_push = body.link_type == LinkType.PUSH

    fallback = np.zeros(body.dim, dtype=np.float32)
    fallback[0] = 1.0

    correction = np.zeros_like(pos)
    hits = np.zeros(body.n_chunks, dtype=np.float32)

    for _ in range(config.iterations):
        delta = pos[a] - pos[b]
        dist = np.linalg.norm(delta, axis=1)

        coincident = dist < _COINCIDENT_EPS
        axis = delta / np.where(coincident, np.float32(1.0), dist)[:, None]
        axis[coincident] = fallback
        error = dist - rest

        # A PULL link is a rope: slack under compression, taut under stretch. A
        # PUSH link is a body: it resists interpenetration and lets go entirely
        # once the chunks separate. NORMAL holds in both directions.
        slack = (is_pull & (error < 0.0)) | (is_push & (error > 0.0))
        active = solvable & ~slack

        share = np.where(active, elasticity * error / inv_sum, np.float32(0.0))
        offset = axis * share[:, None]

        correction[:] = 0.0
        hits[:] = 0.0
        np.add.at(correction, a, -inv_a[:, None] * offset)
        np.add.at(correction, b, inv_b[:, None] * offset)
        np.add.at(hits, a, active)
        np.add.at(hits, b, active)

        correction /= np.maximum(hits, np.float32(1.0))[:, None]
        pos[free] += correction[free]


def _gravity_vector(gravity: tuple[float, ...], dim: int) -> np.ndarray:
    """Trim or zero-pad an authored gravity to the body's dimensionality.

    ``core`` is dimension-agnostic by decision, so the default 3D gravity has to
    survive being handed to a 2D rig without the caller reconfiguring it.
    """
    out = np.zeros(dim, dtype=np.float32)
    shared = min(len(gravity), dim)
    out[:shared] = gravity[:shared]
    return out
