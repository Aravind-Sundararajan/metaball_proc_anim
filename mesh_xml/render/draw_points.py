"""Visual points that chase the physics chunks through their own springs.

This is the single highest-payoff piece of the whole refactor, and it is almost
no code.

Rain World's creatures do not look gelatinous because their physics is soft —
the physics is a handful of circles and distance rods, and on its own it reads
as stiff. They look gelatinous because ``GraphicsModule`` keeps *its own* set of
points which chase the body chunks rather than being them. Each draw point runs
a damped spring toward its chunk, so it arrives late, overshoots, and settles.
That lag is the secondary motion. Drawing the chunks directly throws it away.

The separation buys three things beyond the wobble:

* **Per-point feel.** A head can track tightly while a belly sloshes, from one
  array of stiffnesses, with no change to the simulation.
* **Visual-only effects stay visual.** Squash does not perturb the solver, so it
  cannot destabilise anything.
* **Draw rate is decoupled.** Points tick with the simulation and interpolate for
  the frame, exactly as the body does.

Coefficients here are per-tick, not per-second: ``stiffness`` and ``damping`` are
applied once per fixed tick the way Rain World's are, so they are only meaningful
against a fixed rate. That is the reason :class:`mesh_xml.anim.clock.Clock`
exists. Do not scale them by ``dt``.
"""

from __future__ import annotations

import numpy as np

from mesh_xml.core.body import Body

__all__ = ["DrawPoints"]


class DrawPoints:
    """A smoothed visual copy of a body's chunk positions.

    Each tick decays the existing velocity by ``damping``, then adds ``stiffness``
    of the remaining gap to the chunk. The order matters: damping is momentum
    *carried in*, not a scale on the new impulse. Applied the other way round,
    ``damping=0`` would zero the impulse it had just computed and freeze the point
    in place, rather than meaning "no momentum, go straight there".

    So ``stiffness=1.0`` with ``damping=0.0`` collapses to following the chunks
    exactly, which is the useful sanity case and the baseline lag is measured
    against.

    Both are exposed as per-point arrays after construction, so a rig can author
    a tight head and a loose gut.
    """

    def __init__(
        self,
        body: Body,
        stiffness: float = 0.35,
        damping: float = 0.75,
        squash: float = 0.0,
    ) -> None:
        if not 0.0 <= stiffness <= 1.0:
            raise ValueError(f"stiffness must be in [0, 1], got {stiffness}")
        if not 0.0 <= damping <= 1.0:
            raise ValueError(f"damping must be in [0, 1], got {damping}")
        if squash < 0.0:
            raise ValueError(f"squash must be non-negative, got {squash}")

        self.dim = body.dim
        self.squash = squash
        self._default_stiffness = stiffness
        self._default_damping = damping
        self._ids = np.empty(0, dtype=np.int32)
        self.pos = np.zeros((0, body.dim), dtype=np.float32)
        self.prev_pos = np.zeros((0, body.dim), dtype=np.float32)
        self.vel = np.zeros((0, body.dim), dtype=np.float32)
        self.stiffness = np.zeros(0, dtype=np.float32)
        self.damping = np.zeros(0, dtype=np.float32)
        self.sync(body)

    def sync(self, body: Body) -> None:
        """Match the body's current chunk set, preserving state where ids survive.

        Chunks keep stable ids across removal but not stable slots, so remapping
        by id is what stops an unrelated edit from teleporting every draw point
        and flushing the motion out of the rig.
        """
        if body.dim != self.dim:
            raise ValueError(f"body dim {body.dim} does not match {self.dim}")

        ids = np.asarray(body.chunk_ids(), dtype=np.int32)
        n = len(ids)
        pos = np.zeros((n, self.dim), dtype=np.float32)
        prev_pos = np.zeros((n, self.dim), dtype=np.float32)
        vel = np.zeros((n, self.dim), dtype=np.float32)
        stiffness = np.full(n, self._default_stiffness, dtype=np.float32)
        damping = np.full(n, self._default_damping, dtype=np.float32)

        old_index = {int(cid): i for i, cid in enumerate(self._ids)}
        for slot, cid in enumerate(ids):
            previous = old_index.get(int(cid))
            if previous is None:
                pos[slot] = body.pos[slot]
                prev_pos[slot] = body.pos[slot]
            else:
                pos[slot] = self.pos[previous]
                prev_pos[slot] = self.prev_pos[previous]
                vel[slot] = self.vel[previous]
                stiffness[slot] = self.stiffness[previous]
                damping[slot] = self.damping[previous]

        self._ids = ids
        self.pos = pos
        self.prev_pos = prev_pos
        self.vel = vel
        self.stiffness = stiffness
        self.damping = damping

    def update(self, body: Body, dt: float | None = None) -> None:
        """Advance every draw point one tick toward its chunk.

        ``dt`` is accepted and ignored so this matches the other per-tick update
        signatures; the coefficients are per-tick by construction. Re-syncs
        automatically if the body's chunk set changed.
        """
        if len(self._ids) != body.n_chunks or not np.array_equal(
            self._ids, np.asarray(body.chunk_ids(), dtype=np.int32)
        ):
            self.sync(body)

        self.prev_pos[:] = self.pos
        self.vel *= self.damping[:, None]
        self.vel += (body.pos - self.pos) * self.stiffness[:, None]
        self.pos += self.vel

    def render_pos(self, alpha: float) -> np.ndarray:
        """Draw positions for a frame landing ``alpha`` of the way into the next tick."""
        a = np.float32(min(max(alpha, 0.0), 1.0))
        return self.prev_pos + (self.pos - self.prev_pos) * a

    def render_radius(self, body: Body) -> np.ndarray:
        """Chunk radii thinned by speed, when ``squash`` is enabled.

        A body stretching along its motion has to thin across it. Real
        squash-and-stretch is anisotropic and needs an ellipsoid, which an
        isotropic metaball field cannot express — so this modulates radius only,
        which is the honest cheap version of the effect. ``squash=0`` disables it
        and returns the radii unchanged.
        """
        if self.squash == 0.0:
            return body.radius.copy()
        speed = np.linalg.norm(self.vel, axis=1)
        return body.radius / (1.0 + self.squash * speed)

    def set_stiffness(self, body: Body, chunk_id: int, stiffness: float) -> None:
        if not 0.0 <= stiffness <= 1.0:
            raise ValueError(f"stiffness must be in [0, 1], got {stiffness}")
        self.stiffness[body.slot(chunk_id)] = stiffness

    def snap(self, body: Body) -> None:
        """Drop all lag and velocity. For teleports, where lag would read as a smear."""
        self.sync(body)
        self.pos[:] = body.pos
        self.prev_pos[:] = body.pos
        self.vel[:] = 0.0

    def __repr__(self) -> str:
        return f"<DrawPoints n={len(self._ids)} squash={self.squash}>"
