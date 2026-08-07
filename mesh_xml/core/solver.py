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
from mesh_xml.core.terrain import Terrain

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

    ``ground_y`` is the one piece of world a body can collide with. It is
    ``None`` by default because a solver that silently introduces a floor is
    worse than one with no world at all — a rig authored below the origin would
    be shoved somewhere its author never put it. An application that wants a
    floor asks for one. Without it, and with nothing pinned, a body under
    gravity simply falls forever, which is correct and also useless to look at.

    ``terrain`` is the general world: an SDF the body collides with
    (docs/TERRAIN.md). When set and non-empty it takes precedence over
    ``ground_y`` — having both is not a conflict worth raising over, because
    ``Terrain.flat`` *is* the old floor spelled in the new vocabulary, so the
    terrain simply wins. An empty terrain defers to ``ground_y`` instead: "no
    primitives yet" is an authoring state, not a request to repeal the floor.

    ``self_contact`` turns on collision between a body's own chunks when they
    are strangers in the graph — three or more links apart, or in different
    components. Off by default for the same reason ``ground_y`` is ``None``: a
    rig's own overlap is authored, not accidental. Chunks along a limb are
    placed overlapping so the surface reads as one volume, and a solver that
    silently pushed them apart would reshape every body on load. Linked pairs
    and second neighbours stay exempt even when the pass is on — a folding
    joint drives its second neighbours into contact by design, and the authored
    way to govern that pair is an angle-limit brace, which is itself a link and
    therefore already exempt.

    ``self_contact_elasticity`` is the fraction of an overlap closed per tick —
    a restoring force spelled in position-solver terms. ``1.0`` separates an
    isolated pair in one tick and reads as billiard balls; the ``0.5`` default
    lets limbs squash into each other and recover over a few ticks, which reads
    as flesh. Discovery repeats every tick, so residual overlap is found again
    until it is gone.

    ``ground_friction`` is the Coulomb coefficient ``mu``. Each contact tick
    the floor may undo up to ``mu`` times the chunk's penetration depth of its
    tangential travel; depth is how a position solver expresses normal force,
    so grip scales with the load a foot carries — held outright below the
    Coulomb threshold, sliding above it. At ``0`` a body lands and skates.
    This field used to be a per-tick velocity-bleed fraction, and a bleed
    cannot hold: the slide the constraints cause each tick has already
    happened by the time velocity is bled, so a planted foot crept at every
    setting. Values swept as a bleed (0.15-0.6) remain sensible as ``mu``.
    """

    gravity: tuple[float, ...] = (0.0, -9.81, 0.0)
    damping: float = 0.98
    iterations: int = 4
    max_velocity: float = 100.0
    ground_y: float | None = None
    ground_friction: float = 0.35
    terrain: Terrain | None = None
    self_contact: bool = False
    self_contact_elasticity: float = 0.5


def step(
    body: Body, dt: float, config: SolverConfig, forces: np.ndarray | None = None
) -> None:
    """Advance ``body`` one fixed tick in place.

    ``dt`` must be constant across calls: a Verlet position update infers
    velocity from the previous step's displacement and silently rescales that
    velocity when the step length changes. :func:`fixed_step` supplies it.

    ``forces`` is an optional ``(N, dim)`` accumulator in world units of force,
    which is how procedural controllers get their say. They contribute here and
    the constraint solver reconciles afterward; a controller that instead writes
    positions is fighting the projection that runs immediately after it, and the
    usual result is a rig that buzzes rather than one that moves.
    """
    if body.n_chunks == 0:
        return
    _integrate(body, dt, config, forces)
    _project(body, config)
    # Shape first, separation second, world last: self-contact may push a
    # chunk toward the floor, and the world passes running after it is what
    # guarantees the tick cannot end with a chunk underground.
    _collide_self(body, config)
    # One world per tick: ``Terrain.flat`` and ``ground_y`` describe the same
    # floor, so letting both act would resolve the same contact twice. A set,
    # non-empty terrain is the world and ``ground_y`` stands down; an empty
    # terrain stands down itself, because "no primitives yet" is an authoring
    # state, not a request to repeal a floor the config still asks for.
    if config.terrain is not None and not config.terrain.is_empty:
        _collide_terrain(body, config)
    else:
        _collide_ground(body, config)


def _collide_ground(body: Body, config: SolverConfig) -> None:
    """Lift penetrating chunks onto the floor, with Coulomb friction on the way.

    Resolved by moving position alone. Under Verlet the implicit velocity is
    ``pos - prev_pos``, so lifting a chunk while leaving ``prev_pos`` where it
    was cancels exactly the downward motion that drove it through, with no
    separate velocity term to keep in sync.

    The lift is also the normal impulse, measured in displacement: collision
    runs after projection, so the depth a chunk must be raised this tick
    already carries gravity plus every constraint load transmitted into it —
    a foot bearing a torso is pushed deeper than one bearing only itself.
    Coulomb grip is proportional to exactly that. The floor undoes up to
    ``mu * depth`` of the tangential slide the tick produced, and it acts on
    ``pos``, never on ``prev_pos``: undoing the position erases both the slide
    and the implicit velocity that would replay it, so a full undo is static
    grip — the chunk ends the tick where it began it — and a partial undo
    degrades continuously into kinetic sliding, with no branch between the
    regimes. The previous model bled a fraction of tangential *velocity*
    instead, which arrives too late; the constraints have already displaced
    the chunk by the time friction runs, so a planted foot crept at every
    bleed setting. A chunk at rest regrounds by ``g * dt**2`` each tick, so
    its static grip budget is small and per-tick — that is the honest price
    of expressing normal force as a displacement.

    Runs after projection rather than before, so a chunk cannot be pushed back
    underground by the constraint that put it there.
    """
    if config.ground_y is None or body.dim < 2:
        return

    pos = body.pos
    prev = body.prev_pos
    floor = config.ground_y + body.radius
    depth = floor - pos[:, 1]
    penetrating = (depth > 0.0) & (body.inv_mass > 0.0)
    if not penetrating.any():
        return

    contact = np.where(penetrating)[0]
    pos[contact, 1] = floor[contact]

    mu = max(config.ground_friction, 0.0)
    if mu == 0.0:
        return

    tangential = np.ones(body.dim, dtype=bool)
    tangential[1] = False
    at = np.ix_(contact, tangential)
    slide = pos[at] - prev[at]
    slid = np.linalg.norm(slide, axis=1)
    grip = np.float32(mu) * depth[contact]

    # A zero slide needs no undoing and must stay out of the division; its
    # untouched lane scales a zero vector by one, which is still zero.
    scale = np.ones_like(slid)
    np.divide(grip, slid, out=scale, where=slid > 0.0)
    np.minimum(scale, np.float32(1.0), out=scale)
    pos[at] -= slide * scale[:, None]


def _collide_terrain(body: Body, config: SolverConfig) -> None:
    """Push penetrating chunks out of the terrain field, Coulomb grip included.

    This is :func:`_collide_ground` with the floor's fixed up-axis replaced by
    the field's gradient. The reasoning there — the lift as the tick's normal
    impulse measured in displacement, friction as an undo on ``pos`` with
    ``prev_pos`` never touched, full undo as static grip degrading continuously
    into kinetic sliding with no branch between them, the small per-tick grip
    budget as the honest price of expressing normal force as displacement —
    carries over wholesale and is not repeated here. What the general normal
    changes:

    * Depth is ``radius - distance(pos)`` at the *pre-lift* position, and the
      normal is sampled there too: after the lift the chunk sits on the
      surface, where the gradient no longer remembers which way it sank in.
    * "Tangential" stops meaning "every lane but y" and becomes the tick's
      displacement minus its component along the normal. Measuring it after
      the lift is still exact, not an approximation — the lift is parallel to
      the normal, so the projection removes it along with the approach
      velocity, leaving precisely the slide that happened along the surface.

    That substitution is what makes the slope law hold without tuning: a chunk
    resting on a plane of angle ``theta`` reburies by ``g * dt**2`` vertically
    each tick, which the field reads as ``cos(theta)`` of depth and
    ``sin(theta)`` of slide, so full grip — ``mu * depth >= slide`` — is
    literally ``tan(theta) <= mu``, Coulomb's inequality recovered from
    displacements alone. A zero normal (deep in a blend pocket, where the
    gradient vanishes) pushes nowhere and treats the whole displacement as
    slide, which stalls the chunk rather than handing a NaN downstream.

    The field is evaluated once per call for every chunk and once more for the
    contact set's normals; contacts are masked, never looped over in Python.
    """
    terrain = config.terrain
    if terrain is None or terrain.is_empty:
        return

    pos = body.pos
    prev = body.prev_pos
    d = terrain.distance(pos) - body.radius
    penetrating = (d < 0.0) & (body.inv_mass > 0.0)
    if not penetrating.any():
        return

    contact = np.where(penetrating)[0]
    depth = -d[contact]
    # ``normal`` widens 2-D points to the z = 0 plane and slices its answer
    # back, so the result already matches ``body.dim``.
    n = terrain.normal(pos[contact])
    pos[contact] += depth[:, None] * n

    mu = max(config.ground_friction, 0.0)
    if mu == 0.0:
        return

    disp = pos[contact] - prev[contact]
    slide = disp - (disp * n).sum(axis=1)[:, None] * n
    slid = np.linalg.norm(slide, axis=1)
    grip = mu * depth

    # A zero slide needs no undoing and must stay out of the division; its
    # untouched lane scales a zero vector by one, which is still zero.
    scale = np.ones_like(slid)
    np.divide(grip, slid, out=scale, where=slid > 0.0)
    np.minimum(scale, 1.0, out=scale)
    pos[contact] -= slide * scale[:, None]


def _collide_self(body: Body, config: SolverConfig) -> None:
    """Push overlapping graph-strangers apart: the body may not pass through itself.

    Contacts are discovered fresh each call — every chunk pair that overlaps in
    space while sitting at graph distance three or more — and resolved as
    transient PUSH constraints: corrections shared by inverse mass along the
    pair axis, scaled by ``self_contact_elasticity``. One Jacobi pass, not
    ``config.iterations``: discovery re-runs next tick, so a pile-up resolves
    over ticks rather than within one, and that spread *is* the squash the
    elasticity knob promises.

    The exclusion is structural, not spatial. Distance one is the constraint
    set itself; distance two is the pair every folding joint drives into
    contact, whose governed form is an angle-limit brace — itself a link, and
    so already at distance one. Exclusions and candidates are recomputed from
    the live arrays on every call, like every other derived view of the graph:
    the adjacency matmul is O(n^3) in chunks, microseconds at the body sizes
    the renderer caps out at, and nothing here loops in Python.

    Corrections reuse ``_project``'s symmetric per-pair weight — one over the
    larger of the two endpoints' contact counts — for the same two reasons
    written there: a pile-up must stay a convex combination, and the two ends
    must move equal and opposite in mass-weighted terms or contact becomes a
    momentum source, the artifact that once supplied a fifth of a walker's
    thrust.

    What this pass deliberately does not do is grip: strangers separate
    without friction. A creature holding onto its own body is a controller
    feature, not a contact one.
    """
    n = body.n_chunks
    if not config.self_contact or n < 2:
        return

    pos = body.pos
    radius = body.radius
    inv_mass = body.inv_mass

    linked = np.zeros((n, n), dtype=np.float32)
    linked[body.link_a, body.link_b] = 1.0
    linked[body.link_b, body.link_a] = 1.0
    near = (linked + linked @ linked) > 0.0

    delta_all = pos[:, None, :] - pos[None, :, :]
    dist_sq = np.einsum("ijd,ijd->ij", delta_all, delta_all)
    reach = radius[:, None] + radius[None, :]
    overlapping = dist_sq < reach * reach
    movable = (inv_mass[:, None] + inv_mass[None, :]) > 0.0

    ai, bi = np.nonzero(np.triu(overlapping & ~near & movable, k=1))
    if ai.size == 0:
        return

    elasticity = np.float32(min(max(config.self_contact_elasticity, 0.0), 1.0))
    inv_a = inv_mass[ai]
    inv_b = inv_mass[bi]

    dist = np.sqrt(dist_sq[ai, bi])
    coincident = dist < _COINCIDENT_EPS
    axis = (pos[ai] - pos[bi]) / np.where(coincident, np.float32(1.0), dist)[:, None]
    fallback = np.zeros(body.dim, dtype=np.float32)
    fallback[0] = 1.0
    axis[coincident] = fallback

    # Negative by construction — the depth still to open — so the correction
    # points the ends apart, exactly a PUSH link whose rest is the radii sum.
    error = dist - reach[ai, bi]
    share = elasticity * error / (inv_a + inv_b)

    # Every discovered pair acts, so each endpoint's count is at least one and
    # the weight needs no clamp.
    hits = np.zeros(n, dtype=np.float32)
    np.add.at(hits, ai, np.float32(1.0))
    np.add.at(hits, bi, np.float32(1.0))
    weight = np.float32(1.0) / np.maximum(hits[ai], hits[bi])

    weighted = axis * (share * weight)[:, None]
    correction = np.zeros_like(pos)
    np.add.at(correction, ai, -inv_a[:, None] * weighted)
    np.add.at(correction, bi, inv_b[:, None] * weighted)
    free = inv_mass != 0.0
    pos[free] += correction[free]


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


def _integrate(
    body: Body, dt: float, config: SolverConfig, forces: np.ndarray | None = None
) -> None:
    pos = body.pos
    prev = body.prev_pos
    free = body.inv_mass != 0.0

    velocity = (pos - prev) * np.float32(config.damping)
    limit = np.float32(config.max_velocity * dt)
    speed = np.linalg.norm(velocity, axis=1)
    over = speed > limit
    if over.any():
        velocity[over] *= (limit / speed[over])[:, None]

    accel = np.broadcast_to(
        _gravity_vector(config.gravity, body.dim), (body.n_chunks, body.dim)
    ).astype(np.float32, copy=True)
    if forces is not None:
        # Force, not acceleration, so mass means something to a controller and a
        # pinned chunk absorbs whatever is applied to it rather than needing every
        # caller to remember to skip it.
        accel += forces * body.inv_mass[:, None]
    accel *= np.float32(dt * dt)

    saved = pos.copy()
    pos[free] += velocity[free] + accel[free]
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

        # Overlapping corrections are damped by a per-LINK weight, the same on
        # both ends: one over the larger of the two endpoints' active counts.
        #
        # Two constraints meet here and each once broke the other. Summing raw
        # corrections diverges to NaN at valence four and above, which every
        # cyclic body reaches. But the first cure — dividing each chunk's summed
        # correction by its own count — broke Newton's third law: a link joining
        # a degree-one chunk to a degree-three chunk applied full correction on
        # one side and a third on the other, so every unevenly-connected pair
        # pumped net momentum into the body. A free chain drifted with no force
        # on it, and a fifth of a walking creature's thrust was this artifact.
        #
        # The symmetric weight satisfies both. Per link the two corrections stay
        # equal and opposite in mass-weighted terms (m * inv_m = 1 on each
        # side), so momentum is conserved for any masses; and a chunk of degree
        # d has every incident weight <= 1/d, so its total correction keeps the
        # same bound the per-chunk average had, hub stability included.
        correction[:] = 0.0
        hits[:] = 0.0
        np.add.at(hits, a, active)
        np.add.at(hits, b, active)
        weight = np.float32(1.0) / np.maximum(
            np.maximum(hits[a], hits[b]), np.float32(1.0)
        )
        weighted = offset * weight[:, None]
        np.add.at(correction, a, -inv_a[:, None] * weighted)
        np.add.at(correction, b, inv_b[:, None] * weighted)
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
