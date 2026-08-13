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

    ``self_contact`` turns on collision between a body's own flesh — chunk
    spheres and the round cones along rendered links (docs/SELF_CONTACT.md) —
    wherever the touching regions are strangers in the graph. Off by default
    for the same reason ``ground_y`` is ``None``: a rig's own overlap is
    authored, not accidental. Chunks along a limb are placed overlapping so
    the surface reads as one volume, and a solver that silently pushed them
    apart would reshape every body on load. Linked pairs and second
    neighbours stay exempt even when the pass is on — a folding joint drives
    its second neighbours into contact by design, and the authored way to
    govern that pair is an angle-limit brace, which is itself a link and
    therefore already exempt. Spans inherit the same rule at their end zones,
    and a stranger's mid-span is always real.

    ``self_contact_elasticity`` is the fraction of an overlap closed per tick —
    a restoring force spelled in position-solver terms. ``1.0`` separates an
    isolated pair in one tick and reads as billiard balls; the ``0.5`` default
    lets limbs squash into each other and recover over a few ticks, which reads
    as flesh. Discovery repeats every tick, so residual overlap is found again
    until it is gone.

    ``self_friction`` is the Coulomb ``mu`` between a body's own parts, the
    ground's law brought indoors: a self-contact may undo up to ``mu`` times
    the normal closure it actually applied this tick of the relative
    tangential slide at the contact. ``0.0`` — the default, and the old
    behaviour exactly — separates without grip; raised, a coiled body holds
    its wrap by the load actually crossing the contact. This is passive
    non-slip only: choosing to hold and let go stays a controller feature
    (docs/SELF_CONTACT.md §5).

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
    self_friction: float = 0.0


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


def _segment_params(
    d1: np.ndarray, d2: np.ndarray, r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Clamped closest-point parameters for segment pairs, one lane each.

    The Ericson cascade as lane selects — no data-dependent iteration. Two
    disciplines the lanes owe their callers: the parallel guard is *relative*
    (``denom <= eps * a * e``), because ``a`` and ``e`` are squared lengths
    and an absolute threshold misfires at both large and small scales; and a
    degenerate denominator resolves by projecting the *other* segment's
    midpoint before ``t`` is recomputed — one stated tie-break, deterministic
    instead of wandering the flat valley on numeric noise, and landing inside
    the shared run so an exactly-parallel overlap is found mid-span rather
    than deferred off the end (axis-aligned authoring makes exact
    parallelism common, not measure-zero). Zero-length
    segments are legal inputs (coincident chunks exist transiently, and a
    ``rest=0`` link passes validation); they fall out as the point cases.
    """
    a = np.einsum("id,id->i", d1, d1)
    e = np.einsum("id,id->i", d2, d2)
    b = np.einsum("id,id->i", d1, d2)
    c = np.einsum("id,id->i", d1, r)
    f = np.einsum("id,id->i", d2, r)

    point_eps = np.float32(_COINCIDENT_EPS * _COINCIDENT_EPS)
    a_deg = a <= point_eps
    e_deg = e <= point_eps
    a_safe = np.where(a_deg, np.float32(1.0), a)
    e_safe = np.where(e_deg, np.float32(1.0), e)

    denom = a * e - b * b
    parallel = denom <= np.float32(1e-6) * a * e
    denom_safe = np.where(parallel, np.float32(1.0), denom)

    s = np.clip((b * f - c * e) / denom_safe, 0.0, 1.0)
    midpoint = np.clip((np.float32(0.5) * b - c) / a_safe, 0.0, 1.0)
    s = np.where(parallel, midpoint, s)
    t = (b * s + f) / e_safe
    s = np.where(t < 0.0, np.clip(-c / a_safe, 0.0, 1.0), s)
    s = np.where(t > 1.0, np.clip((b - c) / a_safe, 0.0, 1.0), s)
    t = np.clip(t, 0.0, 1.0)

    s = np.where(e_deg, np.clip(-c / a_safe, 0.0, 1.0), s)
    t = np.where(e_deg, np.float32(0.0), t)
    t = np.where(a_deg & ~e_deg, np.clip(f / e_safe, 0.0, 1.0), t)
    s = np.where(a_deg, np.float32(0.0), s)
    return s.astype(np.float32), t.astype(np.float32)


def _end_owner(
    param: np.ndarray,
    length: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    slot1: np.ndarray,
    slot2: np.ndarray,
) -> np.ndarray:
    """The chunk whose sphere owns this landing, or ``-1`` for the span's mid.

    An end zone is the stretch of centreline within one endpoint radius of
    its end — the cone that endpoint's own sphere dominates, which is the
    flesh the chunk exclusion rule already governs. A parameter inside both
    zones (a short span) belongs to the nearer end.
    """
    in1 = param * length <= r1
    in2 = (np.float32(1.0) - param) * length <= r2
    owner = np.full(param.shape, -1, dtype=np.int64)
    owner[in2] = slot2[in2]
    take1 = in1 & (~in2 | (param <= 0.5))
    owner[take1] = slot1[take1]
    return owner


def _collide_self(body: Body, config: SolverConfig) -> None:
    """Keep the flesh from passing through itself, and let it grip.

    The manifold is the rendered flesh (docs/SELF_CONTACT.md): chunk spheres,
    the round cones along rendered links, and the crossings between them.
    Three families discover fresh each call — sphere–sphere on the landed
    stranger rule, capsule–capsule between rendered spans, sphere–capsule for
    a chunk against a stranger's span — and resolve together as transient
    PUSH constraints in one Jacobi pass: rediscovery next tick is what
    spreads a pile-up over ticks, and that spread *is* the squash the
    elasticity knob promises. Mechanism links carry no flesh and are never
    candidates.

    The exclusion is structural, decided at the *contact*: distance one is
    the constraint set, distance two is the pair a folding joint drives
    together (governed, when governed at all, by an angle-limit brace). A
    landing in a span's end zone belongs to that endpoint chunk and is the
    sphere family's jurisdiction outright — exempt or discovered there by
    the chunk rule — while any landing on a mid-span is real, which is what
    closes the two-fold scissor a pair-level exemption lets through. On a
    well-chunked body the end zones tile every span and the capsule
    families go silent, which is the landed behaviour exactly; they speak
    where the body is sparse, which is the gap they exist to close.
    Candidates and exclusions are recomputed from the live arrays per call —
    derived views, dense masks at renderer scales — but the closest-point
    cascade runs only on prefiltered, compressed survivors: a twenty-step
    cascade over a dense pair grid is the one place the dense idiom must not
    be copied.

    Every contact is a segment pair — a sphere is a zero-length segment — so
    one resolution serves all three families. The share's denominator is the
    *first-power* interpolated inverse mass, not the textbook squared form:
    squared is exact at the contact point but flings the free end of a
    nearly-pinned span by ``error/s``, unbounded, and every correction this
    module applies is bounded by its own error. The price — a mid-span
    contact closes up to half as fast as an endpoint one — is under-closure
    only, and rediscovery absorbs it. One scalar per contact, signs flipped
    between the sides, keeps momentum exact for arbitrary masses (barycentric
    coefficients sum to one and ``m * inv_m == 1``); per-side normalization
    is the natural refactoring mistake and breaks exactly that. The convex
    weight is one shared incidence gather across all families — each contact
    counts one full incidence against each of its chunks, a graze counting a
    whole one on purpose, because weighted counts would forfeit the convexity
    proof — and shared counting is also what lets a clamped capsule contact
    that re-finds a sphere pair sum convexly instead of correcting twice.

    ``self_friction`` is the ground's Coulomb law between a body's own parts:
    after the normal scatter lands, each contact may undo up to ``mu`` times
    the normal closure it actually applied of the relative tangential slide
    at its contact points — read from the settled positions against ``prev``
    with the discovery-time parameters and normal frozen, split equal and
    opposite in mass-weighted terms under the same convex weight, acting on
    ``pos`` and never ``prev``. The budget is the *applied* closure, not the
    discovered overlap: grip must not inflate as elasticity drops, or ``mu``
    becomes a function of solver tuning.
    """
    n = body.n_chunks
    if not config.self_contact or n < 2:
        return

    pos = body.pos
    prev = body.prev_pos
    radius = body.radius
    inv_mass = body.inv_mass

    linked = np.zeros((n, n), dtype=np.float32)
    linked[body.link_a, body.link_b] = 1.0
    linked[body.link_b, body.link_a] = 1.0
    near = (linked + linked @ linked) > 0.0

    fallback = np.zeros(body.dim, dtype=np.float32)
    fallback[0] = 1.0

    parts: list[tuple[np.ndarray, ...]] = []

    # -- sphere–sphere: the landed family, discovery untouched --------------
    delta_all = pos[:, None, :] - pos[None, :, :]
    dist_sq = np.einsum("ijd,ijd->ij", delta_all, delta_all)
    reach = radius[:, None] + radius[None, :]
    overlapping = dist_sq < reach * reach
    movable = (inv_mass[:, None] + inv_mass[None, :]) > 0.0

    ai, bi = np.nonzero(np.triu(overlapping & ~near & movable, k=1))
    if ai.size:
        dist = np.sqrt(dist_sq[ai, bi])
        coincident = dist < _COINCIDENT_EPS
        axis = (pos[ai] - pos[bi]) / np.where(coincident, np.float32(1.0), dist)[
            :, None
        ]
        axis[coincident] = fallback
        zero = np.zeros(ai.size, dtype=np.float32)
        parts.append((ai, ai, zero, bi, bi, zero, axis, dist - reach[ai, bi]))

    # -- the rendered spans, shared by both capsule families ----------------
    ridx = np.nonzero(body.link_render)[0]
    if ridx.size:
        la = body.link_a[ridx]
        lb = body.link_b[ridx]
        p1 = pos[la]
        seg = pos[lb] - pos[la]
        length = np.sqrt(np.einsum("id,id->i", seg, seg))
        r1 = radius[la]
        r2 = radius[lb]
        mid = p1 + np.float32(0.5) * seg
        half = np.float32(0.5) * length
        rmax = np.maximum(r1, r2)

    # -- capsule–capsule: spans of strangers crossing ------------------------
    if ridx.size >= 2:
        shares = (
            (la[:, None] == la[None, :])
            | (la[:, None] == lb[None, :])
            | (lb[:, None] == la[None, :])
            | (lb[:, None] == lb[None, :])
        )
        dmid = mid[:, None, :] - mid[None, :, :]
        mid_sq = np.einsum("ijd,ijd->ij", dmid, dmid)
        sweep = half[:, None] + half[None, :] + rmax[:, None] + rmax[None, :]
        ii, jj = np.nonzero(np.triu((mid_sq < sweep * sweep) & ~shares, k=1))
        if ii.size:
            s, t = _segment_params(seg[ii], seg[jj], p1[ii] - p1[jj])
            ca = p1[ii] + s[:, None] * seg[ii]
            cb = p1[jj] + t[:, None] * seg[jj]
            delta = ca - cb
            dist = np.sqrt(np.einsum("id,id->i", delta, delta))
            span_reach = (
                (np.float32(1.0) - s) * r1[ii]
                + s * r2[ii]
                + (np.float32(1.0) - t) * r1[jj]
                + t * r2[jj]
            )
            # A landing with both sides in end zones is a chunk pair's
            # contact whatever the graph says: exempt when the owners are
            # second neighbours, and the sphere family's own discovery when
            # they are strangers. Either way it is not this family's
            # business — the spheres own the ends, the spans own the middle,
            # and emitting the end case here would perturb the sphere
            # family's exact arithmetic with a fatter duplicate.
            owner_a = _end_owner(s, length[ii], r1[ii], r2[ii], la[ii], lb[ii])
            owner_b = _end_owner(t, length[jj], r1[jj], r2[jj], la[jj], lb[jj])
            deferred = (owner_a >= 0) & (owner_b >= 0)
            keep = (dist < span_reach) & ~deferred
            if keep.any():
                ii, jj, s, t = ii[keep], jj[keep], s[keep], t[keep]
                delta, dist = delta[keep], dist[keep]
                span_reach = span_reach[keep]
                coincident = dist < _COINCIDENT_EPS
                axis = delta / np.where(coincident, np.float32(1.0), dist)[:, None]
                axis[coincident] = fallback
                parts.append(
                    (la[ii], lb[ii], s, la[jj], lb[jj], t, axis, dist - span_reach)
                )

    # -- sphere–capsule: a chunk against a stranger's span ------------------
    if ridx.size:
        slots = np.arange(n)
        endpoint = (slots[:, None] == la[None, :]) | (slots[:, None] == lb[None, :])
        # Both endpoints within distance two means the whole span is
        # second-neighbour flesh — the joint's own geometry, never fought.
        whole_near = near[:, la] & near[:, lb]
        dmid = pos[:, None, :] - mid[None, :, :]
        mid_sq = np.einsum("ijd,ijd->ij", dmid, dmid)
        sweep = half[None, :] + rmax[None, :] + radius[:, None]
        ci, kk = np.nonzero((mid_sq < sweep * sweep) & ~endpoint & ~whole_near)
        if ci.size:
            e = np.einsum("id,id->i", seg[kk], seg[kk])
            e_deg = e <= np.float32(_COINCIDENT_EPS * _COINCIDENT_EPS)
            e_safe = np.where(e_deg, np.float32(1.0), e)
            t = np.einsum("id,id->i", pos[ci] - p1[kk], seg[kk]) / e_safe
            t = np.where(e_deg, np.float32(0.0), t)
            t = np.clip(t, 0.0, 1.0).astype(np.float32)
            cb = p1[kk] + t[:, None] * seg[kk]
            delta = pos[ci] - cb
            dist = np.sqrt(np.einsum("id,id->i", delta, delta))
            span_reach = (
                radius[ci] + (np.float32(1.0) - t) * r1[kk] + t * r2[kk]
            )
            # An end-zone landing is the chunk pair (chunk, owner) — sphere
            # jurisdiction on the same terms as above; only a mid-span
            # landing is this family's contact.
            owner_b = _end_owner(t, length[kk], r1[kk], r2[kk], la[kk], lb[kk])
            keep = (dist < span_reach) & (owner_b < 0)
            if keep.any():
                ci, kk, t = ci[keep], kk[keep], t[keep]
                delta, dist = delta[keep], dist[keep]
                span_reach = span_reach[keep]
                coincident = dist < _COINCIDENT_EPS
                axis = delta / np.where(coincident, np.float32(1.0), dist)[:, None]
                axis[coincident] = fallback
                zero = np.zeros(ci.size, dtype=np.float32)
                parts.append(
                    (ci, ci, zero, la[kk], lb[kk], t, axis, dist - span_reach)
                )

    if not parts:
        return

    i1 = np.concatenate([p[0] for p in parts])
    i2 = np.concatenate([p[1] for p in parts])
    s = np.concatenate([p[2] for p in parts])
    j1 = np.concatenate([p[3] for p in parts])
    j2 = np.concatenate([p[4] for p in parts])
    t = np.concatenate([p[5] for p in parts])
    axis = np.concatenate([p[6] for p in parts])
    error = np.concatenate([p[7] for p in parts])

    one = np.float32(1.0)
    # First power, not squared: the squared form is exact at the contact
    # point but flings the free end of a nearly-pinned span by error/s,
    # unbounded — every correction here stays bounded by its own error.
    denom = (
        (one - s) * inv_mass[i1]
        + s * inv_mass[i2]
        + (one - t) * inv_mass[j1]
        + t * inv_mass[j2]
    )
    live = denom > 0.0
    if not live.all():
        i1, i2, s = i1[live], i2[live], s[live]
        j1, j2, t = j1[live], j2[live], t[live]
        axis, error, denom = axis[live], error[live], denom[live]
    if i1.size == 0:
        return

    elasticity = np.float32(min(max(config.self_contact_elasticity, 0.0), 1.0))
    share = elasticity * error / denom

    # One incidence per distinct chunk per contact, gathered across every
    # family at once — shared counting is what keeps a duplicated end-zone
    # discovery convex instead of double-correcting.
    hits = np.zeros(n, dtype=np.float32)
    np.add.at(hits, i1, np.float32(1.0))
    np.add.at(hits, i2, (i2 != i1).astype(np.float32))
    np.add.at(hits, j1, np.float32(1.0))
    np.add.at(hits, j2, (j2 != j1).astype(np.float32))
    weight = np.float32(1.0) / np.maximum(
        np.maximum(hits[i1], hits[i2]), np.maximum(hits[j1], hits[j2])
    )

    # One scalar per contact, signs flipped between the sides: per-side
    # normalization would make contact a momentum source.
    weighted = axis * (share * weight)[:, None]
    correction = np.zeros_like(pos)
    np.add.at(correction, i1, (-inv_mass[i1] * (one - s))[:, None] * weighted)
    np.add.at(correction, i2, (-inv_mass[i2] * s)[:, None] * weighted)
    np.add.at(correction, j1, (inv_mass[j1] * (one - t))[:, None] * weighted)
    np.add.at(correction, j2, (inv_mass[j2] * t)[:, None] * weighted)
    free = inv_mass != 0.0
    pos[free] += correction[free]

    mu = max(config.self_friction, 0.0)
    if mu == 0.0:
        return

    # Grip, on the ground's terms: slide is read from the settled positions
    # with the discovery-time parameters and normal frozen, and the budget is
    # the relative normal closure this contact actually applied — share and
    # weight and both sides' barycentric mass shares included — never the
    # discovered overlap.
    applied = (
        -(share * weight)
        * (
            (one - s) * (one - s) * inv_mass[i1]
            + s * s * inv_mass[i2]
            + (one - t) * (one - t) * inv_mass[j1]
            + t * t * inv_mass[j2]
        )
    )
    moved_a = (one - s)[:, None] * (pos[i1] - prev[i1]) + s[:, None] * (
        pos[i2] - prev[i2]
    )
    moved_b = (one - t)[:, None] * (pos[j1] - prev[j1]) + t[:, None] * (
        pos[j2] - prev[j2]
    )
    slide = moved_a - moved_b
    slide -= np.einsum("id,id->i", slide, axis)[:, None] * axis
    slid = np.sqrt(np.einsum("id,id->i", slide, slide))

    grip = np.float32(mu) * applied
    scale = np.ones_like(slid)
    np.divide(grip, slid, out=scale, where=slid > 0.0)
    np.minimum(scale, np.float32(1.0), out=scale)
    # The undo is a share like the normal one — divided by the same
    # first-power denominator so the scatter's inverse masses cancel back to
    # a displacement, and tempered by the same convex weight, because grips
    # pile on a chunk where the floor only ever lands one.
    undo = slide * (scale * weight / denom)[:, None]

    correction = np.zeros_like(pos)
    np.add.at(correction, i1, (-inv_mass[i1] * (one - s))[:, None] * undo)
    np.add.at(correction, i2, (-inv_mass[i2] * s)[:, None] * undo)
    np.add.at(correction, j1, (inv_mass[j1] * (one - t))[:, None] * undo)
    np.add.at(correction, j2, (inv_mass[j2] * t)[:, None] * undo)
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
