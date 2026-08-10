"""CPU reference evaluation of the metaball surface.

This module is the oracle. The GLSL raymarcher computes the same numbers on the
GPU and is validated against what is here, so where the two disagree this one is
right by definition. Everything is therefore written for legibility and
exactness rather than speed: float64 throughout, one plain Python loop over
primitives, and no shortcut the shader could not also take. Meshing and camera
framing consume it too, but neither is the reason it exists.

The primitive *math* lives in :mod:`mesh_xml.core.distance` — the world's
terrain shares it, and ``core`` may not import from here — and the GLSL pairing
invariant points there. The primitive *format* is :mod:`mesh_xml.render.sdf`'s
and its two decisions are assumed here without restating them: every record is a round cone, and a sphere
is the degenerate cone whose ends coincide. :data:`~mesh_xml.render.sdf.pack`
emits those degenerate records for unlinked chunks, so the sphere path is not an
edge case to be tolerated — it is normal traffic.

The fold is order-dependent
---------------------------
The smooth minimum is commutative but not associative: for ``k > 0``,
``smin(smin(a, b), c) != smin(a, smin(b, c))``. :func:`field` folds in array
order and the shader must fold in array order too, or the two surfaces disagree
by a fraction of ``k`` wherever three primitives blend at once. That
:func:`~mesh_xml.render.sdf.pack` emits a deterministic order — links first,
then unlinked chunks by slot — is what makes "array order" a contract rather
than an accident. Reordering the array is a visible change to the surface.

The fold is seeded with the first primitive, never a sentinel
-------------------------------------------------------------
Seeding with a large constant looks free, because ``smin(S, d, k)`` is exactly
``d`` whenever ``S - d > k``. But that is a condition on the caller's ``k``
against the caller's scene scale, and nothing enforces it. Where it fails the
sentinel blends against the real surface and donates up to ``k / 4`` of
inflation everywhere at once — the same ``d - k/4`` self-blend that
:mod:`~mesh_xml.render.sdf` avoids by refusing to emit a sphere and its cones
for the same chunk. Seeding with the first primitive retires the question
instead of bounding it.

Sphere tracing is safe here
---------------------------
Sphere tracing may advance by the field value only if the field never *over*
states the distance to its own zero set, which holds for any 1-Lipschitz field.
Exact SDFs are 1-Lipschitz. The usual worry is that smoothing breaks that.

For this smooth minimum it does not. Inside the blend region the polynomial form
collapses to ``(a + b) / 2 - (a - b)**2 / (4k) - k / 4``, whose partials are
``h`` and ``1 - h``: a convex combination. So ``grad f = h*grad a +
(1 - h)*grad b`` and ``|grad f| <= 1`` whenever both inputs are 1-Lipschitz.
Outside the blend region the result is exactly ``min``, which preserves the
bound as well, so the property survives the entire fold. ``tests/test_field.py``
asserts it numerically on a packed body rather than trusting the derivation.

:data:`STEP_RELAXATION` is still below 1. It buys nothing against the maths and
everything against the two things the maths does not cover: float cancellation
on rays that graze the surface at a shallow angle, and callers who supply
primitives that are not exact SDFs. Raise it to 1.0 for speed on a scene already
known good.

Shapes
------
Points are ``(..., 3)`` and the trailing axis is the vector, so a ray grid, a
meshing lattice and a flat ``(N, 3)`` list all work unchanged and the result
keeps the leading shape. A 2-D body is widened onto ``z = 0`` by
:func:`~mesh_xml.render.sdf.pack`, so nothing here takes a dimension.
"""

from __future__ import annotations

import numpy as np

from mesh_xml.core.distance import (
    TETRAHEDRON as _TETRAHEDRON,
)
from mesh_xml.core.distance import (
    as_blend as _as_blend,
)
from mesh_xml.core.distance import (
    as_points as _as_points,
)
from mesh_xml.core.distance import (
    as_primitives as _as_primitives,
)
from mesh_xml.core.distance import (
    DEGENERATE_EPSILON as _DEGENERATE_EPSILON,
)
from mesh_xml.core.distance import (
    round_cone,
    smooth_min,
    smooth_min_weight,
)

__all__ = [
    "NORMAL_EPSILON",
    "SURFACE_EPSILON",
    "STEP_RELAXATION",
    "albedo",
    "bounds",
    "field",
    "normal",
    "raymarch",
    "round_cone",
    "smooth_min",
]

#: Central-difference offset for :func:`normal`. Large enough that float64
#: cancellation in the difference stays negligible, small enough that the
#: estimate is local to one surface feature.
NORMAL_EPSILON = 1e-4

#: Distance below which :func:`raymarch` calls a ray landed. Sphere tracing
#: approaches the surface geometrically and never reaches it, so termination is
#: always a tolerance rather than an equality.
SURFACE_EPSILON = 1e-4

#: Fraction of the field value each sphere-tracing step actually advances. See
#: the module docstring: the maths permits 1.0, this is margin against float
#: error on grazing rays.
STEP_RELAXATION = 0.95


def field(
    points: np.ndarray,
    primitives: np.ndarray,
    k: float = 0.0,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Signed distance to the blended surface of a ``(P, 8)`` primitive array.

    The fold runs in array order, seeded with the first primitive; see the module
    docstring for why both of those are contracts and not implementation
    details. Vectorized over points, looped over primitives — the loop cannot be
    replaced by a reduction for ``k > 0`` because the fold is not associative,
    and writing the ``k == 0`` case as a separate ``min`` reduction would give
    the oracle a code path the shader does not have.

    ``groups`` is a ``(G, 2)`` array of ``(start, count)`` row ranges — the
    blend groups :func:`~mesh_xml.render.sdf.group_table` describes. Each group
    folds with the smooth minimum *within* itself and the groups combine by the
    hard minimum, so two creatures touching stay two skins instead of welding
    into one. Hard min is associative, which is why cross-group order is not a
    contract the way in-group order is. ``None`` keeps the single global fold,
    and one group covering every row computes exactly the same chain, so the
    grouped path is a strict generalisation.

    An empty primitive array yields ``+inf`` everywhere: the union of nothing is
    infinitely far away, which is the only answer that keeps
    ``field(union) == min(field(part), field(rest))`` true for an empty part.
    :func:`normal` and :func:`raymarch` special-case it rather than propagate it.
    """
    prims = _as_primitives(primitives)
    p = _as_points(points)
    blend = _as_blend(k)

    if len(prims) == 0:
        return np.full(p.shape[:-1], np.inf)

    if groups is None:
        d = round_cone(p, prims[0])
        for prim in prims[1:]:
            d = smooth_min(d, round_cone(p, prim), blend)
        return d

    composite: np.ndarray | None = None
    for start, count in np.asarray(groups, dtype=np.int64):
        rows = prims[start : start + count]
        if len(rows) == 0:
            continue
        d = round_cone(p, rows[0])
        for prim in rows[1:]:
            d = smooth_min(d, round_cone(p, prim), blend)
        composite = d if composite is None else np.minimum(composite, d)
    if composite is None:
        return np.full(p.shape[:-1], np.inf)
    return composite


def albedo(
    points: np.ndarray,
    primitives: np.ndarray,
    attributes: np.ndarray,
    k: float = 0.0,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Surface colour of the blended field at ``points``: ``(..., 3)``.

    The CPU twin of ``sd_group_albedo`` (``docs/TEXTURES.md``): the distance
    fold re-run carrying colour beside it, each step blending by the weight
    :func:`~mesh_xml.core.distance.smooth_min_weight` reports — the same
    ``h`` the distance took, so the skin cannot disagree with the flesh
    about where a blend is. Within one primitive the two end colours lerp by
    the orthogonal projection fraction along its own segment. Both fold
    contracts (array order, seeded with the first primitive) are inherited
    from :func:`field` and matter for the same reason.

    ``attributes`` is :func:`~mesh_xml.render.sdf.pack_attributes`' table
    and must pair the primitives row for row; the empty table that function
    packs for a uniform body is *that caller's* signal to skip the fold
    entirely, so it is refused here rather than given a meaning. ``groups``
    selects the winner the way the shading does: the group whose field is
    least at the point — ``sd_nearest_group``'s answer — colours it.

    Meaningful on the surface, defined everywhere; float64 like every other
    oracle here.
    """
    prims = _as_primitives(primitives)
    attrs = np.asarray(attributes, dtype=np.float64)
    if attrs.shape != prims.shape:
        raise ValueError(
            f"attributes must pair the primitives row for row: "
            f"{prims.shape} primitives, {attrs.shape} attributes"
        )
    p = _as_points(points)
    blend = _as_blend(k)
    if len(prims) == 0:
        return np.zeros(p.shape)
    flat = p.reshape(-1, 3)

    def fold(rows: np.ndarray, arows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = round_cone(flat, rows[0])
        colour = _primitive_albedo(flat, rows[0], arows[0])
        for prim, attr in zip(rows[1:], arows[1:]):
            d, h = smooth_min_weight(d, round_cone(flat, prim), blend)
            step = _primitive_albedo(flat, prim, attr)
            colour = step + (colour - step) * h[:, None]
        return d, colour

    if groups is None:
        _, colour = fold(prims, attrs)
        return colour.reshape(p.shape)

    best_d: np.ndarray | None = None
    best_c: np.ndarray | None = None
    for start, count in np.asarray(groups, dtype=np.int64):
        rows = prims[start : start + count]
        arows = attrs[start : start + count]
        if len(rows) == 0:
            continue
        d, colour = fold(rows, arows)
        if best_d is None:
            best_d, best_c = d, colour
        else:
            wins = d < best_d
            best_d = np.where(wins, d, best_d)
            best_c = np.where(wins[:, None], colour, best_c)
    if best_c is None:
        return np.zeros(p.shape)
    return best_c.reshape(p.shape)


#: The pattern constants, mirrored in ``sdf.glsl``'s SDF_* twins: the band
#: level stripes darken to, the factor a countershaded back darkens by, and
#: the fraction of the way to white a countershaded belly pales.
_TAU = 6.28318530717958647692
_STRIPE_BAND = 0.45
_SHADE_DARK = 0.35
_SHADE_PALE = 0.6


def _primitive_albedo(
    flat: np.ndarray, prim: np.ndarray, attr: np.ndarray
) -> np.ndarray:
    """One primitive's colour at each point — ``sd_primitive_albedo``'s twin.

    End colours lerped by the orthogonal projection fraction (a degenerate
    record reads its ``a`` end, ``t = 0``), then the pattern lanes
    modulate: stripes band along the segment's own axis, countershade keys
    on the radial direction's world-y. Same constants, same ``mix``
    algebra — ``mix(x, y, a)`` is ``x + (y - x)·a`` on both sides.
    """
    a = prim[0:3]
    b = prim[4:7]
    ca = attr[0:3]
    cb = attr[4:7]
    ba = b - a
    l2 = float(ba @ ba)
    if l2 < _DEGENERATE_EPSILON * _DEGENERATE_EPSILON:
        t = np.zeros(len(flat))
    else:
        t = np.clip((flat - a) @ ba / l2, 0.0, 1.0)
    base = ca + (cb - ca) * t[:, None]

    pattern = int(attr[3])
    if pattern == 1:
        wave = 0.5 + 0.5 * np.cos(_TAU * float(attr[7]) * t)
        base = base * (1.0 + (_STRIPE_BAND - 1.0) * wave)[:, None]
    elif pattern == 2:
        radial = flat - (a + ba * t[:, None])
        length = np.linalg.norm(radial, axis=1)
        up = radial[:, 1] / np.maximum(length, 1e-6)
        strength = float(attr[7])
        back = strength * np.maximum(up, 0.0)
        base = base * (1.0 + (_SHADE_DARK - 1.0) * back)[:, None]
        belly = _SHADE_PALE * strength * np.maximum(-up, 0.0)
        base = base + (1.0 - base) * belly[:, None]
    return base


def normal(
    points: np.ndarray,
    primitives: np.ndarray,
    k: float = 0.0,
    eps: float = NORMAL_EPSILON,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Unit outward normals, by central differences on the tetrahedron.

    Four taps rather than six, using offsets that sum to zero and whose outer
    products sum to ``4I``, so the combination is a genuine central difference
    and not a forward one wearing a disguise.

    The result points the way the field increases, which is outward from the
    surface — the field is negative inside. Where the gradient vanishes there is
    no normal to report: the exact centre of a sphere, or a local minimum of the
    field in a blend pocket. Those points get the zero vector rather than a NaN,
    so a caller that can reach one has to test for it.
    """
    prims = _as_primitives(primitives)
    p = _as_points(points)
    blend = _as_blend(k)
    if eps <= 0.0 or not np.isfinite(eps):
        raise ValueError(f"eps must be finite and positive, got {eps}")

    if len(prims) == 0:
        return np.zeros(p.shape)

    flat = p.reshape(-1, 3)
    gradient = np.zeros_like(flat)
    for offset in _TETRAHEDRON:
        gradient += offset * field(flat + offset * eps, prims, blend, groups)[:, None]

    length = np.linalg.norm(gradient, axis=1)
    scale = np.where(length > 0.0, length, 1.0)
    return (gradient / scale[:, None]).reshape(p.shape)


def raymarch(
    origins: np.ndarray,
    directions: np.ndarray,
    primitives: np.ndarray,
    k: float = 0.0,
    *,
    max_steps: int = 128,
    max_distance: float = 100.0,
    surface_epsilon: float = SURFACE_EPSILON,
    step_relaxation: float = STEP_RELAXATION,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sphere-trace rays against the field. Returns ``(t, hit)``.

    ``origins`` and ``directions`` broadcast against each other, so one origin
    against a grid of directions is a pinhole camera and needs no tiling.
    Directions are normalized internally, which is what makes ``t`` a distance in
    world units rather than in units of the caller's direction vector; a
    zero-length direction has no such reading and is rejected.

    ``t`` is the surface distance where ``hit``, and clamped to ``max_distance``
    otherwise — meaningful only under the mask. A ray whose origin is already
    inside the surface reports ``t = 0`` and a hit, since the field is already
    below ``surface_epsilon`` at step zero.

    Stepping by the field value is safe because the field is 1-Lipschitz; the
    module docstring derives that and says what ``step_relaxation`` is really
    for. Rays are dropped from the active set as they land or escape, so a scene
    that resolves early costs the steps it needed rather than ``max_steps``.
    """
    prims = _as_primitives(primitives)
    blend = _as_blend(k)
    if max_steps < 1:
        raise ValueError(f"max_steps must be at least 1, got {max_steps}")
    if max_distance <= 0.0:
        raise ValueError(f"max_distance must be positive, got {max_distance}")
    if surface_epsilon <= 0.0:
        raise ValueError(f"surface_epsilon must be positive, got {surface_epsilon}")
    if not 0.0 < step_relaxation <= 1.0:
        raise ValueError(f"step_relaxation must be in (0, 1], got {step_relaxation}")

    origin, direction = np.broadcast_arrays(_as_points(origins), _as_points(directions))
    shape = origin.shape[:-1]
    origin = origin.reshape(-1, 3)
    direction = direction.reshape(-1, 3)

    length = np.linalg.norm(direction, axis=1)
    if not np.all(length > 0.0):
        raise ValueError("directions must be non-zero to measure t in world units")
    direction = direction / length[:, None]

    t = np.zeros(len(origin))
    hit = np.zeros(len(origin), dtype=bool)
    if len(prims) == 0:
        return t.reshape(shape), hit.reshape(shape)

    active = np.arange(len(origin))
    for _ in range(max_steps):
        if active.size == 0:
            break
        d = field(
            origin[active] + direction[active] * t[active, None],
            prims,
            blend,
            groups,
        )
        landed = d < surface_epsilon
        hit[active[landed]] = True
        t[active] += np.where(landed, 0.0, d * step_relaxation)
        active = active[~landed & (t[active] <= max_distance)]

    np.minimum(t, max_distance, out=t)
    return t.reshape(shape), hit.reshape(shape)


def bounds(primitives: np.ndarray, k: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Conservative axis-aligned box around the blended surface.

    A round cone is the convex hull of its two end balls, so the union of those
    balls' boxes bounds it exactly. Blending then pushes the surface outward, by
    at most ``k / 4`` per fold step — see :func:`smooth_min` — and the box is
    inflated by ``k`` to cover it.

    That inflation *is* a proof, and the fold does not accumulate. Writing
    ``u = (a - b) / k``, the smooth minimum equals ``b`` for ``u >= 1``, ``a``
    for ``u <= -1``, and ``b - k * ((1 - u) / 2) ** 2`` between — which is never
    below ``b - k``. By induction the whole fold stays at or above
    ``min(primitive distances) - k`` for any number of primitives, because past
    a separation of ``k`` the smooth minimum simply *is* the hard one. So a pile
    of overlapping primitives converges rather than sinking by ``k / 4`` apiece,
    and the box inflated by ``k`` contains the surface for any ``P``.

    Checked numerically over four hundred random configurations as well: the
    worst came within 0.0012 of ``k`` and none passed it.

    This holds for exact signed distances, which is what
    :func:`~mesh_xml.render.sdf.pack` produces. A caller hand-building records
    that are not exact — a scaled or otherwise non-metric field — is outside the
    argument, which is why :func:`~mesh_xml.render.meshing.polygonise` reports
    ``open_edges`` rather than trusting this.

    An empty primitive array gives a zero-size box at the origin, so a caller
    framing a camera on an empty body gets a degenerate box rather than an
    exception or an inverted one.
    """
    prims = _as_primitives(primitives)
    pad = _as_blend(k)
    if len(prims) == 0:
        return np.zeros(3), np.zeros(3)

    a = prims[:, 0:3]
    ra = prims[:, 3:4]
    b = prims[:, 4:7]
    rb = prims[:, 7:8]
    lo = np.minimum(a - ra, b - rb).min(axis=0) - pad
    hi = np.maximum(a + ra, b + rb).max(axis=0) + pad
    return lo, hi
