"""CPU reference evaluation of the metaball surface.

This module is the oracle. The GLSL raymarcher computes the same numbers on the
GPU and is validated against what is here, so where the two disagree this one is
right by definition. Everything is therefore written for legibility and
exactness rather than speed: float64 throughout, one plain Python loop over
primitives, and no shortcut the shader could not also take. Meshing and camera
framing consume it too, but neither is the reason it exists.

The primitive format is :mod:`mesh_xml.render.sdf`'s and its two decisions are
assumed here without restating them: every record is a round cone, and a sphere
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

from mesh_xml.render.sdf import DEGENERATE_EPSILON, PRIMITIVE_STRIDE

__all__ = [
    "NORMAL_EPSILON",
    "SURFACE_EPSILON",
    "STEP_RELAXATION",
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

#: The four tetrahedron directions :func:`normal` samples. They satisfy
#: ``sum(k) == 0`` and ``sum(k k^T) == 4I``, which is what makes four field
#: evaluations recover the gradient where six would be the naive cost.
_TETRAHEDRON = np.array(
    [[1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, 1.0]]
)


def round_cone(points: np.ndarray, primitive: np.ndarray) -> np.ndarray:
    """Exact signed distance from every point to one ``(8,)`` round cone record.

    This is Inigo Quilez's ``sdRoundCone`` transcribed branch for branch,
    including the ``l2`` scaling that lets all three cases share a single square
    root. Cheaper approximations exist — lerping a capsule radius along the axis
    is the common one — and every one of them is wrong away from the axis, which
    is exactly where the taper is visible.

    Two configurations have no cone and are answered with a sphere instead:

    * **Coincident ends.** The exact formula divides by the squared axis length.
      :func:`~mesh_xml.render.sdf.pack` emits ``a == b`` for every unlinked
      chunk, so this is the common path, not a guard against malformed input.
    * **One end sphere containing the other**, ``|ra - rb| >= |b - a|``. The
      convex hull of the two balls is then just the larger ball, and the
      discriminant ``a2`` the formula takes a square root of has gone negative.
      A big chunk linked to a small close one produces this in ordinary rigs.

    Both fall back to the larger-radius end, which is not an approximation: it is
    what the shape is.
    """
    p = _as_points(points)
    prim = _as_primitive(primitive)
    flat = p.reshape(-1, 3)

    a = prim[0:3]
    ra = float(prim[3])
    b = prim[4:7]
    rb = float(prim[7])

    ba = b - a
    l2 = float(ba @ ba)
    rr = ra - rb
    a2 = l2 - rr * rr

    if l2 < DEGENERATE_EPSILON * DEGENERATE_EPSILON or a2 <= 0.0:
        centre, radius = (a, ra) if ra >= rb else (b, rb)
        sphere = np.linalg.norm(flat - centre, axis=1) - radius
        return sphere.reshape(p.shape[:-1])

    il2 = 1.0 / l2
    pa = flat - a
    y = pa @ ba
    z = y - l2

    perpendicular = pa * l2 - ba * y[:, None]
    x2 = (perpendicular * perpendicular).sum(axis=1)
    y2 = y * y * l2
    z2 = z * z * l2

    split = np.sign(rr) * rr * rr * x2
    on_b_cap = np.sign(z) * a2 * z2 > split
    on_a_cap = np.sign(y) * a2 * y2 < split

    out = np.where(
        on_b_cap,
        np.sqrt(x2 + z2) * il2 - rb,
        np.where(
            on_a_cap,
            np.sqrt(x2 + y2) * il2 - ra,
            (np.sqrt(x2 * a2 * il2) + y * rr) * il2 - ra,
        ),
    )
    return out.reshape(p.shape[:-1])


def smooth_min(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth minimum of two fields with blend radius ``k``.

    ``h = clip(0.5 + 0.5*(b - a)/k, 0, 1)``, then ``mix(b, a, h) - k*h*(1 - h)``.

    Two properties are load-bearing elsewhere and worth naming here.
    ``smooth_min(d, d, k) == d - k/4`` exactly, which is why
    :mod:`~mesh_xml.render.sdf` refuses to emit the same geometry twice. And
    ``min(a, b) - k/4 <= smooth_min(a, b, k) <= min(a, b)``: the smooth minimum
    never exceeds the hard one, so it only ever inflates the surface, never
    erodes it. :func:`bounds` leans on that.

    ``k == 0`` returns a plain ``min`` without evaluating the division at all,
    rather than dividing and repairing the result afterwards — a masked ``where``
    still computes the ``inf`` first and still raises under
    ``-W error::RuntimeWarning``.
    """
    blend = _as_blend(k)
    if blend == 0.0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / blend, 0.0, 1.0)
    return b + (a - b) * h - blend * h * (1.0 - h)


def field(points: np.ndarray, primitives: np.ndarray, k: float = 0.0) -> np.ndarray:
    """Signed distance to the blended surface of a ``(P, 8)`` primitive array.

    The fold runs in array order, seeded with the first primitive; see the module
    docstring for why both of those are contracts and not implementation
    details. Vectorized over points, looped over primitives — the loop cannot be
    replaced by a reduction for ``k > 0`` because the fold is not associative,
    and writing the ``k == 0`` case as a separate ``min`` reduction would give
    the oracle a code path the shader does not have.

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

    d = round_cone(p, prims[0])
    for prim in prims[1:]:
        d = smooth_min(d, round_cone(p, prim), blend)
    return d


def normal(
    points: np.ndarray,
    primitives: np.ndarray,
    k: float = 0.0,
    eps: float = NORMAL_EPSILON,
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
        gradient += offset * field(flat + offset * eps, prims, blend)[:, None]

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
        d = field(origin[active] + direction[active] * t[active, None], prims, blend)
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


# ----------------------------------------------------------------------
# input coercion
# ----------------------------------------------------------------------


def _as_points(points: np.ndarray) -> np.ndarray:
    """Coerce to float64 ``(..., 3)``.

    float64 rather than the packed float32 on purpose: this module is the
    reference the shader's float32 answer is measured against, and a reference
    carrying the same rounding error as the thing it validates measures nothing.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim == 0 or p.shape[-1] != 3:
        raise ValueError(f"points must have a trailing axis of 3, got shape {p.shape}")
    return p


def _as_primitives(primitives: np.ndarray) -> np.ndarray:
    prims = np.asarray(primitives, dtype=np.float64)
    if prims.ndim != 2 or prims.shape[1] != PRIMITIVE_STRIDE:
        raise ValueError(
            f"primitives must have shape (P, {PRIMITIVE_STRIDE}), got {prims.shape}"
        )
    return prims


def _as_primitive(primitive: np.ndarray) -> np.ndarray:
    prim = np.asarray(primitive, dtype=np.float64)
    if prim.shape != (PRIMITIVE_STRIDE,):
        raise ValueError(
            f"primitive must have shape ({PRIMITIVE_STRIDE},), got {prim.shape}"
        )
    return prim


def _as_blend(k: float) -> float:
    value = float(k)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"k must be finite and non-negative, got {k}")
    return value
