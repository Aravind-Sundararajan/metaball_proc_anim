"""The primitive signed-distance math, shared by creature and world.

This began life in :mod:`mesh_xml.render.field` and moved here the day terrain
arrived: the world is a field too (``docs/TERRAIN.md``), and ``core`` may not
import from ``render``. The functions are unchanged by the move — everything
float64, everything shaped ``(..., 3)`` with the trailing axis as the vector.

**The GLSL pairing invariant now points here.** ``render/gl/shaders/sdf.glsl``
is kept line-for-line with *this* module, because the CPU reference is only a
check on the shader while the two compute the same thing. Any edit here needs
the matching edit there, and the numeric comparison tests are what make a
divergence fail rather than merely look odd.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEGENERATE_EPSILON",
    "PRIMITIVE_STRIDE",
    "TETRAHEDRON",
    "as_blend",
    "as_points",
    "as_primitive",
    "as_primitives",
    "round_cone",
    "smooth_min",
]

#: Floats per primitive: ``[ax, ay, az, ra, bx, by, bz, rb]``. The radii are
#: interleaved so each half is a tight ``vec4`` — what a GPU buffer wants.
PRIMITIVE_STRIDE = 8

#: Below this end separation a round cone is treated as a sphere. The exact
#: formula divides by the squared axis length, so the degenerate case has to be
#: branched around rather than left to produce a NaN.
#:
#: .. important::
#:    This covers only the *first* of two degeneracies, and a consumer that
#:    guards on it alone will still render NaN. The second has no epsilon
#:    because it is not a near-miss but an exact geometric condition:
#:    ``abs(ra - rb) >= |b - a|`` — one end ball contains the other, and the
#:    discriminant the formula takes a square root of has gone negative. Both
#:    fall back to the larger-radius end sphere, which is exact rather than an
#:    approximation: the convex hull of two balls where one contains the other
#:    *is* the larger ball. Any GLSL port must carry both branches.
DEGENERATE_EPSILON = 1e-6

#: The four tetrahedron directions gradient estimators sample. They satisfy
#: ``sum(k) == 0`` and ``sum(k k^T) == 4I``, which is what makes four field
#: evaluations recover the gradient where six would be the naive cost.
TETRAHEDRON = np.array(
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
      :func:`mesh_xml.render.sdf.pack` emits ``a == b`` for every unlinked
      chunk, so this is the common path, not a guard against malformed input.
    * **One end sphere containing the other**, ``|ra - rb| >= |b - a|``. The
      convex hull of the two balls is then just the larger ball, and the
      discriminant ``a2`` the formula takes a square root of has gone negative.
      A big chunk linked to a small close one produces this in ordinary rigs.

    Both fall back to the larger-radius end, which is not an approximation: it is
    what the shape is.
    """
    p = as_points(points)
    prim = as_primitive(primitive)
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
    :mod:`mesh_xml.render.sdf` refuses to emit the same geometry twice. And
    ``min(a, b) - k/4 <= smooth_min(a, b, k) <= min(a, b)``: the smooth minimum
    never exceeds the hard one, so it only ever inflates the surface, never
    erodes it. :func:`mesh_xml.render.field.bounds` leans on that.

    ``k == 0`` returns a plain ``min`` without evaluating the division at all,
    rather than dividing and repairing the result afterwards — a masked ``where``
    still computes the ``inf`` first and still raises under
    ``-W error::RuntimeWarning``.
    """
    blend = as_blend(k)
    if blend == 0.0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / blend, 0.0, 1.0)
    return b + (a - b) * h - blend * h * (1.0 - h)


# ----------------------------------------------------------------------
# input coercion
# ----------------------------------------------------------------------


def as_points(points: np.ndarray) -> np.ndarray:
    """Coerce to float64 ``(..., 3)``.

    float64 rather than the packed float32 on purpose: this module is the
    reference the shader's float32 answer is measured against, and a reference
    carrying the same rounding error as the thing it validates measures nothing.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim == 0 or p.shape[-1] != 3:
        raise ValueError(f"points must have a trailing axis of 3, got shape {p.shape}")
    return p


def as_primitives(primitives: np.ndarray) -> np.ndarray:
    prims = np.asarray(primitives, dtype=np.float64)
    if prims.ndim != 2 or prims.shape[1] != PRIMITIVE_STRIDE:
        raise ValueError(
            f"primitives must have shape (P, {PRIMITIVE_STRIDE}), got {prims.shape}"
        )
    return prims


def as_primitive(primitive: np.ndarray) -> np.ndarray:
    prim = np.asarray(primitive, dtype=np.float64)
    if prim.shape != (PRIMITIVE_STRIDE,):
        raise ValueError(
            f"primitive must have shape ({PRIMITIVE_STRIDE},), got {prim.shape}"
        )
    return prim


def as_blend(k: float) -> float:
    value = float(k)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"blend radius must be finite and non-negative, got {k}")
    return value
