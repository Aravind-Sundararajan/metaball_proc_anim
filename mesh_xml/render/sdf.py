"""Packing a body into signed-distance primitives.

This module owns the *format* the metaball surface is built from. The CPU
reference field and the GPU raymarcher both consume it, and they have to agree
exactly or the viewport stops matching what gets exported.

Two decisions are load-bearing.

**Everything is a round cone.** A round cone — a capsule with a different radius
at each end — is the only primitive emitted. A sphere is the degenerate case
where both ends coincide. One record type means the shader has a single loop
with no per-primitive branching and no type tag to unpack, and it means chunk
radii can vary along a chain without the surface stepping between segments the
way uniform capsules would.

**Links carry the geometry, not chunks.** A round cone spanning two chunks
already contains both end spheres, so emitting a sphere *and* its cones would
overlap them. That is not free: the smooth minimum of a shape with itself is
``d - k/4``, so duplicated geometry silently inflates the surface by a fraction
of the blend radius. Chunks therefore contribute a primitive only when nothing
links them.

The whole point of building the surface from links rather than chunks is that
spheres alone read as beads on a string. ``examples/character_armature.xml`` was
hand-authored with ``type="capsule"`` for exactly this reason, and the old
importer threw that away.
"""

from __future__ import annotations

import numpy as np

from mesh_xml.core.body import Body

__all__ = ["PRIMITIVE_STRIDE", "pack", "primitive_count"]

#: Floats per primitive: ``[ax, ay, az, ra, bx, by, bz, rb]``.
#:
#: The radii are interleaved rather than trailing so each half is a tight
#: ``vec4``, which is what a std430 buffer wants and what avoids padding.
PRIMITIVE_STRIDE = 8

#: Below this end separation a round cone is treated as a sphere. The exact
#: formula divides by the squared axis length, so the degenerate case has to be
#: branched around rather than left to produce a NaN.
DEGENERATE_EPSILON = 1e-6


def primitive_count(body: Body) -> int:
    """How many primitives :func:`pack` will emit for ``body``."""
    linked = set()
    for link in body.links():
        linked.add(link.a)
        linked.add(link.b)
    return body.n_links + (body.n_chunks - len(linked))


def pack(
    body: Body,
    positions: np.ndarray | None = None,
    radii: np.ndarray | None = None,
) -> np.ndarray:
    """Build the ``(P, 8) float32`` primitive array for a body.

    ``positions`` and ``radii`` override the body's own, which is how the
    smoothing layer feeds in interpolated draw positions and squashed radii:
    the surface should follow what the viewer sees, not the raw physics.
    Both are indexed by slot, matching :attr:`Body.pos`.

    The array is contiguous and ready to upload; the GPU path should not need to
    touch it again.
    """
    pos = body.pos if positions is None else np.asarray(positions, dtype=np.float32)
    rad = body.radius if radii is None else np.asarray(radii, dtype=np.float32)

    if pos.shape != (body.n_chunks, body.dim):
        raise ValueError(
            f"positions must have shape ({body.n_chunks}, {body.dim}), got {pos.shape}"
        )
    if rad.shape != (body.n_chunks,):
        raise ValueError(f"radii must have shape ({body.n_chunks},), got {rad.shape}")

    count = primitive_count(body)
    out = np.zeros((count, PRIMITIVE_STRIDE), dtype=np.float32)
    if count == 0:
        return out

    n_links = body.n_links
    if n_links:
        a = body.link_a
        b = body.link_b
        out[:n_links, 0:3] = _to_vec3(pos[a], body.dim)
        out[:n_links, 3] = rad[a]
        out[:n_links, 4:7] = _to_vec3(pos[b], body.dim)
        out[:n_links, 7] = rad[b]

    linked = set()
    for link in body.links():
        linked.add(body.slot(link.a))
        linked.add(body.slot(link.b))

    row = n_links
    for slot in range(body.n_chunks):
        if slot in linked:
            continue
        centre = _to_vec3(pos[slot : slot + 1], body.dim)[0]
        out[row, 0:3] = centre
        out[row, 3] = rad[slot]
        out[row, 4:7] = centre
        out[row, 7] = rad[slot]
        row += 1

    return out


def _to_vec3(points: np.ndarray, dim: int) -> np.ndarray:
    """Widen a 2-D body into the 3-D the field always works in, on the z=0 plane."""
    if dim == 3:
        return points.astype(np.float32, copy=False)
    widened = np.zeros((len(points), 3), dtype=np.float32)
    widened[:, :dim] = points
    return widened
