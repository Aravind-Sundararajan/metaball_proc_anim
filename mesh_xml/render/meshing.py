"""Polygonising the metaball field, so a surface can leave the viewport.

The viewport raymarches. Nothing it produces is geometry — it is pixels, and a
pixel cannot be opened in Blender. This module is the only path from the field
to something another tool can read, and :mod:`mesh_xml.io.mesh_export` writes
what it returns. Phase 3 of the refactor plan deferred it deliberately: it was
never wanted for display, only for export, and that is still the contract. Do
not put this in a frame loop.

Surface nets rather than marching cubes
---------------------------------------
Both walk the same lattice and read the same sign changes, so the choice is
entirely about what happens inside one cell.

Marching cubes places vertices *on* grid edges and looks triangles up in a
256-entry case table. A face whose four corners alternate in sign can be joined
two ways, and the classical table picks per-cell without consulting the
neighbour that shares the face; where the two disagree the surface **tears**.
Closing that means the asymptotic decider or one of the topologically
consistent tables — a large amount of data to transcribe correctly and an
awkward amount to test.

Surface nets places one vertex *inside* every cell the surface passes through,
and emits a quad for each grid edge that changes sign, joining the four cells
around that edge. The result is the dual of the lattice, and the construction
cannot produce a boundary edge at all. A dual edge dualises a grid face, and it
belongs to one quad per sign-changing edge of that face; a four-corner loop
always changes sign an even number of times, so the count is zero, two or four.
Two is a manifold edge. Four is the ambiguous face again — and here it costs a
non-manifold edge instead of a hole. For export that is the right trade: a hole
makes a mesh unprintable and unfillable, a non-manifold edge is something a
repair pass can act on and a renderer never notices.

It is also roughly a third of the vertices for the same surface, and there is no
table to get wrong.

Where the mesh could fail to be closed, and why it does not
-----------------------------------------------------------
One place, and it is the usual one: the surface running out through the side of
the sampled box. A lattice edge needs all four of its surrounding cells to exist
before it can become a quad, so a sign change on the outermost shell has nowhere
to attach and would silently become a hole.

It cannot happen for a body :func:`~mesh_xml.render.sdf.pack` produced.
:func:`~mesh_xml.render.field.bounds` describes its own inflation as a working
bound rather than a proof, warning that a pile-up of ``P`` primitives could sink
the field ``(P - 1) * k / 4`` below the hard minimum. That worry does not
survive being worked through. Writing ``u = (a - b) / k``, the smooth minimum is
``b`` for ``u >= 1``, ``a`` for ``u <= -1``, and ``b - k * ((1 - u) / 2)**2``
between — never below ``b - k``. So each fold returns something at least
``min(running, next) - k``, and by induction the whole fold stays at least
``min(every primitive) - k`` no matter how many there are. It converges rather
than accumulating, because past a separation of ``k`` the smooth minimum *is*
the hard one and stops contributing. Outside a box inflated by ``k`` every
primitive is further than ``k`` away, so the field there is positive.
``tests/test_meshing.py`` checks the ``k`` bound numerically on random
configurations rather than resting on the derivation.

:data:`PADDING_CELLS` layers of empty lattice are then added outside that box,
so the outer shell is sampled clear of the surface. :attr:`Mesh.open_edges`
counts sign changes that landed there anyway — the exact number of missing
faces, and :attr:`Mesh.closed` is that count being zero. It is an assertion
carried in the return value rather than an expected outcome, and it is cheap
enough to leave in: a caller can hand-build a primitive array whose records are
not exact SDFs, and that is the one way past the argument above. Exporting for
print should check it; silence is not on offer.

Normals come from the field, not from the faces
-----------------------------------------------
:func:`~mesh_xml.render.field.normal` is a central difference on the real field
and is correct to the field's own precision everywhere, including inside a blend
fillet where the surface curves fastest. Averaging face normals instead would
reproduce the polygonisation's own faceting — which is to say, it would bake the
grid resolution into the shading and make a coarse mesh look coarse even when
smooth-shaded. Sampling the field costs four extra evaluations per vertex and
there are far fewer vertices than grid points, so it is close to free.

Vectorisation, and what resolution is practical
-----------------------------------------------
Sampling is the whole cost: the lattice is ``O(n**3)`` and everything after it
is ``O(n**2)``. Evaluating cells one at a time spends the entire runtime in
Python and numpy call overhead — measured against a nine-primitive body,
:func:`~mesh_xml.render.field.field` costs 207 µs for one point and 0.73 µs per
point in bulk, a factor of 280. That is the difference between eight minutes and
half a second on the 2.4 million lattice points of a resolution-128 ball, and it
only widens as resolution rises.

So the lattice is evaluated in blocks of :data:`SAMPLE_BLOCK` points — one
``field`` call for the whole grid up to resolution 123 on a cube-shaped body,
and up to 256 on anything as elongated as a worm. Blocking rather than one
unconditional call is about peak memory, not speed: the transient ``(M, 3)``
coordinate array and ``field``'s own temporaries together run several times the
size of the samples they produce, so a single call at resolution 256 wants
gigabytes to compute a mesh that occupies megabytes.

Measured on ``examples/chubby_worm.xml`` at ``k = 0.25`` — ten chunks, nine
primitives — resolution 64 takes 0.04 s, 128 takes 0.26 s and 256 takes 1.9 s.
A body whose box is closer to a cube pays for the cube of its longest axis
rather than for its own volume: a unit ball is 0.44 s at 128 and 3.8 s at 256.
96–128 is the useful band. Below it the blend fillets read as facets; above it
the file grows faster than the surface improves, and 128 already holds every
vertex within 1e-4 of an analytic unit sphere.

Threading
---------
:func:`polygonise` reads its arguments and the module constants, writes only
locals, and touches no cache, no scene and no GL state. It can be called on a
worker thread as-is — which is the point, since even a quarter of a second is
too long to spend on the UI thread, and a fine mesh is several.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mesh_xml.render.field import bounds, field, normal
from mesh_xml.render.sdf import PRIMITIVE_STRIDE

__all__ = ["PADDING_CELLS", "SAMPLE_BLOCK", "Mesh", "polygonise"]

#: Layers of lattice added outside :func:`~mesh_xml.render.field.bounds` on
#: every side. One is all the argument in the module docstring needs; two costs
#: about 5% more samples at resolution 64 and covers the case that argument does
#: not — a caller passing records that are not exact signed distances.
PADDING_CELLS = 2

#: Lattice points per :func:`~mesh_xml.render.field.field` call. Sized so the
#: transient coordinate array and the field's internal temporaries stay in a few
#: hundred megabytes; see the module docstring on why this is blocked at all.
SAMPLE_BLOCK = 1 << 21

#: How far a cell vertex is held back from its own cell's faces, as a fraction
#: of the cell. Crossings land exactly on a corner when the field is exactly
#: zero at a lattice point, and an unclamped average of those can coincide with
#: a neighbouring cell's vertex, which turns two triangles into slivers with no
#: area and no normal. The clamp separates any two cells' vertices by at least
#: ``2 * _CELL_MARGIN`` of a cell, at a positional cost far below the sampling
#: error the vertex already carries.
_CELL_MARGIN = 1e-6

#: The four cells around a lattice edge, as offsets in the two axes the edge is
#: not aligned with, wound counter-clockwise seen from the positive edge axis.
#: With ``u = (axis + 1) % 3`` and ``v = (axis + 2) % 3`` the frame ``(u, v,
#: axis)`` is right-handed, so this winding is what makes a face normal point
#: the way the field increases.
_QUAD_CELLS = ((-1, -1), (0, -1), (0, 0), (-1, 0))


@dataclass(frozen=True)
class Mesh:
    """An indexed triangle mesh with one exact field normal per vertex.

    ``faces`` is ``(F, 3) int32`` into ``vertices``, wound counter-clockwise
    seen from outside. ``cell`` is the lattice spacing that produced it, which
    is the scale of every error in the result: no vertex is further than one
    cell diagonal from the true surface.

    ``open_edges`` is the number of faces the surface needed and could not be
    given because it left the sampled box — see the module docstring. It is zero
    on any body :func:`~mesh_xml.render.field.bounds` bounded correctly.
    """

    vertices: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    cell: float
    open_edges: int

    @property
    def closed(self) -> bool:
        """Whether the surface came out with no boundary edges."""
        return self.open_edges == 0

    def __len__(self) -> int:
        return len(self.faces)


def polygonise(
    primitives: np.ndarray,
    blend: float = 0.0,
    resolution: int = 64,
    groups: np.ndarray | None = None,
) -> Mesh:
    """Surface-net the blended field of a ``(P, 8)`` primitive array.

    ``resolution`` is cells along the longest axis of the sampled box. Cells are
    cubes, so the other two axes get however many they need to stay square — a
    long thin body costs proportionally less than its bounding box suggests.

    ``groups`` is the blend-group ranges the field respects — smooth minimum
    within a creature, hard minimum across (:func:`mesh_xml.render.field.field`)
    — so an exported herd keeps the same separate skins the viewport draws
    instead of welding at every contact. The hard minimum of exact fields is
    the exact union field, so the surface nets see one well-signed function
    and the mesh stays as closed as the ungrouped case.

    An empty primitive array gives an empty mesh rather than raising. The union
    of nothing has no surface, and :func:`~mesh_xml.render.field.field` answers
    ``+inf`` everywhere for it, which has no sign change to polygonise; a caller
    exporting a body mid-edit should get an empty file, not an exception. A body
    whose bounding box has no extent at all — every radius zero — is the same
    case and answers the same way.
    """
    prims = np.asarray(primitives, dtype=np.float64)
    if prims.ndim != 2 or prims.shape[1] != PRIMITIVE_STRIDE:
        raise ValueError(
            f"primitives must have shape (P, {PRIMITIVE_STRIDE}), got {prims.shape}"
        )
    if resolution < 1:
        raise ValueError(f"resolution must be at least 1, got {resolution}")

    lo, hi = bounds(prims, blend)
    extent = hi - lo
    span = float(extent.max())
    if len(prims) == 0 or span <= 0.0:
        return _empty_mesh(0.0)

    cell = span / resolution
    counts = np.ceil(extent / cell).astype(np.int64) + 2 * PADDING_CELLS
    origin = 0.5 * (lo + hi) - 0.5 * counts * cell

    values = _sample(prims, blend, origin, cell, counts, groups)
    inside = values < 0.0

    local, hits = _cell_vertices(values, inside)
    active = hits > 0
    if not active.any():
        return _empty_mesh(cell)

    index = np.stack(np.nonzero(active), axis=1)
    vertices = origin + (index + local[active]) * cell

    vertex_id = np.full(tuple(counts), -1, dtype=np.int32)
    vertex_id[active] = np.arange(len(vertices), dtype=np.int32)

    faces, open_edges = _faces(inside, vertex_id, counts)
    return Mesh(
        vertices=vertices,
        normals=normal(vertices, prims, blend, groups=groups),
        faces=faces,
        cell=cell,
        open_edges=open_edges,
    )


# ----------------------------------------------------------------------
# sampling
# ----------------------------------------------------------------------


def _sample(
    prims: np.ndarray,
    blend: float,
    origin: np.ndarray,
    cell: float,
    counts: np.ndarray,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Field values on the ``counts + 1`` lattice, a block of points at a time.

    Coordinates are unravelled from the flat index rather than meshgridded, so
    the largest array alive is one block wide and not one lattice wide.
    """
    shape = tuple(counts + 1)
    axes = [origin[d] + np.arange(shape[d]) * cell for d in range(3)]

    values = np.empty(shape)
    flat = values.reshape(-1)
    points = np.empty((min(SAMPLE_BLOCK, flat.size), 3))

    for start in range(0, flat.size, SAMPLE_BLOCK):
        stop = min(start + SAMPLE_BLOCK, flat.size)
        block = points[: stop - start]
        for d, coordinate in enumerate(np.unravel_index(np.arange(start, stop), shape)):
            block[:, d] = axes[d][coordinate]
        flat[start:stop] = field(block, prims, blend, groups)
    return values


def _cell_vertices(
    values: np.ndarray, inside: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell vertex position in cell-local ``[0, 1]`` coordinates, and hit count.

    The position is the mean of the linearly interpolated crossings on the
    cell's twelve edges, which is what makes a surface net follow a plane
    exactly and a curve to second order. A cell straddles the surface exactly
    when its count is non-zero: the cube's corners form a connected graph, so
    corners of differing sign are always joined by some edge of it.
    """
    cells = tuple(n - 1 for n in values.shape)
    total = np.zeros(cells + (3,))
    hits = np.zeros(cells, dtype=np.int64)

    for axis in range(3):
        bit = 1 << axis
        for c in range(8):
            if c & bit:
                continue
            at0, at1 = _corner_slice(c, cells), _corner_slice(c | bit, cells)
            v0, v1 = values[at0], values[at1]
            in0, in1 = inside[at0], inside[at1]
            crossing = in0 != in1
            fraction = np.where(crossing, v0 / np.where(crossing, v0 - v1, 1.0), 0.0)

            for d in range(3):
                if c & (1 << d):
                    total[..., d] += crossing
            total[..., axis] += fraction
            hits += crossing

    local = total / np.maximum(hits, 1)[..., None]
    return np.clip(local, _CELL_MARGIN, 1.0 - _CELL_MARGIN), hits


def _corner_slice(c: int, cells: tuple[int, ...]) -> tuple[slice, ...]:
    """Every cell's corner ``c``, where ``c``'s bits are its offsets in x, y, z."""
    return tuple(slice((c >> d) & 1, ((c >> d) & 1) + cells[d]) for d in range(3))


# ----------------------------------------------------------------------
# topology
# ----------------------------------------------------------------------


def _faces(
    inside: np.ndarray, vertex_id: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, int]:
    """Triangles for every lattice edge that changes sign and has four cells.

    Returns the faces and the number of sign changes that had fewer than four,
    which is the number of faces the surface is short by.
    """
    cells = tuple(counts)
    quads = []
    open_edges = 0

    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        in0 = inside[_edge_slice(axis, u, v, cells, 0)]
        in1 = inside[_edge_slice(axis, u, v, cells, 1)]
        crossing = in0 != in1

        whole = np.count_nonzero(
            inside[_whole_edge_slice(axis, cells, 0)]
            != inside[_whole_edge_slice(axis, cells, 1)]
        )
        open_edges += whole - int(np.count_nonzero(crossing))
        if not crossing.any():
            continue

        corners = [
            vertex_id[_cell_slice(axis, u, v, cells, du, dv)][crossing]
            for du, dv in _QUAD_CELLS
        ]
        quad = np.stack(corners, axis=1)
        forward = in0[crossing]
        quads.append(np.where(forward[:, None], quad, quad[:, ::-1]))

    if not quads:
        return np.zeros((0, 3), dtype=np.int32), open_edges

    quad = np.concatenate(quads)
    faces = np.concatenate([quad[:, (0, 1, 2)], quad[:, (0, 2, 3)]])
    return faces.astype(np.int32, copy=False), open_edges


def _edge_slice(
    axis: int, u: int, v: int, cells: tuple[int, ...], end: int
) -> tuple[slice, ...]:
    """One end of every lattice edge along ``axis`` that has four cells around it."""
    out = [slice(None)] * 3
    out[axis] = slice(end, cells[axis] + end)
    out[u] = slice(1, cells[u])
    out[v] = slice(1, cells[v])
    return tuple(out)


def _whole_edge_slice(axis: int, cells: tuple[int, ...], end: int) -> tuple[slice, ...]:
    """One end of every lattice edge along ``axis``, four cells or not."""
    out = [slice(None)] * 3
    out[axis] = slice(end, cells[axis] + end)
    return tuple(out)


def _cell_slice(
    axis: int, u: int, v: int, cells: tuple[int, ...], du: int, dv: int
) -> tuple[slice, ...]:
    """One of the four cells around every edge of :func:`_edge_slice`."""
    out = [slice(None)] * 3
    out[axis] = slice(0, cells[axis])
    out[u] = slice(1 + du, cells[u] + du)
    out[v] = slice(1 + dv, cells[v] + dv)
    return tuple(out)


def _empty_mesh(cell: float) -> Mesh:
    return Mesh(
        vertices=np.zeros((0, 3)),
        normals=np.zeros((0, 3)),
        faces=np.zeros((0, 3), dtype=np.int32),
        cell=cell,
        open_edges=0,
    )
