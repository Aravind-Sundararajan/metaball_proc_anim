"""Writing triangle geometry to files other tools can open.

The mesher is not imported here and must not be. A writer that knew about
:class:`~mesh_xml.render.meshing.Mesh` would only be usable on meshes this
project produced, and the useful cases are the other ones — geometry loaded
from elsewhere, decimated by something else, or assembled by hand in a test. So
the interface is three arrays and nothing else, and the caller unpacks whatever
it has into them.

That also means these functions validate rather than trust. A face index outside
the vertex array is the one error worth catching at the door: OBJ has no
checksum and no bounds, so most loaders answer a bad index with a silently
misplaced triangle or a crash a long way from the cause.

OBJ first, STL second
---------------------
OBJ is the format because it is the only trivially writable one that carries
**vertex** normals. That is the whole point of the exercise here — the normals
:mod:`~mesh_xml.render.meshing` produces come from the field itself and are
exact, and a format that dropped them would throw away the reason the mesh looks
smooth at a resolution where its faces are visibly large.

STL is written too because it is fifteen lines and it is what print slicers
want, but it carries per-*facet* normals and no indexing, so the exact vertex
normals are lost and every vertex is repeated three times. The facet normal
stored is the geometric one from the winding rather than an average of the
vertex normals, because that is what the format means by the field and what a
slicer checks the winding against. Use it to print, not to round-trip.

Both are written binary-exact and locale-independent: OBJ through numpy's
formatter with an explicit ``%g`` precision, STL as little-endian regardless of
host, since the format is defined that way and no writer gets to choose.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["OBJ_PRECISION", "write_mesh", "write_obj", "write_stl"]

#: Significant digits per coordinate in OBJ. Nine round-trips float32 exactly
#: and is within a part in ``1e9`` of the float64 the mesher works in, which is
#: several orders below the sampling error any polygonisation carries. More
#: digits only makes the file bigger.
OBJ_PRECISION = 9

#: One binary STL facet: normal, three corners, and the attribute short that the
#: format requires and nothing reads. Declared unaligned so ``itemsize`` is the
#: 50 bytes the format specifies rather than a padded 52.
_STL_FACET = np.dtype(
    [("normal", "<f4", (3,)), ("corners", "<f4", (3, 3)), ("attribute", "<u2")]
)


def write_obj(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray | None = None,
) -> Path:
    """Write Wavefront OBJ. Returns the path written.

    ``vertices`` is ``(V, 3)``, ``faces`` is ``(F, 3)`` zero-based indices into
    it, and ``normals``, when given, is one per vertex. Faces are emitted as
    ``f a//na`` when there are normals and ``f a b c`` when there are not —
    never ``a/vt/vn`` with an empty texture slot, which older loaders read as a
    missing coordinate rather than an omitted one.

    OBJ indices are one-based, which is the single most common way a hand-rolled
    writer produces a mesh that opens with one corner of every triangle wrong.
    """
    v, f, n = _geometry(vertices, faces, normals)
    out = Path(path)

    number = f"%.{OBJ_PRECISION}g"
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# {len(v)} vertices, {len(f)} faces\n")
        if len(v):
            np.savetxt(handle, v, fmt=f"v {number} {number} {number}")
        if n is not None and len(n):
            np.savetxt(handle, n, fmt=f"vn {number} {number} {number}")
        if len(f):
            if n is None:
                np.savetxt(handle, f + 1, fmt="f %d %d %d")
            else:
                index = np.repeat(f + 1, 2, axis=1)
                np.savetxt(handle, index, fmt="f %d//%d %d//%d %d//%d")
    return out


def write_stl(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray | None = None,
) -> Path:
    """Write binary STL. Returns the path written.

    ``normals`` is accepted for signature parity with :func:`write_obj` and
    ignored: the format has nowhere to put a vertex normal, and the facet normal
    it does have is derived from the winding here rather than from an average
    that could contradict it. A facet with no area gets the zero normal, which
    is the format's own convention for "work it out yourself".

    Binary rather than ASCII: an ASCII STL of the same mesh is roughly five
    times the size and parses slower, and nothing gains from being able to read
    it. The header must therefore not begin with ``solid`` — the two formats are
    told apart by that word and nothing else, and a binary file that starts with
    it is read as ASCII and rejected as malformed.
    """
    v, f, _ = _geometry(vertices, faces, normals)
    out = Path(path)

    facets = np.zeros(len(f), dtype=_STL_FACET)
    if len(f):
        corner = v[f]
        edge = corner[:, 1:] - corner[:, :1]
        cross = np.cross(edge[:, 0], edge[:, 1])
        length = np.linalg.norm(cross, axis=1)
        facets["corners"] = corner
        facets["normal"] = cross / np.where(length > 0.0, length, np.inf)[:, None]

    with out.open("wb") as handle:
        handle.write(b"mesh_xml binary STL".ljust(80, b"\0"))
        handle.write(np.array(len(f), dtype="<u4").tobytes())
        handle.write(facets.tobytes())
    return out


def write_mesh(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray | None = None,
) -> Path:
    """Dispatch to :func:`write_obj` or :func:`write_stl` on the suffix."""
    out = Path(path)
    writer = {".obj": write_obj, ".stl": write_stl}.get(out.suffix.lower())
    if writer is None:
        raise ValueError(f"no writer for '{out.suffix}'; expected .obj or .stl")
    return writer(out, vertices, faces, normals)


def _geometry(
    vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Coerce and check the three arrays, so a malformed mesh fails here."""
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must have shape (V, 3), got {v.shape}")

    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3), got {f.shape}")
    if len(f) and (f.min() < 0 or f.max() >= len(v)):
        raise ValueError(
            f"face indices must be in [0, {len(v)}), got [{f.min()}, {f.max()}]"
        )

    if normals is None:
        return v, f, None
    n = np.asarray(normals, dtype=np.float64)
    if n.shape != v.shape:
        raise ValueError(
            f"normals must have one per vertex, got {n.shape} for {v.shape}"
        )
    return v, f, n
