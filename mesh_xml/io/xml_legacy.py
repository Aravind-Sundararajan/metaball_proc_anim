"""Reader for the two pre-versioned ``<armature>`` XML eras.

The v1 exporter changed a child ``position`` from world space to an offset from
its parent and shipped no version marker, so both eras are byte-indistinguishable
at the syntax level and both are present in this repository (refactor plan §2.3).
Reading an absolute-era file with today's relative rules accumulates the offsets:
``tests/data/legacy_v1_absolute_spline.xml`` puts its last sphere at ``x = 20.0``
instead of ``x ≈ 4.444``. That is the regression this module exists to stop.

The auto-detect heuristic
-------------------------
``coords="auto"`` resolves the file twice — once treating child positions as
world space, once as parent offsets — and compares the coefficient of variation
of the resulting parent-to-child distances.

**Relative is the default and absolute must be argued for.** Absolute is chosen
only when the absolute reading yields a chain of near-*identical* segment lengths
(``cv <= 0.05``) that the relative reading would smear out (``cv >= 0.25``). That
asymmetry is deliberate: the absolute era was in practice the spline generator's
era, and evenly spaced spheres along a curve are its fingerprint. A symmetric
"whichever reading looks tidier wins" rule gets ``tests/data/test_export.xml``
wrong, because hand-scattered spheres score slightly tidier read as absolute
(cv 0.137) than as relative (cv 0.209) purely by accident.

Measured on the repository's fixtures:

===================================== ======  ======  =========
file                                  cv_abs  cv_rel  verdict
===================================== ======  ======  =========
legacy_v1_absolute_spline.xml         0.0009  0.400   absolute
chubby_worm.xml                       2.828   0.000   relative
character_armature.xml                2.236   0.000   relative
test_export.xml                       0.137   0.209   relative
test_export_multiple_apendages.xml    0.288   0.236   relative
branching_armature.xml                  n/a     n/a   relative
===================================== ======  ======  =========

How it can be wrong
-------------------
* **An absolute-era file that is not evenly spaced reads as relative.** This is
  the likely failure and it is silent: hand-placed spheres from the absolute era
  have no fingerprint to find. Nothing in the bytes distinguishes them.
* **A relative-era file whose raw offsets happen to be evenly spaced points reads
  as absolute.** Contrived, but a rig whose offsets were snapped to a uniform
  lattice would do it.
* **Files with fewer than two parent-child edges are undecidable** and default to
  relative. For a single-level rig with its root at the origin both readings are
  identical anyway, so the answer does not matter.
* **The test is global.** A file mixing an evenly spaced chain with a scattered
  one is scored as a whole and may go either way.

The heuristic is therefore a convenience, never an authority. Pass
``coords="absolute"`` or ``coords="relative"`` to state the era outright — that
path does no detection at all — and use :func:`detect_coords` to see what auto
would have decided before committing to it. :attr:`LegacyImport.coords` always
reports the mode actually used.

What v1 carried that :class:`Body` cannot hold
----------------------------------------------
``<mesh type=... height=... material=...>`` plus per-bone ``rotation``/``scale``
have nowhere to live on a chunk. The old importer simply dropped them, so every
capsule in ``character_armature.xml`` came back a sphere (defect §2.4). Here they
survive as :class:`LegacyChunkMeta` in :attr:`LegacyImport.meta`, keyed by chunk
id, and callers that only want geometry can keep using :func:`load_legacy`.

A missing ``<mesh>`` is likewise no longer papered over with a silent
``radius = 0.5``. ``default_radius`` is the caller's choice, every chunk that
needed it is listed in :attr:`LegacyImport.missing_mesh`, and passing
``default_radius=None`` refuses the file instead.

Identity in v1 was the bone name
--------------------------------
The old model keyed a dict on the name, so repeated names *were* the same bone,
and the exporter emitted a multiply-parented bone once under each parent. This
reader honours that: bones are coalesced by name, first occurrence winning on
position, and merged names are reported in :attr:`LegacyImport.merged_names`.
Coalescing is also what reproduces the reversed-duplicate-edge corruption of
defect §2.1 — ``A`` parenting ``B`` in one subtree and ``B`` parenting ``A`` in
another — which :meth:`Body.add_link` deduplicates into a single edge.

The constraint is honoured only when *reading* v1. Bodies produced here impose no
uniqueness on names going forward; that was defect §2.6.
"""

from __future__ import annotations

import math
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mesh_xml.core.body import Body

__all__ = [
    "CoordMode",
    "LegacyFormatError",
    "LegacyChunkMeta",
    "LegacyImport",
    "detect_coords",
    "read_legacy",
    "load_legacy",
]

CoordMode = Literal["auto", "absolute", "relative"]

_UNIFORM_CV = 0.05
_SCATTER_CV = 0.25
_LEGACY_DIM = 3


class LegacyFormatError(ValueError):
    """A document is not a readable v1 ``<armature>``."""


@dataclass(frozen=True)
class LegacyChunkMeta:
    """v1 per-bone attributes that :class:`Body` has no field for.

    ``mesh_type``, ``height`` and ``material`` come from the ``<mesh>`` child;
    ``rotation`` and ``scale`` from the ``<bone>`` element. ``has_mesh`` is
    ``False`` when the bone had no ``<mesh>`` child at all, and
    ``radius_defaulted`` is ``True`` whenever the radius came from the caller's
    ``default_radius`` rather than the file — either because ``<mesh>`` was
    absent, or because it carried no usable positive ``radius``.
    """

    mesh_type: str | None = None
    height: float | None = None
    material: str | None = None
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    has_mesh: bool = True
    radius_defaulted: bool = False


@dataclass(frozen=True)
class LegacyImport:
    """A v1 document, split into the part :class:`Body` models and the part it does not."""

    body: Body
    meta: dict[int, LegacyChunkMeta]
    coords: Literal["absolute", "relative"]
    missing_mesh: tuple[int, ...] = ()
    merged_names: tuple[str, ...] = ()


def load_legacy(
    path: str | Path,
    coords: CoordMode = "auto",
    *,
    default_radius: float | None = 0.5,
) -> Body:
    """Read a v1 ``<armature>`` document and return only its :class:`Body`.

    Use :func:`read_legacy` instead when the mesh metadata matters; discarding it
    here is the caller's explicit choice rather than the parser's, which is what
    defect §2.4 got wrong.
    """
    return read_legacy(path, coords, default_radius=default_radius).body


def read_legacy(
    path: str | Path,
    coords: CoordMode = "auto",
    *,
    default_radius: float | None = 0.5,
) -> LegacyImport:
    """Read a v1 ``<armature>`` document, metadata included.

    ``coords`` selects the coordinate era; see the module docstring for what
    ``"auto"`` can and cannot tell apart. ``default_radius`` supplies the radius
    for bones whose ``<mesh>`` is missing or unusable — pass ``None`` to raise
    :class:`LegacyFormatError` on them instead of substituting anything.
    """
    if coords not in ("auto", "absolute", "relative"):
        raise ValueError(f"coords must be auto, absolute or relative, got {coords!r}")

    path = Path(path)
    roots = _parse_document(path)
    resolved = _detect(roots) if coords == "auto" else coords
    _resolve_world(roots, relative=resolved == "relative")

    body = Body(dim=_LEGACY_DIM, name=path.stem)
    meta: dict[int, LegacyChunkMeta] = {}
    missing_mesh: list[int] = []
    merged_names: list[str] = []
    by_name: dict[str, int] = {}
    chunk_of_bone: dict[int, int] = {}

    for bone, parent in _walk(roots):
        chunk_id = by_name.get(bone.name)
        if chunk_id is None:
            chunk_id = _add_chunk(body, bone, path, default_radius)
            by_name[bone.name] = chunk_id
            meta[chunk_id] = LegacyChunkMeta(
                mesh_type=bone.mesh_type,
                height=bone.height,
                material=bone.material,
                rotation=bone.rotation,
                scale=bone.scale,
                has_mesh=bone.has_mesh,
                radius_defaulted=not _usable_radius(bone.radius),
            )
            if not bone.has_mesh:
                missing_mesh.append(chunk_id)
        elif bone.name not in merged_names:
            merged_names.append(bone.name)

        chunk_of_bone[id(bone)] = chunk_id
        if parent is not None:
            parent_id = chunk_of_bone[id(parent)]
            if parent_id != chunk_id:
                body.add_link(parent_id, chunk_id)

    return LegacyImport(
        body=body,
        meta=meta,
        coords=resolved,
        missing_mesh=tuple(missing_mesh),
        merged_names=tuple(merged_names),
    )


def detect_coords(path: str | Path) -> Literal["absolute", "relative"]:
    """Report which era ``"auto"`` would pick for ``path``, without building a body."""
    return _detect(_parse_document(Path(path)))


# ----------------------------------------------------------------------
# document -> bone tree
# ----------------------------------------------------------------------


@dataclass
class _Bone:
    name: str
    raw: tuple[float, float, float]
    rotation: tuple[float, float, float]
    scale: tuple[float, float, float]
    mesh_type: str | None
    radius: float | None
    height: float | None
    material: str | None
    has_mesh: bool
    children: list["_Bone"] = field(default_factory=list)
    world: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _parse_document(path: Path) -> list[_Bone]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise LegacyFormatError(f"{path} is not well-formed XML: {exc}") from None

    if root.tag == "body":
        raise LegacyFormatError(
            f"{path} is a versioned <body> document; read it with mesh_xml.io.xml_v2.load"
        )
    if root.tag != "armature":
        raise LegacyFormatError(f"expected root element <armature>, got <{root.tag}>")

    roots: list[_Bone] = []
    stack: list[tuple[ET.Element, _Bone | None]] = [
        (el, None) for el in reversed(root.findall("bone"))
    ]
    while stack:
        el, parent = stack.pop()
        bone = _bone_from_element(el)
        if parent is None:
            roots.append(bone)
        else:
            parent.children.append(bone)
        stack.extend((child, bone) for child in reversed(el.findall("bone")))
    return roots


def _bone_from_element(el: ET.Element) -> _Bone:
    mesh = el.find("mesh")
    return _Bone(
        name=el.get("name", "Unknown"),
        raw=_vec3(el.get("position"), (0.0, 0.0, 0.0), "position"),
        rotation=_vec3(el.get("rotation"), (0.0, 0.0, 0.0), "rotation"),
        scale=_vec3(el.get("scale"), (1.0, 1.0, 1.0), "scale"),
        mesh_type=None if mesh is None else mesh.get("type"),
        radius=None if mesh is None else _opt_float(mesh.get("radius")),
        height=None if mesh is None else _opt_float(mesh.get("height")),
        material=None if mesh is None else mesh.get("material"),
        has_mesh=mesh is not None,
    )


def _vec3(
    raw: str | None, default: tuple[float, float, float], attr: str
) -> tuple[float, float, float]:
    if raw is None:
        return default
    parts = [p for p in raw.replace(",", " ").split() if p]
    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise LegacyFormatError(f"{attr}={raw!r} is not a vector") from None
    if len(values) != _LEGACY_DIM:
        raise LegacyFormatError(
            f"{attr}={raw!r} has {len(values)} components, expected {_LEGACY_DIM}"
        )
    return (values[0], values[1], values[2])


def _opt_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


# ----------------------------------------------------------------------
# coordinate era
# ----------------------------------------------------------------------


def _resolve_world(roots: list[_Bone], relative: bool) -> None:
    """Fill in ``world`` for every bone. Roots are world space in both eras."""
    for root in roots:
        root.world = root.raw
        stack = [root]
        while stack:
            node = stack.pop()
            for child in node.children:
                child.world = (
                    tuple(p + c for p, c in zip(node.world, child.raw))
                    if relative
                    else child.raw
                )
                stack.append(child)


def _segment_lengths(roots: list[_Bone], relative: bool) -> list[float]:
    _resolve_world(roots, relative)
    return [
        math.dist(parent.world, child.world)
        for parent in _all_bones(roots)
        for child in parent.children
    ]


def _detect(roots: list[_Bone]) -> Literal["absolute", "relative"]:
    absolute_cv = _cv(_segment_lengths(roots, relative=False))
    relative_cv = _cv(_segment_lengths(roots, relative=True))
    if absolute_cv is None or relative_cv is None:
        return "relative"
    if absolute_cv <= _UNIFORM_CV and relative_cv >= _SCATTER_CV:
        return "absolute"
    return "relative"


def _cv(lengths: list[float]) -> float | None:
    """Coefficient of variation, or ``None`` when the sample cannot discriminate.

    Fewer than two segments carries no signal, and a mean at zero means the
    reading collapsed every chunk onto one point — a reading that degenerate is
    not evidence of anything, so the caller falls back to the default era.
    """
    if len(lengths) < 2:
        return None
    mean = statistics.fmean(lengths)
    if mean < 1e-9:
        return None
    return statistics.pstdev(lengths) / mean


# ----------------------------------------------------------------------
# bone tree -> Body
# ----------------------------------------------------------------------


def _all_bones(roots: list[_Bone]) -> list[_Bone]:
    return [bone for bone, _ in _walk(roots)]


def _walk(roots: list[_Bone]) -> list[tuple[_Bone, _Bone | None]]:
    """Depth-first preorder, so chunk ids follow document order.

    Iterative rather than recursive because a spline export is one long chain and
    its depth is the sphere count.
    """
    out: list[tuple[_Bone, _Bone | None]] = []
    stack: list[tuple[_Bone, _Bone | None]] = [(b, None) for b in reversed(roots)]
    while stack:
        bone, parent = stack.pop()
        out.append((bone, parent))
        stack.extend((child, bone) for child in reversed(bone.children))
    return out


def _usable_radius(radius: float | None) -> bool:
    return radius is not None and radius > 0.0


def _add_chunk(
    body: Body, bone: _Bone, path: Path, default_radius: float | None
) -> int:
    radius = bone.radius
    if not _usable_radius(radius):
        if default_radius is None:
            reason = "has no <mesh>" if not bone.has_mesh else "has no usable radius"
            raise LegacyFormatError(
                f"{path}: bone {bone.name!r} {reason} and default_radius is None"
            )
        radius = default_radius
    return body.add_chunk(name=bone.name, pos=bone.world, radius=radius)
