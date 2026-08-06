"""The world as a signed distance field, in the creature's own vocabulary.

``docs/TERRAIN.md`` records why the world is a field rather than a heightfield
or a mesh; the short version is that the creature already is one, so collision
is one subtraction, contact normals are the gradient, and a foothold is a
projection. The vocabulary is deliberately small: infinite half-space planes
plus the same stride-8 round cones bodies are packed into. Ramps are tilted
buried capsules, steps are fat ones, and a plane blending into a ramp gains a
fillet that reads as landscape.

Two folding decisions, both inherited from :mod:`mesh_xml.render.field` and
kept for the same reasons:

* **Order matters and is part of the terrain.** The smooth minimum is not
  associative, so planes fold first (in authoring order), then cones (in
  authoring order). Reordering primitives is a visible change to the ground.
* **The fold is seeded with the first primitive, never a sentinel** — a
  sentinel blends and donates inflation exactly where the scene is larger than
  the caller guessed.

Blending is *within* the terrain only. Compositing against a creature is the
consumer's job and must be a hard minimum: a foot rests on the ground, it does
not smooth into it and become it.

A 2-D body lives on the ``z = 0`` plane, so points with a trailing axis of two
are widened there and answers are sliced back. That is only meaningful for
terrain that is itself symmetric about that plane — a caller pairing a 2-D
body with z-varying ground gets the slice through ``z = 0``, which is the
question it asked, whether or not it is the question it meant.
"""

from __future__ import annotations

import numpy as np

from mesh_xml.core.distance import (
    TETRAHEDRON,
    as_blend,
    round_cone,
    smooth_min,
)

__all__ = ["Terrain"]

#: Central-difference offset for :meth:`Terrain.normal`, matching the field's.
NORMAL_EPSILON = 1e-4

#: Newton-style projection passes in :meth:`Terrain.project`. The field is
#: 1-Lipschitz, so each pass closes most of the remaining gap; three passes
#: land within a millimetre of the surface at world scales around one unit.
_PROJECT_PASSES = 3


class Terrain:
    """Half-space planes and round cones, folded with one blend radius.

    Mutating methods return ``self`` so a world reads as a sentence::

        Terrain.flat(0.0).add_cone((2, -0.4, 0), 0.9, (4, 0.6, 0), 0.9)
    """

    def __init__(self, blend: float = 0.3) -> None:
        self.blend = as_blend(blend)
        self._planes = np.zeros((0, 4), dtype=np.float64)
        self._cones = np.zeros((0, 8), dtype=np.float64)

    # ------------------------------------------------------------------
    # authoring
    # ------------------------------------------------------------------

    @classmethod
    def flat(cls, y: float = 0.0, blend: float = 0.3) -> "Terrain":
        """The old ``ground_y`` world, spelled in the new vocabulary."""
        return cls(blend=blend).add_plane((0.0, 1.0, 0.0), y)

    def add_plane(self, normal, offset: float) -> "Terrain":
        """Add the half-space ``dot(n, p) - offset <= 0`` is inside of.

        ``normal`` points out of the ground and is normalised here, so the
        distance is metric whatever length the caller wrote. ``offset`` is the
        plane's position along the *unit* normal — a floor at height ``y`` is
        ``add_plane((0, 1, 0), y)``, however the normal was scaled.
        """
        n = np.asarray(normal, dtype=np.float64).reshape(3)
        length = float(np.linalg.norm(n))
        if length < 1e-12:
            raise ValueError("plane normal must be non-zero")
        row = np.concatenate([n / length, [float(offset)]])[None, :]
        self._planes = np.vstack([self._planes, row])
        return self

    def add_cone(self, a, ra: float, b, rb: float) -> "Terrain":
        """Add one round cone, the same record bodies pack into."""
        record = np.zeros((1, 8), dtype=np.float64)
        record[0, 0:3] = np.asarray(a, dtype=np.float64).reshape(3)
        record[0, 3] = float(ra)
        record[0, 4:7] = np.asarray(b, dtype=np.float64).reshape(3)
        record[0, 7] = float(rb)
        if record[0, 3] < 0.0 or record[0, 7] < 0.0:
            raise ValueError("cone radii must be non-negative")
        self._cones = np.vstack([self._cones, record])
        return self

    # ------------------------------------------------------------------
    # read access — what serializers and uploaders consume
    # ------------------------------------------------------------------

    @property
    def planes(self) -> np.ndarray:
        """``(P, 4) float64`` rows of ``[nx, ny, nz, offset]``, unit normals."""
        return self._planes

    @property
    def cones(self) -> np.ndarray:
        """``(C, 8) float64`` stride-8 round cone records."""
        return self._cones

    @property
    def is_empty(self) -> bool:
        return len(self._planes) == 0 and len(self._cones) == 0

    # ------------------------------------------------------------------
    # the field
    # ------------------------------------------------------------------

    def distance(self, points) -> np.ndarray:
        """Signed distance to the terrain surface. Empty terrain is ``+inf`` away."""
        p, squeeze = self._widen(points)
        flat = p.reshape(-1, 3)

        d: np.ndarray | None = None
        for plane in self._planes:
            plane_d = flat @ plane[:3] - plane[3]
            d = plane_d if d is None else smooth_min(d, plane_d, self.blend)
        for cone in self._cones:
            cone_d = round_cone(flat, cone)
            d = cone_d if d is None else smooth_min(d, cone_d, self.blend)

        if d is None:
            d = np.full(len(flat), np.inf)
        return d.reshape(p.shape[:-1])

    def normal(self, points, eps: float = NORMAL_EPSILON) -> np.ndarray:
        """Unit outward normals by tetrahedron central differences.

        Zero where the gradient vanishes — deep inside a blend pocket — rather
        than NaN, so a consumer must treat a zero normal as "no direction",
        exactly as :func:`mesh_xml.render.field.normal` behaves.
        """
        p, squeeze = self._widen(points)
        flat = p.reshape(-1, 3)
        if self.is_empty:
            out = np.zeros_like(flat)
        else:
            gradient = np.zeros_like(flat)
            for offset in TETRAHEDRON:
                gradient += offset * self.distance(flat + offset * eps)[:, None]
            length = np.linalg.norm(gradient, axis=1)
            scale = np.where(length > 0.0, length, 1.0)
            out = gradient / scale[:, None]
        out = out.reshape(p.shape)
        return out[..., :2] if squeeze else out

    def project(self, points, clearance: float = 0.0) -> np.ndarray:
        """The nearest point sitting ``clearance`` off the surface.

        Walks ``p -= (d(p) - clearance) * normal(p)`` a fixed number of passes.
        Works from either side: a point buried in the ground is carried out,
        one hanging in the air is brought down. This is what turns a desired
        foothold into an actual one — ``clearance`` is the foot's radius, for
        the same reason :meth:`ControlContext.ground_level` adds one: contact
        holds a chunk a radius off the surface, so aiming at the surface itself
        is aiming underground.

        Empty terrain returns the input unchanged; there is nothing to stand on
        and pretending otherwise would invent a floor.
        """
        p, squeeze = self._widen(points)
        if self.is_empty:
            out = p
        else:
            out = p.astype(np.float64, copy=True)
            flat = out.reshape(-1, 3)
            for _ in range(_PROJECT_PASSES):
                gap = self.distance(flat) - clearance
                flat -= gap[:, None] * self.normal(flat).reshape(-1, 3)
        return out[..., :2] if squeeze else out

    # ------------------------------------------------------------------

    def _widen(self, points) -> tuple[np.ndarray, bool]:
        p = np.asarray(points, dtype=np.float64)
        if p.ndim == 0 or p.shape[-1] not in (2, 3):
            raise ValueError(
                f"points must have a trailing axis of 2 or 3, got shape {p.shape}"
            )
        if p.shape[-1] == 2:
            widened = np.zeros(p.shape[:-1] + (3,), dtype=np.float64)
            widened[..., :2] = p
            return widened, True
        return p, False

    def __repr__(self) -> str:
        return (
            f"<Terrain planes={len(self._planes)} cones={len(self._cones)} "
            f"blend={self.blend}>"
        )
