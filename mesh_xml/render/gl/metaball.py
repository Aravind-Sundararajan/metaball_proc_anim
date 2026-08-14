"""Driving the metaball raymarcher: primitive upload and render settings.

The GPU half of :mod:`mesh_xml.render.field`. The shader in
``shaders/metaball.frag`` does the tracing; this module gets the primitives to
it and supplies the knobs.

Primitives go up as an RGBA32F texture, two texels per primitive, which is the
stride-8 record :func:`mesh_xml.render.sdf.pack` already produces — no repacking,
no per-frame copy beyond the upload itself. A texture rather than a uniform
array because a 3.3 core context has no shader storage buffers and a uniform
array would cap out in the low hundreds; rather than a texture buffer because
``sampler2D`` needs no extension and the row limit is at least 8192 everywhere.

The texture is reallocated only when the primitive count changes. Rig topology
changes when a chunk is added or removed, which is an editing action; positions
change every frame, and those reuse the allocation.

Terrain rides the same machinery. :func:`pack_terrain` lays a
:class:`~mesh_xml.core.terrain.Terrain` out as stride-8 rows — planes first,
padded to the cone record's width, then the cones verbatim — so
:class:`TerrainTexture` is a :class:`PrimitiveTexture` on its own unit with its
own uniform names and inherits the allocate/subimage path unchanged. The
padding costs 16 bytes per plane; a second texel-wide texture format would
save them in exchange for a second upload path to keep correct, on data
measured in tens of rows. Row order in the texture is the fold order
``Terrain.distance`` documents, which is what lets ``sd_terrain`` in
``shaders/sdf.glsl`` fold rows top to bottom and match the CPU exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_CLAMP_TO_EDGE,
    GL_FLOAT,
    GL_NEAREST,
    GL_RGBA,
    GL_RGBA32F,
    GL_RGBA32UI,
    GL_RGBA_INTEGER,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNSIGNED_INT,
    glActiveTexture,
    glBindTexture,
    glDeleteTextures,
    glGenTextures,
    glTexImage2D,
    glTexParameteri,
    glTexSubImage2D,
)

from mesh_xml.core.terrain import Terrain
from mesh_xml.render.look import Look
from mesh_xml.render.sdf import PRIMITIVE_STRIDE

__all__ = [
    "MetaballSettings",
    "PrimitiveTexture",
    "TerrainTexture",
    "GroupTexture",
    "BoundsTexture",
    "TileTexture",
    "fragment_source",
    "pack_terrain",
    "SHADER_NAME",
]

SHADER_NAME = "metaball.frag"

_SHADER_DIR = Path(__file__).parent / "shaders"

#: Texels per primitive. The record is two vec4s: (a, ra) and (b, rb).
TEXELS_PER_PRIMITIVE = PRIMITIVE_STRIDE // 4

#: Texture unit the primitive texture is bound to.
PRIMITIVE_TEXTURE_UNIT = 0

#: Texture unit the terrain texture is bound to. Its own unit so creature and
#: world can both stay bound across a frame.
TERRAIN_TEXTURE_UNIT = 1

#: Texture unit the blend-group table is bound to.
GROUP_TEXTURE_UNIT = 2

#: Texture unit the per-group bounding spheres are bound to.
BOUNDS_TEXTURE_UNIT = 3

#: Texture unit the per-tile group masks are bound to.
TILE_TEXTURE_UNIT = 4

#: Texture unit the per-primitive albedo table is bound to.
#:
#: 6, not 5, and the gap is load-bearing: unit 5 belongs to the cone
#: prepass's seed depths (``context.SEED_TEXTURE_UNIT``), which live in a
#: different module and bind on every prepass frame. Taking 5 here once
#: painted every editor creature the seed's R32F red — while every test
#: stayed green, because only the editor runs the prepass. The unit roster
#: spans *both* files; check context.py before claiming the next number.
ATTRIBUTE_TEXTURE_UNIT = 6


@dataclass
class MetaballSettings:
    """Everything the raymarch shader needs that is not the primitives themselves.

    ``blend`` is the smooth-minimum radius, and it is the one knob that decides
    whether a rig reads as a creature or as a string of beads. It is per-body
    authoring data, not a global constant.

    Note that ``blend`` interacts with authored radii: a chunk joined by two or
    more links reads about ``blend / 4`` fatter than its radius, because both of
    its cones contain its end sphere. See :mod:`mesh_xml.render.sdf`.
    """

    blend: float = 0.25
    max_distance: float = 100.0
    max_steps: int = 128
    surface_epsilon: float = 1e-4
    step_relaxation: float = 0.95
    normal_epsilon: float = 1e-4
    debug_mode: int = 0
    #: The authored look — key light, environment, palette base — composed
    #: whole rather than mirrored field by field, so the record the file
    #: round-trips and the record the frame renders are the same object and
    #: cannot drift (docs/LIGHTING.md §4). ``terrain_colour`` rides it too: a
    #: render preference like the other colours, which is why it is not on
    #: :class:`~mesh_xml.core.terrain.Terrain` — the terrain's *blend* is world
    #: data and travels with the upload instead (:class:`TerrainTexture`).
    look: Look = dataclass_field(default_factory=Look)
    #: Whether the editor draws its reference grid floor. Deliberately absent
    #: from :meth:`uniforms`: the shader's floor also needs a height and a
    #: terrain veto that only the document knows, so the viewport owns the
    #: ``u_floor_*`` uniforms and a standalone view (the test harness) never
    #: sets them — GL's zero default keeps the floor off and the CPU-oracle
    #: comparisons exact.
    show_floor: bool = True

    def uniforms(self) -> dict[str, object]:
        """Uniform values for :attr:`RaymarchView.uniforms`."""
        look = self.look
        return {
            "u_blend": float(self.blend),
            "u_max_distance": float(self.max_distance),
            "u_max_steps": int(self.max_steps),
            "u_surface_epsilon": float(self.surface_epsilon),
            "u_step_relaxation": float(self.step_relaxation),
            "u_normal_epsilon": float(self.normal_epsilon),
            "u_debug_mode": int(self.debug_mode),
            "u_light_direction": tuple(float(v) for v in look.light_direction),
            "u_light_colour": tuple(float(v) for v in look.light_colour),
            "u_light_strength": float(look.light_strength),
            "u_ambient_strength": float(look.ambient_strength),
            "u_ao_strength": float(look.ao_strength),
            # No arithmetic between parse and upload: the shader's == 1.0
            # endpoint branch must see exactly the 1.0 the file or the
            # default stated (docs/CAST_SHADOWS.md 4).
            "u_shadow_strength": float(look.shadow_strength),
            "u_glow_strength": float(look.glow_strength),
            "u_surface_colour": tuple(float(v) for v in look.surface_colour),
            "u_terrain_colour": tuple(float(v) for v in look.terrain_colour),
            "u_background_colour": tuple(float(v) for v in look.background_colour),
            "u_background_top": tuple(float(v) for v in look.background_top),
            "u_primitives": PRIMITIVE_TEXTURE_UNIT,
            # The integer sampler must be pointed off unit 0 even when no tile
            # texture exists: an unset usampler2D defaults to unit 0, where it
            # aliases the float u_primitives sampler, and GL 3.3 forbids two
            # differently-typed samplers on one unit as *draw-time program
            # state* — no fetch has to execute for a conformant driver to
            # refuse the draw. A sampler aimed at an unbound unit is legal.
            "u_tiles": TILE_TEXTURE_UNIT,
            # Float sampler, so unit 0 would be legal — pointed at its own
            # unit anyway, with the count that keeps the fold disarmed, so a
            # view that never uploads attributes states its whole contract.
            "u_attributes": ATTRIBUTE_TEXTURE_UNIT,
            "u_attribute_count": 0,
        }


@dataclass
class PrimitiveTexture:
    """An RGBA32F texture holding the packed primitives.

    Requires a current GL context for every method except construction.
    """

    handle: int = 0
    capacity: int = 0
    count: int = 0
    _unit: int = dataclass_field(default=PRIMITIVE_TEXTURE_UNIT, repr=False)

    def upload(self, primitives: np.ndarray) -> None:
        """Send a ``(P, 8) float32`` array, reallocating only if ``P`` changed."""
        data = np.ascontiguousarray(primitives, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != PRIMITIVE_STRIDE:
            raise ValueError(
                f"primitives must have shape (P, {PRIMITIVE_STRIDE}), got {data.shape}"
            )

        self.count = len(data)
        if self.handle == 0:
            self.handle = int(glGenTextures(1))
            self._configure()

        glActiveTexture(GL_TEXTURE0 + self._unit)
        glBindTexture(GL_TEXTURE_2D, self.handle)

        if self.count == 0:
            return

        if self.count != self.capacity:
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA32F,
                TEXELS_PER_PRIMITIVE,
                self.count,
                0,
                GL_RGBA,
                GL_FLOAT,
                data,
            )
            self.capacity = self.count
        else:
            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                TEXELS_PER_PRIMITIVE,
                self.count,
                GL_RGBA,
                GL_FLOAT,
                data,
            )

    def bind(self) -> None:
        glActiveTexture(GL_TEXTURE0 + self._unit)
        glBindTexture(GL_TEXTURE_2D, self.handle)

    def uniforms(self) -> dict[str, object]:
        return {
            "u_primitives": self._unit,
            "u_primitive_count": int(self.count),
        }

    def dispose(self) -> None:
        if self.handle:
            glDeleteTextures([self.handle])
            self.handle = 0
            self.capacity = 0
            self.count = 0

    def _configure(self) -> None:
        glActiveTexture(GL_TEXTURE0 + self._unit)
        glBindTexture(GL_TEXTURE_2D, self.handle)
        # NEAREST and CLAMP_TO_EDGE are not aesthetic choices. The texture is a
        # data buffer addressed by texelFetch; filtering or wrapping it would
        # interpolate between unrelated primitives.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)


def pack_terrain(terrain: Terrain | None) -> np.ndarray:
    """Lay a :class:`Terrain` out as the ``(P + C, 8) float32`` texture rows.

    Planes first, one row each, ``[nx, ny, nz, offset]`` padded with four
    zeros the shader never fetches; then the cone records verbatim. Row order
    is authoring order twice over — planes, then cones — because the row index
    is the fold index in ``sd_terrain`` and :meth:`Terrain.distance` documents
    exactly that order. Sorting or interleaving here would silently change the
    surface, not just the layout.

    ``None`` and an empty terrain both pack to zero rows, which is what keeps
    the shader's terrain path inert (count 0) when there is no world.
    """
    if terrain is None:
        return np.zeros((0, PRIMITIVE_STRIDE), dtype=np.float32)

    planes = np.asarray(terrain.planes, dtype=np.float32)
    cones = np.asarray(terrain.cones, dtype=np.float32)
    rows = np.zeros((len(planes) + len(cones), PRIMITIVE_STRIDE), dtype=np.float32)
    rows[: len(planes), :4] = planes
    rows[len(planes) :] = cones
    return rows


@dataclass
class TerrainTexture(PrimitiveTexture):
    """The world's half of the primitive upload, on its own texture unit.

    A :class:`PrimitiveTexture` because :func:`pack_terrain` deliberately emits
    the same stride-8 rows, so the whole allocate/subimage path is inherited
    rather than duplicated. What differs is bookkeeping: the shader needs the
    plane/cone split to know which rows are half-spaces, and the blend is the
    terrain's own (world data authored on the :class:`Terrain`, unlike the
    creature's blend which arrives via :class:`MetaballSettings`).
    """

    plane_count: int = 0
    cone_count: int = 0
    blend: float = 0.0
    _unit: int = dataclass_field(default=TERRAIN_TEXTURE_UNIT, repr=False)

    def upload_terrain(self, terrain: Terrain | None) -> None:
        """Send a world to the GPU, or clear it. Requires a current context.

        ``None`` uploads zero rows and zero counts, which renders identically
        to never having uploaded at all — the shader's count check never
        reaches the texture.
        """
        if terrain is None:
            self.plane_count = 0
            self.cone_count = 0
            self.blend = 0.0
        else:
            self.plane_count = len(terrain.planes)
            self.cone_count = len(terrain.cones)
            self.blend = float(terrain.blend)
        self.upload(pack_terrain(terrain))

    def uniforms(self) -> dict[str, object]:
        return {
            "u_terrain": self._unit,
            "u_terrain_plane_count": int(self.plane_count),
            "u_terrain_cone_count": int(self.cone_count),
            "u_terrain_blend": float(self.blend),
        }

    def dispose(self) -> None:
        super().dispose()
        # The counts are what the shader trusts; a disposed texture that still
        # reports rows would have sd_terrain fetch from an unbound unit.
        self.plane_count = 0
        self.cone_count = 0
        self.blend = 0.0


@dataclass
class GroupTexture(PrimitiveTexture):
    """The blend-group table, on its own unit.

    :func:`mesh_xml.render.sdf.group_table` deliberately emits the same
    stride-8 rows as the primitives — ``[start, count, 0, 0][r, g, b, 0]`` —
    so this is a :class:`PrimitiveTexture` with different uniform names and
    nothing else, the same trick :class:`TerrainTexture` pulls. A view that
    never uploads one leaves ``u_group_count`` at GL's zero default, and the
    shader's fold falls back to the single global chain bit-for-bit.
    """

    _unit: int = dataclass_field(default=GROUP_TEXTURE_UNIT, repr=False)

    def uniforms(self) -> dict[str, object]:
        return {
            "u_groups": self._unit,
            "u_group_count": int(self.count),
        }


@dataclass
class BoundsTexture(PrimitiveTexture):
    """Per-group bounding spheres, on their own unit.

    :func:`mesh_xml.render.sdf.group_bounds` rows — ``[cx, cy, cz, r]`` padded
    to stride 8 — re-uploaded every frame the creatures move, which the
    allocate-once/subimage path was built for. The shader arms its cull only
    when this count matches the group table's, so a stale or absent bounds
    texture degrades to folding everything rather than to a wrong image.
    """

    _unit: int = dataclass_field(default=BOUNDS_TEXTURE_UNIT, repr=False)

    def uniforms(self) -> dict[str, object]:
        return {
            "u_group_bounds": self._unit,
            "u_group_bounds_count": int(self.count),
        }


@dataclass
class AttributeTexture(PrimitiveTexture):
    """Per-primitive end colours, on their own unit — the weighted skin's data.

    :func:`mesh_xml.render.sdf.pack_attributes` emits the same stride-8 rows
    as the primitives — ``[ra_r, ra_g, ra_b, 0][rb_r, rb_g, rb_b, 0]`` — so
    this is the group table's trick again: a :class:`PrimitiveTexture` with
    different uniform names. The count doubles as the arming switch: the
    shader folds colour only when it equals the primitive count, so the
    empty table a uniform body packs to leaves the flat path running
    bit-for-bit (``docs/TEXTURES.md``).
    """

    _unit: int = dataclass_field(default=ATTRIBUTE_TEXTURE_UNIT, repr=False)

    def uniforms(self) -> dict[str, object]:
        return {
            "u_attributes": self._unit,
            "u_attribute_count": int(self.count),
        }


@dataclass
class TileTexture:
    """Per-tile group bitmasks as an ``RGBA32UI`` texture — the broad phase.

    Not a :class:`PrimitiveTexture`: the payload is integer bitmasks, not
    float records, so it needs the integer internal format and the integer
    pixel-transfer format, and a ``usampler2D`` on the shader side. The
    allocate-once/subimage discipline is the same — the grid reallocates only
    when the framebuffer's tile dimensions change, which is a resize, not a
    frame.

    ``upload(None)`` clears: the shader's ``u_tiles_on`` goes to zero and the
    trace falls back to the untiled fold, which is always correct.
    """

    handle: int = 0
    shape: tuple[int, int] = (0, 0)
    _unit: int = dataclass_field(default=TILE_TEXTURE_UNIT, repr=False)

    def upload(self, masks: np.ndarray | None) -> None:
        if masks is None:
            self.shape = (0, 0)
            return
        data = np.ascontiguousarray(masks, dtype=np.uint32)
        if data.ndim != 3 or data.shape[2] != 4:
            raise ValueError(f"masks must have shape (rows, cols, 4), got {data.shape}")

        if self.handle == 0:
            self.handle = int(glGenTextures(1))
            glActiveTexture(GL_TEXTURE0 + self._unit)
            glBindTexture(GL_TEXTURE_2D, self.handle)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        rows, cols = data.shape[0], data.shape[1]
        glActiveTexture(GL_TEXTURE0 + self._unit)
        glBindTexture(GL_TEXTURE_2D, self.handle)
        if (rows, cols) != self.shape:
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGBA32UI, cols, rows, 0,
                GL_RGBA_INTEGER, GL_UNSIGNED_INT, data,
            )
            self.shape = (rows, cols)
        else:
            glTexSubImage2D(
                GL_TEXTURE_2D, 0, 0, 0, cols, rows,
                GL_RGBA_INTEGER, GL_UNSIGNED_INT, data,
            )

    def bind(self) -> None:
        if self.handle:
            glActiveTexture(GL_TEXTURE0 + self._unit)
            glBindTexture(GL_TEXTURE_2D, self.handle)

    def uniforms(self) -> dict[str, object]:
        return {
            "u_tiles": self._unit,
            "u_tiles_on": int(self.shape != (0, 0)),
        }

    def dispose(self) -> None:
        if self.handle:
            glDeleteTextures([self.handle])
            self.handle = 0
        self.shape = (0, 0)


def fragment_source() -> str:
    """The raymarch fragment shader, unassembled — ``#include`` lines intact."""
    return (_SHADER_DIR / SHADER_NAME).read_text(encoding="utf-8")


def unpack_depth(pixels: np.ndarray, max_distance: float) -> tuple[np.ndarray, np.ndarray]:
    """Decode ``debug_mode=2`` output back to ``(t, hit)``.

    Inverse of ``pack_depth`` in the shader. ``pixels`` is the RGB read back from
    the framebuffer, either ``uint8`` or already normalised to ``[0, 1]``. A
    fully saturated texel is the miss sentinel.
    """
    rgb = np.asarray(pixels, dtype=np.float64)
    if rgb.dtype == np.float64 and rgb.max(initial=0.0) > 1.0:
        rgb = rgb / 255.0
    rgb = rgb[..., :3]

    miss = np.all(rgb >= 254.0 / 255.0, axis=-1)
    weights = np.array([1.0, 1.0 / 255.0, 1.0 / 65025.0])
    t = (rgb * weights).sum(axis=-1) * max_distance
    return t, ~miss
