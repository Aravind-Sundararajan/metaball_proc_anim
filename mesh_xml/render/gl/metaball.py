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
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    glActiveTexture,
    glBindTexture,
    glDeleteTextures,
    glGenTextures,
    glTexImage2D,
    glTexParameteri,
    glTexSubImage2D,
)

from mesh_xml.render.sdf import PRIMITIVE_STRIDE

__all__ = ["MetaballSettings", "PrimitiveTexture", "fragment_source", "SHADER_NAME"]

SHADER_NAME = "metaball.frag"

_SHADER_DIR = Path(__file__).parent / "shaders"

#: Texels per primitive. The record is two vec4s: (a, ra) and (b, rb).
TEXELS_PER_PRIMITIVE = PRIMITIVE_STRIDE // 4

#: Texture unit the primitive texture is bound to.
PRIMITIVE_TEXTURE_UNIT = 0


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
    light_direction: tuple[float, float, float] = (0.4, 0.8, 0.45)
    surface_colour: tuple[float, float, float] = (0.82, 0.55, 0.42)
    background_colour: tuple[float, float, float] = (0.11, 0.12, 0.14)

    def uniforms(self) -> dict[str, object]:
        """Uniform values for :attr:`RaymarchView.uniforms`."""
        return {
            "u_blend": float(self.blend),
            "u_max_distance": float(self.max_distance),
            "u_max_steps": int(self.max_steps),
            "u_surface_epsilon": float(self.surface_epsilon),
            "u_step_relaxation": float(self.step_relaxation),
            "u_normal_epsilon": float(self.normal_epsilon),
            "u_debug_mode": int(self.debug_mode),
            "u_light_direction": tuple(float(v) for v in self.light_direction),
            "u_surface_colour": tuple(float(v) for v in self.surface_colour),
            "u_background_colour": tuple(float(v) for v in self.background_colour),
            "u_primitives": PRIMITIVE_TEXTURE_UNIT,
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
