"""Orbit camera, as pure numpy.

Deliberately holds no Qt and no GL. A raymarcher needs the *same* primary rays
on both sides of the fence — the CPU reference field in
:mod:`mesh_xml.render.field` and the fragment shader must agree pixel for pixel,
or comparing them proves nothing. So the ray construction lives here, in code
that runs headless and can be unit-tested, and the shader is written to mirror
it (see ``shaders/camera.glsl``).

That is also why ``QMatrix4x4`` is not used. It would drag Qt into the one
module that most needs to be testable without a display, and its ``inverted()``
returns a tuple that is easy to misread.

**Matrix convention.** Matrices are returned row-major in the mathematical
sense: ``M @ [x, y, z, 1]`` transforms a column vector. GL wants column-major
memory, so uploads must transpose — :meth:`ShaderProgram.set_uniform
<mesh_xml.render.gl.program.ShaderProgram.set_uniform>` passes ``GL_TRUE`` for
exactly this reason.

**Row order.** There is one, everywhere: row 0 is the *bottom* of the image,
matching ``gl_FragCoord`` and ``glReadPixels``. Anything that eventually needs
top-down screen coordinates — mouse picking, in a later phase — flips at its own
edge rather than making this module carry two conventions.

**Orientation.** The view rotation is ``Rx(pitch) @ Ry(yaw)``. Composing two
axis rotations rather than running a look-at against a fixed world up is what
lets pitch reach exactly ±90° without the basis collapsing: at the pole a
look-at's cross product goes to zero, but ``Rx(90°) @ Ry(yaw)`` is still a
perfectly good orthonormal frame. That is why the clamp is an inclusive ±90 and
not the ``89.9`` such code usually has to settle for.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "DEFAULT_DISTANCE",
    "MAX_DISTANCE",
    "MIN_DISTANCE",
    "ORBIT_DEGREES_PER_PIXEL",
    "PITCH_LIMIT",
    "ZOOM_PER_WHEEL_UNIT",
    "OrbitCamera",
    "viewport_aspect",
]

#: Degrees of orbit per pixel of drag. Half a degree puts a full turn at a
#: 720-pixel sweep, which is roughly one screen width and reads as neither
#: sluggish nor twitchy.
ORBIT_DEGREES_PER_PIXEL = 0.5

#: World units of dolly per unit of ``QWheelEvent.angleDelta().y()``. One
#: notch of a standard mouse wheel reports 120, so a notch moves 1.2 units.
ZOOM_PER_WHEEL_UNIT = 0.01

#: Default dolly bounds. Instance attributes, not hard limits — a rig authored
#: at a different scale widens them, and the far plane usually moves with them.
MIN_DISTANCE = 1.0
MAX_DISTANCE = 50.0
DEFAULT_DISTANCE = 10.0

#: Inclusive pitch bound in degrees. See the module docstring for why ±90 is
#: safe here and would not be under a look-at.
PITCH_LIMIT = 90.0


def viewport_aspect(width: int, height: int) -> float:
    """Width over height, falling back to 1.0 for a degenerate viewport.

    Qt really does deliver zero-height resizes — a splitter dragged shut, a
    window minimised on some platforms — and an unguarded division surfaces as
    ZeroDivisionError from inside a paint callback, where it is thoroughly
    unhelpful. A square fallback draws a garbage frame nobody can see instead.
    """
    if width <= 0 or height <= 0:
        return 1.0
    return float(width) / float(height)


class OrbitCamera:
    """A camera on a sphere around ``target``, parameterised by yaw and pitch.

    ``distance`` and ``pitch`` clamp on assignment rather than in the input
    handlers, so a camera restored from a file or driven by a controller cannot
    end up in a state the mouse could never produce.
    """

    def __init__(
        self,
        distance: float = DEFAULT_DISTANCE,
        yaw: float = 0.0,
        pitch: float = 0.0,
        target: object = (0.0, 0.0, 0.0),
        fov_y: float = 45.0,
        near: float = 0.1,
        far: float = 100.0,
        min_distance: float = MIN_DISTANCE,
        max_distance: float = MAX_DISTANCE,
        orthographic: bool = False,
        ortho_height: float | None = None,
    ) -> None:
        if not 0.0 < fov_y < 180.0:
            raise ValueError(f"fov_y must be in (0, 180), got {fov_y}")
        if not 0.0 < near < far:
            raise ValueError(f"near/far must satisfy 0 < near < far, got {near}/{far}")
        if not 0.0 < min_distance <= max_distance:
            raise ValueError(
                f"distance bounds must satisfy 0 < min <= max, "
                f"got {min_distance}/{max_distance}"
            )

        self.fov_y = float(fov_y)
        self.orthographic = bool(orthographic)
        self._ortho_height: float | None = None
        self.ortho_height = ortho_height
        self.near = float(near)
        self.far = float(far)
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self._distance = 0.0
        self._yaw = 0.0
        self._pitch = 0.0
        self._target = np.zeros(3, dtype=np.float64)

        self.distance = distance
        self.yaw = yaw
        self.pitch = pitch
        self.target = target

    @property
    def distance(self) -> float:
        return self._distance

    @distance.setter
    def distance(self, value: float) -> None:
        self._distance = min(max(float(value), self.min_distance), self.max_distance)

    @property
    def yaw(self) -> float:
        """Rotation about world Y, in degrees, normalised to ``[0, 360)``.

        Normalising is not cosmetic: an editing session accumulates drag deltas
        indefinitely, and a yaw in the millions loses the low bits that make
        small drags smooth.
        """
        return self._yaw

    @yaw.setter
    def yaw(self, value: float) -> None:
        self._yaw = float(value) % 360.0

    @property
    def pitch(self) -> float:
        return self._pitch

    @pitch.setter
    def pitch(self, value: float) -> None:
        self._pitch = min(max(float(value), -PITCH_LIMIT), PITCH_LIMIT)

    @property
    def target(self) -> np.ndarray:
        return self._target

    @target.setter
    def target(self, value: object) -> None:
        point = np.asarray(value, dtype=np.float64).reshape(-1)
        if point.shape != (3,):
            raise ValueError(f"target must have 3 components, got {point.shape}")
        self._target = point

    @property
    def position(self) -> np.ndarray:
        """World-space eye. Shared by every primary ray under perspective."""
        rotation = self._rotation()
        return (self._target + rotation[2] * self._distance).astype(np.float32)

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World-space ``(right, up, forward)``, orthonormal, forward down -Z."""
        rotation = self._rotation()
        return (
            rotation[0].astype(np.float32),
            rotation[1].astype(np.float32),
            (-rotation[2]).astype(np.float32),
        )

    @property
    def tan_half_fov(self) -> float:
        """Half-height of the near plane over its distance. What rays scale by."""
        return math.tan(math.radians(self.fov_y) * 0.5)

    @property
    def ortho_height(self) -> float:
        """World units visible vertically under orthographic projection.

        Derived from the perspective framing unless overridden, so toggling
        projection keeps the subject roughly the same size on screen instead of
        jumping. That is what makes the toggle usable as a comparison rather than
        as a re-framing exercise.
        """
        if self._ortho_height is not None:
            return self._ortho_height
        return 2.0 * self._distance * self.tan_half_fov

    @ortho_height.setter
    def ortho_height(self, value: float | None) -> None:
        if value is not None and value <= 0.0:
            raise ValueError(f"ortho_height must be positive, got {value}")
        self._ortho_height = None if value is None else float(value)

    def view_matrix(self) -> np.ndarray:
        """World to view, ``(4, 4) float32``."""
        rotation = self._rotation()
        view = np.eye(4, dtype=np.float64)
        view[:3, :3] = rotation
        view[:3, 3] = -rotation @ self._target
        view[2, 3] -= self._distance
        return view.astype(np.float32)

    def projection_matrix(self, width: int, height: int) -> np.ndarray:
        """View to clip, ``(4, 4) float32``. Degenerate viewports fall back to 1:1."""
        aspect = viewport_aspect(width, height)
        focal = 1.0 / self.tan_half_fov
        near, far = self.near, self.far

        projection = np.zeros((4, 4), dtype=np.float64)
        projection[0, 0] = focal / aspect
        projection[1, 1] = focal
        projection[2, 2] = (far + near) / (near - far)
        projection[2, 3] = 2.0 * far * near / (near - far)
        projection[3, 2] = -1.0
        return projection.astype(np.float32)

    def view_projection_matrix(self, width: int, height: int) -> np.ndarray:
        return (
            self.projection_matrix(width, height).astype(np.float64)
            @ self.view_matrix().astype(np.float64)
        ).astype(np.float32)

    def ray_directions(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Primary rays for a whole framebuffer.

        Returns ``(origin, directions)``, both **row 0 at the bottom** to match
        ``gl_FragCoord`` and ``glReadPixels``.

        The two projections put the per-pixel variation in different places, and
        the shapes say which: under perspective one eye serves every pixel and
        the directions fan out, so ``origin`` is ``(3,)`` and materialising
        ``H*W`` copies of it would be waste. Under orthographic every ray shares
        one direction and the origins tile a plane, so ``origin`` is
        ``(H, W, 3)``. :func:`mesh_xml.render.field.raymarch` broadcasts the two
        against each other, so both shapes feed it unchanged.

        Samples pixel centres, which is what the rasteriser interpolates to, so
        a CPU render lines up with the shader's instead of sitting half a pixel
        off.
        """
        if width <= 0 or height <= 0:
            empty = np.zeros((max(height, 0), max(width, 0), 3), dtype=np.float32)
            return self.position, empty

        right, up, forward = self.basis()
        aspect = viewport_aspect(width, height)

        ndc_x = 2.0 * (np.arange(width, dtype=np.float64) + 0.5) / width - 1.0
        ndc_y = 2.0 * (np.arange(height, dtype=np.float64) + 0.5) / height - 1.0

        if self.orthographic:
            half = self.ortho_height * 0.5
            origins = (
                self.position[None, None, :].astype(np.float64)
                + right[None, None, :] * (ndc_x * aspect * half)[None, :, None]
                + up[None, None, :] * (ndc_y * half)[:, None, None]
            )
            directions = np.broadcast_to(forward, (height, width, 3)).copy()
            return origins.astype(np.float32), directions.astype(np.float32)

        scale = self.tan_half_fov
        directions = (
            forward[None, None, :]
            + right[None, None, :] * (ndc_x * aspect * scale)[None, :, None]
            + up[None, None, :] * (ndc_y * scale)[:, None, None]
        )
        directions /= np.linalg.norm(directions, axis=2, keepdims=True)
        return self.position, directions.astype(np.float32)

    def orbit(self, dx: float, dy: float) -> None:
        """Apply a mouse drag in pixels.

        ``dy`` follows Qt's downward-positive mouse delta, and a downward drag
        raises the eye — the object tips toward the viewer, the grab-and-turn
        reading rather than the fly-the-camera one.
        """
        self.yaw = self._yaw + dx * ORBIT_DEGREES_PER_PIXEL
        self.pitch = self._pitch + dy * ORBIT_DEGREES_PER_PIXEL

    def zoom(self, wheel_delta: float) -> None:
        """Apply ``QWheelEvent.angleDelta().y()``. Positive scrolls closer."""
        self.distance = self._distance - wheel_delta * ZOOM_PER_WHEEL_UNIT

    def pan(self, dx: float, dy: float, viewport_height: int) -> None:
        """Slide the target in the camera plane by a drag in pixels.

        Scaled by distance and field of view so one pixel of drag moves the
        world by one pixel at the target's depth — the grab stays under the
        cursor instead of accelerating away as you dolly out.
        """
        right, up, _ = self.basis()
        world_per_pixel = (
            2.0 * self._distance * self.tan_half_fov / max(viewport_height, 1)
        )
        self._target = (
            self._target
            + (-right * dx + up * dy).astype(np.float64) * world_per_pixel
        )

    def _rotation(self) -> np.ndarray:
        """``Rx(pitch) @ Ry(yaw)``: rows are right, up, backward."""
        pitch = math.radians(self._pitch)
        yaw = math.radians(self._yaw)
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        return np.array(
            [
                [cos_y, 0.0, sin_y],
                [sin_p * sin_y, cos_p, -sin_p * cos_y],
                [-cos_p * sin_y, sin_p, cos_p * cos_y],
            ],
            dtype=np.float64,
        )

    def __repr__(self) -> str:
        target = ", ".join(f"{v:.2f}" for v in self._target)
        return (
            f"<OrbitCamera d={self._distance:.2f} yaw={self._yaw:.1f} "
            f"pitch={self._pitch:.1f} target=({target})>"
        )
