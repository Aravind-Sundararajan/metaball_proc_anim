"""The viewport that draws a document and lets the mouse edit it.

:mod:`mesh_xml.render.gl.context` refuses to do any of this on purpose — it
says so in its own docstring — because picking and dragging need a document and
a command stack, and a viewport that reaches into the model directly is how a
repaint ends up being the thing that changes it. Those two now exist, so this
module is the layer that was being waited for: it embeds :class:`RaymarchView`
and adds the editing, and every edit leaves through :meth:`Document.do`.

Four decisions here are load-bearing.

**Frames are not free and are not unconditional.** ``main.py:134`` in the old
app ran a 16 ms ``QTimer`` straight into ``update()`` for the whole of its life,
so a rig sitting still on screen repainted a raymarched fullscreen shader sixty
times a second forever. :meth:`MetaballViewport.tick` instead asks whether
anything can have changed — the simulation is running, the document was edited,
the camera moved, or the draw-point springs have not finished settling — and
returns without touching the GPU when the answer is no. :attr:`~MetaballViewport.idle`
reports that, so the behaviour is assertable rather than a claim.

**The elapsed time is measured, never assumed.** A frame that took 40 ms and one
that took 4 ms must not advance the simulation by the same amount, or the rig
runs at a speed that depends on what else the machine is doing.
:class:`~mesh_xml.anim.Clock` already converts wall-clock seconds into whole
fixed ticks; this module's only job is to hand it a real number.

**One drag is one command.** Recording a :class:`MoveChunk` per mouse-move event
would put several hundred entries on the undo stack for one gesture and make
undo useless exactly where it is most wanted. So the drag writes the body
position live for the user's benefit, and on release *rewinds* the body to where
the gesture started before submitting a single command — which is what makes
:class:`MoveChunk` capture the pre-drag position rather than the dragged one,
since it reads the current position when it applies.

**The rig is drawn over the surface with QPainter.** The metaball shader has no
per-primitive colour: everything melts into one blended field, which is the
entire point of it and also means selection cannot be shown in the shader
without giving up the blend. So the skeleton is projected with
:meth:`OrbitCamera.view_projection_matrix` and painted on top, and it is
toggleable because the surface is what the tool is for.

Picking and the overlay both use *body* positions, not the smoothed draw
positions the surface is built from. The draw points deliberately lag — that lag
is the secondary motion — so pinning the editing affordances to them would mean
the thing you grab drifts away from the cursor while you hold it. The body is
what an edit edits; the surface eases after it, exactly as
:meth:`Document.advance` describes.

:func:`ray_through` lives here rather than on the camera because
:mod:`mesh_xml.render.gl.camera` keeps one row order — row 0 at the bottom, to
match ``gl_FragCoord`` — and says that anything needing top-down screen
coordinates flips at its own edge. This is that edge: it is the module that has
the mouse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QHideEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QShowEvent,
)

from mesh_xml.core import Body
from mesh_xml.editor.document import Document
from mesh_xml.render import field
from mesh_xml.render.gl.camera import OrbitCamera, viewport_aspect
from mesh_xml.render.gl.context import RaymarchView

if TYPE_CHECKING:
    from mesh_xml.core.terrain import Terrain
    from mesh_xml.render.gl.metaball import PrimitiveTexture, TerrainTexture

__all__ = [
    "FRAME_INTERVAL_MS",
    "MetaballViewport",
    "chunk_centres",
    "gizmo_handles",
    "gizmo_hit",
    "nearest_hit",
    "plane_hit",
    "project",
    "ray_through",
]

#: Frame timer period. The ceiling on the refresh rate, not the rate itself:
#: :meth:`MetaballViewport.tick` decides per frame whether to do any work.
FRAME_INTERVAL_MS = 16

#: How close the draw points must be to their chunks, in world units and in
#: units per tick, before the springs count as finished and the viewport is
#: allowed to stop drawing. Below this the motion is under a hundredth of a
#: pixel at any sane zoom.
SETTLE_EPSILON = 1e-4

#: Slack :meth:`MetaballViewport.frame_all` leaves around the rig's bounding
#: sphere, so a framed body does not sit flush against the window edge.
FRAME_MARGIN = 1.15

#: Below this dot product a ray is parallel to the drag plane and the
#: intersection runs off to infinity.
PARALLEL_EPSILON = 1e-6

LINK_COLOUR = QColor(120, 190, 235, 150)
CHUNK_COLOUR = QColor(232, 238, 248, 210)
SELECTED_COLOUR = QColor(255, 186, 72, 255)

LINK_WIDTH = 1.4
CHUNK_RADIUS_PX = 3.0
SELECTED_RADIUS_PX = 6.5

#: Orientation gizmo layout, in device-independent pixels from the top-right
#: corner. The handle radius doubles as the click target's core; the hit test
#: pads it slightly.
GIZMO_RADIUS_PX = 40.0
GIZMO_MARGIN_PX = 16.0
GIZMO_HANDLE_PX = 8.0

#: The gizmo's handles as ``(snap view, world axis)``. Named by view rather
#: than by axis because clicking the handle at +axis puts the eye there, which
#: is exactly what :meth:`OrbitCamera.snap_view` of that name does.
GIZMO_AXES = (
    ("right", (1.0, 0.0, 0.0)),
    ("left", (-1.0, 0.0, 0.0)),
    ("top", (0.0, 1.0, 0.0)),
    ("bottom", (0.0, -1.0, 0.0)),
    ("front", (0.0, 0.0, 1.0)),
    ("back", (0.0, 0.0, -1.0)),
)

_GIZMO_COLOUR = {
    "right": QColor(213, 92, 82),
    "left": QColor(213, 92, 82),
    "top": QColor(112, 189, 96),
    "bottom": QColor(112, 189, 96),
    "front": QColor(94, 138, 219),
    "back": QColor(94, 138, 219),
}
_GIZMO_LABEL = {"right": "X", "top": "Y", "front": "Z"}


def gizmo_handles(
    camera: OrbitCamera, width: int, height: int
) -> list[tuple[str, float, float, float]]:
    """Screen layout of the orientation gizmo: ``(view, x, y, depth)`` handles.

    Pure geometry, so the tests need no widget. Each world axis lands at the
    gizmo centre plus ``(axis . right, -axis . up)`` arms — the same projection
    the camera applies to everything, minus the perspective it does not need —
    and ``depth`` is ``axis . forward``: positive points into the screen.
    Sorted by descending depth, which is back-to-front paint order.
    """
    right, up, forward = camera.basis()
    cx = width - GIZMO_MARGIN_PX - GIZMO_RADIUS_PX
    cy = GIZMO_MARGIN_PX + GIZMO_RADIUS_PX
    arm = GIZMO_RADIUS_PX - GIZMO_HANDLE_PX * 0.5
    handles = []
    for view, axis in GIZMO_AXES:
        a = np.asarray(axis, dtype=np.float64)
        handles.append(
            (
                view,
                cx + float(a @ right) * arm,
                cy - float(a @ up) * arm,
                float(a @ forward),
            )
        )
    handles.sort(key=lambda handle: handle[3], reverse=True)
    return handles


def gizmo_hit(
    camera: OrbitCamera, width: int, height: int, x: float, y: float
) -> str | None:
    """The snap view under a click, nearest handle first, or ``None``.

    Walks the paint order backwards so an overlapping pair resolves to the
    handle the user can actually see.
    """
    reach = (GIZMO_HANDLE_PX + 2.0) ** 2
    for view, hx, hy, _depth in reversed(gizmo_handles(camera, width, height)):
        if (hx - x) ** 2 + (hy - y) ** 2 <= reach:
            return view
    return None


def ray_through(
    camera: OrbitCamera, x: float, y: float, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Primary ray through one pixel, in Qt's y-down window coordinates.

    The single-pixel counterpart of :meth:`OrbitCamera.ray_directions`, and it
    has to stay one: a pick that disagrees with the render by half a pixel puts
    the selection on the wrong chunk near a silhouette. ``(x, y)`` is a pixel
    index and the ray goes through that pixel's centre, so
    ``ray_through(cam, x, y, w, h)`` is ``ray_directions(w, h)[h - 1 - y, x]``.

    Returns ``(origin, direction)`` in float64 with ``direction`` normalized.
    """
    right, up, forward = (v.astype(np.float64) for v in camera.basis())
    scale = camera.tan_half_fov
    aspect = viewport_aspect(width, height)

    ndc_x = 2.0 * (float(x) + 0.5) / max(width, 1) - 1.0
    ndc_y = 1.0 - 2.0 * (float(y) + 0.5) / max(height, 1)

    direction = forward + right * (ndc_x * aspect * scale) + up * (ndc_y * scale)
    return camera.position.astype(np.float64), direction / np.linalg.norm(direction)


def nearest_hit(
    origin: np.ndarray,
    direction: np.ndarray,
    centres: np.ndarray,
    radii: np.ndarray,
) -> tuple[int, float] | None:
    """Index of the first sphere along the ray, with the distance to it.

    A ray starting inside a sphere reports the exit point rather than missing,
    so clicking a chunk the camera has been dollied into still selects it.
    """
    if len(centres) == 0:
        return None

    to_origin = np.asarray(origin, dtype=np.float64) - np.asarray(centres, np.float64)
    half_b = to_origin @ np.asarray(direction, dtype=np.float64)
    c = (to_origin * to_origin).sum(axis=1) - np.asarray(radii, np.float64) ** 2
    discriminant = half_b * half_b - c

    root = np.sqrt(np.where(discriminant >= 0.0, discriminant, 0.0))
    near = -half_b - root
    t = np.where(near >= 0.0, near, -half_b + root)

    valid = (discriminant >= 0.0) & (t >= 0.0)
    if not valid.any():
        return None

    distances = np.where(valid, t, np.inf)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def plane_hit(
    origin: np.ndarray,
    direction: np.ndarray,
    point: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray | None:
    """Where a ray meets the plane through ``point``, or ``None`` if parallel."""
    denominator = float(np.dot(direction, normal))
    if abs(denominator) < PARALLEL_EPSILON:
        return None
    t = float(np.dot(np.asarray(point, np.float64) - origin, normal)) / denominator
    return np.asarray(origin, np.float64) + np.asarray(direction, np.float64) * t


def project(
    matrix: np.ndarray, points: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """World points to Qt window coordinates, with a mask of what is in front.

    ``matrix`` is a view-projection in this codebase's ``M @ v`` row-major
    sense. Points at or behind the eye have a non-positive clip ``w`` and would
    project to a mirrored ghost in front of the camera, so they are masked out
    rather than drawn.
    """
    positions = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.concatenate(
        [positions, np.ones((len(positions), 1))], axis=1
    )
    clip = homogeneous @ np.asarray(matrix, dtype=np.float64).T

    w = clip[:, 3]
    visible = w > PARALLEL_EPSILON
    ndc = clip[:, :3] / np.where(visible, w, 1.0)[:, None]

    screen = np.empty((len(positions), 2), dtype=np.float64)
    screen[:, 0] = (ndc[:, 0] + 1.0) * 0.5 * width
    screen[:, 1] = (1.0 - ndc[:, 1]) * 0.5 * height
    return screen, visible


def chunk_centres(body: Body) -> np.ndarray:
    """Chunk positions as ``(N, 3) float64``, widening a 2-D body onto ``z = 0``.

    The same widening :func:`mesh_xml.render.sdf.pack` applies, so a pick tests
    against the spheres that were actually rendered.
    """
    centres = np.zeros((body.n_chunks, 3), dtype=np.float64)
    centres[:, : body.dim] = body.pos
    return centres


@dataclass
class _ChunkDrag:
    """State for one grab-and-move gesture, from press to release."""

    chunk_id: int
    start: tuple[float, ...]
    anchor: np.ndarray
    normal: np.ndarray
    offset: np.ndarray
    moved: bool = False


class MetaballViewport(RaymarchView):
    """A raymarched view of a :class:`Document` that the mouse can edit.

    Left-drag on empty space orbits, left-drag on a chunk moves it, middle or
    right drag pans, the wheel zooms. Click selects, shift-click toggles,
    clicking empty space clears.

    The frame loop starts when the widget is shown and stops when it is hidden;
    :meth:`tick` is public so a test can drive frames without an event loop.
    """

    idle_changed = Signal(bool)

    def __init__(
        self,
        document: Document,
        camera: OrbitCamera | None = None,
        parent: object = None,
    ) -> None:
        from mesh_xml.render.gl.metaball import (
            SHADER_NAME,
            PrimitiveTexture,
            TerrainTexture,
            fragment_source,
        )

        super().__init__(camera=camera, parent=parent)
        self.set_fragment_source(fragment_source(), SHADER_NAME)

        self._document = document
        self._texture: PrimitiveTexture = PrimitiveTexture()
        self._terrain: TerrainTexture = TerrainTexture()
        # The Terrain object last sent to the GPU. Identity, not equality:
        # a world does not move per tick, so the only re-upload trigger is the
        # document swapping the object itself (Document.set_terrain says so).
        self._terrain_source: Terrain | None = None
        self._uploaded: np.ndarray | None = None
        self._pending: np.ndarray | None = None
        self._upload_count = 0
        self._frame_count = 0

        self._idle = True
        self._show_rig = True
        self._needs_upload = True
        self._needs_repaint = True
        self._drag: _ChunkDrag | None = None

        self._elapsed = QElapsedTimer()
        self._elapsed.start()
        self._timer = QTimer(self)
        self._timer.setInterval(FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self.tick)

        document.body_replaced.connect(self._invalidate_geometry)
        document.topology_changed.connect(self._invalidate_geometry)
        document.chunk_changed.connect(self._on_chunk_changed)
        document.settings_changed.connect(self._invalidate_frame)
        # A terrain-only change must buy a frame or the new world never shows;
        # the paint itself notices the object swap and re-uploads.
        document.terrain_changed.connect(self._invalidate_frame)
        document.selection_changed.connect(self._invalidate_frame)
        document.running_changed.connect(self._on_running_changed)
        self.camera_changed.connect(self._invalidate_frame)

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    @property
    def document(self) -> Document:
        return self._document

    @property
    def texture(self) -> PrimitiveTexture:
        """The primitive texture. Exposed so a test can watch it reallocate."""
        return self._texture

    @property
    def terrain_texture(self) -> TerrainTexture:
        """The world's texture. Exposed so a test can watch what was uploaded."""
        return self._terrain

    @property
    def idle(self) -> bool:
        """Whether the last :meth:`tick` decided there was nothing to draw."""
        return self._idle

    @property
    def frames(self) -> int:
        """Frames :meth:`tick` has asked for. Flat while idle, by design."""
        return self._frame_count

    @property
    def uploads(self) -> int:
        """Primitive uploads performed. Flat while the packed array is unchanged."""
        return self._upload_count

    @property
    def show_rig(self) -> bool:
        return self._show_rig

    @show_rig.setter
    def show_rig(self, show: bool) -> None:
        show = bool(show)
        if show == self._show_rig:
            return
        self._show_rig = show
        self._invalidate_frame()

    @property
    def looping(self) -> bool:
        """Whether the frame timer is live.

        Not spelled ``running``: that already means "the simulation is playing"
        on :class:`Document`, and a viewport whose loop is turning while the
        simulation is paused is the normal state, not a contradiction.
        """
        return self._timer.isActive()

    def start(self) -> None:
        self._elapsed.restart()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    # ------------------------------------------------------------------
    # the frame loop
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """One pass of the frame loop, skipped entirely when nothing can change.

        The elapsed timer restarts unconditionally, including on a skipped
        frame. Otherwise a viewport that sat idle for ten seconds would hand
        those ten seconds to the simulation the moment something woke it.
        """
        elapsed = self._elapsed.restart() / 1000.0
        if not self._needs_frame():
            self._set_idle(True)
            return

        self._set_idle(False)
        self._document.advance(elapsed)
        self._repack()
        self._needs_repaint = False
        self._frame_count += 1
        self.update()

    def _needs_frame(self) -> bool:
        return (
            self._document.running
            or self._needs_upload
            or self._needs_repaint
            or not self._settled()
        )

    def _settled(self) -> bool:
        """Whether the draw-point springs have finished chasing the chunks.

        A paused document is not a still one: after a drag or an undo the draw
        points are still easing toward the body, and stopping the frame loop
        while they are would freeze the surface halfway there.
        """
        draw = self._document.draw
        body = self._document.body
        if len(draw.pos) != body.n_chunks:
            return False
        if body.n_chunks == 0:
            return True
        gap = float(np.abs(body.pos - draw.pos).max())
        speed = float(np.abs(draw.vel).max())
        return gap <= SETTLE_EPSILON and speed <= SETTLE_EPSILON

    def _repack(self) -> None:
        """Pack the primitives, queueing an upload only if they differ.

        The comparison is what keeps a still-but-not-idle frame — the camera
        orbiting around a paused rig — from re-uploading identical bytes every
        frame.

        It compares against whatever is queued *or* already on the GPU, in that
        order. Against the GPU alone, a chunk nudged and then nudged back
        before the paint landed would match what was uploaded, leave the queue
        untouched, and ship the intermediate position.
        """
        self._needs_upload = False
        primitives = self._document.primitives()
        latest = self._pending if self._pending is not None else self._uploaded
        if latest is not None and np.array_equal(primitives, latest):
            return
        self._pending = primitives

    def _set_idle(self, idle: bool) -> None:
        if idle != self._idle:
            self._idle = idle
            self.idle_changed.emit(idle)

    def _invalidate_frame(self) -> None:
        self._needs_repaint = True

    def _invalidate_geometry(self) -> None:
        self._needs_upload = True

    def _on_chunk_changed(self, chunk_id: int) -> None:
        del chunk_id
        self._invalidate_geometry()

    def _on_running_changed(self, running: bool) -> None:
        del running
        self._elapsed.restart()
        self._invalidate_frame()

    # ------------------------------------------------------------------
    # painting
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        super().initializeGL()
        context = self.context()
        if context is not None:
            context.aboutToBeDestroyed.connect(self._release_texture)

    def paintGL(self) -> None:
        self._flush_upload()
        self._flush_terrain()
        self.uniforms.update(self._document.metaball.uniforms())
        self.uniforms.update(self._texture.uniforms())
        self.uniforms.update(self._terrain.uniforms())
        self.uniforms.update(self._floor_uniforms())
        self._texture.bind()
        self._terrain.bind()
        super().paintGL()
        self._paint_overlay()

    def _floor_uniforms(self) -> dict[str, object]:
        """The reference grid's uniforms, owned here rather than by settings.

        The floor needs two things only the document can supply: the solver's
        floor height, and a veto when a terrain is loaded — the terrain *is*
        the world, and a grid plane slicing through its hills would read as a
        second, contradictory floor.
        """
        terrain = self._document.terrain
        enabled = self._document.metaball.show_floor and (
            terrain is None or terrain.is_empty
        )
        height = self._document.solver_config.ground_y
        return {
            "u_floor_enabled": int(enabled),
            "u_floor_height": float(height if height is not None else 0.0),
        }

    def _flush_upload(self) -> None:
        """Push queued primitives to the GPU. Requires the context current.

        Uploading from :meth:`paintGL` rather than from :meth:`tick` avoids a
        ``makeCurrent``/``doneCurrent`` pair per frame, and means a repaint Qt
        asks for on its own — an expose, a resize — finds the texture populated
        even if no tick has run yet.
        """
        if self._pending is None:
            if self._uploaded is not None:
                return
            self._pending = self._document.primitives()

        self._texture.upload(self._pending)
        self._uploaded = self._pending
        self._pending = None
        self._upload_count += 1

    def _flush_terrain(self) -> None:
        """Upload the world once per object, not per frame. Context must be current.

        The creature re-packs every frame because it moves every tick; the
        terrain does neither, so the per-frame cost here is one identity
        check. Anything that replaces the document's terrain — open, adopt,
        :meth:`Document.set_terrain` — installs a new object, and that is the
        entire re-upload protocol.
        """
        terrain = self._document.terrain
        if terrain is self._terrain_source:
            return
        self._terrain.upload_terrain(terrain)
        self._terrain_source = terrain

    def _paint_overlay(self) -> None:
        """Draw the rig and the gizmo over the surface, then put GL back as it was.

        ``QPainter`` on a ``QOpenGLWidget`` renders through GL into the same
        framebuffer, and the ordering is not a style preference: begun before
        the raymarch pass, every stroke is overwritten by the fullscreen
        triangle and the overlay is not merely dimmed but *entirely absent*.
        ``tests/test_viewport.py`` asserts the overlay changes pixels for
        exactly that reason.

        The state restore afterwards is insurance rather than a fix. Measured
        on this driver the frame is byte-identical with or without it, because
        the fullscreen pass rebinds its own program and VAO every frame. But
        the pass also inherits ``glDisable(GL_DEPTH_TEST)`` from
        :meth:`RaymarchView.initializeGL`, which runs *once*, and the painter's
        engine is free to leave the enable bits and the viewport wherever it
        finished — so the one invariant that is set once is the one worth
        re-establishing.
        """
        body = self._document.body
        width, height = max(self.width(), 1), max(self.height(), 1)
        painter = QPainter(self)
        try:
            if self._show_rig and body.n_chunks:
                matrix = self.camera.view_projection_matrix(width, height)
                screen, visible = project(matrix, chunk_centres(body), width, height)
                self._stroke_rig(painter, body, screen, visible)
            self._stroke_gizmo(painter, width, height)
        finally:
            painter.end()
        self._restore_gl_state()

    def _stroke_gizmo(self, painter: QPainter, width: int, height: int) -> None:
        """The orientation triad: axis handles around a disc, click-to-snap.

        Painted in :func:`gizmo_handles`' back-to-front order so the handle a
        click resolves to is the one on top. Handles pointing at the viewer are
        full strength and the positive three carry their axis letter; handles
        pointing away dim, which is what makes the third dimension legible on
        a flat disc.
        """
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        cx = width - GIZMO_MARGIN_PX - GIZMO_RADIUS_PX
        cy = GIZMO_MARGIN_PX + GIZMO_RADIUS_PX
        centre = QPointF(cx, cy)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(8, 9, 11, 60))
        pad = GIZMO_HANDLE_PX + 6.0
        painter.drawEllipse(centre, GIZMO_RADIUS_PX + pad, GIZMO_RADIUS_PX + pad)

        for view, x, y, depth in gizmo_handles(self.camera, width, height):
            colour = QColor(_GIZMO_COLOUR[view])
            colour.setAlpha(255 if depth <= 0.0 else 120)
            point = QPointF(x, y)

            if view in _GIZMO_LABEL:
                painter.setPen(QPen(colour, 1.8))
                painter.drawLine(centre, point)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(colour)
                painter.drawEllipse(point, GIZMO_HANDLE_PX, GIZMO_HANDLE_PX)
                painter.setPen(QPen(QColor(12, 13, 16, colour.alpha())))
                painter.drawText(
                    QRectF(
                        x - GIZMO_HANDLE_PX,
                        y - GIZMO_HANDLE_PX,
                        GIZMO_HANDLE_PX * 2.0,
                        GIZMO_HANDLE_PX * 2.0,
                    ),
                    Qt.AlignmentFlag.AlignCenter,
                    _GIZMO_LABEL[view],
                )
            else:
                painter.setPen(QPen(colour, 1.6))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(
                    point, GIZMO_HANDLE_PX - 1.5, GIZMO_HANDLE_PX - 1.5
                )

    def _stroke_rig(
        self,
        painter: QPainter,
        body: Body,
        screen: np.ndarray,
        visible: np.ndarray,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        painter.setPen(QPen(LINK_COLOUR, LINK_WIDTH))
        for link in body.links():
            a, b = body.slot(link.a), body.slot(link.b)
            if visible[a] and visible[b]:
                painter.drawLine(
                    QPointF(*screen[a].tolist()), QPointF(*screen[b].tolist())
                )

        selection = self._document.selection
        for slot in range(body.n_chunks):
            if not visible[slot]:
                continue
            centre = QPointF(*screen[slot].tolist())
            selected = body.id_at(slot) in selection
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(SELECTED_COLOUR if selected else CHUNK_COLOUR)
            painter.drawEllipse(centre, CHUNK_RADIUS_PX, CHUNK_RADIUS_PX)
            if selected:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(SELECTED_COLOUR, LINK_WIDTH + 0.4))
                painter.drawEllipse(centre, SELECTED_RADIUS_PX, SELECTED_RADIUS_PX)

    def _restore_gl_state(self) -> None:
        from OpenGL.GL import (
            GL_BLEND,
            GL_DEPTH_TEST,
            GL_SCISSOR_TEST,
            GL_STENCIL_TEST,
            GL_TEXTURE0,
            glActiveTexture,
            glDisable,
            glViewport,
        )

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glDisable(GL_SCISSOR_TEST)
        glDisable(GL_STENCIL_TEST)
        glActiveTexture(GL_TEXTURE0)
        glViewport(0, 0, *self._framebuffer)

    def _release_texture(self) -> None:
        self.makeCurrent()
        self._texture.dispose()
        self._terrain.dispose()
        self.doneCurrent()
        self._uploaded = None
        self._pending = None
        # Forget what was uploaded so a fresh context gets a fresh upload.
        self._terrain_source = None

    # ------------------------------------------------------------------
    # camera
    # ------------------------------------------------------------------

    def frame_all(self) -> None:
        """Fit the camera to the rig's blended bounds.

        Widens the camera's dolly limits and far plane to reach, which
        :class:`OrbitCamera` documents as the intended response to a rig
        authored at a different scale — the defaults assume a body a few units
        across and silently clamp anything larger.

        Deliberately frames the CREATURE, not the world: a terrain plane is an
        infinite half-space, so bounds that included it would put the camera
        at infinity looking at everything and showing nothing. Framing is for
        the thing being edited, and the ground will be wherever the rig is.
        """
        lo, hi = field.bounds(self._document.primitives(), self._document.metaball.blend)
        radius = float(np.linalg.norm(hi - lo)) * 0.5
        self.camera.target = (lo + hi) * 0.5

        if radius > 0.0:
            aspect = viewport_aspect(max(self.width(), 1), max(self.height(), 1))
            fit = radius / (self.camera.tan_half_fov * min(1.0, aspect)) * FRAME_MARGIN
            self.camera.min_distance = min(self.camera.min_distance, fit)
            self.camera.max_distance = max(self.camera.max_distance, fit)
            self.camera.far = max(self.camera.far, fit + radius * 2.0)
            self.camera.distance = fit

        self.camera_changed.emit()

    # ------------------------------------------------------------------
    # picking
    # ------------------------------------------------------------------

    def ray_through(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        """The primary ray under a mouse position, in widget coordinates."""
        return ray_through(self.camera, x, y, max(self.width(), 1), max(self.height(), 1))

    def pick(self, x: float, y: float) -> int | None:
        """Id of the nearest chunk under a mouse position, or ``None``."""
        body = self._document.body
        origin, direction = self.ray_through(x, y)
        hit = nearest_hit(
            origin, direction, chunk_centres(body), np.asarray(body.radius, np.float64)
        )
        return None if hit is None else body.id_at(hit[0])

    def rig_screen_points(self) -> dict[int, tuple[float, float]]:
        """Where each chunk lands on screen. What the overlay draws, by chunk id."""
        body = self._document.body
        width, height = max(self.width(), 1), max(self.height(), 1)
        matrix = self.camera.view_projection_matrix(width, height)
        screen, visible = project(matrix, chunk_centres(body), width, height)
        return {
            body.id_at(slot): (float(screen[slot, 0]), float(screen[slot, 1]))
            for slot in range(body.n_chunks)
            if visible[slot]
        }

    # ------------------------------------------------------------------
    # mouse
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            position = event.position()
            view = gizmo_hit(
                self.camera,
                max(self.width(), 1),
                max(self.height(), 1),
                position.x(),
                position.y(),
            )
            if view is not None:
                self.camera.snap_view(view)
                self.camera_changed.emit()
                self.update()
                return

            chunk_id = self.pick(position.x(), position.y())
            shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

            if chunk_id is not None and shift:
                self._document.toggle_selection(chunk_id)
                return
            if chunk_id is not None:
                if chunk_id not in self._document.selection:
                    self._document.set_selection((chunk_id,))
                self._begin_drag(chunk_id, position.x(), position.y())
                return
            if not shift:
                self._document.set_selection(())

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag is not None:
            position = event.position()
            self._drag_to(position.x(), position.y())
            return

        if self._drag_button in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            position = event.position().toPoint()
            delta = position - self._last_pos
            self._last_pos = position
            self.camera.pan(delta.x(), delta.y(), max(self.height(), 1))
            self.camera_changed.emit()
            self.update()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag is not None and event.button() == Qt.MouseButton.LeftButton:
            self._end_drag()
            return
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    # dragging
    # ------------------------------------------------------------------

    def _begin_drag(self, chunk_id: int, x: float, y: float) -> None:
        """Grab a chunk on the plane through it facing the camera.

        The grab offset is recorded so the chunk does not jump its own centre
        under the cursor on the first pixel of movement, and the plane is fixed
        at press time so the gesture stays in one plane for its whole duration.
        """
        body = self._document.body
        anchor = chunk_centres(body)[body.slot(chunk_id)]
        normal = self.camera.basis()[2].astype(np.float64)

        origin, direction = self.ray_through(x, y)
        grabbed = plane_hit(origin, direction, anchor, normal)
        if grabbed is None:
            return

        self._drag = _ChunkDrag(
            chunk_id=chunk_id,
            start=body.chunk(chunk_id).pos,
            anchor=anchor,
            normal=normal,
            offset=anchor - grabbed,
        )
        # The hand owns the chunk for the duration: kinematic drivers stand
        # down for held chunks instead of overwriting the drag every tick.
        self._document.controllers.held.add(chunk_id)

    def _drag_to(self, x: float, y: float) -> None:
        """Preview the move by writing the body directly — the one place that happens.

        Not an edit: no command, no dirty flag, and :meth:`_end_drag` rewinds it
        before the real command is submitted. Going through :meth:`Document.do`
        here instead is what would flood the undo stack.
        """
        drag = self._drag
        if drag is None:
            return

        origin, direction = self.ray_through(x, y)
        landed = plane_hit(origin, direction, drag.anchor, drag.normal)
        if landed is None:
            return

        body = self._document.body
        target = landed + drag.offset
        body.set_pos(drag.chunk_id, tuple(target[: body.dim]))
        drag.moved = True
        self._invalidate_geometry()

    def _end_drag(self) -> None:
        """Rewind the preview and record the whole gesture as one command."""
        drag = self._drag
        self._drag = None
        if drag is None:
            return
        self._document.controllers.held.discard(drag.chunk_id)

        body = self._document.body
        if not drag.moved or not body.has_chunk(drag.chunk_id):
            return

        final = body.chunk(drag.chunk_id).pos
        if final == drag.start:
            return

        body.set_pos(drag.chunk_id, drag.start)
        self._document.commit_drag(drag.chunk_id, final)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self.start()

    def hideEvent(self, event: QHideEvent) -> None:
        self.stop()
        super().hideEvent(event)

    def __repr__(self) -> str:
        return (
            f"<MetaballViewport frames={self._frame_count} uploads={self._upload_count} "
            f"idle={self._idle} rig={self._show_rig}>"
        )
