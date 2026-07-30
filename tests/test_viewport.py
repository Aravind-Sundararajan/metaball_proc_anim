"""Tests for the editing viewport.

Split deliberately in two. The geometry — pixel rays, ray/sphere picking, the
drag plane, overlay projection — is plain numpy against a known camera and runs
anywhere, because that is the half most likely to be silently wrong: a picking
ray that is off by a half pixel or a flipped row still selects *something*, and
the bug only shows up as the wrong chunk near a silhouette.

The other half needs a real widget and skips without a core-profile context,
following ``tests/test_metaball.py``. Two things there cannot be established any
other way: that the ``QPainter`` overlay survives being composited over a
fullscreen GL pass, and that :class:`PrimitiveTexture` reallocates only when the
primitive count changes. Check the report, not just the exit code.
"""

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent

from mesh_xml.core import Body
from mesh_xml.editor.document import Document, MoveChunk
from mesh_xml.editor.viewport import (
    MetaballViewport,
    chunk_centres,
    nearest_hit,
    plane_hit,
    project,
    ray_through,
)
from mesh_xml.render.gl.camera import OrbitCamera

gl_context = pytest.importorskip("mesh_xml.render.gl.context")

needs_gl = pytest.mark.skipif(
    not gl_context.gl_available(),
    reason="no core-profile 3.3 context available",
)

SIZE = (192, 144)


def worm(n: int = 5, spacing: float = 0.9, radius: float = 0.3) -> Body:
    body = Body()
    ids = [
        body.add_chunk(f"s{i}", [i * spacing - n * spacing * 0.5, 0.0, 0.0], radius=radius)
        for i in range(n)
    ]
    for a, b in zip(ids, ids[1:]):
        body.add_link(a, b)
    return body


def document_with(body: Body) -> Document:
    """A document holding ``body``, wired up the way :meth:`Document.open` leaves it."""
    document = Document()
    document.body = body
    document.draw.sync(body)
    document.draw.snap(body)
    document.capture_rest_pose()
    return document


def press(view, x, y, button=Qt.MouseButton.LeftButton, modifiers=Qt.KeyboardModifier.NoModifier):
    view.mousePressEvent(_event(QMouseEvent.Type.MouseButtonPress, x, y, button, modifiers))


def move(view, x, y, button=Qt.MouseButton.LeftButton, modifiers=Qt.KeyboardModifier.NoModifier):
    view.mouseMoveEvent(_event(QMouseEvent.Type.MouseMove, x, y, Qt.MouseButton.NoButton, modifiers, button))


def release(view, x, y, button=Qt.MouseButton.LeftButton, modifiers=Qt.KeyboardModifier.NoModifier):
    view.mouseReleaseEvent(_event(QMouseEvent.Type.MouseButtonRelease, x, y, button, modifiers))


def _event(kind, x, y, button, modifiers, buttons=None):
    point = QPointF(float(x), float(y))
    return QMouseEvent(
        kind,
        point,
        point,
        button,
        buttons if buttons is not None else button,
        modifiers,
    )


# ----------------------------------------------------------------------
# geometry — no Qt widget, no GL
# ----------------------------------------------------------------------


class TestPixelRays:
    def test_matches_the_camera_row_for_row(self):
        """The pick ray and the render ray must be the same ray, flipped.

        ray_directions puts row 0 at the bottom; Qt's mouse puts row 0 at the
        top. If this ever drifts, picking lands on the wrong chunk near a
        silhouette and nothing else looks wrong.
        """
        camera = OrbitCamera(distance=6.0, yaw=31.0, pitch=-17.0, target=(0.4, 1.0, -2.0))
        width, height = 33, 21
        _, grid = camera.ray_directions(width, height)

        for x, y in ((0, 0), (width - 1, 0), (0, height - 1), (16, 10), (7, 13)):
            origin, direction = ray_through(camera, x, y, width, height)
            assert origin == pytest.approx(camera.position, abs=1e-6)
            assert direction == pytest.approx(grid[height - 1 - y, x], abs=1e-6)

    def test_the_centre_pixel_looks_down_the_view_axis(self):
        camera = OrbitCamera(yaw=44.0, pitch=22.0)
        _, direction = ray_through(camera, 32, 16, 65, 33)
        assert direction == pytest.approx(camera.basis()[2], abs=1e-6)

    def test_dragging_the_mouse_down_lowers_the_ray(self):
        """A y-down mouse coordinate must not produce a y-up ray."""
        camera = OrbitCamera()
        _, high = ray_through(camera, 50, 10, 100, 100)
        _, low = ray_through(camera, 50, 90, 100, 100)
        up = camera.basis()[1]
        assert high @ up > 0.0
        assert low @ up < 0.0

    def test_a_degenerate_viewport_does_not_divide_by_zero(self):
        _, direction = ray_through(OrbitCamera(), 0, 0, 0, 0)
        assert np.isfinite(direction).all()


class TestPicking:
    """The camera sits at +Z looking down -Z, so 'nearest' is 'largest z'."""

    @staticmethod
    def scene():
        camera = OrbitCamera(distance=10.0, yaw=0.0, pitch=0.0)
        assert camera.position == pytest.approx([0.0, 0.0, 10.0])
        return camera

    def test_the_nearest_sphere_along_the_ray_wins(self):
        camera = self.scene()
        origin, direction = ray_through(camera, 50, 50, 101, 101)

        centres = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
        radii = np.array([0.5, 0.5])

        index, distance = nearest_hit(origin, direction, centres, radii)
        assert index == 1
        assert distance == pytest.approx(6.5, abs=1e-6)

    def test_order_in_the_array_does_not_decide_it(self):
        camera = self.scene()
        origin, direction = ray_through(camera, 50, 50, 101, 101)
        centres = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]])
        radii = np.array([0.5, 0.5])
        assert nearest_hit(origin, direction, centres, radii)[0] == 0

    def test_a_ray_that_misses_everything_reports_nothing(self):
        camera = self.scene()
        origin, direction = ray_through(camera, 0, 0, 101, 101)
        centres = np.array([[0.0, 0.0, 0.0]])
        assert nearest_hit(origin, direction, centres, np.array([0.1])) is None

    def test_an_empty_body_reports_nothing(self):
        origin, direction = ray_through(self.scene(), 5, 5, 11, 11)
        assert nearest_hit(origin, direction, np.zeros((0, 3)), np.zeros(0)) is None

    def test_a_sphere_behind_the_eye_is_not_a_hit(self):
        camera = self.scene()
        origin, direction = ray_through(camera, 50, 50, 101, 101)
        centres = np.array([[0.0, 0.0, 20.0]])
        assert nearest_hit(origin, direction, centres, np.array([0.5])) is None

    def test_a_sphere_the_eye_is_inside_still_hits(self):
        camera = self.scene()
        origin, direction = ray_through(camera, 50, 50, 101, 101)
        centres = np.array([[0.0, 0.0, 10.0]])
        index, distance = nearest_hit(origin, direction, centres, np.array([2.0]))
        assert index == 0
        assert distance == pytest.approx(2.0, abs=1e-6)

    def test_a_two_dimensional_body_is_widened_onto_z_zero(self):
        body = Body(dim=2)
        body.add_chunk("a", [1.0, 2.0], radius=0.3)
        centres = chunk_centres(body)
        assert centres.shape == (1, 3)
        assert centres[0] == pytest.approx([1.0, 2.0, 0.0])


class TestOverlayProjection:
    def test_projecting_a_chunk_and_picking_it_back_agree(self):
        """Two independent derivations: the matrix path and the analytic ray path.

        The overlay dot has to sit where a click on it would land, or selection
        and its own feedback disagree.
        """
        camera = OrbitCamera(distance=7.0, yaw=37.0, pitch=-21.0, target=(0.2, 0.1, 0.0))
        width, height = SIZE
        body = worm()
        centres = chunk_centres(body)

        matrix = camera.view_projection_matrix(width, height)
        screen, visible = project(matrix, centres, width, height)
        assert visible.all()

        for slot in range(body.n_chunks):
            x, y = screen[slot]
            origin, direction = ray_through(camera, x - 0.5, y - 0.5, width, height)
            hit = nearest_hit(origin, direction, centres, np.asarray(body.radius, float))
            assert hit is not None, f"chunk {slot} projected off its own sphere"
            assert hit[0] == slot

    def test_the_centre_of_the_view_projects_to_the_centre_of_the_widget(self):
        camera = OrbitCamera(distance=5.0, yaw=63.0, pitch=12.0, target=(1.0, -2.0, 3.0))
        width, height = 200, 100
        screen, visible = project(
            camera.view_projection_matrix(width, height), camera.target[None, :], width, height
        )
        assert visible[0]
        assert screen[0] == pytest.approx([width / 2.0, height / 2.0], abs=1e-4)

    def test_a_point_behind_the_eye_is_masked_not_mirrored(self):
        """Without the w test it would project to a ghost in front of the camera."""
        camera = OrbitCamera(distance=5.0)
        behind = np.array([[0.0, 0.0, 20.0]])
        _, visible = project(camera.view_projection_matrix(*SIZE), behind, *SIZE)
        assert not visible[0]

    def test_moving_up_in_the_world_moves_up_the_screen(self):
        camera = OrbitCamera(distance=8.0)
        points = np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
        screen, _ = project(camera.view_projection_matrix(*SIZE), points, *SIZE)
        assert screen[1, 1] < screen[0, 1]


class TestDragPlane:
    def test_a_ray_lands_on_the_plane_through_the_chunk(self):
        origin = np.array([0.0, 0.0, 10.0])
        direction = np.array([0.0, 0.0, -1.0])
        landed = plane_hit(origin, direction, np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, -1.0]))
        assert landed == pytest.approx([0.0, 0.0, 3.0])

    def test_a_ray_parallel_to_the_plane_reports_nothing(self):
        origin = np.array([0.0, 0.0, 10.0])
        direction = np.array([1.0, 0.0, 0.0])
        assert plane_hit(origin, direction, np.zeros(3), np.array([0.0, 0.0, -1.0])) is None


# ----------------------------------------------------------------------
# the widget
# ----------------------------------------------------------------------


@needs_gl
class TestViewport:
    @staticmethod
    def viewport(document: Document | None = None, size=SIZE) -> MetaballViewport:
        gl_context.ensure_application()
        document = document if document is not None else document_with(worm())
        view = MetaballViewport(document)
        view.resize(*size)
        view.camera.distance = 8.0
        return view

    @staticmethod
    def image(view: MetaballViewport) -> np.ndarray:
        from PySide6.QtGui import QImage

        frame = view.grabFramebuffer().convertToFormat(QImage.Format.Format_RGB888)
        buffer = np.frombuffer(memoryview(frame.constBits()), np.uint8)
        row = frame.bytesPerLine() // 3
        return buffer.reshape(frame.height(), row, 3)[:, : frame.width()].copy()

    # -- rendering ---------------------------------------------------

    def test_it_renders_the_document_without_a_shader_error(self):
        view = self.viewport()
        failures: list[str] = []
        view.shader_failed.connect(failures.append)

        rgb = self.image(view)

        assert not failures, failures[0]
        assert len(np.unique(rgb.reshape(-1, 3), axis=0)) > 50, "nothing but background"

    def test_an_empty_document_still_renders(self):
        view = self.viewport(Document())
        failures: list[str] = []
        view.shader_failed.connect(failures.append)
        self.image(view)
        assert not failures, failures[0]

    def test_live_metaball_settings_reach_the_shader(self):
        """The panels edit document.metaball; the frame must pick it up with no wiring."""
        view = self.viewport()
        view.show_rig = False

        background = np.array(view.document.metaball.background_colour) * 255.0

        def surface_pixels(rgb):
            return int((np.abs(rgb - background) > 12.0).any(axis=2).sum())

        view.document.metaball.blend = 0.0
        thin = surface_pixels(self.image(view))
        view.document.metaball.blend = 0.6
        fat = surface_pixels(self.image(view))

        assert thin > 0
        assert fat > thin, "a larger blend radius did not thicken the surface"

    # -- the overlay -------------------------------------------------

    def test_the_rig_overlay_actually_reaches_the_framebuffer(self):
        """QPainter begun before the raymarch pass draws nothing at all."""
        view = self.viewport()
        view.show_rig = False
        clean = self.image(view)
        view.show_rig = True
        overlaid = self.image(view)

        changed = int((clean != overlaid).any(axis=2).sum())
        assert changed > 0, "the overlay did not composite over the GL surface"

    def test_the_surface_survives_the_painter(self):
        """A second overlay frame must match the first, or GL state leaked."""
        view = self.viewport()
        view.show_rig = True
        first = self.image(view)
        second = self.image(view)
        assert np.array_equal(first, second)

        view.show_rig = False
        after = self.image(view)
        view_clean = self.viewport(document_with(worm()))
        view_clean.show_rig = False
        assert np.array_equal(after, self.image(view_clean))

    def test_selection_changes_what_the_overlay_draws(self):
        view = self.viewport()
        view.show_rig = True
        unselected = self.image(view)
        view.document.set_selection([view.document.body.chunk_ids()[0]])
        selected = self.image(view)
        assert not np.array_equal(unselected, selected)

    def test_hiding_the_rig_leaves_only_the_surface(self):
        view = self.viewport()
        view.show_rig = True
        assert view.show_rig
        with_rig = self.image(view)
        view.show_rig = False
        assert not np.array_equal(with_rig, self.image(view))

    def test_the_overlay_lands_on_the_projected_chunks(self):
        """A dot drawn in the wrong place is still 'the overlay reached the frame'."""
        view = self.viewport()
        view.show_rig = False
        clean = self.image(view)
        view.show_rig = True
        overlaid = self.image(view)

        changed = np.argwhere((clean != overlaid).any(axis=2))
        assert len(changed) > 0

        points = np.array(list(view.rig_screen_points().values()))
        assert len(points) == view.document.body.n_chunks

        drawn = changed[:, ::-1] / view.devicePixelRatioF()
        for point in points:
            distance = np.linalg.norm(drawn - point, axis=1).min()
            assert distance < 4.0, f"nothing was drawn at the chunk projected to {point}"

    # -- the frame loop ----------------------------------------------

    def test_a_still_paused_document_stops_asking_for_frames(self):
        """The defect this replaces: main.py repainted at 60fps forever."""
        view = self.viewport()
        for _ in range(20):
            view.tick()
        assert view.idle

        settled = view.frames
        for _ in range(30):
            view.tick()
        assert view.frames == settled
        assert view.idle

    def test_a_running_document_never_goes_idle(self):
        view = self.viewport()
        for _ in range(20):
            view.tick()
        view.document.set_running(True)

        before = view.frames
        for _ in range(5):
            view.tick()
        assert not view.idle
        assert view.frames == before + 5

    def test_moving_the_camera_wakes_it_for_one_frame(self):
        view = self.viewport()
        for _ in range(20):
            view.tick()
        assert view.idle

        before = view.frames
        view.camera.orbit(12.0, 0.0)
        view.camera_changed.emit()
        view.tick()
        assert view.frames == before + 1

        view.tick()
        assert view.idle, "a camera nudge should buy one frame, not a stream"

    def test_editing_the_document_wakes_it_until_the_springs_settle(self):
        view = self.viewport()
        for _ in range(60):
            view.tick()
        assert view.idle

        chunk_id = view.document.body.chunk_ids()[0]
        view.document.do(MoveChunk(chunk_id, (5.0, 5.0, 0.0)))
        view.tick()
        assert not view.idle

        for _ in range(400):
            view.tick()
        assert view.idle, "the draw-point springs never converged"
        assert view.document.draw.pos[0] == pytest.approx([5.0, 5.0, 0.0], abs=1e-3)

    def test_the_frame_loop_can_be_stopped_and_restarted(self):
        view = self.viewport()
        assert not view.looping, "an unshown viewport must not burn frames"
        view.start()
        assert view.looping
        view.stop()
        assert not view.looping

    def test_the_idle_transition_is_announced(self):
        view = self.viewport()
        seen: list[bool] = []
        view.idle_changed.connect(seen.append)
        for _ in range(60):
            view.tick()
        assert seen[-1] is True

    def test_it_advances_by_measured_time_not_a_fixed_step(self):
        """A frame that took 200ms must buy more ticks than one that took none."""
        view = self.viewport()
        view.document.set_running(True)
        view.tick()

        from PySide6.QtCore import QThread

        before = view.document.clock.ticks
        view.tick()
        immediate = view.document.clock.ticks - before

        QThread.msleep(200)
        before = view.document.clock.ticks
        view.tick()
        delayed = view.document.clock.ticks - before

        assert immediate == 0
        assert delayed > 0

    # -- uploads -----------------------------------------------------

    def test_the_texture_reallocates_only_when_the_count_changes(self):
        from mesh_xml.render.gl import metaball

        allocations: list[tuple] = []
        real = metaball.glTexImage2D

        def spy(*args):
            allocations.append(args[:6])
            return real(*args)

        view = self.viewport()
        self.image(view)
        assert view.uploads == 1

        metaball.glTexImage2D = spy
        try:
            chunk_id = view.document.body.chunk_ids()[0]
            view.document.do(MoveChunk(chunk_id, (0.0, 1.5, 0.0)))
            view.tick()
            self.image(view)
            assert view.uploads == 2
            assert allocations == [], "a move must reuse the allocation"

            view.document.body.add_chunk("extra", [0.0, -2.0, 0.0], radius=0.3)
            view.document.topology_changed.emit()
            view.tick()
            self.image(view)
            assert view.uploads == 3
            assert len(allocations) == 1, "a new chunk must reallocate exactly once"
        finally:
            metaball.glTexImage2D = real

    def test_it_does_not_re_upload_an_unchanged_rig(self):
        view = self.viewport()
        self.image(view)
        for _ in range(40):
            view.tick()
        settled = view.uploads

        view.camera.orbit(20.0, 5.0)
        view.camera_changed.emit()
        view.tick()
        self.image(view)
        assert view.uploads == settled, "orbiting re-uploaded identical primitives"

    def test_the_first_paint_populates_the_texture_without_a_tick(self):
        view = self.viewport()
        self.image(view)
        assert view.texture.count == 4
        assert view.texture.capacity == 4

    # -- picking -----------------------------------------------------

    def test_clicking_a_chunk_selects_it(self):
        view = self.viewport()
        chunk_id = view.document.body.chunk_ids()[3]
        x, y = view.rig_screen_points()[chunk_id]

        press(view, x, y)
        release(view, x, y)
        assert view.document.selection == {chunk_id}

    def test_clicking_empty_space_clears_the_selection(self):
        view = self.viewport()
        view.document.set_selection(view.document.body.chunk_ids())
        press(view, 1, 1)
        release(view, 1, 1)
        assert view.document.selection == set()

    def test_a_plain_click_replaces_and_shift_click_toggles(self):
        view = self.viewport()
        points = view.rig_screen_points()
        first, second = view.document.body.chunk_ids()[:2]

        press(view, *points[first])
        release(view, *points[first])
        assert view.document.selection == {first}

        press(view, *points[second], modifiers=Qt.KeyboardModifier.ShiftModifier)
        release(view, *points[second], modifiers=Qt.KeyboardModifier.ShiftModifier)
        assert view.document.selection == {first, second}

        press(view, *points[second], modifiers=Qt.KeyboardModifier.ShiftModifier)
        release(view, *points[second], modifiers=Qt.KeyboardModifier.ShiftModifier)
        assert view.document.selection == {first}

        press(view, *points[second])
        release(view, *points[second])
        assert view.document.selection == {second}

    def test_selection_goes_through_the_document_not_around_it(self):
        view = self.viewport()
        emitted: list[int] = []
        view.document.selection_changed.connect(lambda: emitted.append(1))
        chunk_id = view.document.body.chunk_ids()[0]
        press(view, *view.rig_screen_points()[chunk_id])
        assert emitted, "the document never learned about the selection"

    def test_clicking_empty_space_still_orbits(self):
        view = self.viewport()
        before = view.camera.yaw
        press(view, 2, 2)
        move(view, 42, 2)
        release(view, 42, 2)
        assert view.camera.yaw != before

    # -- dragging ----------------------------------------------------

    def test_one_drag_is_exactly_one_command(self):
        """The whole reason the drag previews on the body instead of committing."""
        view = self.viewport()
        chunk_id = view.document.body.chunk_ids()[2]
        start = view.document.body.chunk(chunk_id).pos
        x, y = view.rig_screen_points()[chunk_id]

        press(view, x, y)
        move(view, x + 8, y + 3)
        move(view, x + 17, y + 9)
        move(view, x + 26, y + 14)
        release(view, x + 26, y + 14)

        assert view.document.can_undo
        assert view.document.undo_text == "Move Chunk"

        moved = view.document.body.chunk(chunk_id).pos
        assert moved != start

        view.document.undo()
        assert view.document.body.chunk(chunk_id).pos == pytest.approx(start)
        assert not view.document.can_undo, "the drag left more than one command behind"

    def test_the_command_captures_the_position_before_the_drag(self):
        """MoveChunk reads the current position when it applies, so the preview
        has to be rewound first or undo becomes a no-op."""
        view = self.viewport()
        chunk_id = view.document.body.chunk_ids()[1]
        start = view.document.body.chunk(chunk_id).pos
        x, y = view.rig_screen_points()[chunk_id]

        press(view, x, y)
        for step in range(1, 6):
            move(view, x + step * 6, y - step * 4)
        release(view, x + 30, y - 20)

        dragged = view.document.body.chunk(chunk_id).pos
        view.document.undo()
        assert view.document.body.chunk(chunk_id).pos == pytest.approx(start)
        view.document.redo()
        assert view.document.body.chunk(chunk_id).pos == pytest.approx(dragged)

    def test_the_body_moves_live_during_the_drag(self):
        view = self.viewport()
        chunk_id = view.document.body.chunk_ids()[2]
        start = view.document.body.chunk(chunk_id).pos
        x, y = view.rig_screen_points()[chunk_id]

        press(view, x, y)
        move(view, x + 30, y)
        assert view.document.body.chunk(chunk_id).pos != start
        assert not view.document.can_undo, "a mouse move must not record a command"
        release(view, x + 30, y)

    def test_a_drag_stays_in_the_plane_facing_the_camera(self):
        view = self.viewport()
        view.camera.yaw = 0.0
        view.camera.pitch = 0.0
        chunk_id = view.document.body.chunk_ids()[2]
        before = view.document.body.chunk(chunk_id).pos
        x, y = view.rig_screen_points()[chunk_id]

        press(view, x, y)
        move(view, x + 40, y - 25)
        release(view, x + 40, y - 25)

        after = view.document.body.chunk(chunk_id).pos
        assert after[2] == pytest.approx(before[2], abs=1e-5)
        assert after[0] > before[0]
        assert after[1] > before[1]

    def test_the_chunk_does_not_jump_to_the_cursor_on_grab(self):
        """Pressing off-centre but inside a chunk must not recentre it."""
        view = self.viewport()
        chunk_id = view.document.body.chunk_ids()[2]
        before = view.document.body.chunk(chunk_id).pos
        x, y = view.rig_screen_points()[chunk_id]

        press(view, x + 4, y + 4)
        move(view, x + 4, y + 4)
        assert view.document.body.chunk(chunk_id).pos == pytest.approx(before, abs=1e-5)
        release(view, x + 4, y + 4)
        assert not view.document.can_undo

    def test_a_press_and_release_with_no_movement_records_nothing(self):
        view = self.viewport()
        chunk_id = view.document.body.chunk_ids()[0]
        x, y = view.rig_screen_points()[chunk_id]
        press(view, x, y)
        release(view, x, y)
        assert not view.document.can_undo

    def test_dragging_a_chunk_does_not_orbit_the_camera(self):
        view = self.viewport()
        yaw, pitch = view.camera.yaw, view.camera.pitch
        chunk_id = view.document.body.chunk_ids()[2]
        x, y = view.rig_screen_points()[chunk_id]

        press(view, x, y)
        move(view, x + 50, y + 30)
        release(view, x + 50, y + 30)

        assert view.camera.yaw == yaw
        assert view.camera.pitch == pitch

    # -- camera ------------------------------------------------------

    def test_right_drag_pans(self):
        view = self.viewport()
        before = view.camera.target.copy()
        press(view, 40, 40, button=Qt.MouseButton.RightButton)
        move(view, 70, 60, button=Qt.MouseButton.RightButton)
        release(view, 70, 60, button=Qt.MouseButton.RightButton)
        assert not np.allclose(view.camera.target, before)

    def test_middle_drag_pans(self):
        view = self.viewport()
        before = view.camera.target.copy()
        press(view, 40, 40, button=Qt.MouseButton.MiddleButton)
        move(view, 70, 60, button=Qt.MouseButton.MiddleButton)
        release(view, 70, 60, button=Qt.MouseButton.MiddleButton)
        assert not np.allclose(view.camera.target, before)

    def test_frame_all_fits_the_rig_in_view(self):
        view = self.viewport(document_with(worm(n=9, spacing=1.6, radius=0.5)))
        view.frame_all()

        width, height = view.width(), view.height()
        matrix = view.camera.view_projection_matrix(width, height)
        screen, visible = project(matrix, chunk_centres(view.document.body), width, height)

        assert visible.all()
        assert (screen[:, 0] >= 0).all() and (screen[:, 0] <= width).all()
        assert (screen[:, 1] >= 0).all() and (screen[:, 1] <= height).all()

    def test_frame_all_reaches_past_the_default_dolly_limit(self):
        """The default max_distance is 50; a big rig silently clamps without this."""
        view = self.viewport(document_with(worm(n=12, spacing=8.0, radius=2.0)))
        view.frame_all()
        assert view.camera.distance > 50.0
        assert view.camera.far > view.camera.distance

    def test_frame_all_on_an_empty_document_does_not_explode(self):
        view = self.viewport(Document())
        view.frame_all()
        assert np.isfinite(view.camera.target).all()
        assert view.camera.distance > 0.0

    def test_frame_all_centres_on_the_rig(self):
        body = worm()
        for chunk_id in body.chunk_ids():
            pos = body.chunk(chunk_id).pos
            body.set_pos(chunk_id, (pos[0] + 6.0, pos[1] + 3.0, pos[2]))
        view = self.viewport(document_with(body))
        view.frame_all()
        assert view.camera.target[0] == pytest.approx(6.0, abs=0.5)
        assert view.camera.target[1] == pytest.approx(3.0, abs=0.5)
