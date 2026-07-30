"""Tests for the orbit camera.

No GL anywhere in this file, by design — it imports the camera submodule
directly rather than the ``gl`` package so nothing pulls in PyOpenGL. The
camera is the piece the raymarcher is most likely to be silently wrong about,
and a wrong ray produces a plausible-looking image, so most of these assert
geometric relationships rather than literal numbers.
"""

import math

import numpy as np
import pytest

from mesh_xml.render.gl.camera import (
    MAX_DISTANCE,
    MIN_DISTANCE,
    ORBIT_DEGREES_PER_PIXEL,
    PITCH_LIMIT,
    OrbitCamera,
    viewport_aspect,
)


def unproject(camera: OrbitCamera, ndc: tuple[float, float], width: int, height: int):
    """World-space ray for an NDC point, derived from the matrices alone.

    Independent of ray_directions on purpose: it goes through the projection and
    view matrices, so agreeing with it means the analytic ray construction and
    the matrices describe the same camera.
    """
    matrix = camera.view_projection_matrix(width, height).astype(np.float64)
    inverse = np.linalg.inv(matrix)
    near = inverse @ np.array([ndc[0], ndc[1], -1.0, 1.0])
    far = inverse @ np.array([ndc[0], ndc[1], 1.0, 1.0])
    near = near[:3] / near[3]
    far = far[:3] / far[3]
    direction = far - near
    return near, direction / np.linalg.norm(direction)


class TestRays:
    def test_the_centre_ray_is_the_view_direction(self):
        """Odd dimensions so one pixel sits exactly on the centre."""
        camera = OrbitCamera(yaw=37.0, pitch=-14.0)
        forward = camera.basis()[2]

        origin, directions = camera.ray_directions(65, 33)

        assert origin == pytest.approx(camera.position)
        assert directions[16, 32] == pytest.approx(forward, abs=1e-6)

    def test_opposite_corners_are_symmetric(self):
        """Their directions must mirror through the view axis, or the frustum is skewed."""
        camera = OrbitCamera(yaw=25.0, pitch=10.0)
        _, directions = camera.ray_directions(64, 32)
        right, up, forward = camera.basis()

        bottom_left = directions[0, 0].astype(np.float64)
        top_right = directions[-1, -1].astype(np.float64)

        assert bottom_left @ forward == pytest.approx(top_right @ forward, abs=1e-6)
        assert bottom_left @ right == pytest.approx(-(top_right @ right), abs=1e-6)
        assert bottom_left @ up == pytest.approx(-(top_right @ up), abs=1e-6)

        bottom_right = directions[0, -1].astype(np.float64)
        top_left = directions[-1, 0].astype(np.float64)
        assert bottom_right @ right == pytest.approx(-(top_left @ right), abs=1e-6)
        assert bottom_right @ up == pytest.approx(-(top_left @ up), abs=1e-6)

    def test_every_direction_is_unit_length(self):
        camera = OrbitCamera(yaw=-63.0, pitch=41.0, fov_y=80.0)
        _, directions = camera.ray_directions(37, 21)

        lengths = np.linalg.norm(directions.astype(np.float64), axis=2)
        assert lengths == pytest.approx(1.0, abs=1e-6)

    def test_rays_match_the_projection_matrix(self):
        """The strongest check available headless: two independent derivations agree."""
        camera = OrbitCamera(distance=7.5, yaw=118.0, pitch=-33.0, target=(1.0, -2.0, 3.0))
        width, height = 16, 9
        origin, directions = camera.ray_directions(width, height)

        for row, column in ((0, 0), (4, 11), (8, 15), (5, 5)):
            ndc_x = 2.0 * (column + 0.5) / width - 1.0
            ndc_y = 2.0 * (row + 0.5) / height - 1.0
            point, expected = unproject(camera, (ndc_x, ndc_y), width, height)

            assert directions[row, column] == pytest.approx(expected, abs=1e-5)
            # The unprojected near-plane point must lie on the ray from the eye.
            offset = point - origin.astype(np.float64)
            assert offset / np.linalg.norm(offset) == pytest.approx(expected, abs=1e-5)

    def test_wider_viewports_spread_rays_horizontally_only(self):
        """Compared as angles, not components: normalising couples the two axes."""
        camera = OrbitCamera()
        right, up, forward = (v.astype(np.float64) for v in camera.basis())
        _, square = camera.ray_directions(33, 33)
        _, wide = camera.ray_directions(65, 33)

        square_top = square[-1, 16].astype(np.float64)
        wide_top = wide[-1, 32].astype(np.float64)
        assert square_top @ right == pytest.approx(0.0, abs=1e-6)
        assert wide_top @ right == pytest.approx(0.0, abs=1e-6)
        assert (square_top @ up) / (square_top @ forward) == pytest.approx(
            (wide_top @ up) / (wide_top @ forward), abs=1e-6
        )

        square_edge = square[16, -1].astype(np.float64)
        wide_edge = wide[16, -1].astype(np.float64)
        assert abs(wide_edge @ right / (wide_edge @ forward)) > abs(
            square_edge @ right / (square_edge @ forward)
        )

    def test_row_zero_is_the_bottom_of_the_image(self):
        """The single row-order convention, and the one the GPU comparison rests on.

        Getting it backwards produces a vertically mirrored render that still
        looks plausible until something asymmetric is in frame.
        """
        camera = OrbitCamera()
        up = camera.basis()[1]
        _, directions = camera.ray_directions(16, 16)

        assert directions[0, 8] @ up < 0.0
        assert directions[-1, 8] @ up > 0.0
        assert directions[0, 8] @ up == pytest.approx(-(directions[-1, 8] @ up), abs=1e-6)


class TestDegenerateViewports:
    def test_zero_height_does_not_divide_by_zero(self):
        """Qt delivers zero-height resizes, and an unguarded aspect division raises
        from inside a paint callback where the traceback is useless."""
        camera = OrbitCamera()

        assert viewport_aspect(800, 0) == 1.0
        projection = camera.projection_matrix(800, 0)
        assert np.isfinite(projection).all()

        origin, directions = camera.ray_directions(800, 0)
        assert directions.shape == (0, 800, 3)
        assert np.isfinite(origin).all()

    @pytest.mark.parametrize(
        ("width", "height"), [(0, 0), (0, 600), (-4, 10), (10, -4)]
    )
    def test_every_degenerate_size_is_survivable(self, width, height):
        camera = OrbitCamera()
        assert viewport_aspect(width, height) == 1.0
        assert np.isfinite(camera.projection_matrix(width, height)).all()
        assert np.isfinite(camera.ray_directions(width, height)[1]).all()


class TestProjection:
    def test_aspect_ratio_scales_x_only(self):
        camera = OrbitCamera(fov_y=45.0)
        square = camera.projection_matrix(100, 100)
        wide = camera.projection_matrix(200, 100)

        assert wide[0, 0] == pytest.approx(square[0, 0] / 2.0, rel=1e-6)
        assert wide[1, 1] == pytest.approx(square[1, 1], rel=1e-6)

    def test_vertical_field_of_view_is_the_authored_one(self):
        camera = OrbitCamera(fov_y=60.0)
        projection = camera.projection_matrix(400, 300)

        assert projection[1, 1] == pytest.approx(1.0 / math.tan(math.radians(30.0)), rel=1e-6)

    def test_near_and_far_map_to_the_ndc_range(self):
        camera = OrbitCamera(near=0.25, far=80.0)
        projection = camera.projection_matrix(64, 64).astype(np.float64)

        for depth, expected in ((0.25, -1.0), (80.0, 1.0)):
            clip = projection @ np.array([0.0, 0.0, -depth, 1.0])
            assert clip[2] / clip[3] == pytest.approx(expected, abs=1e-5)

    def test_the_target_projects_to_the_screen_centre(self):
        camera = OrbitCamera(distance=6.0, yaw=44.0, pitch=-18.0, target=(2.0, 1.0, -3.0))
        matrix = camera.view_projection_matrix(1280, 720).astype(np.float64)

        clip = matrix @ np.array([*camera.target, 1.0])
        assert clip[3] > 0.0
        assert clip[0] / clip[3] == pytest.approx(0.0, abs=1e-6)
        assert clip[1] / clip[3] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("bad", [0.0, -10.0, 180.0, 200.0])
    def test_impossible_field_of_view_rejected(self, bad):
        with pytest.raises(ValueError):
            OrbitCamera(fov_y=bad)

    @pytest.mark.parametrize(("near", "far"), [(0.0, 10.0), (-1.0, 10.0), (10.0, 1.0)])
    def test_impossible_clip_planes_rejected(self, near, far):
        with pytest.raises(ValueError):
            OrbitCamera(near=near, far=far)


class TestView:
    def test_the_view_basis_is_orthonormal(self):
        camera = OrbitCamera(yaw=143.0, pitch=-67.0)
        view = camera.view_matrix().astype(np.float64)

        assert view[:3, :3] @ view[:3, :3].T == pytest.approx(np.eye(3), abs=1e-6)
        assert np.linalg.det(view[:3, :3]) == pytest.approx(1.0, abs=1e-6)

    def test_the_view_matrix_puts_the_target_straight_ahead(self):
        camera = OrbitCamera(distance=4.0, yaw=70.0, pitch=20.0, target=(-1.0, 5.0, 2.0))
        view = camera.view_matrix().astype(np.float64)

        seen = view @ np.array([*camera.target, 1.0])
        assert seen[:3] == pytest.approx([0.0, 0.0, -4.0], abs=1e-5)

    def test_the_eye_sits_distance_away_from_the_target(self):
        camera = OrbitCamera(distance=9.0, yaw=-31.0, pitch=55.0, target=(3.0, 0.0, -2.0))
        offset = camera.position.astype(np.float64) - camera.target

        assert np.linalg.norm(offset) == pytest.approx(9.0, abs=1e-5)

    def test_the_default_camera_looks_down_negative_z(self):
        """At rest the camera sits on +Z looking back at the origin, GL's own default."""
        camera = OrbitCamera(distance=10.0)

        assert camera.position == pytest.approx([0.0, 0.0, 10.0], abs=1e-6)
        assert camera.basis()[2] == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)

    def test_the_basis_survives_the_pitch_pole(self):
        """A look-at against world up degenerates here. The Rx@Ry composition does not."""
        camera = OrbitCamera(pitch=PITCH_LIMIT, yaw=33.0)
        right, up, forward = (v.astype(np.float64) for v in camera.basis())

        assert forward == pytest.approx([0.0, -1.0, 0.0], abs=1e-6)
        for a, b in ((right, up), (right, forward), (up, forward)):
            assert a @ b == pytest.approx(0.0, abs=1e-6)
            assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-6)
            assert np.linalg.norm(b) == pytest.approx(1.0, abs=1e-6)


class TestInteraction:
    def test_pitch_clamps_at_both_poles(self):
        camera = OrbitCamera()

        camera.orbit(0.0, 100000.0)
        assert camera.pitch == PITCH_LIMIT

        camera.orbit(0.0, -100000.0)
        assert camera.pitch == -PITCH_LIMIT

    def test_pitch_clamps_on_direct_assignment_too(self):
        """The invariant belongs to the camera, not to the mouse handler."""
        camera = OrbitCamera()
        camera.pitch = 275.0
        assert camera.pitch == PITCH_LIMIT
        camera.pitch = -400.0
        assert camera.pitch == -PITCH_LIMIT

        assert OrbitCamera(pitch=1e6).pitch == PITCH_LIMIT

    def test_orbit_applies_the_documented_sensitivity(self):
        camera = OrbitCamera()
        camera.orbit(10.0, 4.0)

        assert camera.yaw == pytest.approx(10.0 * ORBIT_DEGREES_PER_PIXEL)
        assert camera.pitch == pytest.approx(4.0 * ORBIT_DEGREES_PER_PIXEL)

    def test_a_downward_drag_raises_the_eye(self):
        """Grab-and-turn, not fly-the-camera. Reversing this is the classic
        inverted-orbit complaint."""
        camera = OrbitCamera()
        camera.orbit(0.0, 30.0)

        assert camera.position[1] > 0.0

    def test_a_full_turn_returns_the_original_matrix(self):
        camera = OrbitCamera(yaw=37.0, pitch=-12.0, distance=6.0)
        before = camera.view_matrix()

        pixels = int(360.0 / ORBIT_DEGREES_PER_PIXEL)
        for _ in range(pixels):
            camera.orbit(1.0, 0.0)

        assert camera.yaw == pytest.approx(37.0)
        assert camera.view_matrix() == pytest.approx(before, abs=1e-6)

    def test_yaw_stays_bounded_over_a_long_session(self):
        """Unnormalised yaw loses the low bits that make small drags smooth."""
        camera = OrbitCamera()
        for _ in range(20000):
            camera.orbit(7.0, 0.0)

        assert 0.0 <= camera.yaw < 360.0

    def test_zoom_clamps_at_both_ends(self):
        camera = OrbitCamera()

        camera.zoom(120.0 * 1000)
        assert camera.distance == MIN_DISTANCE

        camera.zoom(-120.0 * 1000)
        assert camera.distance == MAX_DISTANCE

    def test_the_dolly_bounds_move_with_the_rig_scale(self):
        """A rig authored at a hundred times the default scale must still be
        viewable; a hardcoded ceiling would make it permanently half off-screen."""
        camera = OrbitCamera(
            distance=400.0, near=1.0, far=5000.0, min_distance=10.0, max_distance=2000.0
        )
        assert camera.distance == 400.0

        camera.zoom(-120.0 * 10000)
        assert camera.distance == 2000.0

    @pytest.mark.parametrize(("low", "high"), [(0.0, 10.0), (-1.0, 10.0), (10.0, 1.0)])
    def test_impossible_dolly_bounds_rejected(self, low, high):
        with pytest.raises(ValueError):
            OrbitCamera(min_distance=low, max_distance=high)

    def test_a_wheel_notch_moves_closer(self):
        camera = OrbitCamera(distance=10.0)
        camera.zoom(120.0)

        assert camera.distance == pytest.approx(10.0 - 1.2)

    def test_pan_slides_the_target_in_the_camera_plane(self):
        camera = OrbitCamera(yaw=30.0, pitch=15.0)
        right, up, forward = (v.astype(np.float64) for v in camera.basis())
        before = camera.target.copy()

        camera.pan(20.0, -10.0, 600)
        moved = camera.target - before

        assert moved @ forward == pytest.approx(0.0, abs=1e-6)
        assert moved @ right < 0.0
        assert moved @ up < 0.0

    def test_pan_tracks_the_cursor_at_the_target_depth(self):
        """One pixel of drag must move the world one pixel, or the grab slips."""
        camera = OrbitCamera(distance=20.0)
        width, height = 800, 600
        before = camera.target.copy()

        camera.pan(-40.0, 0.0, height)

        matrix = camera.view_projection_matrix(width, height).astype(np.float64)
        clip = matrix @ np.array([*before, 1.0])
        ndc_x = clip[0] / clip[3]
        assert ndc_x * width / 2.0 == pytest.approx(-40.0, abs=1e-3)

    def test_panning_further_out_moves_further(self):
        near = OrbitCamera(distance=2.0)
        far = OrbitCamera(distance=40.0)
        near.pan(50.0, 0.0, 600)
        far.pan(50.0, 0.0, 600)

        assert np.linalg.norm(far.target) > np.linalg.norm(near.target)

    def test_a_malformed_target_is_rejected(self):
        with pytest.raises(ValueError):
            OrbitCamera(target=(1.0, 2.0))

    def test_repr_is_readable(self):
        assert "OrbitCamera" in repr(OrbitCamera())


class TestOrthographic:
    """The projection moves the per-pixel variation from directions to origins."""

    def test_all_directions_are_parallel(self):
        camera = OrbitCamera(distance=6.0, yaw=30.0, pitch=20.0, orthographic=True)
        _, directions = camera.ray_directions(16, 12)

        assert np.allclose(directions, directions[0, 0], atol=1e-6)

    def test_origins_tile_a_plane(self):
        camera = OrbitCamera(distance=6.0, orthographic=True, ortho_height=4.0)
        origins, _ = camera.ray_directions(16, 12)

        assert origins.shape == (12, 16, 3)
        # Pixel centres, so the sampled span is one pixel short of the full height.
        span = float(np.linalg.norm(origins[-1, 0] - origins[0, 0]))
        assert span == pytest.approx(4.0 * (1 - 1 / 12), rel=1e-4)

    def test_perspective_still_shares_one_origin(self):
        camera = OrbitCamera(distance=6.0)
        origin, directions = camera.ray_directions(16, 12)

        assert origin.shape == (3,)
        assert directions.shape == (12, 16, 3)

    def test_ortho_height_defaults_to_the_perspective_framing(self):
        """Toggling projection should compare, not re-frame."""
        camera = OrbitCamera(distance=10.0, fov_y=45.0, orthographic=True)
        assert camera.ortho_height == pytest.approx(2.0 * 10.0 * camera.tan_half_fov)

    def test_ortho_height_can_be_overridden_and_cleared(self):
        camera = OrbitCamera(distance=10.0, orthographic=True)
        camera.ortho_height = 3.0
        assert camera.ortho_height == pytest.approx(3.0)
        camera.ortho_height = None
        assert camera.ortho_height == pytest.approx(2.0 * 10.0 * camera.tan_half_fov)

    @pytest.mark.parametrize("bad", [0.0, -2.0])
    def test_non_positive_ortho_height_rejected(self, bad):
        with pytest.raises(ValueError):
            OrbitCamera(orthographic=True, ortho_height=bad)

    def test_a_degenerate_viewport_is_still_safe(self):
        camera = OrbitCamera(orthographic=True)
        origin, directions = camera.ray_directions(0, 0)
        assert directions.size == 0
