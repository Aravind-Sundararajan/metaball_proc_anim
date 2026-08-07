"""Tests for the per-tile group masks — the raymarcher's broad phase.

Conservatism is the whole contract (tiles.py says why), so the load-bearing
test is a property: for random cameras and random spheres, every pixel whose
ray passes near a sphere must find that group's bit set in its tile. The
converse — bits set for groups a tile never needs — costs speed, not
correctness, and is not asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from mesh_xml.render.gl.camera import OrbitCamera
from mesh_xml.render.tiles import MAX_GROUPS, TILE_SIZE, tile_masks

SIZE = (96, 64)  # 6 x 4 tiles


def ray_sphere_distance(
    origins: np.ndarray, directions: np.ndarray, centre: np.ndarray, far: float
) -> np.ndarray:
    """Min distance from each forward ray segment [0, far] to ``centre``."""
    o = np.broadcast_to(origins, directions.shape).reshape(-1, 3)
    d = directions.reshape(-1, 3)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    along = np.clip(((centre - o) * d).sum(axis=1), 0.0, far)
    closest = o + d * along[:, None]
    return np.linalg.norm(closest - centre, axis=1).reshape(directions.shape[:2])


def bit_of(masks: np.ndarray, row: int, col: int, group: int) -> bool:
    return bool(masks[row, col, group >> 5] & np.uint32(1 << (group & 31)))


def sphere_rows(spheres: list[tuple[tuple[float, float, float], float]]) -> np.ndarray:
    rows = np.zeros((len(spheres), 8), dtype=np.float32)
    for i, (centre, radius) in enumerate(spheres):
        rows[i, 0:3] = centre
        rows[i, 3] = radius
    return rows


class TestConservatism:
    """No influencing group may ever be missing from a tile's mask."""

    @pytest.mark.parametrize("orthographic", [False, True])
    @pytest.mark.parametrize("seed", [3, 11, 29])
    def test_every_near_ray_finds_its_group(self, orthographic, seed):
        rng = np.random.default_rng(seed)
        camera = OrbitCamera(
            distance=float(rng.uniform(5.0, 10.0)),
            yaw=float(rng.uniform(0.0, 360.0)),
            pitch=float(rng.uniform(-90.0, 90.0)),
            target=tuple(rng.uniform(-1.0, 1.0, size=3)),
            orthographic=orthographic,
        )
        spheres = [
            (tuple(rng.uniform(-4.0, 4.0, size=3)), float(rng.uniform(0.3, 1.5)))
            for _ in range(6)
        ]
        bounds = sphere_rows(spheres)
        width, height = SIZE
        masks = tile_masks(bounds, camera, width, height, slack=0.5)
        assert masks is not None

        origins, directions = camera.ray_directions(width, height)
        for group, (centre, radius) in enumerate(spheres):
            # 0.45 < the 0.5 slack: strictly inside what the mask must cover,
            # so the assertion cannot flake on the boundary itself.
            near = ray_sphere_distance(
                np.asarray(origins), directions, np.asarray(centre), far=200.0
            ) < (radius + 0.45)
            ys, xs = np.nonzero(near)
            for y, x in zip(ys, xs):
                assert bit_of(masks, y // TILE_SIZE, x // TILE_SIZE, group), (
                    f"group {group} missing from tile of pixel ({x}, {y}) "
                    f"[camera seed {seed}, ortho={orthographic}]"
                )


class TestExtremeCoordinates:
    """The float32 regression an adversarial sweep found and the fix retired.

    Building view-space coordinates through ``camera.view_matrix()`` — a
    float32 rounding *different* from the one the shader's rays are built
    from — drifted by ``2^-24`` of the coordinate magnitude, which passed the
    fixed slack once a rig sat millions of units from the origin. The mask
    now derives from ``camera.position``/``basis()`` — the rays' own inputs —
    so conservatism must hold out where the old path first failed (3e6) and
    far beyond it.
    """

    @pytest.mark.parametrize("magnitude", [3e6, 1e7, 1e8])
    def test_conservatism_survives_far_from_the_origin(self, magnitude):
        rng = np.random.default_rng(int(magnitude) % 997)
        target = np.array([magnitude, -magnitude / 3.0, magnitude / 7.0])
        camera = OrbitCamera(
            distance=10.0,
            yaw=float(rng.uniform(0.0, 360.0)),
            pitch=float(rng.uniform(-60.0, 60.0)),
            target=tuple(target),
            max_distance=1e9,
        )
        spheres = []
        for _ in range(4):
            centre = np.float32(target + rng.uniform(-3.0, 3.0, size=3))
            spheres.append((tuple(float(v) for v in centre), float(rng.uniform(0.3, 0.8))))
        bounds = sphere_rows(spheres)
        width, height = SIZE
        masks = tile_masks(bounds, camera, width, height, slack=0.5)
        assert masks is not None

        origins, directions = camera.ray_directions(width, height)
        for group, (centre, radius) in enumerate(spheres):
            near = ray_sphere_distance(
                np.asarray(origins), directions, np.asarray(centre), far=200.0
            ) < (radius + 0.45)
            ys, xs = np.nonzero(near)
            for y, x in zip(ys, xs):
                assert bit_of(masks, y // TILE_SIZE, x // TILE_SIZE, group), (
                    f"group {group} missing at ({x}, {y}), |coords|~{magnitude:g}"
                )


class TestEdges:
    def test_a_sphere_strictly_behind_the_eye_is_nowhere(self):
        camera = OrbitCamera(distance=5.0)  # eye at +z looking -z
        bounds = sphere_rows([((0.0, 0.0, 12.0), 1.0)])
        masks = tile_masks(bounds, camera, *SIZE)
        assert masks is not None
        assert int(masks.sum()) == 0

    def test_a_sphere_around_the_camera_is_everywhere(self):
        camera = OrbitCamera(distance=5.0)
        bounds = sphere_rows([((0.0, 0.0, 5.0), 2.0)])
        masks = tile_masks(bounds, camera, *SIZE)
        assert masks is not None
        assert bool((masks[:, :, 0] & np.uint32(1)).all())

    def test_a_sphere_far_off_screen_is_nowhere(self):
        camera = OrbitCamera(distance=5.0)
        bounds = sphere_rows([((80.0, 0.0, 0.0), 1.0)])
        masks = tile_masks(bounds, camera, *SIZE)
        assert masks is not None
        assert int(masks.sum()) == 0

    def test_row_zero_is_the_bottom(self):
        """A sphere well above the view axis must mark only upper tile rows,
        because gl_FragCoord's y grows upward and the grid must agree."""
        camera = OrbitCamera(distance=8.0)
        bounds = sphere_rows([((0.0, 2.5, 0.0), 0.6)])
        masks = tile_masks(bounds, camera, *SIZE, slack=0.1)
        assert masks is not None
        rows_hit = np.nonzero(masks.sum(axis=(1, 2)))[0]
        assert rows_hit.size > 0
        assert rows_hit.min() >= (SIZE[1] // TILE_SIZE) // 2

    def test_no_groups_or_too_many_groups_fall_back(self):
        camera = OrbitCamera(distance=5.0)
        assert tile_masks(np.zeros((0, 8), np.float32), camera, *SIZE) is None
        many = np.zeros((MAX_GROUPS + 1, 8), np.float32)
        many[:, 3] = 0.5
        assert tile_masks(many, camera, *SIZE) is None

    def test_a_degenerate_viewport_falls_back(self):
        camera = OrbitCamera(distance=5.0)
        bounds = sphere_rows([((0.0, 0.0, 0.0), 1.0)])
        assert tile_masks(bounds, camera, 0, 0) is None
