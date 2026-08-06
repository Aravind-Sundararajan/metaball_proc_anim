"""Tests for the terrain field and the primitive-math move that enabled it.

The move test matters as much as the terrain tests: ``render/field.py`` now
delegates its primitive math to ``core/distance.py``, and the GLSL pairing
invariant points at the new home. If delegation ever drifts into duplication,
the CPU oracle and the shader stop being checked against the same numbers.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning", "error::DeprecationWarning")

from mesh_xml.core import distance
from mesh_xml.core.terrain import Terrain
from mesh_xml.render import field


class TestTheMove:
    def test_field_delegates_rather_than_duplicates(self):
        assert field.round_cone is distance.round_cone
        assert field.smooth_min is distance.smooth_min

    def test_the_format_constants_have_one_home(self):
        from mesh_xml.render import sdf

        assert sdf.PRIMITIVE_STRIDE is distance.PRIMITIVE_STRIDE
        assert sdf.DEGENERATE_EPSILON is distance.DEGENERATE_EPSILON


class TestPlanes:
    def test_a_flat_floor_measures_height(self):
        terrain = Terrain.flat(0.0)
        points = np.array([[0.0, 2.0, 0.0], [5.0, 0.5, -3.0], [0.0, -1.0, 0.0]])

        assert terrain.distance(points) == pytest.approx([2.0, 0.5, -1.0])

    def test_offset_is_along_the_unit_normal_whatever_the_scaling(self):
        scaled = Terrain(blend=0.0).add_plane((0.0, 2.0, 0.0), 1.5)
        unit = Terrain(blend=0.0).add_plane((0.0, 1.0, 0.0), 1.5)
        point = np.array([[0.0, 4.0, 0.0]])

        assert scaled.distance(point) == pytest.approx(unit.distance(point))

    def test_a_tilted_plane_is_metric(self):
        # 45 degrees: normal (1, 1, 0) / sqrt(2) through the origin.
        terrain = Terrain(blend=0.0).add_plane((1.0, 1.0, 0.0), 0.0)
        d = float(terrain.distance(np.array([[1.0, 1.0, 0.0]]))[0])

        assert d == pytest.approx(np.sqrt(2.0))

    def test_a_zero_normal_is_rejected(self):
        with pytest.raises(ValueError):
            Terrain().add_plane((0.0, 0.0, 0.0), 1.0)


class TestCones:
    def test_a_terrain_cone_is_the_body_cone(self):
        """Same record, same math — by construction, not by coincidence."""
        record = np.array([1.0, 0.0, 0.0, 0.5, 3.0, 0.0, 0.0, 0.8])
        terrain = Terrain(blend=0.0).add_cone(record[0:3], record[3], record[4:7], record[7])
        points = np.random.default_rng(3).normal(0.0, 2.0, (64, 3))

        expected = distance.round_cone(points, record)
        assert terrain.distance(points) == pytest.approx(expected)

    def test_negative_radius_rejected(self):
        with pytest.raises(ValueError):
            Terrain().add_cone((0, 0, 0), -0.1, (1, 0, 0), 0.2)


class TestBlending:
    def test_blend_never_exceeds_the_hard_minimum(self):
        terrain = Terrain(blend=0.5).add_plane((0, 1, 0), 0.0)
        terrain.add_cone((0.0, 0.2, 0.0), 0.6, (0.0, 0.2, 0.0), 0.6)
        hard = Terrain(blend=0.0).add_plane((0, 1, 0), 0.0)
        hard.add_cone((0.0, 0.2, 0.0), 0.6, (0.0, 0.2, 0.0), 0.6)

        points = np.random.default_rng(5).normal(0.0, 1.5, (128, 3))
        soft_d = terrain.distance(points)
        hard_d = hard.distance(points)

        assert (soft_d <= hard_d + 1e-12).all()
        assert (soft_d >= hard_d - 0.5).all()

    def test_zero_blend_is_the_exact_minimum(self):
        terrain = Terrain(blend=0.0).add_plane((0, 1, 0), 0.0)
        terrain.add_cone((2.0, 0.0, 0.0), 1.0, (2.0, 0.0, 0.0), 1.0)
        point = np.array([[2.0, 0.4, 0.0]])

        plane_d = 0.4
        sphere_d = float(np.linalg.norm([0.0, 0.4, 0.0]) - 1.0)
        assert float(terrain.distance(point)[0]) == pytest.approx(min(plane_d, sphere_d))


class TestNormals:
    def test_floor_normal_is_up(self):
        terrain = Terrain.flat(0.0)
        n = terrain.normal(np.array([[3.0, 0.5, -2.0]]))

        assert n[0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)

    def test_normals_are_unit_and_outward(self):
        terrain = Terrain.flat(0.0).add_cone((0, -0.5, 0), 1.2, (0, -0.5, 0), 1.2)
        points = np.random.default_rng(7).normal(0.0, 2.0, (64, 3))
        points[:, 1] = np.abs(points[:, 1]) + 0.2

        normals = terrain.normal(points)
        lengths = np.linalg.norm(normals, axis=1)
        assert lengths == pytest.approx(np.ones(len(points)), abs=1e-5)

        ahead = terrain.distance(points + normals * 1e-3)
        behind = terrain.distance(points - normals * 1e-3)
        assert (ahead > behind).all()


class TestProjection:
    def test_project_lands_at_the_clearance(self):
        terrain = Terrain.flat(0.0)
        hovering = np.array([[1.0, 3.0, 2.0]])
        buried = np.array([[1.0, -2.0, 2.0]])

        for start in (hovering, buried):
            landed = terrain.project(start, clearance=0.13)
            assert terrain.distance(landed)[0] == pytest.approx(0.13, abs=1e-3)

    def test_project_on_a_slope_lands_on_the_slope(self):
        terrain = Terrain(blend=0.0).add_plane((1.0, 1.0, 0.0), 0.0)
        landed = terrain.project(np.array([[0.0, 2.0, 0.0]]), clearance=0.1)

        assert terrain.distance(landed)[0] == pytest.approx(0.1, abs=1e-3)
        # The nearest point slides down-normal, not straight down.
        assert landed[0][0] < 0.0

    def test_empty_terrain_refuses_to_invent_a_floor(self):
        terrain = Terrain()
        points = np.array([[1.0, 2.0, 3.0]])

        assert np.isinf(terrain.distance(points)).all()
        assert terrain.normal(points) == pytest.approx(np.zeros((1, 3)))
        assert terrain.project(points, clearance=0.1) == pytest.approx(points)


class TestTwoDimensional:
    def test_2d_points_are_answered_in_2d(self):
        terrain = Terrain.flat(0.0)
        d = terrain.distance(np.array([[3.0, 1.5]]))
        n = terrain.normal(np.array([[3.0, 1.5]]))
        p = terrain.project(np.array([[3.0, 1.5]]), clearance=0.2)

        assert float(d[0]) == pytest.approx(1.5)
        assert n.shape == (1, 2)
        assert n[0] == pytest.approx([0.0, 1.0], abs=1e-6)
        assert p.shape == (1, 2)
        assert p[0][1] == pytest.approx(0.2, abs=1e-3)


class TestConfigPassthrough:
    def test_the_context_reads_the_config_terrain(self):
        from mesh_xml.anim import ControlContext, ControllerStack
        from mesh_xml.core import Body
        from mesh_xml.core.solver import SolverConfig

        body = Body()
        body.add_chunk("a", [0.0, 1.0, 0.0])
        stack = ControllerStack(body)
        world = Terrain.flat(0.0)

        seen = {}

        class Spy:
            enabled = True

            def update(self, ctx: ControlContext) -> None:
                seen["terrain"] = ctx.terrain

        stack.add(Spy())
        stack.tick(body, 1 / 40, SolverConfig(terrain=world))

        assert seen["terrain"] is world
