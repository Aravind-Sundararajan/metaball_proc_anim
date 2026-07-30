"""Tests for the visual smoothing layer.

The load-bearing property is that draw points are *late*. Several tests assert
lag rather than accuracy, which is the opposite of what a test usually wants —
a draw point that tracks its chunk exactly has thrown away the entire effect.
"""

import numpy as np
import pytest

from mesh_xml.core import Body
from mesh_xml.render import DrawPoints


def one_chunk(dim: int = 3) -> tuple[Body, int]:
    body = Body(dim=dim)
    return body, body.add_chunk("a", [0.0] * dim)


class TestLag:
    def test_draw_points_start_on_their_chunks(self):
        body, a = one_chunk()
        draw = DrawPoints(body)
        assert draw.pos[0] == pytest.approx(body.pos[0])

    def test_a_draw_point_trails_a_moving_chunk(self):
        """The whole point. Arriving on time means no secondary motion."""
        body, a = one_chunk()
        draw = DrawPoints(body, stiffness=0.35, damping=0.75)
        body.pos[0] = [10.0, 0.0, 0.0]
        draw.update(body)

        assert 0.0 < draw.pos[0][0] < 10.0

    def test_it_catches_up_once_the_chunk_settles(self):
        body, a = one_chunk()
        draw = DrawPoints(body)
        body.pos[0] = [10.0, 0.0, 0.0]
        for _ in range(400):
            draw.update(body)

        assert draw.pos[0] == pytest.approx(body.pos[0], abs=1e-3)

    def test_low_damping_overshoots(self):
        """Overshoot is the wobble. A critically damped point looks dead."""
        body, a = one_chunk()
        draw = DrawPoints(body, stiffness=0.5, damping=0.98)
        body.pos[0] = [1.0, 0.0, 0.0]

        peak = 0.0
        for _ in range(200):
            draw.update(body)
            peak = max(peak, float(draw.pos[0][0]))

        assert peak > 1.0

    def test_full_stiffness_no_damping_tracks_exactly(self):
        """The degenerate case, and the baseline the lag is measured against."""
        body, a = one_chunk()
        draw = DrawPoints(body, stiffness=1.0, damping=0.0)
        body.pos[0] = [3.0, -2.0, 1.0]
        draw.update(body)

        assert draw.pos[0] == pytest.approx(body.pos[0])

    def test_stiffer_points_lag_less(self):
        body = Body()
        body.add_chunk("loose", [0.0, 0.0, 0.0])
        body.add_chunk("tight", [0.0, 0.0, 0.0])
        draw = DrawPoints(body, damping=0.75)
        draw.stiffness[0] = 0.1
        draw.stiffness[1] = 0.9

        body.pos[:] = [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
        draw.update(body)

        assert draw.pos[1][0] > draw.pos[0][0]


class TestRenderInterpolation:
    def test_alpha_zero_is_the_previous_tick(self):
        body, a = one_chunk()
        draw = DrawPoints(body)
        body.pos[0] = [4.0, 0.0, 0.0]
        draw.update(body)

        assert draw.render_pos(0.0) == pytest.approx(draw.prev_pos)

    def test_alpha_one_is_the_current_tick(self):
        body, a = one_chunk()
        draw = DrawPoints(body)
        body.pos[0] = [4.0, 0.0, 0.0]
        draw.update(body)

        assert draw.render_pos(1.0) == pytest.approx(draw.pos)

    def test_alpha_half_is_the_midpoint(self):
        body, a = one_chunk()
        draw = DrawPoints(body)
        body.pos[0] = [4.0, 0.0, 0.0]
        draw.update(body)

        expected = (draw.prev_pos + draw.pos) / 2.0
        assert draw.render_pos(0.5) == pytest.approx(expected)

    def test_alpha_is_clamped(self):
        body, a = one_chunk()
        draw = DrawPoints(body)
        body.pos[0] = [4.0, 0.0, 0.0]
        draw.update(body)

        assert draw.render_pos(5.0) == pytest.approx(draw.pos)
        assert draw.render_pos(-3.0) == pytest.approx(draw.prev_pos)


class TestTopologyChanges:
    def test_new_chunks_are_seeded_in_place(self):
        body, a = one_chunk()
        draw = DrawPoints(body)
        b = body.add_chunk("b", [7.0, 0.0, 0.0])
        draw.update(body)

        assert draw.pos[body.slot(b)] == pytest.approx([7.0, 0.0, 0.0])

    def test_surviving_points_keep_their_state_across_a_removal(self):
        """Slots shuffle on removal; remapping by id is what stops a visual flush."""
        body = Body()
        a = body.add_chunk("a", [0.0, 0.0, 0.0])
        body.add_chunk("b", [1.0, 0.0, 0.0])
        c = body.add_chunk("c", [2.0, 0.0, 0.0])
        draw = DrawPoints(body)

        body.pos[:] = [[0.0, 5.0, 0.0], [1.0, 5.0, 0.0], [2.0, 5.0, 0.0]]
        for _ in range(3):
            draw.update(body)
        moving_c = draw.vel[body.slot(c)].copy()
        assert np.linalg.norm(moving_c) > 0.1

        body.remove_chunk(a)
        draw.update(body)

        # A rebuild-from-scratch sync would seed c at its chunk with zero
        # velocity. Carrying momentum through the edit is the whole point.
        assert np.linalg.norm(draw.vel[body.slot(c)]) > 0.1
        assert draw.pos[body.slot(c)] != pytest.approx(body.pos[body.slot(c)])

    def test_update_auto_syncs_when_the_chunk_set_changes(self):
        body, a = one_chunk()
        draw = DrawPoints(body)
        for _ in range(5):
            body.add_chunk("more", [1.0, 0.0, 0.0])
        draw.update(body)

        assert len(draw.pos) == body.n_chunks

    def test_snap_drops_all_lag(self):
        body, a = one_chunk()
        draw = DrawPoints(body)
        body.pos[0] = [9.0, 0.0, 0.0]
        draw.update(body)
        draw.snap(body)

        assert draw.pos[0] == pytest.approx(body.pos[0])
        assert draw.vel[0] == pytest.approx([0.0, 0.0, 0.0])


class TestSquash:
    def test_disabled_by_default(self):
        body, a = one_chunk()
        draw = DrawPoints(body)
        body.pos[0] = [10.0, 0.0, 0.0]
        draw.update(body)

        assert draw.render_radius(body) == pytest.approx(body.radius)

    def test_speed_thins_the_radius(self):
        body, a = one_chunk()
        draw = DrawPoints(body, squash=0.5)
        body.pos[0] = [10.0, 0.0, 0.0]
        draw.update(body)

        assert draw.render_radius(body)[0] < body.radius[0]

    def test_a_resting_point_is_unthinned(self):
        body, a = one_chunk()
        draw = DrawPoints(body, squash=0.5)
        for _ in range(200):
            draw.update(body)

        assert draw.render_radius(body) == pytest.approx(body.radius, abs=1e-4)


class TestStability:
    def test_a_long_run_stays_finite(self):
        body = Body()
        for i in range(20):
            body.add_chunk(f"c{i}", [i, 0.0, 0.0])
        draw = DrawPoints(body, stiffness=0.9, damping=0.99)

        for tick in range(3000):
            body.pos[:, 1] = np.sin(tick * 0.1) * 3.0
            draw.update(body)

        assert np.isfinite(draw.pos).all()
        assert np.abs(draw.pos).max() < 1e3

    def test_dim_two_works(self):
        body, a = one_chunk(dim=2)
        draw = DrawPoints(body)
        body.pos[0] = [3.0, 4.0]
        draw.update(body)

        assert draw.pos.shape == (1, 2)

    def test_mismatched_dim_rejected(self):
        body, a = one_chunk(dim=3)
        draw = DrawPoints(body)
        with pytest.raises(ValueError):
            draw.sync(Body(dim=2))

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_out_of_range_coefficients_rejected(self, bad):
        body, a = one_chunk()
        with pytest.raises(ValueError):
            DrawPoints(body, stiffness=bad)
        with pytest.raises(ValueError):
            DrawPoints(body, damping=bad)
