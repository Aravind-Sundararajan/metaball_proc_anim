"""Tests for the fixed-timestep clock."""

import pytest

from mesh_xml.anim import Clock


class TestTickYield:
    def test_exact_multiples_yield_exact_ticks(self):
        clock = Clock(dt=0.025)
        assert len(clock.advance(0.1)) == 4
        assert clock.ticks == 4

    def test_partial_time_is_banked_not_lost(self):
        clock = Clock(dt=0.025)
        assert len(clock.advance(0.03)) == 1
        assert len(clock.advance(0.02)) == 1
        assert clock.ticks == 2

    def test_sub_tick_frames_accumulate_into_a_tick(self):
        clock = Clock(dt=0.025)
        for _ in range(4):
            assert len(clock.advance(0.006)) == 0
        assert len(clock.advance(0.006)) == 1

    def test_advance_returns_a_range_so_ticks_cannot_be_banked_and_forgotten(self):
        clock = Clock(dt=0.025)
        assert list(clock.advance(0.05)) == [0, 1]

    def test_negative_elapsed_is_ignored(self):
        clock = Clock(dt=0.025)
        clock.advance(0.03)
        before = clock.accumulator
        assert len(clock.advance(-5.0)) == 0
        assert clock.accumulator == pytest.approx(before)


class TestCatchUpCap:
    def test_a_long_stall_is_capped(self):
        clock = Clock(dt=0.025, max_ticks=8)
        assert len(clock.advance(10.0)) == 8

    def test_overshoot_is_dropped_not_carried(self):
        """Carrying the backlog forward is the spiral of death; running slow is not."""
        # dt is a negative power of two so the tick count is exact and the
        # assertion below is testing the cap, not float representation.
        clock = Clock(dt=0.25, max_ticks=8)
        assert len(clock.advance(10.0)) == 8
        assert clock.dropped == 32
        assert len(clock.advance(0.0)) == 0
        assert clock.accumulator < clock.dt

    def test_steady_frames_never_drop(self):
        clock = Clock(dt=1 / 40)
        for _ in range(600):
            clock.advance(1 / 60)
        assert clock.dropped == 0
        assert clock.ticks == pytest.approx(400, abs=1)


class TestAlpha:
    def test_alpha_is_zero_on_an_exact_boundary(self):
        clock = Clock(dt=0.025)
        clock.advance(0.05)
        assert clock.alpha == pytest.approx(0.0)

    def test_alpha_reports_partial_progress(self):
        clock = Clock(dt=0.025)
        clock.advance(0.0375)
        assert clock.alpha == pytest.approx(0.5)

    def test_alpha_stays_in_unit_interval(self):
        clock = Clock(dt=1 / 40)
        for elapsed in (0.001, 0.017, 0.033, 0.4, 2.0):
            clock.advance(elapsed)
            assert 0.0 <= clock.alpha < 1.0


class TestLifecycle:
    def test_reset_drops_banked_time(self):
        clock = Clock(dt=0.025)
        clock.advance(0.02)
        clock.reset()
        assert clock.alpha == 0.0
        assert len(clock.advance(0.0)) == 0

    @pytest.mark.parametrize("dt", [0.0, -1.0])
    def test_non_positive_dt_rejected(self, dt):
        with pytest.raises(ValueError):
            Clock(dt=dt)

    def test_max_ticks_below_one_rejected(self):
        with pytest.raises(ValueError):
            Clock(max_ticks=0)
