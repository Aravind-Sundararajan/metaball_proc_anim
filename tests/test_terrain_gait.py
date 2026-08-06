"""Integration: the walking creature on real ground.

The packages under test landed separately — terrain collision in the solver,
terrain footholds in the picker, each with its own unit suite. What none of
them could test alone is the composition: a self-propelled walker whose feet
are placed by projection onto the same field its body collides with.

The slope sweep is the honest headline. The *static* law says grip fails at
``tan θ = μ`` — 19.3° for μ = 0.35 — and the resting-chunk tests bracket that
to a tenth of a degree. A *walking* creature stalls well below it, here around
12–14°, because propulsion spends the same friction budget standing still gets
to keep: the push that drives the body forward is itself tangential force at
the foot. A creature that walked right up to its static limit would be the
suspicious result.
"""

import numpy as np

from mesh_xml.core.solver import TICK_DT, SolverConfig
from mesh_xml.core.terrain import Terrain
from tests.test_balance import centre_x, walker

RAMP_TICKS = 2500
SLOPE_TICKS = 1600
MU = 0.35


def ramp_world() -> Terrain:
    """A flat floor with a long shallow ramp: rises 4.6 over 26, about 10 deg."""
    return Terrain.flat(0.0, blend=0.4).add_cone(
        (4.0, -1.2, 0.0), 1.4, (30.0, 3.4, 0.0), 1.4
    )


def uphill_plane(theta_degrees: float) -> Terrain:
    theta = np.radians(theta_degrees)
    return Terrain(blend=0.0).add_plane((-np.sin(theta), np.cos(theta), 0.0), 0.0)


def walk_on(terrain: Terrain, ticks: int, drive: float = 0.4):
    """Drive the standard walker across ``terrain`` and audit its footholds.

    The pickers are built with only their legacy ``ground_y``; the world reaches
    them through ``ctx.terrain``, which is the wiring a loaded document uses.
    That the creature needs no per-limb configuration to change worlds is part
    of what is being tested.
    """
    body, stack, back, feet, pickers = walker(drive=drive)
    config = SolverConfig(
        ground_y=0.0, ground_friction=MU, iterations=8, terrain=terrain
    )

    start = centre_x(body)
    residuals: list[float] = []
    for _ in range(ticks):
        stack.tick(body, TICK_DT, config)
        for picker, foot in zip(pickers, feet):
            if picker.foothold is not None:
                surface = float(terrain.distance(np.asarray(picker.foothold)[None, :])[0])
                residuals.append(surface - float(body.radius[body.slot(foot)]))

    assert np.isfinite(body.pos).all()
    return centre_x(body) - start, np.abs(residuals).max()


class TestRampWalk:
    def test_it_crosses_the_floor_and_climbs_the_ramp(self):
        travel, worst = walk_on(ramp_world(), RAMP_TICKS)

        assert travel > 15.0
        # Every foothold sat on the terrain surface, not at any global height.
        assert worst < 1e-3


class TestSlopeLawWalking:
    def test_it_climbs_gentle_ground(self):
        travel, worst = walk_on(uphill_plane(5.0), SLOPE_TICKS)
        assert travel > 4.0
        assert worst < 1e-3

    def test_it_still_makes_way_at_ten_degrees(self):
        travel, _ = walk_on(uphill_plane(10.0), SLOPE_TICKS)
        assert travel > 1.5

    def test_it_cannot_hold_a_slope_past_its_friction(self):
        """25 deg is beyond what walking can spend: it loses ground outright."""
        travel, _ = walk_on(uphill_plane(25.0), SLOPE_TICKS)
        assert travel < -10.0

    def test_the_stall_sits_below_the_static_ceiling(self):
        """Walking must stall before tan(theta) = mu, never after.

        If a future change lets the walker climb *steeper* than its static
        friction limit, the contact model has started inventing grip and this
        is the test that says so.
        """
        static_ceiling = np.degrees(np.arctan(MU))
        climbing, _ = walk_on(uphill_plane(5.0), SLOPE_TICKS)
        at_ceiling, _ = walk_on(uphill_plane(static_ceiling + 1.0), SLOPE_TICKS)

        assert climbing > 0.0
        assert at_ceiling < 0.0
