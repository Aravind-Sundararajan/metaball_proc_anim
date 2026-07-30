"""Behavioural tests for the Verlet + position-projection solver.

Pure math: no Qt, no GL, no files. Most cases use zero gravity so that a
constraint's effect is the only thing moving anything, and the assertions can be
exact rather than approximate.
"""

from __future__ import annotations

import numpy as np
import pytest

from mesh_xml.core.body import Body, LinkType
from mesh_xml.core.solver import (
    TICK_DT,
    TICK_HZ,
    SolverConfig,
    fixed_step,
    render_pos,
    step,
)

NO_GRAVITY = (0.0, 0.0, 0.0)


def still(**overrides) -> SolverConfig:
    return SolverConfig(gravity=NO_GRAVITY, **overrides)


def separation(body: Body, id_a: int, id_b: int) -> float:
    return float(np.linalg.norm(body.pos[body.slot(id_a)] - body.pos[body.slot(id_b)]))


def pair(start: float, rest: float, type: LinkType = LinkType.NORMAL) -> tuple:
    """Two unit-mass chunks on the x axis, ``start`` apart, joined by one link."""
    body = Body(dim=3)
    a = body.add_chunk("a", (0.0, 0.0, 0.0))
    b = body.add_chunk("b", (start, 0.0, 0.0))
    body.add_link(a, b, rest=rest, type=type)
    return body, a, b


def chain(n: int, spacing: float = 1.0) -> tuple[Body, list[int]]:
    """``n`` chunks along +x with the first pinned, NORMAL-linked end to end."""
    body = Body(dim=3, name="chain")
    ids = [
        body.add_chunk(f"c{i}", (i * spacing, 0.0, 0.0), pinned=(i == 0))
        for i in range(n)
    ]
    for i in range(1, n):
        body.add_link(ids[i - 1], ids[i], rest=spacing)
    return body, ids


def square(pinned_corner: bool = False) -> tuple[Body, list[int]]:
    """Four chunks on a unit square plus both diagonals — a cycle, on purpose."""
    body = Body(dim=3, name="square")
    corners = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    ids = [
        body.add_chunk(f"q{i}", c, pinned=(pinned_corner and i == 0))
        for i, c in enumerate(corners)
    ]
    for i in range(4):
        body.add_link(ids[i], ids[(i + 1) % 4])
    body.add_link(ids[0], ids[2])
    body.add_link(ids[1], ids[3])
    return body, ids


def run(body: Body, ticks: int, config: SolverConfig) -> None:
    for _ in range(ticks):
        step(body, TICK_DT, config)


# ----------------------------------------------------------------------
# integration
# ----------------------------------------------------------------------


def test_free_chunk_falls_along_gravity() -> None:
    body = Body(dim=3)
    body.add_chunk("c", (0.0, 0.0, 0.0))
    run(body, 40, SolverConfig())

    assert body.pos[0][1] < -0.05
    assert body.pos[0][0] == 0.0
    assert body.pos[0][2] == 0.0


def test_gravity_is_trimmed_to_body_dim() -> None:
    body = Body(dim=2)
    body.add_chunk("c", (0.0, 0.0))
    run(body, 40, SolverConfig())

    assert body.pos[0][0] == 0.0
    assert body.pos[0][1] < -0.05


def test_gravity_is_padded_to_body_dim() -> None:
    body = Body(dim=4)
    body.add_chunk("c", (0.0, 0.0, 0.0, 0.0))
    run(body, 40, SolverConfig(gravity=(1.0, -9.81)))

    assert body.pos[0][0] > 0.0
    assert body.pos[0][1] < 0.0
    assert body.pos[0][2] == 0.0
    assert body.pos[0][3] == 0.0


def test_damping_bleeds_off_velocity() -> None:
    body = Body(dim=3)
    body.add_chunk("c", (0.0, 0.0, 0.0))
    body.pos[0] = (1.0, 0.0, 0.0)

    run(body, 500, still(damping=0.5))
    travelled = float(body.pos[0][0])

    run(body, 500, still(damping=0.5))
    assert float(body.pos[0][0]) == pytest.approx(travelled, abs=1e-5)


def test_max_velocity_clamps_a_teleported_chunk() -> None:
    body = Body(dim=3)
    body.add_chunk("c", (0.0, 0.0, 0.0))
    body.pos[0] = (1e6, 0.0, 0.0)

    config = still(max_velocity=10.0)
    step(body, TICK_DT, config)

    assert float(body.pos[0][0] - 1e6) == pytest.approx(10.0 * TICK_DT, rel=1e-3)


def test_empty_body_is_a_noop() -> None:
    step(Body(dim=3), TICK_DT, SolverConfig())


def test_body_without_links_still_integrates() -> None:
    body = Body(dim=3)
    body.add_chunk("a", (0.0, 0.0, 0.0))
    body.add_chunk("b", (5.0, 0.0, 0.0))
    run(body, 20, SolverConfig())

    assert np.isfinite(body.pos).all()
    assert body.pos[0][1] == pytest.approx(body.pos[1][1])


# ----------------------------------------------------------------------
# pinning
# ----------------------------------------------------------------------


def test_pinned_chunk_never_moves_under_gravity() -> None:
    body = Body(dim=3)
    body.add_chunk("anchor", (1.0, 2.0, 3.0), pinned=True)

    for _ in range(500):
        step(body, TICK_DT, SolverConfig())
        assert tuple(body.pos[0]) == (1.0, 2.0, 3.0)

    assert tuple(body.prev_pos[0]) == (1.0, 2.0, 3.0)


def test_pinned_chunk_never_moves_under_constraint_pressure() -> None:
    body = Body(dim=3)
    anchor = body.add_chunk("anchor", (0.0, 0.0, 0.0), pinned=True)
    heavy = body.add_chunk("heavy", (5.0, 0.0, 0.0), mass=1000.0)
    body.add_link(anchor, heavy, rest=1.0)

    for _ in range(500):
        step(body, TICK_DT, SolverConfig())
        assert tuple(body.pos[body.slot(anchor)]) == (0.0, 0.0, 0.0)

    assert separation(body, anchor, heavy) == pytest.approx(1.0, abs=1e-4)


def test_two_pinned_chunks_on_a_violated_link_are_left_alone() -> None:
    body = Body(dim=3)
    a = body.add_chunk("a", (0.0, 0.0, 0.0), pinned=True)
    b = body.add_chunk("b", (9.0, 0.0, 0.0), pinned=True)
    body.add_link(a, b, rest=1.0)

    run(body, 50, SolverConfig())

    assert np.isfinite(body.pos).all()
    assert separation(body, a, b) == 9.0


# ----------------------------------------------------------------------
# NORMAL links
# ----------------------------------------------------------------------


@pytest.mark.parametrize("start", [3.0, 0.2])
def test_normal_link_converges_to_rest(start: float) -> None:
    body, a, b = pair(start, rest=1.0)
    config = still()

    step(body, TICK_DT, config)
    assert separation(body, a, b) == pytest.approx(1.0, abs=1e-5)

    run(body, 400, config)
    assert separation(body, a, b) == pytest.approx(1.0, abs=1e-5)
    assert np.isfinite(body.pos).all()


@pytest.mark.parametrize("start", [3.0, 0.2])
def test_normal_link_preserves_the_midpoint_for_equal_masses(start: float) -> None:
    body, a, b = pair(start, rest=1.0)
    midpoint = body.pos.mean(axis=0).copy()

    run(body, 200, still())

    assert body.pos.mean(axis=0) == pytest.approx(midpoint, abs=1e-4)


def test_elasticity_scales_the_correction() -> None:
    body, a, b = pair(3.0, rest=1.0)
    body.link_elasticity[0] = 0.5

    step(body, TICK_DT, still(iterations=1))

    assert separation(body, a, b) == pytest.approx(2.0, abs=1e-5)


# ----------------------------------------------------------------------
# PULL / PUSH — the point of the exercise
# ----------------------------------------------------------------------


def test_pull_does_not_push_apart_when_compressed() -> None:
    body, a, b = pair(0.4, rest=2.0, type=LinkType.PULL)
    before = body.pos.copy()

    run(body, 200, still())

    assert (body.pos == before).all()


def test_pull_corrects_a_stretch() -> None:
    body, a, b = pair(3.0, rest=1.0, type=LinkType.PULL)

    step(body, TICK_DT, still())

    assert separation(body, a, b) == pytest.approx(1.0, abs=1e-5)


def test_push_does_not_pull_together_when_stretched() -> None:
    body, a, b = pair(3.0, rest=1.0, type=LinkType.PUSH)
    before = body.pos.copy()

    run(body, 200, still())

    assert (body.pos == before).all()


def test_push_corrects_an_overlap() -> None:
    body, a, b = pair(0.3, rest=1.0, type=LinkType.PUSH)

    step(body, TICK_DT, still())

    assert separation(body, a, b) == pytest.approx(1.0, abs=1e-5)


def test_push_lets_the_separation_it_created_keep_going() -> None:
    """Resolving an overlap hands the chunks real velocity, and PUSH never brakes.

    A position change *is* a velocity change under Verlet, so the separation
    impulse persists after the constraint goes slack. That is the intended
    behaviour, and it is why a PUSH link cannot substitute for a NORMAL one.
    """
    body, a, b = pair(0.3, rest=1.0, type=LinkType.PUSH)

    run(body, 200, still())

    assert separation(body, a, b) > 1.0
    assert np.isfinite(body.pos).all()


# ----------------------------------------------------------------------
# inverse-mass weighting
# ----------------------------------------------------------------------


def test_light_chunk_takes_the_larger_share_of_the_correction() -> None:
    body = Body(dim=3)
    light = body.add_chunk("light", (0.0, 0.0, 0.0), mass=1.0)
    heavy = body.add_chunk("heavy", (3.0, 0.0, 0.0), mass=100.0)
    body.add_link(light, heavy, rest=1.0)

    before = body.pos.copy()
    step(body, TICK_DT, still(iterations=1))
    moved = np.linalg.norm(body.pos - before, axis=1)

    assert moved[body.slot(light)] / moved[body.slot(heavy)] == pytest.approx(
        100.0, rel=1e-3
    )
    assert moved.sum() == pytest.approx(2.0, abs=1e-4)


def test_pinned_neighbour_hands_over_the_entire_correction() -> None:
    body = Body(dim=3)
    anchor = body.add_chunk("anchor", (0.0, 0.0, 0.0), pinned=True)
    free = body.add_chunk("free", (3.0, 0.0, 0.0))
    body.add_link(anchor, free, rest=1.0)

    before = body.pos.copy()
    step(body, TICK_DT, still(iterations=1))
    moved = np.linalg.norm(body.pos - before, axis=1)

    assert moved[body.slot(anchor)] == 0.0
    assert moved[body.slot(free)] == pytest.approx(2.0, abs=1e-5)


# ----------------------------------------------------------------------
# degeneracies
# ----------------------------------------------------------------------


def test_coincident_chunks_produce_no_nan() -> None:
    body = Body(dim=3)
    a = body.add_chunk("a", (2.0, -1.0, 0.5))
    b = body.add_chunk("b", (2.0, -1.0, 0.5))
    body.add_link(a, b, rest=1.0)

    run(body, 200, still())

    assert np.isfinite(body.pos).all()
    assert np.isfinite(body.prev_pos).all()
    assert separation(body, a, b) == pytest.approx(1.0, abs=1e-5)


def test_coincident_chunks_with_zero_rest_stay_put() -> None:
    body = Body(dim=3)
    a = body.add_chunk("a", (1.0, 2.0, 3.0))
    b = body.add_chunk("b", (1.0, 2.0, 3.0))
    body.add_link(a, b, rest=0.0)

    run(body, 200, still())

    assert np.isfinite(body.pos).all()
    assert separation(body, a, b) == 0.0


def test_coincident_pinned_and_free_produces_no_nan() -> None:
    body = Body(dim=3)
    anchor = body.add_chunk("anchor", (0.0, 0.0, 0.0), pinned=True)
    free = body.add_chunk("free", (0.0, 0.0, 0.0))
    body.add_link(anchor, free, rest=1.0)

    run(body, 100, still())

    assert np.isfinite(body.pos).all()
    assert tuple(body.pos[body.slot(anchor)]) == (0.0, 0.0, 0.0)
    assert separation(body, anchor, free) == pytest.approx(1.0, abs=1e-5)


# ----------------------------------------------------------------------
# stability
# ----------------------------------------------------------------------


def test_long_chain_does_not_diverge() -> None:
    body, ids = chain(20)
    config = SolverConfig()

    motion = []
    for _ in range(2000):
        step(body, TICK_DT, config)
        motion.append(float(np.abs(body.pos - body.prev_pos).sum()))

    assert np.isfinite(body.pos).all()
    assert not np.isnan(body.pos).any()
    assert float(np.abs(body.pos).max()) < 100.0
    assert motion[-1] < motion[99]
    assert motion[-1] < 1e-4

    lengths = [separation(body, ids[i], ids[i + 1]) for i in range(19)]
    assert min(lengths) > 0.9
    assert max(lengths) < 1.5
    assert tuple(body.pos[body.slot(ids[0])]) == (0.0, 0.0, 0.0)


def test_cyclic_topology_is_stable() -> None:
    body, ids = square()
    body.set_pos(ids[0], (0.5, -0.4, 0.3))

    run(body, 2000, still())

    assert np.isfinite(body.pos).all()
    assert float(np.abs(body.pos).max()) < 10.0
    for i in range(4):
        assert separation(body, ids[i], ids[(i + 1) % 4]) == pytest.approx(
            1.0, abs=1e-3
        )
    assert separation(body, ids[0], ids[2]) == pytest.approx(np.sqrt(2.0), abs=1e-3)
    assert separation(body, ids[1], ids[3]) == pytest.approx(np.sqrt(2.0), abs=1e-3)


def test_cyclic_topology_survives_gravity_on_a_single_pin() -> None:
    body, ids = square(pinned_corner=True)

    run(body, 2000, SolverConfig())

    assert np.isfinite(body.pos).all()
    assert float(np.abs(body.pos).max()) < 10.0
    assert tuple(body.pos[body.slot(ids[0])]) == (0.0, 0.0, 0.0)
    for i in range(4):
        assert separation(body, ids[i], ids[(i + 1) % 4]) == pytest.approx(
            1.0, abs=0.05
        )


@pytest.mark.parametrize("spokes", [3, 4, 6, 8])
def test_high_valence_chunk_is_stable(spokes: int) -> None:
    """A chunk pulled by many links at once must not accumulate their sum.

    This is the case that separates a hip girdle from a chain, and it is the one
    that reaches NaN if overlapping corrections are simply added together.
    """
    body = Body(dim=3)
    hub = body.add_chunk("hub", (0.05, 0.0, 0.0))
    for i in range(spokes):
        theta = 2.0 * np.pi * i / spokes
        anchor = body.add_chunk(
            f"a{i}", (np.cos(theta), np.sin(theta), 0.0), pinned=True
        )
        body.add_link(hub, anchor, rest=1.0)

    run(body, 400, still())

    assert np.isfinite(body.pos).all()
    assert float(np.abs(body.pos[body.slot(hub)]).max()) < 1.0


def test_dense_cycle_of_mixed_link_types_stays_bounded() -> None:
    body, ids = square(pinned_corner=True)
    body.link_type[4] = LinkType.PUSH
    body.link_type[5] = LinkType.PULL

    run(body, 2000, SolverConfig())

    assert np.isfinite(body.pos).all()
    assert float(np.abs(body.pos).max()) < 50.0


# ----------------------------------------------------------------------
# fixed timestep
# ----------------------------------------------------------------------


def test_tick_rate_default() -> None:
    assert TICK_HZ == 40.0
    assert TICK_DT == pytest.approx(0.025)


def test_fixed_step_runs_whole_ticks_and_returns_the_remainder() -> None:
    driven, _ = chain(4)
    manual, _ = chain(4)
    config = SolverConfig()

    accumulator, alpha = fixed_step(driven, TICK_DT * 2.5, config)
    run(manual, 2, config)

    assert accumulator == pytest.approx(TICK_DT * 0.5)
    assert alpha == pytest.approx(0.5)
    assert driven.pos == pytest.approx(manual.pos)


def test_fixed_step_carries_the_accumulator_between_frames() -> None:
    driven, _ = chain(4)
    manual, _ = chain(4)
    config = SolverConfig()

    accumulator = 0.0
    for _ in range(10):
        accumulator, alpha = fixed_step(driven, TICK_DT * 0.35, config, accumulator)
        assert 0.0 <= alpha <= 1.0
    run(manual, 3, config)

    assert driven.pos == pytest.approx(manual.pos)
    assert accumulator == pytest.approx(TICK_DT * 0.5)


def test_fixed_step_below_one_tick_does_not_advance() -> None:
    body, _ = chain(4)
    before = body.pos.copy()

    accumulator, alpha = fixed_step(body, TICK_DT * 0.25, SolverConfig())

    assert body.pos == pytest.approx(before)
    assert accumulator == pytest.approx(TICK_DT * 0.25)
    assert alpha == pytest.approx(0.25)


def test_fixed_step_drops_the_overflow_instead_of_spiralling() -> None:
    driven, _ = chain(4)
    manual, _ = chain(4)
    config = SolverConfig()

    accumulator, alpha = fixed_step(driven, TICK_DT * 500.0, config, max_ticks=8)
    run(manual, 8, config)

    assert driven.pos == pytest.approx(manual.pos)
    assert 0.0 <= accumulator < TICK_DT
    assert 0.0 <= alpha <= 1.0


def test_fixed_step_ignores_a_negative_elapsed() -> None:
    body, _ = chain(4)
    before = body.pos.copy()

    accumulator, alpha = fixed_step(body, -5.0, SolverConfig())

    assert body.pos == pytest.approx(before)
    assert accumulator == 0.0
    assert alpha == 0.0


def test_fixed_step_rejects_a_non_positive_dt() -> None:
    body, _ = chain(4)
    with pytest.raises(ValueError):
        fixed_step(body, 1.0, SolverConfig(), dt=0.0)


def test_render_pos_interpolates_between_ticks() -> None:
    body = Body(dim=3)
    body.add_chunk("c", (0.0, 0.0, 0.0))
    step(body, TICK_DT, SolverConfig())
    body.pos[0] = (10.0, 0.0, 0.0)
    body.prev_pos[0] = (0.0, 0.0, 0.0)

    assert render_pos(body, 0.0) == pytest.approx(body.prev_pos)
    assert render_pos(body, 1.0) == pytest.approx(body.pos)
    assert render_pos(body, 0.25)[0][0] == pytest.approx(2.5)


def test_render_pos_does_not_alias_the_body() -> None:
    body = Body(dim=3)
    body.add_chunk("c", (0.0, 0.0, 0.0))

    drawn = render_pos(body, 0.5)
    drawn[0] = (99.0, 99.0, 99.0)

    assert tuple(body.pos[0]) == (0.0, 0.0, 0.0)


class TestGroundCollision:
    """A floor is opt-in: the library default is no world at all.

    Without it a body under gravity with nothing pinned falls forever, which is
    correct and also the reason an editor has to ask for a floor explicitly.
    """

    @staticmethod
    def _chain(n: int = 8, ground: float | None = None, height: float = 4.0) -> tuple[Body, SolverConfig]:
        body = Body()
        ids = [body.add_chunk(f"c{i}", [i * 0.5, height, 0.0], radius=0.2) for i in range(n)]
        for a, b in zip(ids, ids[1:]):
            body.add_link(a, b)
        return body, SolverConfig(ground_y=ground, iterations=8)

    def test_without_a_floor_a_body_falls_without_bound(self):
        body, config = self._chain(ground=None)
        for _ in range(600):
            step(body, TICK_DT, config)
        assert body.pos[:, 1].min() < -50.0

    def test_a_floor_stops_it(self):
        body, config = self._chain(ground=0.0)
        for _ in range(600):
            step(body, TICK_DT, config)

        assert (body.pos[:, 1] >= 0.2 - 1e-3).all()
        assert np.isfinite(body.pos).all()

    def test_it_rests_at_exactly_one_radius_above_the_floor(self):
        body, config = self._chain(ground=0.0)
        for _ in range(600):
            step(body, TICK_DT, config)
        assert body.pos[:, 1] == pytest.approx(0.2, abs=1e-3)

    def test_a_non_zero_floor_is_respected(self):
        body, config = self._chain(ground=-3.0)
        for _ in range(600):
            step(body, TICK_DT, config)
        assert body.pos[:, 1] == pytest.approx(-2.8, abs=1e-3)

    def test_a_pinned_chunk_ignores_the_floor(self):
        """Pinned means the authored position wins, floor included."""
        body = Body()
        pinned = body.add_chunk("pin", [0.0, -5.0, 0.0], radius=0.2, pinned=True)
        config = SolverConfig(ground_y=0.0)
        for _ in range(50):
            step(body, TICK_DT, config)
        assert body.chunk(pinned).pos[1] == pytest.approx(-5.0)

    def test_friction_shortens_a_slide(self):
        def slide(friction: float) -> float:
            body = Body()
            body.add_chunk("c", [0.0, 0.2, 0.0], radius=0.2)
            body.pos[0] = [0.3, 0.2, 0.0]
            config = SolverConfig(ground_y=0.0, ground_friction=friction)
            for _ in range(80):
                step(body, TICK_DT, config)
            return float(body.pos[0, 0])

        assert slide(0.35) < slide(0.0)

    def test_collision_runs_after_projection(self):
        """Otherwise a constraint can shove a chunk back through the floor."""
        body = Body()
        anchor = body.add_chunk("anchor", [0.0, -2.0, 0.0], radius=0.1, pinned=True)
        hanging = body.add_chunk("hanging", [0.0, 1.0, 0.0], radius=0.1)
        body.add_link(anchor, hanging, rest=3.0)
        config = SolverConfig(ground_y=0.0, iterations=8)
        for _ in range(200):
            step(body, TICK_DT, config)

        assert body.chunk(hanging).pos[1] >= 0.1 - 1e-3

    def test_a_two_dimensional_body_collides_on_y(self):
        body = Body(dim=2)
        body.add_chunk("c", [0.0, 3.0], radius=0.25)
        config = SolverConfig(ground_y=0.0, gravity=(0.0, -9.81))
        for _ in range(300):
            step(body, TICK_DT, config)
        assert body.pos[0, 1] == pytest.approx(0.25, abs=1e-3)
