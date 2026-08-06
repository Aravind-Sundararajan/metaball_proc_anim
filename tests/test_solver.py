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
from mesh_xml.core.terrain import Terrain

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


class TestCoulombFriction:
    """The lift is the normal impulse, so grip is ``mu`` times the lift.

    A chunk resting under gravity regrounds by ``g * dt**2`` each tick, so its
    per-tick grip budget is ``mu * g * dt**2`` of tangential displacement —
    which is precisely a lateral-force threshold of ``mu * m * g``, Coulomb's
    law recovered from displacements alone. Every case here leans on that
    identity: hold below the threshold, slide above it, threshold proportional
    to load, and nothing discontinuous at the boundary.
    """

    MU = 0.5
    THRESHOLD = MU * 9.81
    """Lateral force a resting unit mass can just barely be held against."""

    @staticmethod
    def _resting(mass: float = 1.0) -> Body:
        body = Body()
        body.add_chunk("c", [0.0, 0.2, 0.0], radius=0.2, mass=mass)
        return body

    @staticmethod
    def _world(world: str, mu: float) -> SolverConfig:
        """The same floor twice: the fast path, and the SDF that must equal it.

        The guarantees in this class are what the terrain path is not allowed
        to lose (docs/TERRAIN.md), so the key cases run against both worlds.
        """
        if world == "ground_y":
            return SolverConfig(ground_y=0.0, ground_friction=mu)
        return SolverConfig(terrain=Terrain.flat(0.0), ground_friction=mu)

    @classmethod
    def _drift(
        cls,
        lateral: float,
        mass: float = 1.0,
        ticks: int = 400,
        world: str = "ground_y",
    ) -> float:
        body = cls._resting(mass)
        config = cls._world(world, cls.MU)
        push = np.zeros((1, 3), dtype=np.float32)
        push[0, 0] = lateral
        for _ in range(ticks):
            step(body, TICK_DT, config, forces=push)
        return float(body.pos[0, 0])

    @pytest.mark.parametrize("world", ["ground_y", "terrain"])
    def test_a_lateral_force_below_the_threshold_causes_no_creep(self, world):
        """Zero drift, not slow drift — the claim the bleed model could not make.

        A bleed removes velocity after the tick's displacement has already
        happened, so even at full bleed this chunk kept its ``F * dt**2`` of
        slide every tick: 1.9 units over this run, at any bleed setting. The
        position-level cap undoes the slide itself, and the bound here is three
        orders of magnitude below what any bleed leaks.
        """
        assert abs(self._drift(3.0, ticks=1000, world=world)) < 1e-3

    @pytest.mark.parametrize("world", ["ground_y", "terrain"])
    def test_the_holdable_force_scales_with_the_load(self, world):
        """Double the mass is double the weight is double the grip."""
        between = self.THRESHOLD * 1.4
        assert self._drift(between, mass=1.0, world=world) > 1.0
        assert abs(self._drift(between, mass=2.0, world=world)) < 1e-3
        assert self._drift(between * 2.0, mass=2.0, world=world) > 1.0

    def test_load_transmitted_through_a_link_adds_grip(self):
        """Depth carries constraint load, not just gravity.

        Collision runs after projection, so a pinned chunk pressing down
        through a compressed link deepens the penetration the floor must undo
        each tick, and the grip budget grows with it. The same lateral force
        that flings an unloaded chunk cannot move a loaded one.
        """
        shove = self.THRESHOLD * 10.0

        body = self._resting()
        press = body.add_chunk("press", [0.0, 1.2, 0.0], radius=0.1, pinned=True)
        body.add_link(press, body.id_at(0), rest=1.5)
        config = SolverConfig(ground_y=0.0, ground_friction=self.MU)
        push = np.zeros((2, 3), dtype=np.float32)
        push[0, 0] = shove
        for _ in range(400):
            step(body, TICK_DT, config, forces=push)

        assert abs(float(body.pos[0, 0])) < 1e-3
        assert self._drift(shove) > 1.0

    def test_sliding_grows_with_the_force_that_drives_it(self):
        drifts = [self._drift(f) for f in (6.0, 8.0, 10.0)]

        assert all(d > 0.05 for d in drifts)
        assert drifts[0] < drifts[1] < drifts[2]

    def test_a_slider_stops_once_the_force_is_removed(self):
        """Kinetic friction sheds energy; it does not just cap a slide."""
        body = self._resting()
        config = SolverConfig(ground_y=0.0, ground_friction=self.MU)
        push = np.zeros((1, 3), dtype=np.float32)
        push[0, 0] = 10.0
        for _ in range(200):
            step(body, TICK_DT, config, forces=push)
        launched = float(body.pos[0, 0])

        for _ in range(300):
            step(body, TICK_DT, config)
        settled = float(body.pos[0, 0])
        for _ in range(100):
            step(body, TICK_DT, config)

        assert launched > 0.5
        assert float(body.pos[0, 0]) == pytest.approx(settled, abs=1e-4)
        assert float(np.abs(body.pos - body.prev_pos).max()) < 1e-5

    @pytest.mark.parametrize("world", ["ground_y", "terrain"])
    def test_zero_mu_is_exactly_frictionless(self, world):
        """Bit-for-bit, not approximately: the propulsion suite's control
        assumes the floor at ``mu = 0`` exerts no horizontal force at all, so a
        chunk skating on it must match one coasting in empty space. The terrain
        path keeps the guarantee because a flat field's normal has exactly zero
        horizontal components, so the lift writes ``x += depth * 0.0``."""
        skating = self._resting()
        skating.pos[0, 0] = 0.3
        coasting = self._resting()
        coasting.pos[0, 0] = 0.3

        floor = self._world(world, mu=0.0)
        space = SolverConfig(gravity=NO_GRAVITY)
        for _ in range(200):
            step(skating, TICK_DT, floor)
            step(coasting, TICK_DT, space)

        assert float(skating.pos[0, 0]) == float(coasting.pos[0, 0])

    def test_displacement_is_continuous_through_the_threshold(self):
        """A hinge, not a step: zero up to the threshold, linear past it.

        While sliding, each tick adds ``(F/m) * dt**2`` and the floor undoes
        ``mu * g * dt**2``, so drift is proportional to the excess force alone
        and vanishes as the threshold is approached from above. A pop at the
        static-kinetic boundary would show up here as a nonzero intercept.
        """
        below = self._drift(self.THRESHOLD - 0.5)
        near = self._drift(self.THRESHOLD + 0.5)
        double = self._drift(self.THRESHOLD + 1.0)
        far = self._drift(self.THRESHOLD + 2.0)

        assert abs(below) < 1e-3
        assert 0.0 < near < double < far
        assert double / near == pytest.approx(2.0, rel=0.05)
        assert far / near == pytest.approx(4.0, rel=0.05)


class TestTerrainCollision:
    """The SDF world: everything the flat floor promised, plus directions.

    ``Terrain.flat`` must reproduce ``ground_y`` because the flat path stays
    as the fast path and the two must never disagree about the same floor.
    The ramp, overhang and 2-D cases are the ground the generalisation buys:
    surfaces whose normals a heightfield could never express.
    """

    @staticmethod
    def _drop_chain(config: SolverConfig, ticks: int = 600) -> Body:
        body = Body()
        ids = [
            body.add_chunk(f"c{i}", [i * 0.5, 4.0, 0.0], radius=0.2)
            for i in range(8)
        ]
        for a, b in zip(ids, ids[1:]):
            body.add_link(a, b)
        for _ in range(ticks):
            step(body, TICK_DT, config)
        return body

    def test_flat_terrain_reproduces_the_ground_plane(self):
        """Same fall, same landing, same rest — through either code path.

        Not asserted bit-for-bit because the flat path computes in float32
        lanes while the field answers in float64, but the tolerance here is
        below a float32 ulp of the coordinates involved: during static grip
        both paths round to identical positions, so the only divergence is a
        few kinetic settling ticks.
        """
        on_plane = self._drop_chain(SolverConfig(ground_y=0.0, iterations=8))
        on_field = self._drop_chain(
            SolverConfig(terrain=Terrain.flat(0.0), iterations=8)
        )

        assert on_field.pos == pytest.approx(on_plane.pos, abs=1e-5)

    def test_terrain_wins_when_both_worlds_are_set(self):
        """Both set is not an error — the terrain is simply the world."""
        body = Body()
        body.add_chunk("c", [0.0, 3.0, 0.0], radius=0.2)
        config = SolverConfig(terrain=Terrain.flat(1.0), ground_y=-5.0)
        for _ in range(400):
            step(body, TICK_DT, config)

        assert float(body.pos[0, 1]) == pytest.approx(1.2, abs=1e-3)

    def test_an_empty_terrain_defers_to_ground_y(self):
        """An empty world means "nothing authored yet", not "no floor"."""
        body = Body()
        body.add_chunk("c", [0.0, 3.0, 0.0], radius=0.2)
        config = SolverConfig(terrain=Terrain(), ground_y=0.0)
        for _ in range(400):
            step(body, TICK_DT, config)

        assert float(body.pos[0, 1]) == pytest.approx(0.2, abs=1e-3)

    def test_a_pinned_chunk_ignores_the_terrain(self):
        """Pinned means the authored position wins, worlds included."""
        body = Body()
        body.add_chunk("pin", [0.0, -5.0, 0.0], radius=0.2, pinned=True)
        config = SolverConfig(terrain=Terrain.flat(0.0))
        for _ in range(50):
            step(body, TICK_DT, config)

        assert float(body.pos[0, 1]) == pytest.approx(-5.0)

    def test_a_ball_rests_on_a_buried_capsule_ramp(self):
        """Rest is ``distance == radius`` on the incline, not a global height.

        The ramp is TERRAIN.md's vocabulary sentence: a plane with a tilted
        buried capsule on top. The ball must come to rest a radius off the
        capsule's surface, well above where the buried plane would hold it —
        the assertion is against the field itself, so it cannot be satisfied
        by any flat-floor behaviour.
        """
        terrain = Terrain.flat(-2.0).add_cone(
            (0.0, -1.0, 0.0), 1.0, (4.0, 1.0, 0.0), 1.0
        )
        body = Body()
        body.add_chunk("ball", [1.0, 2.0, 0.0], radius=0.25)
        config = SolverConfig(terrain=terrain, ground_friction=0.7)
        for _ in range(400):
            step(body, TICK_DT, config)

        assert float(terrain.distance(body.pos)[0]) == pytest.approx(
            0.25, abs=1e-3
        )
        assert float(body.pos[0, 1]) > 0.4
        assert float(np.abs(body.pos - body.prev_pos).max()) < 1e-4

    def test_an_overhang_stops_a_chunk_pushed_up_into_it(self):
        """The anti-heightfield case: this world can say "ceiling".

        A chunk driven upward by more than its weight must be held under the
        overhang, resting a radius below it, and must never pass through on
        the way — the push-out along a downward normal is the same code that
        lifts a chunk off a floor.
        """
        terrain = Terrain.flat(0.0).add_plane((0.0, -1.0, 0.0), -2.0)
        body = Body()
        body.add_chunk("c", [0.0, 1.0, 0.0], radius=0.2)
        config = SolverConfig(terrain=terrain)
        push = np.zeros((1, 3), dtype=np.float32)
        push[0, 1] = 30.0

        highest = -np.inf
        for _ in range(300):
            step(body, TICK_DT, config, forces=push)
            highest = max(highest, float(body.pos[0, 1]))

        assert float(body.pos[0, 1]) == pytest.approx(1.8, abs=1e-3)
        assert highest <= 1.8 + 1e-3

    def test_a_two_dimensional_body_rests_on_symmetric_terrain(self):
        """The field widens 2-D points and slices its answers back to 2-D.

        A slope with no z component is symmetric about the ``z = 0`` plane a
        2-D body lives on, so dropping onto it must land, grip (30 degrees is
        inside ``mu = 0.7``), and rest a radius off the surface — the whole
        contact pipeline in two dimensions.
        """
        theta = np.radians(30.0)
        terrain = Terrain().add_plane((-np.sin(theta), np.cos(theta), 0.0), 0.0)
        body = Body(dim=2)
        body.add_chunk("c", [0.0, 1.0], radius=0.25)
        config = SolverConfig(
            terrain=terrain, ground_friction=0.7, gravity=(0.0, -9.81)
        )
        for _ in range(300):
            step(body, TICK_DT, config)

        assert float(terrain.distance(body.pos)[0]) == pytest.approx(
            0.25, abs=1e-3
        )
        assert abs(float(body.pos[0, 0])) < 0.1
        assert float(np.abs(body.pos - body.prev_pos).max()) < 1e-4


class TestSlopeLaw:
    """A chunk on an incline holds while ``tan(theta) < mu`` and slides above.

    docs/TERRAIN.md names this the acceptance test, and it needs no tuning
    because the identity is exact: a resting chunk reburies by ``g * dt**2``
    vertically each tick, which the field reads as ``cos(theta)`` of depth
    and ``sin(theta)`` of tangential slide, so the full-grip condition
    ``mu * depth >= slide`` is literally ``mu >= tan(theta)``. The brackets
    below sit two to five degrees around each threshold (``atan 0.7`` is
    34.99 degrees, ``atan 0.4`` is 21.80); if they ever need widening, the
    contact model is lying somewhere.
    """

    @staticmethod
    def _slide(mu: float, theta_deg: float, ticks: int = 400) -> float:
        """Signed downhill travel of a chunk that starts at rest on the slope."""
        theta = np.radians(theta_deg)
        normal = np.array([-np.sin(theta), np.cos(theta), 0.0])
        downhill = np.array([-np.cos(theta), -np.sin(theta), 0.0])

        body = Body()
        body.add_chunk("c", normal * 0.2, radius=0.2)
        start = body.pos[0].copy()
        config = SolverConfig(
            terrain=Terrain().add_plane(normal, 0.0), ground_friction=mu
        )
        for _ in range(ticks):
            step(body, TICK_DT, config)

        assert np.isfinite(body.pos).all()
        return float((body.pos[0] - start) @ downhill)

    @pytest.mark.parametrize(
        "mu, theta_deg",
        [(0.7, 30.0), (0.7, 34.0), (0.4, 15.0), (0.4, 21.0)],
    )
    def test_holds_below_the_coulomb_angle(self, mu, theta_deg):
        """Held outright — no creep — right up against the threshold."""
        assert abs(self._slide(mu, theta_deg)) < 1e-3

    @pytest.mark.parametrize(
        "mu, theta_deg",
        [(0.7, 36.0), (0.7, 40.0), (0.4, 23.0), (0.4, 30.0)],
    )
    def test_slides_above_the_coulomb_angle(self, mu, theta_deg):
        """Past the threshold the chunk slides, and it slides *downhill*."""
        assert self._slide(mu, theta_deg) > 0.05


class TestMomentumConservation:
    """Projection must not be a force. A free body's centre of mass stays put.

    The first overlap cure — dividing each chunk's summed correction by its own
    link count — quietly broke this: a link joining chunks of unequal degree
    applied full correction on one side and a fraction on the other, so every
    unevenly-connected pair pumped momentum into the body. A fifth of the first
    self-propelled walker's thrust turned out to be this artifact.
    """

    @staticmethod
    def _drift(build) -> float:
        body = build()
        com0 = body.pos.mean(axis=0).copy()
        config = SolverConfig(gravity=(0.0, 0.0, 0.0), damping=1.0)
        for _ in range(300):
            step(body, TICK_DT, config)
        assert np.isfinite(body.pos).all()
        return float(np.linalg.norm(body.pos.mean(axis=0) - com0))

    def test_an_uneven_chain_does_not_drift(self):
        """Degree 1 against degree 2 — the smallest case that leaked."""

        def build() -> Body:
            body = Body()
            ids = [body.add_chunk(f"c{i}", [x, 0.0, 0.0]) for i, x in enumerate((0.0, 1.0, 2.2))]
            body.add_link(ids[0], ids[1], rest=1.0)
            body.add_link(ids[1], ids[2], rest=1.0)
            return body

        assert self._drift(build) < 1e-3

    def test_a_stretched_star_does_not_drift(self):
        """A hub is the maximally uneven topology."""

        def build() -> Body:
            body = Body()
            hub = body.add_chunk("hub", [0.3, 0.1, 0.0])
            for k in range(5):
                angle = 2.0 * np.pi * k / 5
                tip = body.add_chunk(f"t{k}", [1.6 * np.cos(angle), 1.6 * np.sin(angle), 0.0])
                body.add_link(hub, tip, rest=1.0)
            return body

        assert self._drift(build) < 1e-3

    def test_unequal_masses_conserve_momentum_not_position(self):
        """The heavy chunk moves less; the mass-weighted sum still cancels."""
        body = Body()
        light = body.add_chunk("light", [0.0, 0.0, 0.0], mass=1.0)
        heavy = body.add_chunk("heavy", [2.0, 0.0, 0.0], mass=9.0)
        body.add_link(light, heavy, rest=1.0)

        masses = np.array([1.0, 9.0])[:, None]
        com0 = (body.pos * masses).sum(axis=0) / masses.sum()
        config = SolverConfig(gravity=(0.0, 0.0, 0.0), damping=1.0)
        for _ in range(300):
            step(body, TICK_DT, config)

        com1 = (body.pos * masses).sum(axis=0) / masses.sum()
        assert np.linalg.norm(com1 - com0) < 1e-3
        assert np.linalg.norm(body.pos[1] - body.pos[0]) == pytest.approx(1.0, abs=1e-3)
