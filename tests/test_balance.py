"""Postural control, and the creature that walks on it.

Two halves. The first drives :class:`Posture` and :class:`Level` directly and
asserts on the force accumulator, because both are claims about sums — a lift
that is charged back to the feet, a levelling pair that cancels — and a sum is
only exactly a sum before the solver has had it.

The second half is the point of the exercise, and it is a whole creature:
nothing pinned, nothing writing positions, no path carrying the body along. If
it moves it moves because a foot pushed on the floor. That claim is checked
three ways rather than by watching the number go up, since a number going up is
the easiest thing in the world to arrange by accident here:

* it reverses exactly when the drive reverses;
* it collapses to nothing when the reaction is switched off, which is the same
  gait with the same footfalls and no way to spend them;
* it does **not** survive removing ground friction. That control is the
  important one. Gravity is vertical and a reacted limb contributes zero net
  force, so friction is the only external horizontal force in the model — a
  creature that still travelled without it would be travelling on a solver
  artifact, not on its feet.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pytest

from mesh_xml.anim.controller import ControlContext, ControllerStack
from mesh_xml.anim.controllers.balance import Level, Posture
from mesh_xml.anim.controllers.limb import FootholdPicker, LimbIK, LimbMode
from mesh_xml.core.body import Body
from mesh_xml.core.solver import TICK_DT, SolverConfig

GROUND = 0.0
BACK_Y = 1.05
HIP_Y = 0.9
BONE = 0.5
FOOT_R = 0.08
HALF_X = 0.35
HALF_Z = 0.24

#: Trunk corners, front pair first.
CORNERS = ((HALF_X, -HALF_Z), (HALF_X, HALF_Z), (-HALF_X, -HALF_Z), (-HALF_X, HALF_Z))

#: Front knees fold back, rear knees fold forward, as a quadruped's do.
POLES = (-1.0, -1.0, 1.0, 1.0)

#: Diagonal pairs share a phase, which is a trot.
PHASES = (0.0, 0.5, 0.5, 0.0)

#: Long enough for the walk to reach a steady speed and several strides past it.
WALK_TICKS = 1000


def context(body: Body, time: float = 0.0) -> ControlContext:
    return ControlContext(
        body=body,
        forces=np.zeros((body.n_chunks, body.dim), dtype=np.float32),
        dt=TICK_DT,
        time=time,
        rng=np.random.default_rng(0),
    )


def chain(heights: list[float], dim: int = 3) -> tuple[Body, list[int]]:
    body = Body(dim=dim, name="chain")
    ids = []
    for i, y in enumerate(heights):
        point = [i * 0.4, y, 0.0][:dim]
        ids.append(body.add_chunk(f"c{i}", point, radius=0.1))
    for a, b in zip(ids, ids[1:]):
        body.add_link(a, b)
    return body, ids


# ----------------------------------------------------------------------
# the contract: force only, never a position
# ----------------------------------------------------------------------


def test_posture_contributes_force_without_moving_anything() -> None:
    body, ids = chain([0.4, 0.4, 0.4])
    ctx = context(body)
    pos = body.pos.copy()
    prev = body.prev_pos.copy()

    Posture(ids, height=1.0).update(ctx)

    assert (body.pos == pos).all()
    assert (body.prev_pos == prev).all()
    assert float(np.abs(ctx.forces).sum()) > 0.0


def test_level_contributes_force_without_moving_anything() -> None:
    body, ids = chain([0.4, 0.9, 0.2])
    ctx = context(body)
    pos = body.pos.copy()
    prev = body.prev_pos.copy()

    Level(ids).update(ctx)

    assert (body.pos == pos).all()
    assert (body.prev_pos == prev).all()
    assert float(np.abs(ctx.forces).sum()) > 0.0


# ----------------------------------------------------------------------
# Posture
# ----------------------------------------------------------------------


def test_a_trunk_below_its_height_is_lifted_and_one_above_it_is_pressed_down() -> None:
    body, ids = chain([0.4, 1.6])
    ctx = context(body)

    Posture(ids, height=1.0, damping=0.0).update(ctx)

    assert float(ctx.forces[body.slot(ids[0]), 1]) > 0.0
    assert float(ctx.forces[body.slot(ids[1]), 1]) < 0.0


def test_only_the_vertical_axis_is_touched() -> None:
    body, ids = chain([0.4, 0.4, 0.4])
    ctx = context(body)

    Posture(ids, height=1.0).update(ctx)

    assert not ctx.forces[:, 0].any()
    assert not ctx.forces[:, 2].any()


def test_the_lift_is_proportional_to_the_error_until_it_clamps() -> None:
    body, ids = chain([0.9, 0.8, 0.0])
    ctx = context(body)

    Posture(ids, height=1.0, strength=100.0, damping=0.0, max_force=15.0).update(ctx)

    assert float(ctx.forces[body.slot(ids[0]), 1]) == pytest.approx(10.0, abs=1e-3)
    assert float(ctx.forces[body.slot(ids[1]), 1]) == pytest.approx(15.0, abs=1e-3)
    assert float(ctx.forces[body.slot(ids[2]), 1]) == pytest.approx(15.0, abs=1e-3)


def test_damping_opposes_a_trunk_that_is_already_rising() -> None:
    body, ids = chain([0.5])
    body.prev_pos[body.slot(ids[0]), 1] -= 0.05
    ctx = context(body)

    Posture(ids, height=1.0, strength=0.0, damping=4.0, max_force=None).update(ctx)

    assert float(ctx.forces[body.slot(ids[0]), 1]) < 0.0


def test_naming_the_feet_makes_the_lift_an_internal_pair() -> None:
    body, ids = chain([0.4, 0.4])
    feet = [
        body.add_chunk(f"foot{i}", (float(i), 0.08, 0.0), radius=0.08) for i in (0, 1)
    ]
    ctx = context(body)

    Posture(ids, height=1.0, support=feet).update(ctx)

    assert float(ctx.forces[body.slot(ids[0]), 1]) > 0.0
    assert float(ctx.forces[body.slot(feet[0]), 1]) < 0.0
    assert ctx.forces.sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-4)


def test_without_support_the_lift_comes_from_nowhere() -> None:
    """The honest contrast case, since the docstring says so and it is a real cost."""
    body, ids = chain([0.4, 0.4])
    ctx = context(body)

    Posture(ids, height=1.0).update(ctx)

    assert float(ctx.forces.sum(axis=0)[1]) > 0.0


def test_a_body_dropped_from_a_height_is_caught_and_held() -> None:
    body, ids = chain([2.5, 2.5])
    feet = [
        body.add_chunk(f"foot{i}", (i * 0.4, 2.5 - 0.9, 0.0), radius=0.08)
        for i in (0, 1)
    ]
    for held, foot in zip(ids, feet):
        body.add_link(held, foot, rest=0.9)
    stack = ControllerStack(body, seed=1)
    stack.add(Posture(ids, height=1.0, strength=300.0, damping=20.0, support=feet))
    config = SolverConfig(ground_y=GROUND, iterations=6)

    heights = []
    for _ in range(600):
        stack.tick(body, TICK_DT, config)
        heights.append(float(body.pos[body.slot(ids[0]), 1]))

    assert np.isfinite(body.pos).all()
    assert max(heights) < 3.0
    assert float(np.mean(heights[-100:])) == pytest.approx(1.0, abs=0.25)


def test_a_posture_over_deleted_chunks_is_silent() -> None:
    body, ids = chain([0.4, 0.4])
    posture = Posture(list(ids), height=1.0, support=list(ids))
    for chunk_id in ids:
        body.remove_chunk(chunk_id)
    ctx = context(body)

    posture.update(ctx)

    assert not ctx.forces.any()


def test_a_posture_whose_support_is_gone_still_lifts() -> None:
    body, ids = chain([0.4, 0.4])
    foot = body.add_chunk("foot", (0.0, 0.08, 0.0), radius=0.08)
    posture = Posture(list(ids), height=1.0, support=[foot])
    body.remove_chunk(foot)
    ctx = context(body)

    posture.update(ctx)

    assert float(ctx.forces.sum(axis=0)[1]) > 0.0


def test_posture_works_on_a_two_dimensional_body() -> None:
    body, ids = chain([0.4, 0.4], dim=2)
    ctx = context(body)

    Posture(ids, height=1.0).update(ctx)

    assert (ctx.forces[:, 1] > 0.0).all()


# ----------------------------------------------------------------------
# Level
# ----------------------------------------------------------------------


def test_a_tilted_chain_is_pushed_flat() -> None:
    body, ids = chain([0.0, 0.6])
    ctx = context(body)

    Level(ids, damping=0.0).update(ctx)

    assert float(ctx.forces[body.slot(ids[0]), 1]) > 0.0
    assert float(ctx.forces[body.slot(ids[1]), 1]) < 0.0


def test_levelling_is_an_equal_and_opposite_pair() -> None:
    body, ids = chain([0.0, 0.6, -0.3, 0.9])
    ctx = context(body)

    Level(ids).update(ctx)

    assert ctx.forces.any()
    assert ctx.forces.sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-5)


def test_a_chain_that_is_already_level_costs_nothing() -> None:
    body, ids = chain([0.5, 0.5, 0.5])
    ctx = context(body)

    Level(ids).update(ctx)

    assert not ctx.forces.any()


def test_levelling_clamps_a_wildly_tilted_chain() -> None:
    body, ids = chain([0.0, 400.0])
    ctx = context(body)

    Level(ids, strength=50.0, damping=0.0, max_force=25.0).update(ctx)

    assert float(ctx.forces[body.slot(ids[0]), 1]) == pytest.approx(25.0, abs=1e-3)
    assert float(ctx.forces[body.slot(ids[1]), 1]) == pytest.approx(-25.0, abs=1e-3)


def test_a_chain_of_fewer_than_two_live_chunks_is_silent() -> None:
    body, ids = chain([0.0, 0.6])
    level = Level(list(ids))
    body.remove_chunk(ids[1])
    ctx = context(body)

    level.update(ctx)

    assert not ctx.forces.any()


def test_levelling_settles_a_tilted_chain_over_time() -> None:
    body, ids = chain([0.4, 1.4, 0.4])
    stack = ControllerStack(body, seed=2)
    stack.add(Level(ids, strength=90.0, damping=9.0))
    config = SolverConfig(gravity=(0.0, 0.0, 0.0), iterations=6)
    tilt = float(body.pos[:, 1].max() - body.pos[:, 1].min())

    for _ in range(600):
        stack.tick(body, TICK_DT, config)

    assert np.isfinite(body.pos).all()
    assert float(body.pos[:, 1].max() - body.pos[:, 1].min()) < 0.25 * tilt


# ----------------------------------------------------------------------
# a creature that moves under its own power
# ----------------------------------------------------------------------


def walker(
    reaction: float = 1.0, drive: float = 0.5, seed: int = 5
) -> tuple[Body, ControllerStack, int, list[int], list[FootholdPicker]]:
    """A quadruped with nothing pinned and nothing writing positions.

    The trunk is a braced cage rather than a chain, and that is not decoration.
    A distance link is isotropic, so a hip hung off a spine by one link swings
    inward and downward under load until it sits on the centreline — the
    quadruped example hits this and answers it by pinning the hips. Nothing is
    pinned here, so the four hips are instead tied to each other and to a back
    chunk above them, which is a rigid truss and holds its own shape.

    Gains sit under the ceiling explicit Verlet imposes rather than at the
    stiffest setting that happens to hold: at 40 Hz nothing here comes within a
    factor of two of ``max_force``, and the walk survives four to sixteen solver
    iterations with the same speed to within ten percent.
    """
    body = Body(dim=3, name="walker")
    stack = ControllerStack(body, seed=seed)

    hips = [
        body.add_chunk(f"hip{i}", (x, HIP_Y, z), radius=0.10, mass=1.0)
        for i, (x, z) in enumerate(CORNERS)
    ]
    back = body.add_chunk("back", (0.0, BACK_Y, 0.0), radius=0.16, mass=3.0)
    for i in range(len(hips)):
        body.add_link(back, hips[i])
        for j in range(i + 1, len(hips)):
            body.add_link(hips[i], hips[j])

    feet: list[int] = []
    pickers: list[FootholdPicker] = []
    for i, (hip, (x, z)) in enumerate(zip(hips, CORNERS)):
        knee = body.add_chunk(f"knee{i}", (x, HIP_Y - BONE, z), radius=0.08, mass=0.4)
        foot = body.add_chunk(
            f"foot{i}", (x, GROUND + FOOT_R, z), radius=FOOT_R, mass=0.4
        )
        body.add_link(hip, knee, rest=BONE)
        body.add_link(knee, foot, rest=BONE)
        feet.append(foot)
        picker = FootholdPicker(
            stance=(0.0, -(HIP_Y - GROUND - FOOT_R), 0.0),
            ground_y=GROUND + FOOT_R,
            step_threshold=0.20,
            step_height=0.40,
            step_duration=0.24,
            lead_time=0.18,
            period=0.8,
            phase=PHASES[i],
            duty=0.45,
            drive=None if drive == 0.0 else (drive, 0.0, 0.0),
        )
        pickers.append(picker)
        stack.add(
            LimbIK(
                hip,
                knee,
                foot,
                target=picker,
                pole=(POLES[i], 0.0, 0.0),
                strength=150.0,
                damping=11.0,
                max_force=120.0,
                reaction=reaction,
            )
        )

    stack.add(
        Posture([back], BACK_Y, strength=300.0, damping=20.0, max_force=120.0,
                support=feet)
    )
    stack.add(
        Posture(hips, HIP_Y, strength=300.0, damping=20.0, max_force=120.0,
                support=feet)
    )
    stack.add(Level(hips, strength=60.0, damping=6.0))
    return body, stack, back, feet, pickers


def centre_x(body: Body) -> float:
    """Mass-weighted, because that is the thing an external force has to move."""
    mass = np.where(body.inv_mass > 0.0, 1.0 / np.maximum(body.inv_mass, 1e-9), 0.0)
    return float((body.pos[:, 0] * mass).sum() / mass.sum())


def simulate(
    ticks: int = WALK_TICKS,
    friction: float = 0.35,
    iterations: int = 8,
    **kwargs,
) -> tuple[Body, ControllerStack, int, list[FootholdPicker], list[set], float]:
    body, stack, back, _, pickers = walker(**kwargs)
    config = SolverConfig(
        ground_y=GROUND, iterations=iterations, ground_friction=friction
    )
    start = centre_x(body)
    plants: list[set] = [set() for _ in pickers]
    for _ in range(ticks):
        stack.tick(body, TICK_DT, config)
        for seen, picker in zip(plants, pickers):
            if picker.foothold is not None:
                seen.add(round(float(picker.foothold[0]), 2))
    return body, stack, back, pickers, plants, centre_x(body) - start


@dataclass(frozen=True)
class Walk:
    """The summary of one run, and the reason :func:`walk` may be memoised.

    A tick of this creature is not cheap and half the cases below want the same
    2000 of them for a different reason, so the runs are cached — which is only
    honest because nothing here is a live view: the numbers are read out once
    and the body is left behind.
    """

    travelled: float
    back_height: float
    plants: tuple[int, ...]
    finite: bool
    highest: float
    lowest: float
    widest: float


@lru_cache(maxsize=None)
def walk(
    ticks: int = WALK_TICKS,
    friction: float = 0.35,
    iterations: int = 8,
    reaction: float = 1.0,
    drive: float = 0.5,
) -> Walk:
    body, stack, back, _, plants, travelled = simulate(
        ticks=ticks,
        friction=friction,
        iterations=iterations,
        reaction=reaction,
        drive=drive,
    )
    return Walk(
        travelled=travelled,
        back_height=float(body.pos[body.slot(back), 1]),
        plants=tuple(len(seen) for seen in plants),
        finite=bool(np.isfinite(body.pos).all() and np.isfinite(stack.forces).all()),
        highest=float(body.pos[:, 1].max()),
        lowest=float(body.pos[:, 1].min()),
        widest=float(np.abs(body.pos[:, 2]).max()),
    )


def test_the_creature_travels_forward_on_its_own_footfalls() -> None:
    run = walk()

    assert run.finite
    assert run.travelled > 3.5
    assert run.back_height > 0.75
    assert min(run.plants) > 15


def test_switching_the_reaction_off_leaves_the_same_gait_going_nowhere() -> None:
    """Same footfalls, same posture, no way to spend them."""
    driven = walk()
    idle = walk(reaction=0.0)

    assert min(idle.plants) > 15
    assert abs(idle.travelled) < 0.6
    assert driven.travelled > 6.0 * abs(idle.travelled)


def test_reversing_the_drive_reverses_the_travel() -> None:
    forward = walk(drive=0.5)
    backward = walk(drive=-0.5)

    assert forward.travelled > 3.5
    assert backward.travelled == pytest.approx(-forward.travelled, rel=0.05)


def test_a_creature_told_to_stand_still_stands_still() -> None:
    assert abs(walk(drive=0.0).travelled) < 0.6


def test_without_ground_friction_there_is_nothing_to_push_against() -> None:
    """The control that makes the claim mean something; see the module docstring."""
    gripped = walk(friction=0.0, drive=0.0)
    slipping = walk(friction=0.0, drive=0.5)

    assert abs(gripped.travelled) < 0.6
    assert slipping.travelled < 0.6
    assert walk().travelled > 3.5


@pytest.mark.parametrize("iterations", [4, 16])
def test_the_walk_does_not_depend_on_the_solver_iteration_count(iterations) -> None:
    """The failure this codebase warns about is stable at four and gone at sixteen."""
    assert walk(iterations=iterations).travelled == pytest.approx(
        walk().travelled, rel=0.15
    )


def test_the_whole_creature_writes_no_positions_during_accumulate() -> None:
    body, stack, _, _, _, _ = simulate(ticks=200)
    pos = body.pos.copy()
    prev = body.prev_pos.copy()

    forces = stack.accumulate(body, TICK_DT, SolverConfig(ground_y=GROUND))

    assert (body.pos == pos).all()
    assert (body.prev_pos == prev).all()
    assert np.isfinite(forces).all()


def test_three_thousand_ticks_of_walking_stay_finite_and_upright() -> None:
    run = walk(ticks=3000)

    assert run.finite
    assert run.travelled > 3.5 * (3000 / WALK_TICKS)
    assert run.highest < 3.0
    assert run.lowest > -1.0
    assert run.widest < 2.0
    assert run.back_height > 0.75
    # Bounded and finite is not enough: a creature that fell over on tick nine
    # and slid the rest of the way would pass every line above.
    assert min(run.plants) > 40


def test_a_walker_that_loses_a_leg_mid_stride_does_not_raise() -> None:
    body, stack, back, feet, pickers = walker()
    config = SolverConfig(ground_y=GROUND, iterations=8)
    for _ in range(200):
        stack.tick(body, TICK_DT, config)

    body.remove_chunk(feet[0])
    for _ in range(400):
        stack.tick(body, TICK_DT, config)

    assert np.isfinite(body.pos).all()
    assert any(picker.mode is LimbMode.PLANTED for picker in pickers[1:])
