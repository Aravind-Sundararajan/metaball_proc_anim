"""Behavioural tests for two-bone IK and the foothold state machine.

Pure math and pure state: no Qt, no GL, no files.

The first test in the file is the one that matters most. A controller that
writes positions passes every geometry test in here and still destabilises the
rig, because the failure only shows up as a fight with the projection pass that
runs immediately afterward. So the contract is tested directly: run the
controller, and assert the body's arrays are *bit*-identical to what they were.
Only the force accumulator may change.

The IK cases are checked against closed-form ground truth rather than against a
previous run, so they measure the geometry rather than pinning the
implementation. The foothold cases mostly drive :meth:`FootholdPicker.tip_target`
with synthetic :class:`LimbFrame` snapshots — that is the whole seam between the
two halves, and driving it directly is what lets a step's arc be measured to the
tick without a solver in the way.

The reaction and drive cases are the same shape. A reaction is a claim about a
*sum* — that the three forces a limb contributes cancel — so it is asserted on
the force accumulator rather than on where anything ended up, which is the only
place the claim is exactly true. What the pair is *for* needs a whole creature
and lives in ``test_balance.py``.

The terrain cases assert in the field's own terms — ``terrain.distance()`` of a
foothold against the tip radius — because "on the slope" is a claim the SDF can
check exactly at any incline, where a hand-computed height would re-derive the
projection and prove nothing. The ramp walk itself is integration and lands
after all the terrain packages do, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from mesh_xml.anim import serialize
from mesh_xml.anim.controller import ControlContext, ControllerStack
from mesh_xml.anim.controllers.limb import (
    FootholdPicker,
    LimbFrame,
    LimbIK,
    LimbMode,
    two_bone_ik,
)
from mesh_xml.core.body import Body
from mesh_xml.core.solver import TICK_DT, SolverConfig
from mesh_xml.core.terrain import Terrain

POLE = (0.0, 0.0, 1.0)


def leg(root_y: float = 1.2, upper: float = 0.8, lower: float = 0.8) -> tuple:
    """Hip / knee / foot hanging straight down. The hip is pinned so a test drives it.

    Bones of 0.8 give a span of 1.6 and a committed reach of 1.52, comfortably
    clear of the 1.2 drop from hip to ground.
    """
    body = Body(dim=3, name="leg")
    hip = body.add_chunk("hip", (0.0, root_y, 0.0), radius=0.1, pinned=True)
    knee = body.add_chunk("knee", (0.0, root_y - upper, 0.0), radius=0.08)
    foot = body.add_chunk("foot", (0.0, root_y - upper - lower, 0.0), radius=0.08)
    body.add_link(hip, knee, rest=upper)
    body.add_link(knee, foot, rest=lower)
    return body, hip, knee, foot


def context(
    body: Body,
    time: float = 0.0,
    dt: float = TICK_DT,
    seed: int = 0,
    config: SolverConfig | None = None,
) -> ControlContext:
    return ControlContext(
        body=body,
        forces=np.zeros((body.n_chunks, body.dim), dtype=np.float32),
        dt=dt,
        time=time,
        rng=np.random.default_rng(seed),
        config=config,
    )


def frame_at(
    x: float, speed: float = 0.0, root_y: float = 1.0, tip_radius: float = 0.0
) -> LimbFrame:
    """A limb whose shoulder is at ``x`` travelling along +x, foot on the ground."""
    return LimbFrame(
        root=np.array([x, root_y, 0.0], dtype=np.float32),
        tip=np.array([x, 0.0, 0.0], dtype=np.float32),
        velocity=np.array([speed, 0.0, 0.0], dtype=np.float32),
        reach=1.4,
        tip_radius=tip_radius,
    )


def longest_run(flags: list[bool]) -> int:
    best = 0
    run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def drive(
    picker: FootholdPicker,
    ticks: int,
    speed: float,
    seed: int = 0,
    tip_radius: float = 0.0,
) -> list:
    """Walk a picker along +x with synthetic frames, collecting its targets."""
    body = Body(dim=3)
    body.add_chunk("c", (0.0, 0.0, 0.0))
    rng = np.random.default_rng(seed)
    out = []
    for i in range(ticks):
        time = i * TICK_DT
        ctx = ControlContext(
            body=body,
            forces=np.zeros((1, 3), dtype=np.float32),
            dt=TICK_DT,
            time=time,
            rng=rng,
        )
        out.append(
            picker.tip_target(ctx, frame_at(speed * time, speed, tip_radius=tip_radius))
        )
    return out


# ----------------------------------------------------------------------
# the contract: force only, never a position
# ----------------------------------------------------------------------


def test_update_contributes_force_without_moving_anything() -> None:
    body, hip, knee, foot = leg()
    limb = LimbIK(hip, knee, foot, target=(0.6, 0.9, 0.2), pole=POLE)
    ctx = context(body)
    pos = body.pos.copy()
    prev = body.prev_pos.copy()

    limb.update(ctx)

    assert (body.pos == pos).all()
    assert (body.prev_pos == prev).all()
    assert float(np.abs(ctx.forces).sum()) > 0.0


def test_the_root_is_never_pushed() -> None:
    body, hip, knee, foot = leg()
    limb = LimbIK(hip, knee, foot, target=(0.6, 0.9, 0.2), pole=POLE)
    ctx = context(body)

    limb.update(ctx)

    assert (ctx.forces[body.slot(hip)] == 0.0).all()
    assert ctx.forces[body.slot(knee)].any()
    assert ctx.forces[body.slot(foot)].any()


def test_a_reaction_still_moves_nothing_during_update() -> None:
    """The contract does not soften because the limb pushes back."""
    body, hip, knee, foot = leg()
    limb = LimbIK(hip, knee, foot, target=(0.6, 0.9, 0.2), pole=POLE, reaction=1.0)
    ctx = context(body)
    pos = body.pos.copy()
    prev = body.prev_pos.copy()

    limb.update(ctx)

    assert (body.pos == pos).all()
    assert (body.prev_pos == prev).all()
    assert float(np.abs(ctx.forces).sum()) > 0.0


def test_a_whole_stack_of_limbs_moves_nothing_during_accumulate() -> None:
    body, hip, knee, foot = leg()
    picker = FootholdPicker(stance=(0.0, -1.2, 0.0), ground_y=0.08)
    stack = ControllerStack(body, seed=3)
    stack.add(LimbIK(hip, knee, foot, target=picker, pole=POLE))
    config = SolverConfig(ground_y=0.0)
    for _ in range(15):
        stack.tick(body, TICK_DT, config)

    pos = body.pos.copy()
    prev = body.prev_pos.copy()
    forces = stack.accumulate(body, TICK_DT)

    assert (body.pos == pos).all()
    assert (body.prev_pos == prev).all()
    assert np.isfinite(forces).all()


def test_a_limb_without_a_target_contributes_nothing() -> None:
    body, hip, knee, foot = leg()
    ctx = context(body)

    LimbIK(hip, knee, foot, target=None).update(ctx)

    assert not ctx.forces.any()


# ----------------------------------------------------------------------
# two-bone IK against closed-form ground truth
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "offset, upper, lower",
    [
        ((1.0, -1.0, 0.0), 1.0, 1.0),
        ((0.5, -0.2, 0.3), 1.0, 0.7),
        ((-1.4, 0.0, 0.6), 0.9, 1.3),
        ((0.0, 0.0, 0.75), 0.5, 0.5),
        ((0.05, 0.0, 0.0), 1.0, 1.0),
    ],
)
def test_the_solved_pose_satisfies_both_bones_and_lands_on_the_target(
    offset, upper: float, lower: float
) -> None:
    root = np.array([0.2, 1.0, -0.3], dtype=np.float32)
    target = root + np.asarray(offset, dtype=np.float32)

    mid, tip = two_bone_ik(root, target, upper, lower, POLE)

    assert float(np.linalg.norm(mid - root)) == pytest.approx(upper, abs=1e-5)
    assert float(np.linalg.norm(tip - mid)) == pytest.approx(lower, abs=1e-5)
    assert tip == pytest.approx(target, abs=1e-5)


def test_a_reachable_target_actually_bends_the_elbow_off_the_axis() -> None:
    """Otherwise every case above would also pass for a limb that never bends."""
    root = np.zeros(3, dtype=np.float32)
    mid, tip = two_bone_ik(root, (1.0, 0.0, 0.0), 1.0, 1.0, POLE)

    assert float(mid[2]) == pytest.approx(np.sqrt(0.75), abs=1e-5)
    assert float(mid[0]) == pytest.approx(0.5, abs=1e-5)


def test_an_unreachable_target_extends_straight_to_full_span() -> None:
    root = np.zeros(3, dtype=np.float32)

    mid, tip = two_bone_ik(root, (50.0, 0.0, 0.0), 0.6, 0.9, (0.0, 1.0, 0.0))

    assert np.isfinite(mid).all()
    assert np.isfinite(tip).all()
    assert float(np.linalg.norm(tip - root)) == pytest.approx(1.5, abs=1e-5)
    assert float(np.linalg.norm(mid - root)) == pytest.approx(0.6, abs=1e-5)
    assert float(np.linalg.norm(tip - mid)) == pytest.approx(0.9, abs=1e-5)
    assert np.cross(mid - root, tip - root) == pytest.approx(np.zeros(3), abs=1e-5)


def test_a_target_on_the_root_folds_the_limb_without_nan() -> None:
    root = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    mid, tip = two_bone_ik(root, root, 1.0, 1.0, POLE)

    assert np.isfinite(mid).all()
    assert np.isfinite(tip).all()
    assert float(np.linalg.norm(mid - root)) == pytest.approx(1.0, abs=1e-5)
    assert float(np.linalg.norm(tip - mid)) == pytest.approx(1.0, abs=1e-5)
    assert tip == pytest.approx(root, abs=1e-5)


def test_a_target_inside_the_fold_distance_is_clamped_outward() -> None:
    root = np.zeros(3, dtype=np.float32)

    mid, tip = two_bone_ik(root, (0.05, 0.0, 0.0), 1.0, 0.4, POLE)

    assert float(np.linalg.norm(tip - root)) == pytest.approx(0.6, abs=1e-5)
    assert float(np.linalg.norm(mid - root)) == pytest.approx(1.0, abs=1e-5)
    assert float(np.linalg.norm(tip - mid)) == pytest.approx(0.4, abs=1e-5)


@pytest.mark.parametrize("pole", [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-3.0, 0.0, 0.0)])
def test_a_pole_parallel_to_the_limb_axis_still_bends(pole) -> None:
    root = np.zeros(3, dtype=np.float32)

    mid, tip = two_bone_ik(root, (1.0, 0.0, 0.0), 1.0, 1.0, pole)

    assert np.isfinite(mid).all()
    assert float(np.linalg.norm(mid - root)) == pytest.approx(1.0, abs=1e-5)
    assert float(mid[0]) == pytest.approx(0.5, abs=1e-5)
    assert float(np.linalg.norm(mid[1:])) == pytest.approx(np.sqrt(0.75), abs=1e-5)


def test_zero_length_bones_collapse_onto_the_root() -> None:
    root = np.array([4.0, -1.0, 2.0], dtype=np.float32)

    mid, tip = two_bone_ik(root, (9.0, 9.0, 9.0), 0.0, 0.0, POLE)

    assert mid == pytest.approx(root)
    assert tip == pytest.approx(root)


def test_the_flip_sign_mirrors_the_elbow_across_the_limb_axis() -> None:
    root = np.zeros(3, dtype=np.float32)
    target = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    bone = float(np.sqrt(2.0))

    left, left_tip = two_bone_ik(root, target, bone, bone, (0.0, 1.0, 0.0), flip=1.0)
    right, right_tip = two_bone_ik(root, target, bone, bone, (0.0, 1.0, 0.0), flip=-1.0)

    assert left == pytest.approx((1.0, 1.0, 0.0), abs=1e-5)
    assert right == pytest.approx((1.0, -1.0, 0.0), abs=1e-5)
    assert (left + right) / 2.0 == pytest.approx((1.0, 0.0, 0.0), abs=1e-5)
    assert left_tip == pytest.approx(right_tip, abs=1e-6)


def test_a_two_dimensional_limb_bends_from_the_flip_sign_alone() -> None:
    """The 3-D pole trims to nothing in 2-D, which is Rain World's actual case."""
    root = np.zeros(2, dtype=np.float32)
    target = np.array([2.0, 0.0], dtype=np.float32)
    bone = float(np.sqrt(2.0))

    left, _ = two_bone_ik(root, target, bone, bone, POLE, flip=1.0)
    right, _ = two_bone_ik(root, target, bone, bone, POLE, flip=-1.0)

    assert abs(float(left[1])) == pytest.approx(1.0, abs=1e-5)
    assert float(left[1]) == pytest.approx(-float(right[1]), abs=1e-6)


def test_the_bone_lengths_come_from_the_links_not_the_squashed_pose() -> None:
    body, hip, knee, foot = leg(upper=0.8, lower=0.5)
    body.set_pos(knee, (0.0, 0.1, 0.0))

    assert LimbIK(hip, knee, foot).bone_lengths(body) == pytest.approx((0.8, 0.5))


def test_explicit_lengths_override_the_links() -> None:
    body, hip, knee, foot = leg()

    assert LimbIK(hip, knee, foot, lengths=(0.3, 0.4)).bone_lengths(body) == (0.3, 0.4)


def test_the_tip_converges_on_a_reachable_fixed_target() -> None:
    body, hip, knee, foot = leg()
    goal = np.array([0.7, 0.5, 0.2], dtype=np.float32)
    stack = ControllerStack(body)
    stack.add(LimbIK(hip, knee, foot, target=goal, pole=POLE))
    config = SolverConfig(gravity=(0.0, 0.0, 0.0), iterations=8)

    for _ in range(400):
        stack.tick(body, TICK_DT, config)

    assert float(np.linalg.norm(body.pos[body.slot(foot)] - goal)) < 0.05
    assert np.isfinite(body.pos).all()


# ----------------------------------------------------------------------
# foothold commitment — the point of the exercise
# ----------------------------------------------------------------------


def test_a_planted_foothold_does_not_move_while_the_body_does() -> None:
    picker = FootholdPicker(
        stance=(0.0, -1.2, 0.0), ground_y=0.08, step_threshold=0.5, lead_time=0.3
    )
    body, hip, knee, foot = leg()
    stack = ControllerStack(body, seed=1)
    stack.add(LimbIK(hip, knee, foot, target=picker, pole=POLE))
    config = SolverConfig(ground_y=0.0, iterations=6)

    for _ in range(20):
        stack.tick(body, TICK_DT, config)
    assert picker.mode is LimbMode.PLANTED
    planted = picker.foothold.copy()
    start = float(body.pos[body.slot(hip), 0])

    for _ in range(60):
        body.pos[body.slot(hip), 0] += 0.15 * TICK_DT
        stack.tick(body, TICK_DT, config)
        assert picker.mode is LimbMode.PLANTED
        assert (picker.foothold == planted).all()

    assert float(body.pos[body.slot(hip), 0]) - start > 0.2


def test_the_foot_replants_ahead_once_the_body_outruns_the_plant() -> None:
    picker = FootholdPicker(
        stance=(0.0, -1.2, 0.0),
        ground_y=0.08,
        step_threshold=0.4,
        lead_time=0.3,
        step_duration=0.25,
    )
    body, hip, knee, foot = leg()
    stack = ControllerStack(body, seed=1)
    stack.add(LimbIK(hip, knee, foot, target=picker, pole=POLE))
    config = SolverConfig(ground_y=0.0, iterations=6)

    for _ in range(20):
        stack.tick(body, TICK_DT, config)
    first = picker.foothold.copy()

    swung = False
    replanted = None
    for _ in range(400):
        body.pos[body.slot(hip), 0] += 0.5 * TICK_DT
        stack.tick(body, TICK_DT, config)
        swung |= picker.mode is LimbMode.REACHING
        if picker.mode is LimbMode.PLANTED and not np.array_equal(
            picker.foothold, first
        ):
            replanted = picker.foothold.copy()
            break

    assert swung
    assert replanted is not None
    assert float(replanted[0]) > float(first[0])
    assert float(replanted[0] - first[0]) > 0.4
    assert float(replanted[1]) == pytest.approx(0.08, abs=1e-5)


def test_the_swing_arcs_over_the_ground_and_lands_back_on_it() -> None:
    picker = FootholdPicker(
        stance=(0.0, -1.0, 0.0),
        ground_y=0.0,
        step_height=0.4,
        step_duration=0.25,
        lead_time=0.4,
    )
    body = Body(dim=3)
    body.add_chunk("c", (0.0, 0.0, 0.0))
    frame = LimbFrame(
        root=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        tip=np.array([-0.5, 0.0, 0.0], dtype=np.float32),
        velocity=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        reach=1.4,
    )

    swing = [
        picker.tip_target(context(body, time=i * TICK_DT), frame) for i in range(20)
    ]
    heights = [float(point[1]) for point in swing]

    assert picker.mode is LimbMode.PLANTED
    assert max(heights) == pytest.approx(0.4, abs=1e-4)
    assert heights[0] == pytest.approx(0.0, abs=1e-6)
    assert heights[-1] == pytest.approx(0.0, abs=1e-6)
    assert min(heights) >= -1e-6
    assert float(swing[0][0]) == pytest.approx(-0.5, abs=1e-5)
    assert float(swing[-1][0]) == pytest.approx(0.2, abs=1e-5)


def test_the_foot_is_off_the_ground_for_exactly_the_step_duration() -> None:
    picker = FootholdPicker(
        stance=(0.0, -1.0, 0.0),
        ground_y=0.0,
        step_height=0.3,
        step_duration=0.5,
        step_threshold=0.4,
        lead_time=0.2,
    )
    targets = drive(picker, ticks=200, speed=0.6)
    airborne = [float(point[1]) > 1e-6 for point in targets]

    assert picker.mode is not LimbMode.DANGLING
    # 0.5 s of swing is 20 ticks, of which the two endpoints sit on the ground.
    assert 17 <= longest_run(airborne) <= 20


def test_a_limb_with_nothing_in_reach_dangles_and_contributes_no_force() -> None:
    body, hip, knee, foot = leg(root_y=3.0)
    picker = FootholdPicker(stance=(0.0, -3.0, 0.0), ground_y=0.0)
    ctx = context(body)

    LimbIK(hip, knee, foot, target=picker, pole=POLE).update(ctx)

    assert picker.mode is LimbMode.DANGLING
    assert picker.foothold is None
    assert not ctx.forces.any()


def test_a_dangling_limb_takes_the_ground_back_when_it_comes_into_reach() -> None:
    picker = FootholdPicker(stance=(0.0, -1.0, 0.0), ground_y=0.0)
    body = Body(dim=3)
    body.add_chunk("c", (0.0, 0.0, 0.0))

    high = LimbFrame(
        root=np.array([0.0, 4.0, 0.0], dtype=np.float32),
        tip=np.array([0.0, 3.0, 0.0], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        reach=1.4,
    )
    assert picker.tip_target(context(body), high) is None
    assert picker.mode is LimbMode.DANGLING

    for i in range(1, 20):
        picker.tip_target(context(body, time=i * TICK_DT), frame_at(0.0))

    assert picker.mode is LimbMode.PLANTED
    assert float(picker.foothold[1]) == pytest.approx(0.0, abs=1e-6)


def test_the_plant_is_dropped_onto_the_ground_plane() -> None:
    picker = FootholdPicker(stance=(0.0, -0.3, 0.0), ground_y=-2.5)
    body = Body(dim=3)
    body.add_chunk("c", (0.0, 0.0, 0.0))
    frame = LimbFrame(
        root=np.array([0.0, -1.6, 0.0], dtype=np.float32),
        tip=np.array([0.0, -2.5, 0.0], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        reach=1.4,
    )

    for i in range(20):
        picker.tip_target(context(body, time=i * TICK_DT), frame)

    assert float(picker.foothold[1]) == pytest.approx(-2.5, abs=1e-6)


def test_without_a_ground_plane_the_plant_keeps_the_stance_height() -> None:
    picker = FootholdPicker(stance=(0.0, -1.0, 0.0), ground_y=None)
    targets = drive(picker, ticks=20, speed=0.0)

    assert picker.mode is LimbMode.PLANTED
    assert float(targets[-1][1]) == pytest.approx(0.0, abs=1e-6)


# ----------------------------------------------------------------------
# reaction — the limb as a closed actuator rather than a thruster
# ----------------------------------------------------------------------


def test_the_reaction_is_equal_and_opposite() -> None:
    body, hip, knee, foot = leg()
    limb = LimbIK(hip, knee, foot, target=(0.6, 0.9, 0.2), pole=POLE, reaction=1.0)
    ctx = context(body)

    limb.update(ctx)

    root = ctx.forces[body.slot(hip)]
    assert root.any()
    assert root == pytest.approx(
        -(ctx.forces[body.slot(knee)] + ctx.forces[body.slot(foot)])
    )
    assert ctx.forces.sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-4)


@pytest.mark.parametrize("reaction", [0.25, 0.5, 1.0])
def test_the_reaction_is_charged_at_its_own_fraction(reaction: float) -> None:
    body, hip, knee, foot = leg()
    target = (0.6, 0.9, 0.2)
    full = context(body)
    LimbIK(hip, knee, foot, target=target, pole=POLE, reaction=1.0).update(full)
    part = context(body)

    LimbIK(hip, knee, foot, target=target, pole=POLE, reaction=reaction).update(part)

    assert part.forces[body.slot(hip)] == pytest.approx(
        full.forces[body.slot(hip)] * reaction
    )
    assert part.forces[body.slot(knee)] == pytest.approx(full.forces[body.slot(knee)])


def test_the_reaction_pays_back_the_clamped_force_not_the_requested_one() -> None:
    """Paying back the requested force would break the pair exactly when it bites."""
    body, hip, knee, foot = leg()
    limb = LimbIK(
        hip, knee, foot, target=(0.6, 0.4, 0.2), pole=POLE, reaction=1.0, max_force=10.0
    )
    ctx = context(body)

    limb.update(ctx)

    assert float(np.linalg.norm(ctx.forces[body.slot(foot)])) == pytest.approx(10.0)
    assert float(np.linalg.norm(ctx.forces[body.slot(knee)])) == pytest.approx(10.0)
    assert ctx.forces.sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-4)


def test_a_limb_with_nothing_to_stand_on_charges_the_root_nothing() -> None:
    """Not planted reaches the root as zero because it reaches it as nothing."""
    body, hip, knee, foot = leg(root_y=3.0)
    picker = FootholdPicker(stance=(0.0, -3.0, 0.0), ground_y=0.0)
    ctx = context(body)

    LimbIK(hip, knee, foot, target=picker, pole=POLE, reaction=1.0).update(ctx)

    assert picker.mode is LimbMode.DANGLING
    assert not ctx.forces.any()


def test_a_limb_with_no_target_at_all_charges_the_root_nothing() -> None:
    body, hip, knee, foot = leg()
    ctx = context(body)

    LimbIK(hip, knee, foot, target=None, reaction=1.0).update(ctx)

    assert not ctx.forces.any()


def test_an_unreacted_limb_injects_momentum_over_a_whole_gait() -> None:
    """The thruster, measured: what an unpaired seek hands the world every tick."""
    config = SolverConfig(ground_y=0.0, iterations=6)

    def injected(reaction: float) -> float:
        body, hip, knee, foot = leg()
        picker = FootholdPicker(
            stance=(0.0, -1.2, 0.0), ground_y=0.08, step_threshold=0.3
        )
        stack = ControllerStack(body, seed=6)
        stack.add(LimbIK(hip, knee, foot, target=picker, pole=POLE, reaction=reaction))
        total = np.zeros(3, dtype=np.float64)
        for _ in range(400):
            body.pos[body.slot(hip), 0] += 0.4 * TICK_DT
            stack.tick(body, TICK_DT, config)
            total += stack.forces.sum(axis=0)
        return float(np.linalg.norm(total))

    assert injected(0.0) > 100.0
    assert injected(1.0) < 1e-2


def test_a_reacting_limb_stays_finite_reaching_far_past_its_span() -> None:
    body, hip, knee, foot = leg()
    body.set_pinned(hip, False)
    stack = ControllerStack(body)
    stack.add(
        LimbIK(hip, knee, foot, target=(500.0, -900.0, 300.0), pole=POLE, reaction=1.0)
    )
    config = SolverConfig(iterations=6)

    for _ in range(1000):
        stack.tick(body, TICK_DT, config)

    assert np.isfinite(body.pos).all()
    reach = float(np.linalg.norm(body.pos[body.slot(foot)] - body.pos[body.slot(hip)]))
    assert reach < 1.7


# ----------------------------------------------------------------------
# drive — a stance that pushes rather than merely holds
# ----------------------------------------------------------------------


def test_a_driven_plant_retreats_at_exactly_the_drive_speed() -> None:
    picker = FootholdPicker(
        stance=(0.0, -1.0, 0.0),
        ground_y=0.0,
        step_threshold=10.0,
        drive=(0.8, 0.0, 0.0),
    )
    body = Body(dim=3)
    body.add_chunk("c", (0.0, 0.0, 0.0))
    picker.tip_target(context(body), frame_at(0.0))
    for i in range(1, 20):
        picker.tip_target(context(body, time=i * TICK_DT), frame_at(0.0))
    assert picker.mode is LimbMode.PLANTED
    start = picker.foothold.copy()

    for i in range(20, 60):
        picker.tip_target(context(body, time=i * TICK_DT), frame_at(0.0))

    assert float(start[0] - picker.foothold[0]) == pytest.approx(
        0.8 * 40 * TICK_DT, abs=1e-4
    )
    assert float(picker.foothold[1]) == pytest.approx(0.0, abs=1e-6)


def test_an_undriven_plant_is_still_a_fixed_point() -> None:
    picker = FootholdPicker(stance=(0.0, -1.0, 0.0), ground_y=0.0, step_threshold=10.0)
    body = Body(dim=3)
    body.add_chunk("c", (0.0, 0.0, 0.0))
    for i in range(20):
        picker.tip_target(context(body, time=i * TICK_DT), frame_at(0.0))
    planted = picker.foothold.copy()

    for i in range(20, 60):
        picker.tip_target(context(body, time=i * TICK_DT), frame_at(0.0))

    assert (picker.foothold == planted).all()


def test_a_drive_places_new_footholds_that_much_further_ahead() -> None:
    """Read at touchdown, before the plant has begun retreating out from under it."""

    def first_plant(intent) -> np.ndarray:
        picker = FootholdPicker(
            stance=(0.0, -1.0, 0.0), ground_y=0.0, lead_time=0.25, drive=intent
        )
        body = Body(dim=3)
        body.add_chunk("c", (0.0, 0.0, 0.0))
        for i in range(40):
            picker.tip_target(context(body, time=i * TICK_DT), frame_at(0.0))
            if picker.planted:
                return picker.foothold.copy()
        raise AssertionError("the picker never planted")

    still = first_plant(None)
    driven = first_plant((0.6, 0.0, 0.0))

    assert float(driven[0] - still[0]) == pytest.approx(0.6 * 0.25, abs=1e-4)


def test_a_drive_eventually_forces_the_foot_to_step() -> None:
    """The retreating plant is what outruns ``step_threshold`` on a still body."""
    picker = FootholdPicker(
        stance=(0.0, -1.0, 0.0),
        ground_y=0.0,
        step_threshold=0.3,
        step_duration=0.2,
        lead_time=0.1,
        drive=(0.5, 0.0, 0.0),
    )
    body = Body(dim=3)
    body.add_chunk("c", (0.0, 0.0, 0.0))
    plants = set()
    for i in range(400):
        picker.tip_target(context(body, time=i * TICK_DT), frame_at(0.0))
        if picker.foothold is not None:
            plants.add(round(float(picker.foothold[0]), 2))

    assert len(plants) > 20
    assert picker.mode is not LimbMode.DANGLING


def test_the_planted_flag_tracks_the_mode() -> None:
    picker = FootholdPicker(stance=(0.0, -1.0, 0.0), ground_y=0.0)
    assert not picker.planted

    drive(picker, ticks=20, speed=0.0)
    assert picker.planted

    picker.reset()
    assert not picker.planted


# ----------------------------------------------------------------------
# terrain — footholds on real geometry
# ----------------------------------------------------------------------

TIP_R = 0.08
"""Tip radius carried by the terrain-case frames, matching ``leg()``'s foot."""


def ramp(degrees: float = 20.0) -> Terrain:
    """A bare incline through the origin, rising along +x."""
    theta = np.radians(degrees)
    return Terrain().add_plane((-np.sin(theta), np.cos(theta), 0.0), 0.0)


def frame_on(
    terrain: Terrain, x: float, drop: float = 1.0, speed: float = 0.0
) -> LimbFrame:
    """A stance ``drop`` above the local surface at ``x``, foot resting on it."""
    foot = terrain.project(np.array([x, 0.0, 0.0]), TIP_R)
    return LimbFrame(
        root=(foot + np.array([0.0, drop, 0.0])).astype(np.float32),
        tip=foot.astype(np.float32),
        velocity=np.array([speed, 0.0, 0.0], dtype=np.float32),
        reach=1.4,
        tip_radius=TIP_R,
    )


def settle(
    picker: FootholdPicker,
    frame: LimbFrame,
    ticks: int = 20,
    config: SolverConfig | None = None,
) -> None:
    """Hold one stance for ``ticks``, long enough for any swing to complete."""
    body = Body(dim=3)
    body.add_chunk("c", (0.0, 0.0, 0.0))
    for i in range(ticks):
        picker.tip_target(context(body, time=i * TICK_DT, config=config), frame)


@pytest.mark.parametrize("x", [-1.0, 0.0, 0.75, 2.5])
def test_a_foothold_on_a_tilted_plane_lies_on_the_plane(x: float) -> None:
    """``|distance - tip_radius| ~ 0`` is "on the slope" in the field's own terms."""
    terrain = ramp(20.0)
    picker = FootholdPicker(stance=(0.0, -1.0, 0.0), terrain=terrain)

    settle(picker, frame_on(terrain, x))

    assert picker.mode is LimbMode.PLANTED
    assert abs(float(terrain.distance(picker.foothold)) - TIP_R) < 1e-3


def test_the_terrain_supersedes_ground_y_when_both_are_set() -> None:
    """The solver's precedence, mirrored: two floors would mean aiming at a lie."""
    terrain = ramp(20.0)
    picker = FootholdPicker(stance=(0.0, -1.0, 0.0), ground_y=-5.0, terrain=terrain)

    settle(picker, frame_on(terrain, 1.5))

    assert picker.mode is LimbMode.PLANTED
    assert float(picker.foothold[1]) > 0.0, "the plant took the dead ground_y"
    assert abs(float(terrain.distance(picker.foothold)) - TIP_R) < 1e-3


def test_a_foothold_on_a_mound_lands_on_its_local_surface() -> None:
    """The plane alone is out of reach from this stance, so planting at all
    proves the candidate stood on the mound rather than at any global height."""
    terrain = (
        Terrain(blend=0.1)
        .add_plane((0.0, 1.0, 0.0), 0.0)
        .add_cone((0.0, -0.5, 0.0), 0.9, (0.0, -0.4, 0.0), 0.9)
    )
    frame = LimbFrame(
        root=np.array([0.0, 1.5, 0.0], dtype=np.float32),
        tip=np.array([0.0, 0.58, 0.0], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        reach=1.4,
        tip_radius=TIP_R,
    )

    bare = FootholdPicker(stance=(0.0, -1.0, 0.0), terrain=Terrain.flat(0.0))
    settle(bare, frame)
    assert bare.mode is LimbMode.DANGLING, "the fixture must put the plane too far"

    picker = FootholdPicker(stance=(0.0, -1.0, 0.0), terrain=terrain)
    settle(picker, frame)

    assert picker.mode is LimbMode.PLANTED
    assert float(picker.foothold[1]) > 0.4
    assert abs(float(terrain.distance(picker.foothold)) - TIP_R) < 1e-3


def platforms() -> Terrain:
    """Two fat capsules whose tops sit at y = 0, a chasm between their ends."""
    return (
        Terrain(blend=0.1)
        .add_cone((-4.0, -0.8, 0.0), 0.8, (-1.5, -0.8, 0.0), 0.8)
        .add_cone((1.5, -0.8, 0.0), 0.8, (4.0, -0.8, 0.0), 0.8)
    )


def stance_over(x: float) -> LimbFrame:
    return LimbFrame(
        root=np.array([x, 1.0, 0.0], dtype=np.float32),
        tip=np.array([x, 0.08, 0.0], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        reach=1.4,
        tip_radius=TIP_R,
    )


def test_a_stance_over_a_gap_dangles_where_one_over_a_platform_plants() -> None:
    """Dangling as a statement about the world: projection drags the candidate
    to the nearest edge, and an edge beyond reach is a hole, not a bad rig."""
    terrain = platforms()
    body = Body(dim=3)
    body.add_chunk("c", (0.0, 0.0, 0.0))
    over_gap = FootholdPicker(stance=(0.0, -1.0, 0.0), terrain=terrain)
    over_slab = FootholdPicker(stance=(0.0, -1.0, 0.0), terrain=terrain)

    for i in range(15):
        ctx = context(body, time=i * TICK_DT)
        assert over_gap.tip_target(ctx, stance_over(0.1)) is None
        over_slab.tip_target(ctx, stance_over(-2.75))

    assert over_gap.mode is LimbMode.DANGLING
    assert over_gap.foothold is None
    assert over_slab.mode is LimbMode.PLANTED
    assert abs(float(terrain.distance(over_slab.foothold)) - TIP_R) < 1e-3


def test_a_driven_plant_on_a_slope_retreats_along_the_slope() -> None:
    """Re-projection is what keeps a sliding plant on the incline: the commanded
    travel is horizontal, so its normal component is annihilated every tick and
    the foothold slides back at ``drive * cos^2(theta)`` horizontally, one tip
    radius off the surface the whole way down."""
    terrain = ramp(20.0)
    picker = FootholdPicker(
        stance=(0.0, -1.0, 0.0),
        terrain=terrain,
        max_reach=5.0,
        step_threshold=10.0,
        drive=(0.6, 0.0, 0.0),
    )
    body = Body(dim=3)
    body.add_chunk("c", (0.0, 0.0, 0.0))
    frame = frame_on(terrain, 0.0)

    for i in range(20):
        picker.tip_target(context(body, time=i * TICK_DT), frame)
    assert picker.mode is LimbMode.PLANTED
    start = picker.foothold.copy()

    gaps = []
    for i in range(20, 80):
        picker.tip_target(context(body, time=i * TICK_DT), frame)
        gaps.append(float(terrain.distance(picker.foothold)))

    slid = float(start[0] - picker.foothold[0])
    assert slid == pytest.approx(
        0.6 * 60 * TICK_DT * float(np.cos(np.radians(20.0)) ** 2), abs=5e-3
    )
    assert float(start[1] - picker.foothold[1]) > 0.2, "it left the incline"
    assert max(abs(gap - TIP_R) for gap in gaps) < 1e-3


def test_flat_terrain_reproduces_the_ground_plane_footholds() -> None:
    """``Terrain.flat(y)`` against ``ground_y = y + tip_radius``: the old world
    spelled two ways must place the same feet, or every plane rig that migrates
    changes gait."""

    def gait(picker: FootholdPicker) -> list:
        return drive(picker, ticks=300, speed=0.5, tip_radius=TIP_R)

    plane = gait(
        FootholdPicker(
            stance=(0.0, -1.0, 0.0), ground_y=0.4 + TIP_R, step_threshold=0.3
        )
    )
    field = gait(
        FootholdPicker(
            stance=(0.0, -1.0, 0.0),
            ground_y=None,
            terrain=Terrain.flat(0.4),
            step_threshold=0.3,
        )
    )

    assert [point is None for point in plane] == [point is None for point in field]
    assert any(point is not None for point in plane)
    for a, b in zip(plane, field):
        if a is not None:
            assert a == pytest.approx(b, abs=1e-5)


def test_a_picker_with_no_terrain_of_its_own_stands_on_the_worlds() -> None:
    """``ctx.terrain`` is the solver's world; following it is what lets one
    saved creature be dropped into any world without re-wiring each limb."""
    terrain = ramp(20.0)
    config = SolverConfig(terrain=terrain)
    picker = FootholdPicker(stance=(0.0, -1.0, 0.0))

    settle(picker, frame_on(terrain, 1.5), config=config)

    assert picker.mode is LimbMode.PLANTED
    assert float(picker.foothold[1]) > 0.3, "the plant fell back to ground_y"
    assert abs(float(terrain.distance(picker.foothold)) - TIP_R) < 1e-3


def test_the_pickers_own_terrain_wins_over_the_contexts() -> None:
    """The explicit choice is the one the author made; the context is only the
    default a creature inherits from wherever it happens to be standing."""
    own = Terrain.flat(1.0)
    world = Terrain.flat(0.0)
    picker = FootholdPicker(stance=(0.0, -1.0, 0.0), terrain=own)
    frame = LimbFrame(
        root=np.array([0.0, 2.0, 0.0], dtype=np.float32),
        tip=np.array([0.0, 1.08, 0.0], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        reach=1.4,
        tip_radius=TIP_R,
    )

    settle(picker, frame, config=SolverConfig(terrain=world))

    # The world's floor is out of this limb's reach, so standing at all means
    # the constructor's surface won.
    assert picker.mode is LimbMode.PLANTED
    assert float(picker.foothold[1]) == pytest.approx(1.0 + TIP_R, abs=1e-5)


def test_the_flat_path_is_untouched_when_no_terrain_exists_anywhere() -> None:
    """No picker terrain and no config terrain must stay the plane behaviour,
    bit for bit — that is every rig authored before terrain existed."""
    with_config = FootholdPicker(stance=(0.0, -1.0, 0.0), ground_y=0.3)
    without = FootholdPicker(stance=(0.0, -1.0, 0.0), ground_y=0.3)
    body = Body(dim=3)
    body.add_chunk("c", (0.0, 0.0, 0.0))

    for i in range(40):
        a = with_config.tip_target(
            context(body, time=i * TICK_DT, config=SolverConfig(ground_y=0.2)),
            frame_at(0.3 * i * TICK_DT, 0.3, tip_radius=TIP_R),
        )
        b = without.tip_target(
            context(body, time=i * TICK_DT),
            frame_at(0.3 * i * TICK_DT, 0.3),
        )
        assert (a is None) == (b is None)
        if a is not None:
            assert (a == b).all()

    assert float(with_config.foothold[1]) == pytest.approx(0.3, abs=1e-6)


def test_a_terrain_picker_refuses_to_serialize_where_a_plain_one_does_not() -> None:
    """The world belongs to the document — ``SolverConfig.terrain``, and the
    document section that will carry it — not to one foot of one creature. So
    ``terrain=None`` round-trips exactly as the plane configuration always has,
    and a picker holding a live surface is refused loudly rather than written
    without it and reloaded standing somewhere else."""
    plain = FootholdPicker(ground_y=0.13, lead_time=1.0 / 3.0)

    loaded = serialize.decode_controller(serialize.encode_controller(plain))

    assert loaded.terrain is None
    assert loaded.ground_y == 0.13
    assert loaded.lead_time == 1.0 / 3.0

    with pytest.raises(
        serialize.ControllerFormatError, match=r"FootholdPicker\.terrain"
    ):
        serialize.encode_controller(FootholdPicker(terrain=Terrain.flat(0.0)))


# ----------------------------------------------------------------------
# gait coordination
# ----------------------------------------------------------------------


def gait_pair(period: float, phase_b: float) -> tuple[FootholdPicker, FootholdPicker]:
    settings = dict(
        stance=(0.0, -1.0, 0.0),
        ground_y=0.0,
        step_threshold=0.3,
        step_duration=0.3,
        lead_time=0.25,
        period=period,
        duty=0.5,
    )
    return (
        FootholdPicker(phase=0.0, **settings),
        FootholdPicker(phase=phase_b, **settings),
    )


def walk_pair(pair, ticks: int, speed: float) -> list[int]:
    """Step both pickers off the same frames; return how often each started a swing."""
    body = Body(dim=3)
    body.add_chunk("c", (0.0, 0.0, 0.0))
    starts = [0, 0]
    for i in range(ticks):
        time = i * TICK_DT
        ctx = context(body, time=time)
        frame = frame_at(speed * time, speed)
        for index, picker in enumerate(pair):
            before = picker.mode
            picker.tip_target(ctx, frame)
            if picker.mode is LimbMode.REACHING and before is not LimbMode.REACHING:
                starts[index] += 1
        assert not all(picker.mode is LimbMode.REACHING for picker in pair)
    return starts


def test_antiphase_limbs_never_swing_at_the_same_time() -> None:
    starts = walk_pair(gait_pair(period=1.0, phase_b=0.5), ticks=1200, speed=0.5)

    assert min(starts) >= 10


def test_ungated_limbs_step_in_lockstep() -> None:
    """The contrast case: without a period, identical legs are perfectly in sync."""
    pair = gait_pair(period=0.0, phase_b=0.5)
    body = Body(dim=3)
    body.add_chunk("c", (0.0, 0.0, 0.0))
    together = 0
    for i in range(400):
        time = i * TICK_DT
        ctx = context(body, time=time)
        frame = frame_at(0.5 * time, 0.5)
        for picker in pair:
            picker.tip_target(ctx, frame)
        together += all(picker.mode is LimbMode.REACHING for picker in pair)

    assert together > 0
    assert pair[0].mode is pair[1].mode


def test_a_swing_that_cannot_fit_its_duty_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        FootholdPicker(period=1.0, duty=0.25, step_duration=0.3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"step_duration": 0.0},
        {"step_threshold": -1.0},
        {"step_height": -0.1},
        {"scatter": -0.1},
        {"duty": 0.0},
        {"duty": 1.5},
    ],
)
def test_nonsense_tuning_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        FootholdPicker(**kwargs)


# ----------------------------------------------------------------------
# randomness and reset
# ----------------------------------------------------------------------


def test_scatter_draws_from_the_context_rng() -> None:
    def plant(seed: int) -> np.ndarray:
        picker = FootholdPicker(stance=(0.0, -1.0, 0.0), ground_y=0.0, scatter=0.05)
        return drive(picker, ticks=20, speed=0.0, seed=seed)[-1]

    assert plant(4) == pytest.approx(plant(4))
    assert float(np.linalg.norm(plant(4) - plant(9))) > 1e-4


def test_scatter_left_at_zero_does_not_disturb_the_shared_stream() -> None:
    picker = FootholdPicker(stance=(0.0, -1.0, 0.0), ground_y=0.0)
    body = Body(dim=3)
    body.add_chunk("c", (0.0, 0.0, 0.0))
    ctx = context(body, seed=17)

    for i in range(20):
        picker.tip_target(
            ControlContext(
                body=body,
                forces=ctx.forces,
                dt=TICK_DT,
                time=i * TICK_DT,
                rng=ctx.rng,
            ),
            frame_at(0.0),
        )

    assert ctx.rng.random() == np.random.default_rng(17).random()


def test_reset_forgets_the_plant_and_the_mode() -> None:
    picker = FootholdPicker(stance=(0.0, -1.0, 0.0), ground_y=0.0)
    drive(picker, ticks=20, speed=0.0)
    assert picker.mode is LimbMode.PLANTED

    picker.reset()

    assert picker.mode is LimbMode.DANGLING
    assert picker.foothold is None


def test_a_reset_picker_replays_the_same_gait() -> None:
    picker = FootholdPicker(
        stance=(0.0, -1.0, 0.0),
        ground_y=0.0,
        step_threshold=0.3,
        scatter=0.04,
        period=0.8,
        duty=0.5,
        step_duration=0.3,
    )
    first = drive(picker, ticks=300, speed=0.5, seed=5)
    picker.reset()
    second = drive(picker, ticks=300, speed=0.5, seed=5)

    assert [point is None for point in first] == [point is None for point in second]
    for a, b in zip(first, second):
        if a is not None:
            assert a == pytest.approx(b)


def test_limb_reset_delegates_to_its_target() -> None:
    picker = FootholdPicker(stance=(0.0, -1.0, 0.0), ground_y=0.0)
    drive(picker, ticks=20, speed=0.0)

    LimbIK(0, 1, 2, target=picker).reset()

    assert picker.mode is LimbMode.DANGLING


def test_limb_reset_tolerates_a_target_that_has_no_state() -> None:
    LimbIK(0, 1, 2, target=(1.0, 2.0, 3.0)).reset()
    LimbIK(0, 1, 2, target=None).reset()


# ----------------------------------------------------------------------
# stale ids — the editor deletes chunks out from under a controller
# ----------------------------------------------------------------------


@pytest.mark.parametrize("doomed", ["hip", "knee", "foot"])
def test_a_deleted_chunk_is_skipped_rather_than_raising(doomed: str) -> None:
    body, hip, knee, foot = leg()
    limb = LimbIK(hip, knee, foot, target=(0.5, 0.2, 0.0), pole=POLE)
    body.remove_chunk({"hip": hip, "knee": knee, "foot": foot}[doomed])
    ctx = context(body)

    limb.update(ctx)

    assert not ctx.forces.any()


def test_a_stale_limb_does_not_raise_mid_tick() -> None:
    body, hip, knee, foot = leg()
    picker = FootholdPicker(stance=(0.0, -1.2, 0.0), ground_y=0.08)
    stack = ControllerStack(body, seed=2)
    stack.add(LimbIK(hip, knee, foot, target=picker, pole=POLE))
    config = SolverConfig(ground_y=0.0)

    for _ in range(20):
        stack.tick(body, TICK_DT, config)
    body.remove_chunk(knee)
    for _ in range(50):
        stack.tick(body, TICK_DT, config)

    assert np.isfinite(body.pos).all()
    assert not stack.forces.any()


def test_a_limb_whose_chunk_returns_is_picked_back_up() -> None:
    """Ids are stable across removal, so re-adding one revives the controller."""
    body, hip, knee, foot = leg()
    limb = LimbIK(hip, knee, foot, target=(0.5, 0.9, 0.0), pole=POLE)
    body.remove_chunk(knee)
    body.add_chunk("knee", (0.0, 0.4, 0.0), radius=0.08, id=knee)
    body.add_link(hip, knee, rest=0.8)
    body.add_link(knee, foot, rest=0.8)

    ctx = context(body)
    limb.update(ctx)

    assert ctx.forces.any()


# ----------------------------------------------------------------------
# stability
# ----------------------------------------------------------------------


def test_three_thousand_ticks_of_a_driven_walker_stay_finite() -> None:
    body = Body(dim=3, name="walker")
    stack = ControllerStack(body, seed=7)
    hips = [
        body.add_chunk(f"hip{i}", (i * 0.6, 0.95, 0.0), radius=0.12, pinned=True)
        for i in range(2)
    ]
    body.add_link(hips[0], hips[1])
    pickers: list[FootholdPicker] = []
    feet: list[int] = []
    for i, hip in enumerate(hips):
        knee = body.add_chunk(f"knee{i}", (i * 0.6, 0.40, 0.0), radius=0.09)
        foot = body.add_chunk(f"foot{i}", (i * 0.6, -0.15, 0.0), radius=0.09)
        feet.append(foot)
        picker = FootholdPicker(
            stance=(0.0, -0.9, 0.0),
            ground_y=0.09,
            step_threshold=0.3,
            step_height=0.2,
            step_duration=0.3,
            lead_time=0.25,
            scatter=0.02,
            period=0.8,
            phase=0.5 * i,
            duty=0.5,
        )
        pickers.append(picker)
        body.add_link(hip, knee, rest=0.55)
        body.add_link(knee, foot, rest=0.55)
        stack.add(LimbIK(hip, knee, foot, target=picker, pole=POLE, flip=1.0 - 2.0 * i))

    config = SolverConfig(ground_y=0.0, iterations=6)
    speed = 0.45
    plants: list[set] = [set(), set()]
    tracking = []
    for _ in range(3000):
        for hip in hips:
            body.pos[body.slot(hip), 0] += speed * TICK_DT
        stack.tick(body, TICK_DT, config)
        assert np.isfinite(body.pos).all()
        for i, picker in enumerate(pickers):
            if picker.foothold is not None:
                plants[i].add(round(float(picker.foothold[0]), 3))
                tracking.append(
                    float(
                        np.linalg.norm(
                            body.pos[body.slot(feet[i])] - picker.foothold
                        )
                    )
                )

    travel = speed * 3000 * TICK_DT
    assert not np.isnan(body.pos).any()
    assert float(body.pos[:, 1].max()) < 3.0
    assert float(body.pos[:, 1].min()) > -1.0
    assert float(np.abs(body.pos[:, 2]).max()) < 3.0
    assert float(body.pos[:, 0].min()) > -1.0
    assert float(body.pos[:, 0].max()) < travel + 3.0
    assert np.isfinite(stack.forces).all()

    # Bounded and finite is not enough on its own: a rig that dangled for
    # seventy-five seconds would pass every line above.
    assert min(len(seen) for seen in plants) > 20
    assert float(np.mean(tracking)) < 0.1


def test_a_limb_reaching_far_past_its_span_does_not_explode() -> None:
    body, hip, knee, foot = leg()
    stack = ControllerStack(body)
    stack.add(LimbIK(hip, knee, foot, target=(500.0, -900.0, 300.0), pole=POLE))
    config = SolverConfig(iterations=6)

    for _ in range(1000):
        stack.tick(body, TICK_DT, config)

    assert np.isfinite(body.pos).all()
    reach = float(np.linalg.norm(body.pos[body.slot(foot)] - body.pos[body.slot(hip)]))
    assert reach < 1.7
