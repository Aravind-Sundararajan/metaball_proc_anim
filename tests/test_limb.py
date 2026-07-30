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
"""

from __future__ import annotations

import numpy as np
import pytest

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
    body: Body, time: float = 0.0, dt: float = TICK_DT, seed: int = 0
) -> ControlContext:
    return ControlContext(
        body=body,
        forces=np.zeros((body.n_chunks, body.dim), dtype=np.float32),
        dt=dt,
        time=time,
        rng=np.random.default_rng(seed),
    )


def frame_at(x: float, speed: float = 0.0, root_y: float = 1.0) -> LimbFrame:
    """A limb whose shoulder is at ``x`` travelling along +x, foot on the ground."""
    return LimbFrame(
        root=np.array([x, root_y, 0.0], dtype=np.float32),
        tip=np.array([x, 0.0, 0.0], dtype=np.float32),
        velocity=np.array([speed, 0.0, 0.0], dtype=np.float32),
        reach=1.4,
    )


def longest_run(flags: list[bool]) -> int:
    best = 0
    run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def drive(picker: FootholdPicker, ticks: int, speed: float, seed: int = 0) -> list:
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
        out.append(picker.tip_target(ctx, frame_at(speed * time, speed)))
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
