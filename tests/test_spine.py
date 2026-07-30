"""Behavioural tests for the spine, look-at and breathing controllers.

Pure math: no Qt, no GL, no files. The first section is the important one. A
controller that writes positions passes every behavioural test in this file and
still destabilises the rig, because the damage shows up as buzz rather than as a
wrong number, so the contract is tested directly: run ``update`` and assert the
body's arrays came out bit-identical.

Joint angles are recomputed here from raw positions rather than borrowed from
the controller, so a sign or convention error in the bend limit cannot pass by
being measured with its own ruler.
"""

from __future__ import annotations

import numpy as np
import pytest

from mesh_xml.anim.controller import ControlContext, ControllerStack
from mesh_xml.anim.controllers.spine import Breathe, LookAt, SpineFollow
from mesh_xml.core.body import Body
from mesh_xml.core.solver import TICK_DT, SolverConfig

NO_GRAVITY = (0.0, 0.0, 0.0)


def still(**overrides) -> SolverConfig:
    return SolverConfig(gravity=NO_GRAVITY, **overrides)


def chain(n: int = 8, spacing: float = 1.0, pinned: set[int] = frozenset()) -> tuple:
    """``n`` chunks along +x, NORMAL-linked end to end. Index 0 is the head."""
    body = Body(dim=3, name="spine")
    ids = [
        body.add_chunk(f"c{i}", (i * spacing, 0.0, 0.0), pinned=(i in pinned))
        for i in range(n)
    ]
    for i in range(1, n):
        body.add_link(ids[i - 1], ids[i], rest=spacing)
    return body, ids


def context(body: Body, time: float = 0.0, seed: int = 0) -> ControlContext:
    return ControlContext(
        body=body,
        forces=np.zeros((body.n_chunks, body.dim), dtype=np.float32),
        dt=TICK_DT,
        time=time,
        rng=np.random.default_rng(seed),
    )


def joint_degrees(body: Body, ids: list[int]) -> np.ndarray:
    """Interior angle at each interior chunk. ``180`` is straight, ``0`` folded."""
    pos = np.array([body.pos[body.slot(i)] for i in ids], dtype=np.float64)
    incoming = pos[:-2] - pos[1:-1]
    outgoing = pos[2:] - pos[1:-1]
    cosine = (incoming * outgoing).sum(axis=1) / (
        np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1)
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def at(body: Body, chunk_id: int) -> np.ndarray:
    return body.pos[body.slot(chunk_id)].copy()


def run(stack: ControllerStack, body: Body, ticks: int, config: SolverConfig) -> None:
    for _ in range(ticks):
        stack.tick(body, TICK_DT, config)


def curled(bend_strength: float, ticks: int = 4000) -> tuple[Body, list[int], float]:
    """Drag a tail-pinned chain's head back onto its own anchor and let it settle.

    The head cannot reach: eight chunks curling at forty degrees a joint close a
    two-hundred-and-forty degree arc whose chord is still nearly three units, so
    the lead pulls forever and every joint is pressed against its limit rather
    than resting comfortably inside it.
    """
    body, ids = chain(8, pinned={7})
    spine = SpineFollow(
        chain=list(ids),
        strength=0.0,
        damping=3.0,
        max_bend_degrees=40.0,
        bend_strength=bend_strength,
        lead=tuple(float(v) for v in at(body, ids[-1])),
        lead_strength=4.0,
    )
    stack = ControllerStack(body, seed=3)
    stack.add(spine)
    body.pos[body.slot(ids[0])] += np.array([0.0, 0.05, 0.0], dtype=np.float32)

    run(stack, body, ticks, still())
    return body, ids, spine.min_joint_degrees


# ----------------------------------------------------------------------
# the contract: force only, never a position
# ----------------------------------------------------------------------


def bent_chain() -> tuple[Body, list[int]]:
    body, ids = chain(6)
    body.pos[body.slot(ids[0])] = (4.0, 3.0, 1.0)
    body.pos[body.slot(ids[3])] = (2.0, -2.0, 0.5)
    body.prev_pos[body.slot(ids[1])] = (0.4, 0.2, 0.1)
    return body, ids


def test_spine_follow_moves_nothing_by_itself() -> None:
    body, ids = bent_chain()
    ctx = context(body)
    before, before_prev = body.pos.tobytes(), body.prev_pos.tobytes()

    SpineFollow(chain=list(ids), lead=(9.0, -4.0, 2.0)).update(ctx)

    assert body.pos.tobytes() == before
    assert body.prev_pos.tobytes() == before_prev
    assert np.abs(ctx.forces).sum() > 0.0


def test_look_at_moves_nothing_by_itself() -> None:
    body, ids = bent_chain()
    ctx = context(body)
    before, before_prev = body.pos.tobytes(), body.prev_pos.tobytes()

    LookAt(chain=list(ids[:3]), target=(0.0, 8.0, -3.0)).update(ctx)

    assert body.pos.tobytes() == before
    assert body.prev_pos.tobytes() == before_prev
    assert np.abs(ctx.forces).sum() > 0.0


def test_breathe_moves_nothing_by_itself() -> None:
    body, ids = bent_chain()
    ctx = context(body, time=0.7)
    before, before_prev = body.pos.tobytes(), body.prev_pos.tobytes()

    Breathe(chunks=list(ids), amplitude=5.0).update(ctx)

    assert body.pos.tobytes() == before
    assert body.prev_pos.tobytes() == before_prev
    assert np.abs(ctx.forces).sum() > 0.0


def test_the_whole_stack_only_reaches_the_body_through_forces() -> None:
    body, ids = bent_chain()
    stack = ControllerStack(body, seed=11)
    stack.add(SpineFollow(chain=list(ids), lead=(6.0, 6.0, 6.0)))
    stack.add(LookAt(chain=list(ids[:3]), target=(-5.0, 1.0, 0.0)))
    stack.add(Breathe(chunks=list(ids)))
    before = body.pos.tobytes()

    stack.accumulate(body, TICK_DT)

    assert body.pos.tobytes() == before
    assert np.abs(stack.forces).sum() > 0.0


# ----------------------------------------------------------------------
# following a lead
# ----------------------------------------------------------------------


def test_the_head_tracks_a_moving_lead() -> None:
    body, ids = chain(8)
    spine = SpineFollow(chain=list(ids), lead=tuple(float(v) for v in at(body, ids[0])))
    stack = ControllerStack(body, seed=1)
    stack.add(spine)

    for tick in range(400):
        spine.lead = (0.0, 0.008 * tick, 0.0)
        stack.tick(body, TICK_DT, still())

    assert float(np.linalg.norm(at(body, ids[0]) - spine.lead)) < 1.0


def test_the_chain_is_dragged_after_its_head() -> None:
    body, ids = chain(8)
    spine = SpineFollow(chain=list(ids), lead=(0.0, 12.0, 0.0))
    stack = ControllerStack(body, seed=1)
    stack.add(spine)

    run(stack, body, 400, still())

    height = np.array([float(at(body, cid)[1]) for cid in ids])
    assert height[0] > 8.0
    assert (np.diff(height) < 0.0).all()


def test_the_tail_arrives_after_the_head() -> None:
    """The lag is the point: a chain whose ends move together reads as a plank."""
    body, ids = chain(8)
    stack = ControllerStack(body, seed=1)
    stack.add(SpineFollow(chain=list(ids), lead=(0.0, 12.0, 0.0)))

    arrival = {ids[0]: None, ids[-1]: None}
    for tick in range(600):
        stack.tick(body, TICK_DT, still())
        for cid in arrival:
            if arrival[cid] is None and at(body, cid)[1] > 1.0:
                arrival[cid] = tick

    assert arrival[ids[0]] is not None
    assert arrival[ids[-1]] is not None
    assert arrival[ids[-1]] > arrival[ids[0]] + 10


# ----------------------------------------------------------------------
# the bend limit
# ----------------------------------------------------------------------


def test_a_hairpin_settles_no_tighter_than_the_limit() -> None:
    body, ids, limit = curled(bend_strength=1500.0)

    angles = joint_degrees(body, ids)

    assert np.isfinite(body.pos).all()
    assert angles.min() > limit - 2.0


def test_the_same_hairpin_folds_right_through_itself_without_the_bend_force() -> None:
    """The control run. Without it the assertion above proves only that the
    hairpin never formed."""
    body, ids, limit = curled(bend_strength=0.0)

    assert joint_degrees(body, ids).min() < limit - 30.0


@pytest.mark.parametrize("max_bend", [25.0, 55.0, 85.0])
def test_the_limit_tracks_its_configured_angle(max_bend: float) -> None:
    body, ids = chain(8, pinned={7})
    spine = SpineFollow(
        chain=list(ids),
        strength=0.0,
        max_bend_degrees=max_bend,
        bend_strength=1500.0,
        lead=tuple(float(v) for v in at(body, ids[-1])),
        lead_strength=4.0,
    )
    stack = ControllerStack(body, seed=3)
    stack.add(spine)
    body.pos[body.slot(ids[0])] += np.array([0.0, 0.05, 0.0], dtype=np.float32)

    run(stack, body, 4000, still())

    assert joint_degrees(body, ids).min() > spine.min_joint_degrees - 3.0


def test_a_straight_chain_is_left_alone_by_the_bend_limit() -> None:
    body, ids = chain(6)
    ctx = context(body)

    SpineFollow(chain=list(ids), strength=0.0, max_bend_degrees=40.0).update(ctx)

    assert np.abs(ctx.forces).max() == 0.0


def test_the_bend_force_adds_no_net_momentum() -> None:
    """Equal and opposite on the joint's neighbours, so a chain pressed against
    its limit does not walk off across the world."""
    body, ids = chain(5)
    for i, cid in enumerate(ids):
        body.pos[body.slot(cid)] = (float(i % 2), float(i) * 0.2, 0.0)
    ctx = context(body)

    SpineFollow(chain=list(ids), strength=0.0, max_bend_degrees=10.0).update(ctx)

    assert np.abs(ctx.forces).sum() > 0.0
    assert ctx.forces.sum(axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)


# ----------------------------------------------------------------------
# stability
# ----------------------------------------------------------------------


def test_three_thousand_ticks_stay_finite_and_bounded() -> None:
    body, ids = chain(12, pinned={0})
    stack = ControllerStack(body, seed=7)
    stack.add(SpineFollow(chain=list(ids), lead=(2.0, 3.0, -1.0)))
    stack.add(LookAt(chain=list(ids[:3]), target=(5.0, 5.0, 5.0)))
    stack.add(Breathe(chunks=list(ids)))
    config = SolverConfig(ground_y=-6.0)

    for _ in range(3000):
        stack.tick(body, TICK_DT, config)
        assert np.isfinite(body.pos).all()

    assert np.abs(body.pos).max() < 50.0


def test_a_chain_driven_by_a_thrashing_lead_stays_bounded() -> None:
    body, ids = chain(10)
    spine = SpineFollow(chain=list(ids))
    stack = ControllerStack(body, seed=5)
    stack.add(spine)
    rng = np.random.default_rng(5)

    for _ in range(3000):
        spine.lead = tuple(rng.uniform(-8.0, 8.0, size=3))
        stack.tick(body, TICK_DT, still())

    assert np.isfinite(body.pos).all()
    assert np.abs(body.pos).max() < 200.0


def test_more_solver_iterations_do_not_destabilise_the_rig() -> None:
    """A controller fighting the projection is stable at four passes and
    diverges at sixteen. One that only contributes force does not care."""
    reach = []
    for iterations in (2, 16):
        body, ids = chain(10, pinned={0})
        stack = ControllerStack(body, seed=2)
        stack.add(SpineFollow(chain=list(ids), lead=(0.0, 5.0, 3.0)))
        run(stack, body, 1200, still(iterations=iterations))
        assert np.isfinite(body.pos).all()
        reach.append(float(np.abs(body.pos).max()))

    assert max(reach) < 20.0


# ----------------------------------------------------------------------
# breathing
# ----------------------------------------------------------------------


def breathing_trace(seed: int, ticks: int = 60) -> tuple[np.ndarray, Body, list[int]]:
    body, ids = chain(5, pinned=set(range(5)))
    stack = ControllerStack(body, seed=seed)
    stack.add(
        Breathe(chunks=list(ids), amplitude=5.0, frequency=0.5, axis=(0.0, 1.0, 0.0))
    )
    trace = []
    for _ in range(ticks):
        stack.tick(body, TICK_DT, still())
        trace.append(stack.forces.copy())
    return np.array(trace), body, ids


def test_breathing_repeats_exactly_for_one_seed() -> None:
    first, _, _ = breathing_trace(seed=4)
    second, _, _ = breathing_trace(seed=4)

    assert first.tobytes() == second.tobytes()


def test_breathing_differs_between_seeds() -> None:
    first, _, _ = breathing_trace(seed=4)
    second, _, _ = breathing_trace(seed=5)

    assert not np.allclose(first, second)


def test_two_chunks_do_not_pulse_in_unison() -> None:
    """In phase or exactly anti-phase would make the two series proportional,
    which is what a rig that pulses like one object looks like."""
    trace, body, ids = breathing_trace(seed=4, ticks=80)

    first = trace[:, body.slot(ids[0]), 1]
    second = trace[:, body.slot(ids[1]), 1]

    assert np.abs(first).max() > 0.5
    assert abs(float(np.corrcoef(first, second)[0, 1])) < 0.99


def test_breathing_reads_the_clock_rather_than_counting_calls() -> None:
    """``ControllerStack.reset`` rewinds ``ctx.time`` and nothing else, so a
    controller keeping its own counter would not rewind with it."""
    body, ids = chain(4)
    breathe = Breathe(chunks=list(ids), amplitude=4.0, frequency=0.3)
    times = [0.0, 0.35, 0.7, 1.05]

    forward = []
    for moment in times:
        ctx = context(body, time=moment)
        breathe.update(ctx)
        forward.append(ctx.forces.copy())

    assert np.abs(np.diff(forward, axis=0)).max() > 0.1
    for moment, expected in zip(reversed(times), reversed(forward)):
        ctx = context(body, time=moment)
        breathe.update(ctx)
        assert ctx.forces == pytest.approx(expected)


def test_extending_the_breathing_group_redraws_rather_than_over_indexing() -> None:
    body, ids = chain(4)
    breathe = Breathe(chunks=list(ids[:2]), amplitude=4.0)
    breathe.update(context(body, time=0.2))

    breathe.chunks = list(ids)
    ctx = context(body, time=0.2)
    breathe.update(ctx)

    assert np.abs(ctx.forces[body.slot(ids[3])]).sum() > 0.0


def test_breathing_swells_about_the_group_centroid_by_default() -> None:
    body, ids = chain(5)
    ctx = context(body, time=0.4)

    Breathe(chunks=list(ids), amplitude=6.0).update(ctx)

    assert np.abs(ctx.forces).sum() > 0.0
    assert np.abs(ctx.forces[:, 1]).max() == 0.0


# ----------------------------------------------------------------------
# stale ids: the editor deletes chunks while the simulation runs
# ----------------------------------------------------------------------


def controllers_for(ids: list[int]) -> list:
    return [
        SpineFollow(chain=list(ids), lead=(3.0, 4.0, 0.0)),
        LookAt(chain=list(ids), target=(3.0, 4.0, 0.0)),
        Breathe(chunks=list(ids), amplitude=4.0),
    ]


@pytest.mark.parametrize("index", [0, 3, 5])
def test_a_deleted_chunk_is_skipped_not_raised(index: int) -> None:
    body, ids = chain(6)
    controllers = controllers_for(ids)
    body.remove_chunk(ids[index])

    ctx = context(body, time=0.6)
    for controller in controllers:
        controller.update(ctx)

    assert np.isfinite(ctx.forces).all()


def test_every_configured_chunk_deleted_is_a_quiet_no_op() -> None:
    body, ids = chain(4)
    controllers = controllers_for(ids)
    for cid in ids:
        body.remove_chunk(cid)
    body.add_chunk("survivor", (0.0, 0.0, 0.0))

    ctx = context(body, time=0.6)
    for controller in controllers:
        controller.update(ctx)

    assert np.abs(ctx.forces).max() == 0.0


def test_a_chunk_deleted_mid_run_does_not_break_the_tick() -> None:
    body, ids = chain(8, pinned={0})
    stack = ControllerStack(body, seed=9)
    for controller in controllers_for(ids):
        stack.add(controller)

    run(stack, body, 200, still())
    body.remove_chunk(ids[4])
    body.remove_chunk(ids[7])
    run(stack, body, 400, still())

    assert np.isfinite(body.pos).all()
    assert body.n_chunks == 6


def test_a_two_chunk_chain_shortened_to_one_still_seeks_its_lead() -> None:
    body, ids = chain(2)
    spine = SpineFollow(chain=list(ids), lead=(0.0, 5.0, 0.0))
    body.remove_chunk(ids[1])

    ctx = context(body)
    spine.update(ctx)

    assert ctx.forces[body.slot(ids[0])][1] > 0.0


# ----------------------------------------------------------------------
# pinning
# ----------------------------------------------------------------------


def test_a_pinned_chunk_is_unmoved_by_every_controller() -> None:
    body, ids = chain(7, pinned={3})
    stack = ControllerStack(body, seed=6)
    stack.add(SpineFollow(chain=list(ids), lead=(-6.0, 7.0, 2.0)))
    stack.add(LookAt(chain=list(ids[:4]), target=(0.0, -9.0, 4.0)))
    stack.add(Breathe(chunks=list(ids), amplitude=8.0))
    anchor = tuple(at(body, ids[3]))

    for _ in range(400):
        stack.tick(body, TICK_DT, still())
        assert tuple(at(body, ids[3])) == anchor

    assert np.abs(stack.forces[body.slot(ids[3])]).sum() > 0.0


# ----------------------------------------------------------------------
# look at
# ----------------------------------------------------------------------


def test_look_at_swings_the_sub_chain_onto_its_target() -> None:
    body, ids = chain(5, pinned={0})
    stack = ControllerStack(body, seed=8)
    stack.add(LookAt(chain=list(ids), target=(0.0, 6.0, 0.0), strength=40.0))

    run(stack, body, 600, still())

    aim = at(body, ids[-1]) - at(body, ids[0])
    aim /= np.linalg.norm(aim)
    assert aim[1] > 0.9


def test_look_at_pivots_about_its_base_rather_than_dragging_it() -> None:
    body, ids = chain(4, pinned={0})
    ctx = context(body)

    LookAt(chain=list(ids), target=(0.0, 6.0, 0.0)).update(ctx)

    assert np.abs(ctx.forces[body.slot(ids[0])]).max() == 0.0
    assert np.abs(ctx.forces[body.slot(ids[1])]).max() > 0.0


def test_look_at_without_a_target_is_inert() -> None:
    body, ids = chain(4)
    ctx = context(body)

    LookAt(chain=list(ids)).update(ctx)

    assert np.abs(ctx.forces).max() == 0.0


def test_a_target_on_the_base_is_skipped_rather_than_normalised() -> None:
    body, ids = chain(4)
    ctx = context(body)

    LookAt(chain=list(ids), target=tuple(at(body, ids[0]))).update(ctx)

    assert np.isfinite(ctx.forces).all()
    assert np.abs(ctx.forces).max() == 0.0


# ----------------------------------------------------------------------
# disabled controllers
# ----------------------------------------------------------------------


def test_a_disabled_controller_contributes_nothing() -> None:
    body, ids = chain(6)
    stack = ControllerStack(body, seed=0)
    for controller in controllers_for(ids):
        controller.enabled = False
        stack.add(controller)

    forces = stack.accumulate(body, TICK_DT)

    assert np.abs(forces).max() == 0.0
