"""Behavioural tests for angle limits compiled to distance links.

Pure math: no Qt, no GL. The claims worth guarding are the ones the compilation
was bought for — that the emitted rest *is* the law-of-cosines chord, that a
limit is a hard stop rather than a spring that rings, that it conserves
momentum because it is an ordinary link in the just-fixed projection, that it
carries no flesh, and that a knee stop lets a leg hold a load up with no
postural spring at all.

One deliberate deviation from the design sketch, argued where it happens: the
payoff rig stands on *two* mirrored two-bone legs, because a single leg on a
point contact under a top-heavy load is an inverted pendulum whose lean e-folds
in about a third of a second — it topples within a couple of simulated seconds
whatever its knee does, and the knee stop's claim is structural, not postural.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mesh_xml.core.angles import angle_limit, brace, joint_degrees
from mesh_xml.core.body import Body, LinkType
from mesh_xml.core.solver import TICK_DT, SolverConfig, step
from mesh_xml.io import xml_v2
from mesh_xml.render import sdf

NO_GRAVITY = (0.0, 0.0, 0.0)

#: (r1, r2, interior degrees) combos exercising unequal bones and both halves
#: of the angle range.
CHORD_CASES = [
    (1.0, 1.0, 60.0),
    (1.0, 0.8, 90.0),
    (2.0, 0.5, 120.0),
    (0.7, 1.3, 45.0),
    (1.0, 1.0, 170.0),
]


def chord(r1: float, r2: float, degrees: float) -> float:
    theta = math.radians(degrees)
    return math.sqrt(r1 * r1 + r2 * r2 - 2.0 * r1 * r2 * math.cos(theta))


def triple(
    r1: float = 1.0,
    r2: float = 0.8,
    degrees: float = 90.0,
    pinned_arm: bool = False,
) -> tuple[Body, int, int, int]:
    """``a-b-c`` posed at ``degrees``, with the ``a-b`` arm along +x.

    Pinning the arm fixes the joint's frame of reference so a test can drive
    ``c`` and read the angle as the only thing that moved.
    """
    body = Body(dim=3)
    theta = math.radians(degrees)
    a = body.add_chunk("a", (r1, 0.0, 0.0), pinned=pinned_arm)
    b = body.add_chunk("b", (0.0, 0.0, 0.0), pinned=pinned_arm)
    c = body.add_chunk("c", (r2 * math.cos(theta), r2 * math.sin(theta), 0.0))
    body.add_link(a, b, rest=r1)
    body.add_link(b, c, rest=r2)
    return body, a, b, c


def press(
    body: Body,
    a: int,
    b: int,
    c: int,
    ticks: int,
    config: SolverConfig,
    magnitude: float = 40.0,
    apart: bool = False,
) -> list[float]:
    """Shove ``c`` straight at (or away from) ``a``, re-aimed every tick.

    Toward folds the joint shut, away hyperextends it; either way the force is
    strong enough to carry an unlimited joint to its geometric extreme, which
    is what makes the limited runs a contrast rather than a coincidence.
    """
    angles = []
    forces = np.zeros((body.n_chunks, body.dim), dtype=np.float32)
    for _ in range(ticks):
        axis = body.pos[body.slot(a)] - body.pos[body.slot(c)]
        norm = float(np.linalg.norm(axis))
        forces[:] = 0.0
        if norm > 1e-9:
            direction = axis / np.float32(norm)
            forces[body.slot(c)] = direction * np.float32(
                -magnitude if apart else magnitude
            )
        step(body, TICK_DT, config, forces)
        angles.append(joint_degrees(body, a, b, c))
    return angles


# ----------------------------------------------------------------------
# the law of cosines, both directions
# ----------------------------------------------------------------------


@pytest.mark.parametrize("r1,r2,degrees", CHORD_CASES)
def test_a_min_limit_compiles_to_a_push_at_the_analytic_chord(r1, r2, degrees):
    body, a, b, c = triple(r1, r2, degrees=min(degrees + 5.0, 180.0))

    angle_limit(body, a, b, c, min_degrees=degrees)

    link = body.link(a, c)
    assert link.type is LinkType.PUSH
    assert link.render is False
    assert link.rest == pytest.approx(chord(r1, r2, degrees), rel=1e-6)


@pytest.mark.parametrize("r1,r2,degrees", CHORD_CASES)
def test_a_max_limit_compiles_to_a_pull_at_the_analytic_chord(r1, r2, degrees):
    body, a, b, c = triple(r1, r2, degrees=max(degrees - 5.0, 1.0))

    angle_limit(body, a, b, c, max_degrees=degrees)

    link = body.link(a, c)
    assert link.type is LinkType.PULL
    assert link.render is False
    assert link.rest == pytest.approx(chord(r1, r2, degrees), rel=1e-6)


@pytest.mark.parametrize("r1,r2,degrees", CHORD_CASES)
def test_joint_degrees_reads_back_the_built_pose(r1, r2, degrees):
    body, _, _, _ = triple(r1, r2, degrees)
    assert joint_degrees(body, 0, 1, 2) == pytest.approx(degrees, abs=0.01)


def test_the_emitted_rest_comes_from_authored_rests_not_positions():
    body, a, b, c = triple(1.0, 0.8, degrees=90.0)
    body.set_pos(c, (0.0, 0.4, 0.0))

    angle_limit(body, a, b, c, min_degrees=90.0)

    assert body.link(a, c).rest == pytest.approx(chord(1.0, 0.8, 90.0), rel=1e-6)


def test_elasticity_reaches_the_emitted_link():
    body, a, b, c = triple()
    angle_limit(body, a, b, c, min_degrees=60.0, elasticity=0.5)
    assert body.link(a, c).elasticity == pytest.approx(0.5)


# ----------------------------------------------------------------------
# a limit is a hard stop, not a suggestion
# ----------------------------------------------------------------------


@pytest.mark.parametrize("iterations,tolerance", [(4, 2.0), (8, 1.0), (16, 0.5)])
def test_a_min_limit_stops_a_folding_press_at_the_limit(iterations, tolerance):
    """Settles pressed against the stop — measured 0.5/0.2/0.03 degrees past
    at 4/8/16 iterations — and never punches through or rings around it."""
    body, a, b, c = triple(1.0, 0.8, degrees=150.0, pinned_arm=True)
    angle_limit(body, a, b, c, min_degrees=90.0)
    config = SolverConfig(gravity=NO_GRAVITY, iterations=iterations)

    angles = press(body, a, b, c, ticks=600, config=config)

    settled = angles[-100:]
    assert min(angles[150:]) > 90.0 - tolerance
    assert float(np.mean(settled)) == pytest.approx(90.0, abs=tolerance)
    assert max(settled) - min(settled) < 0.25


def test_without_the_limit_the_same_press_folds_far_past():
    body, a, b, c = triple(1.0, 0.8, degrees=150.0, pinned_arm=True)
    config = SolverConfig(gravity=NO_GRAVITY, iterations=8)

    angles = press(body, a, b, c, ticks=600, config=config)

    assert angles[-1] < 30.0


@pytest.mark.parametrize("iterations,tolerance", [(4, 3.5), (8, 2.0), (16, 1.0)])
def test_a_max_limit_stops_a_hyperextending_press_at_the_limit(
    iterations, tolerance
):
    """Softer than the min side under the same press — the chord flattens as
    sin(theta) toward straight, exactly as the module docstring warns —
    but still 2.1/1.0/0.4 degrees at 4/8/16 iterations, with no ringing."""
    body, a, b, c = triple(1.0, 0.8, degrees=90.0, pinned_arm=True)
    angle_limit(body, a, b, c, max_degrees=140.0)
    config = SolverConfig(gravity=NO_GRAVITY, iterations=iterations)

    angles = press(body, a, b, c, ticks=600, config=config, apart=True)

    settled = angles[-100:]
    assert max(angles[150:]) < 140.0 + tolerance
    assert float(np.mean(settled)) == pytest.approx(140.0, abs=tolerance)
    assert max(settled) - min(settled) < 0.25


def test_without_the_limit_the_same_press_hyperextends_to_straight():
    body, a, b, c = triple(1.0, 0.8, degrees=90.0, pinned_arm=True)
    config = SolverConfig(gravity=NO_GRAVITY, iterations=8)

    angles = press(body, a, b, c, ticks=600, config=config, apart=True)

    assert angles[-1] > 170.0


def test_a_limit_only_engages_on_its_own_side():
    """Inside the allowed range the joint must feel nothing at all."""
    body, a, b, c = triple(1.0, 0.8, degrees=120.0)
    angle_limit(body, a, b, c, min_degrees=60.0)
    before = body.pos.copy()

    for _ in range(200):
        step(body, TICK_DT, SolverConfig(gravity=NO_GRAVITY))

    assert (body.pos == before).all()


# ----------------------------------------------------------------------
# momentum
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "build_degrees,limit",
    [
        (60.0, {"min_degrees": 120.0}),
        (170.0, {"max_degrees": 120.0}),
    ],
)
def test_an_engaged_limit_conserves_momentum(build_degrees, limit):
    """A free chain whose limit is violated at t=0 snaps open and tumbles;
    the centre of mass must not care."""
    body, a, b, c = triple(1.0, 1.0, degrees=build_degrees)
    angle_limit(body, a, b, c, **limit)
    com0 = body.pos.mean(axis=0).copy()
    config = SolverConfig(gravity=NO_GRAVITY, damping=1.0)

    for _ in range(500):
        step(body, TICK_DT, config)

    assert np.isfinite(body.pos).all()
    assert float(np.linalg.norm(body.pos.mean(axis=0) - com0)) < 2e-3


# ----------------------------------------------------------------------
# the payoff: a leg becomes a column
# ----------------------------------------------------------------------


def standing_rig(with_stops: bool) -> tuple[Body, int, list[tuple[int, int, int]]]:
    """A load over a hip over two mirrored two-bone legs, nothing pinned.

    Two legs rather than the design sketch's one, for a reason no knee
    constraint can answer: a single point contact under a top-heavy load is an
    inverted pendulum — its lean e-folds in roughly ``sqrt(I / (M*g*h))`` of a
    second, about a third of a second here — so a lone free-standing leg
    topples within seconds whatever its knee does. A knee stop rigidifies a
    leg; it cannot balance one, and balance.py owns staying over the feet.
    Two legs in the x-y plane are the smallest structure whose standing is
    decided by the knees alone: the mirror symmetry is exact in float32, so
    nothing ever leaves the plane or leans, and the only way down is the
    drawing-compass fold the stops exist to forbid. The foot-to-foot tie is
    the authored skeleton's own — without it the feet creep apart under the
    bleed-model friction and the contrast would be measuring the floor, not
    the knees.
    """
    body = Body(dim=3, name="column")
    load = body.add_chunk("load", (0.0, 1.55, 0.0), radius=0.12, mass=4.0)
    hip = body.add_chunk("hip", (0.0, 1.15, 0.0), radius=0.1, mass=1.0)
    body.add_link(load, hip)

    joints = []
    feet = []
    for side, out in (("l", -1.0), ("r", 1.0)):
        knee = body.add_chunk(
            f"knee_{side}", (out * 0.34, 0.62, 0.0), radius=0.07, mass=0.3
        )
        foot = body.add_chunk(
            f"foot_{side}", (out * 0.4, 0.08, 0.0), radius=0.08, mass=0.4
        )
        body.add_link(hip, knee)
        body.add_link(knee, foot)
        joints.append((hip, knee, foot))
        feet.append(foot)
    body.add_link(feet[0], feet[1])

    if with_stops:
        for hip_id, knee_id, foot_id in joints:
            stop = joint_degrees(body, hip_id, knee_id, foot_id) - 1.5
            angle_limit(body, hip_id, knee_id, foot_id, min_degrees=stop)
    return body, load, joints


def load_heights(body: Body, load: int, settle: int, hold: int) -> tuple[float, float]:
    """Height after the elastic settle, and again ``hold`` ticks later.

    The stack compresses for the first hundred-odd ticks — links squashing
    under the arriving weight, the stops taking up their softness — and that
    settling is elastic, not collapse. What must then be true of a column is
    that the second reading equals the first.
    """
    config = SolverConfig(ground_y=0.0, iterations=8)
    for _ in range(settle):
        step(body, TICK_DT, config)
    settled = float(body.pos[body.slot(load), 1])
    for _ in range(hold):
        step(body, TICK_DT, config)
    assert np.isfinite(body.pos).all()
    return settled, float(body.pos[body.slot(load), 1])


def test_payoff_a_knee_stop_turns_a_leg_into_a_column():
    """The design's acceptance line: a load held with NO Posture controller.

    Nothing here but chunks, links and the floor — no controller stack, no
    height spring, no pins. The only difference between the two runs is the
    pair of compiled knee stops, and it is the difference between standing
    at 94 percent of authored height for thirty simulated seconds and folding
    to a third of it with the hip on the ground.
    """
    held_body, held_load, joints = standing_rig(with_stops=True)
    start = float(held_body.pos[held_body.slot(held_load), 1])

    settled, held = load_heights(held_body, held_load, settle=300, hold=1000)
    assert held > start - 0.15
    assert held == pytest.approx(settled, abs=0.005)

    for hip_id, knee_id, foot_id in joints:
        limit = held_body.link(hip_id, foot_id)
        assert limit.type is LinkType.PUSH
        assert joint_degrees(held_body, hip_id, knee_id, foot_id) > 138.0

    collapsed_body, collapsed_load, _ = standing_rig(with_stops=False)
    _, collapsed = load_heights(collapsed_body, collapsed_load, settle=300, hold=1000)
    assert collapsed < 0.7
    assert held - collapsed > 0.85


def test_a_braced_knee_holds_the_column_too():
    body, load, joints = standing_rig(with_stops=False)
    for hip_id, knee_id, foot_id in joints:
        brace(body, hip_id, knee_id, foot_id)
    start = float(body.pos[body.slot(load), 1])

    settled, held = load_heights(body, load, settle=300, hold=1000)

    assert held > start - 0.15
    assert held == pytest.approx(settled, abs=0.005)


# ----------------------------------------------------------------------
# no flesh
# ----------------------------------------------------------------------


def test_a_limit_adds_no_surface_geometry():
    body, a, b, c = triple()
    before = sdf.primitive_count(body)

    angle_limit(body, a, b, c, min_degrees=60.0)

    assert sdf.primitive_count(body) == before
    assert len(sdf.pack(body)) == before
    assert body.link(a, c).render is False


def test_a_brace_adds_no_surface_geometry():
    body, a, b, c = triple()
    before = sdf.primitive_count(body)

    brace(body, a, b, c)

    assert sdf.primitive_count(body) == before
    assert body.link(a, c).render is False


# ----------------------------------------------------------------------
# round trip
# ----------------------------------------------------------------------


def test_a_limit_round_trips_through_xml_v2_and_still_stops(tmp_path):
    body, a, b, c = triple(1.0, 0.8, degrees=150.0, pinned_arm=True)
    angle_limit(body, a, b, c, min_degrees=90.0)
    path = tmp_path / "limited.xml"

    xml_v2.save(body, path)
    loaded = xml_v2.load(path)

    link = loaded.link(a, c)
    assert link.type is LinkType.PUSH
    assert link.render is False
    assert link.rest == pytest.approx(chord(1.0, 0.8, 90.0), rel=1e-6)

    config = SolverConfig(gravity=NO_GRAVITY, iterations=8)
    angles = press(loaded, a, b, c, ticks=600, config=config)
    assert min(angles[150:]) > 89.0
    assert float(np.mean(angles[-100:])) == pytest.approx(90.0, abs=1.0)


# ----------------------------------------------------------------------
# recompiling
# ----------------------------------------------------------------------


def test_recompiling_a_limit_updates_the_link_in_place():
    body, a, b, c = triple()
    angle_limit(body, a, b, c, min_degrees=60.0)
    links = body.n_links

    angle_limit(body, a, b, c, min_degrees=100.0, elasticity=0.75)

    assert body.n_links == links
    link = body.link(a, c)
    assert link.rest == pytest.approx(chord(1.0, 0.8, 100.0), rel=1e-6)
    assert link.elasticity == pytest.approx(0.75)


def test_recompiling_can_turn_a_floor_into_a_ceiling():
    body, a, b, c = triple()
    angle_limit(body, a, b, c, min_degrees=60.0)

    angle_limit(body, a, b, c, max_degrees=150.0)

    link = body.link(a, c)
    assert link.type is LinkType.PULL
    assert link.rest == pytest.approx(chord(1.0, 0.8, 150.0), rel=1e-6)


def test_a_brace_and_a_limit_overwrite_each_other():
    body, a, b, c = triple(degrees=120.0)
    brace(body, a, b, c)
    assert body.link(a, c).type is LinkType.NORMAL

    angle_limit(body, a, b, c, min_degrees=90.0)
    assert body.link(a, c).type is LinkType.PUSH

    brace(body, a, b, c)
    assert body.link(a, c).type is LinkType.NORMAL
    assert body.n_links == 3


def test_a_flesh_triangle_refuses_a_limit():
    body, a, b, c = triple()
    body.add_link(a, c)

    with pytest.raises(ValueError, match=f"joint {a}-{b}-{c}"):
        angle_limit(body, a, b, c, min_degrees=90.0)
    with pytest.raises(ValueError, match=f"joint {a}-{b}-{c}"):
        brace(body, a, b, c)


# ----------------------------------------------------------------------
# brace
# ----------------------------------------------------------------------


def test_a_brace_freezes_the_current_angle_against_both_presses():
    config = SolverConfig(gravity=NO_GRAVITY, iterations=8)
    for apart in (False, True):
        body, a, b, c = triple(1.0, 0.8, degrees=120.0, pinned_arm=True)
        brace(body, a, b, c)
        assert body.link(a, c).rest == pytest.approx(
            chord(1.0, 0.8, 120.0), rel=1e-6
        )

        angles = press(body, a, b, c, ticks=400, config=config, apart=apart)

        assert float(np.mean(angles[-100:])) == pytest.approx(120.0, abs=3.0)


def test_a_brace_measures_the_angle_but_not_the_squash():
    body, a, b, c = triple(1.0, 0.8, degrees=90.0)
    body.set_pos(c, (0.0, 0.4, 0.0))

    brace(body, a, b, c)

    assert body.link(a, c).rest == pytest.approx(chord(1.0, 0.8, 90.0), rel=1e-6)


# ----------------------------------------------------------------------
# degenerates and refusals
# ----------------------------------------------------------------------


def test_a_min_of_exactly_180_holds_the_joint_straight():
    body, a, b, c = triple(1.0, 0.8, degrees=140.0)
    angle_limit(body, a, b, c, min_degrees=180.0)
    assert body.link(a, c).rest == pytest.approx(1.8, rel=1e-6)

    for _ in range(600):
        step(body, TICK_DT, SolverConfig(gravity=NO_GRAVITY))

    assert joint_degrees(body, a, b, c) > 174.0


def test_a_max_of_180_compiles_to_nothing():
    body, a, b, c = triple()
    links = body.n_links

    angle_limit(body, a, b, c, max_degrees=180.0)

    assert body.n_links == links
    assert not body.has_link(a, c)


def test_a_floor_may_accompany_the_vacuous_ceiling():
    body, a, b, c = triple()

    angle_limit(body, a, b, c, min_degrees=120.0, max_degrees=180.0)

    link = body.link(a, c)
    assert link.type is LinkType.PUSH
    assert link.rest == pytest.approx(chord(1.0, 0.8, 120.0), rel=1e-6)


def test_a_floor_and_a_real_ceiling_cannot_share_a_joint():
    body, a, b, c = triple()
    with pytest.raises(ValueError, match="one link per pair"):
        angle_limit(body, a, b, c, min_degrees=60.0, max_degrees=150.0)


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"min_degrees": 0.0},
        {"min_degrees": -10.0},
        {"min_degrees": 190.0},
        {"max_degrees": 0.0},
        {"max_degrees": 181.0},
        {"min_degrees": 140.0, "max_degrees": 120.0},
        {"min_degrees": 120.0, "max_degrees": 120.0},
    ],
)
def test_bad_degree_arguments_are_rejected(bad):
    body, a, b, c = triple()
    with pytest.raises(ValueError):
        angle_limit(body, a, b, c, **bad)


def test_a_missing_bone_link_raises_by_name():
    body = Body(dim=3)
    a = body.add_chunk("a", (1.0, 0.0, 0.0))
    b = body.add_chunk("b", (0.0, 0.0, 0.0))
    c = body.add_chunk("c", (0.0, 1.0, 0.0))
    body.add_link(a, b)

    with pytest.raises(ValueError, match=f"no {b}-{c} link"):
        angle_limit(body, a, b, c, min_degrees=90.0)


def test_a_zero_length_bone_is_rejected():
    body = Body(dim=3)
    a = body.add_chunk("a", (0.0, 0.0, 0.0))
    b = body.add_chunk("b", (0.0, 0.0, 0.0))
    c = body.add_chunk("c", (0.0, 1.0, 0.0))
    body.add_link(a, b)
    body.add_link(b, c)

    with pytest.raises(ValueError, match="zero rest length"):
        angle_limit(body, a, b, c, min_degrees=90.0)


def test_the_triple_must_be_three_distinct_chunks():
    body, a, b, _ = triple()
    with pytest.raises(ValueError, match="distinct"):
        angle_limit(body, a, b, a, min_degrees=90.0)


def test_a_missing_chunk_raises_the_bodys_keyerror():
    body, a, b, _ = triple()
    with pytest.raises(KeyError):
        joint_degrees(body, a, b, 99)
    with pytest.raises(KeyError):
        angle_limit(body, a, b, 99, min_degrees=90.0)


def test_joint_degrees_needs_measurable_arms():
    body = Body(dim=3)
    a = body.add_chunk("a", (0.0, 0.0, 0.0))
    b = body.add_chunk("b", (0.0, 0.0, 0.0))
    c = body.add_chunk("c", (0.0, 1.0, 0.0))

    with pytest.raises(ValueError, match="no length"):
        joint_degrees(body, a, b, c)
