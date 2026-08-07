"""Tests for topology analysis and controller fitting.

The decisive test is :meth:`TestTrunk.test_legs_do_not_become_the_trunk`. The
longest path through a quadruped genuinely does run down one leg and up another;
it is the correct answer to the wrong question, and nothing about it looks like
a bug from the outside.
"""

import numpy as np
import pytest

from mesh_xml.anim import autorig
from mesh_xml.core import Body
from mesh_xml.core.solver import SolverConfig


def chain(n: int = 8, spacing: float = 0.5, height: float = 1.0) -> Body:
    body = Body()
    ids = [body.add_chunk(f"c{i}", [i * spacing, height, 0.0], radius=0.15) for i in range(n)]
    for a, b in zip(ids, ids[1:]):
        body.add_link(a, b)
    return body


def quadruped() -> tuple[Body, list[int], list[list[int]]]:
    body = Body()
    spine = [body.add_chunk(f"s{i}", [-i * 0.5, 1.0, 0.0], radius=0.2) for i in range(5)]
    for a, b in zip(spine, spine[1:]):
        body.add_link(a, b)

    legs = []
    for index, (anchor, z) in enumerate(
        ((spine[1], 0.3), (spine[1], -0.3), (spine[3], 0.3), (spine[3], -0.3))
    ):
        x = float(body.chunk(anchor).pos[0])
        root = body.add_chunk(f"L{index}r", [x, 1.0, z], radius=0.1)
        knee = body.add_chunk(f"L{index}k", [x, 0.55, z], radius=0.09)
        foot = body.add_chunk(f"L{index}f", [x, 0.1, z], radius=0.09)
        body.add_link(anchor, root)
        body.add_link(root, knee)
        body.add_link(knee, foot)
        legs.append([root, knee, foot])
    return body, spine, legs


class TestTrunk:
    def test_a_plain_chain_is_all_trunk(self):
        body = chain(8)
        topology = autorig.analyse(body)

        assert len(topology.trunk) == 8
        assert topology.appendages == ()
        assert topology.is_single_chain

    def test_legs_do_not_become_the_trunk(self):
        """The longest path really does run leg-to-leg. That is not a spine."""
        body, spine, legs = quadruped()
        topology = autorig.analyse(body, ground_y=0.0)

        assert set(topology.trunk.chunks) == set(spine)
        assert len(topology.leg_candidates) == 4
        for leg in legs:
            assert not set(leg) & set(topology.trunk.chunks)

    def test_without_a_ground_the_legs_are_not_recognised(self):
        """No floor means no way to tell a leg from a neck, and it says so."""
        body, spine, _ = quadruped()
        topology = autorig.analyse(body)

        assert topology.leg_candidates == ()
        assert set(topology.trunk.chunks) != set(spine)

    def test_a_branch_that_is_not_a_leg_becomes_an_appendage(self):
        body = chain(6)
        ids = body.chunk_ids()
        # Kept short on purpose: a branch longer than the stub it competes with
        # legitimately becomes the trunk, which is correct and not what is under
        # test here.
        arm = body.add_chunk("arm", [1.0, 1.5, 0.0], radius=0.1)
        body.add_link(ids[2], arm)

        topology = autorig.analyse(body, ground_y=0.0)
        assert set(topology.trunk.chunks) == set(ids)
        assert len(topology.appendages) == 1
        assert topology.appendages[0].tip == arm

    def test_an_appendage_is_oriented_outward(self):
        """Its first chunk must be the attachment, or the follow chain runs backwards."""
        body = chain(6)
        ids = body.chunk_ids()
        arm = body.add_chunk("arm", [1.0, 1.5, 0.0], radius=0.1)
        body.add_link(ids[2], arm)

        appendage = autorig.analyse(body, ground_y=0.0).appendages[0]
        assert appendage.base == ids[2]

    def test_an_empty_body_analyses_to_nothing(self):
        topology = autorig.analyse(Body())
        assert topology.trunk is None
        assert not topology.is_single_chain

    def test_a_lone_chunk_is_a_trunk_of_one(self):
        body = Body()
        body.add_chunk("only", [0.0, 0.0, 0.0])
        assert len(autorig.analyse(body).trunk) == 1

    def test_a_ring_terminates(self):
        """No node has degree one, so the endpoint-driven walk has to cope."""
        body = Body()
        ids = [
            body.add_chunk(f"r{i}", [np.cos(i), 1.0, np.sin(i)], radius=0.1) for i in range(6)
        ]
        for a, b in zip(ids, ids[1:]):
            body.add_link(a, b)
        body.add_link(ids[-1], ids[0])

        topology = autorig.analyse(body)
        assert topology.trunk is not None
        assert len(topology.trunk) >= 2


class TestFitting:
    def test_the_anchor_is_pinned(self):
        body = chain(8)
        rig = autorig.auto_controllers(body)

        assert rig.anchor is not None
        assert body.inv_mass[body.slot(rig.anchor)] == 0.0

    def test_the_anchor_is_a_trunk_end(self):
        body = chain(8)
        rig = autorig.auto_controllers(body)
        assert rig.anchor in (rig.topology.trunk.base, rig.topology.trunk.tip)

    def test_the_highest_end_anchors_by_default(self):
        body = Body()
        low = body.add_chunk("low", [0.0, 0.0, 0.0], radius=0.1)
        mid = body.add_chunk("mid", [0.0, 1.0, 0.0], radius=0.1)
        high = body.add_chunk("high", [0.0, 2.0, 0.0], radius=0.1)
        body.add_link(low, mid)
        body.add_link(mid, high)

        assert autorig.auto_controllers(body).anchor == high

    def test_an_explicit_anchor_is_honoured(self):
        body = chain(8)
        wanted = body.chunk_ids()[-1]
        assert autorig.auto_controllers(body, anchor=wanted).anchor == wanted

    def test_an_unknown_anchor_falls_back(self):
        body = chain(8)
        rig = autorig.auto_controllers(body, anchor=9999)
        assert rig.anchor in body.chunk_ids()

    def test_the_trunk_runs_away_from_the_anchor(self):
        """SpineFollow chases chain[0]; anchored at the far end it would invert."""
        body = chain(8)
        wanted = body.chunk_ids()[-1]
        rig = autorig.auto_controllers(body, anchor=wanted)
        spine = next(c for c in rig.stack.controllers if hasattr(c, "chain"))

        assert spine.chain[0] == wanted

    def test_an_empty_body_fits_nothing_and_does_not_raise(self):
        rig = autorig.auto_controllers(Body())
        assert rig.anchor is None
        assert rig.stack.controllers == []


class TestItActuallyAnimates:
    @pytest.mark.parametrize("length", [4, 10, 20])
    def test_a_chain_moves_and_stays_finite(self, length):
        body = chain(length)
        rig = autorig.auto_controllers(body, ground_y=0.0)
        config = SolverConfig(ground_y=0.0, iterations=10)

        start = body.pos.copy()
        for _ in range(600):
            rig.stack.tick(body, 1 / 40, config)

        assert np.isfinite(body.pos).all()
        assert np.abs(body.pos - start).max() > 0.01
        assert np.abs(body.pos).max() < 1e3

    def test_the_anchor_never_moves(self):
        body = chain(10)
        rig = autorig.auto_controllers(body, ground_y=0.0)
        anchored = body.chunk(rig.anchor).pos

        for _ in range(400):
            rig.stack.tick(body, 1 / 40, SolverConfig(ground_y=0.0))

        assert body.chunk(rig.anchor).pos == pytest.approx(anchored)

    def test_every_shipped_rig_animates(self):
        """The examples carry their gaits: open one, tick it, it moves.

        This is the promise the editor's Open Example menu makes. The rigs
        were upgraded from bare v1 armatures to v3 documents with an autorig
        plus a WaveAnchor driver baked in, precisely because a shipped example
        that hangs motionless under Play reads as a broken app.
        """
        from mesh_xml.io import xml_v2

        for name in ("chubby_worm", "character_armature", "simple_armature"):
            rig = xml_v2.read(f"examples/{name}.xml")
            body = rig.body
            assert rig.controllers.controllers, name
            start = body.pos.copy()
            for _ in range(300):
                rig.controllers.tick(
                    body, 1 / 40, SolverConfig(ground_y=0.0, iterations=10)
                )

            assert np.isfinite(body.pos).all(), name
            assert float(np.abs(body.pos - start).max()) > 0.3, name


def legged_worm() -> tuple[Body, list[int], dict[int, list[list[int]]]]:
    """The editor-authored shape: a spine with four legs grown by Add.

    Geometry deliberately lands on the proven walker's proportions — spine at
    the back height, hips a bone below, feet a radius off the floor — so the
    span-normalised gait numbers compile to the values the balance suite
    measured. Each leg is three chunks hanging from a spine chunk, exactly
    what growing with the Add gesture produces.
    """
    body = Body(name="legged")
    spine = [
        body.add_chunk(f"s{i}", (i * 0.42, 1.05, 0.0), radius=0.2, mass=1.5)
        for i in range(6)
    ]
    for a, b in zip(spine, spine[1:]):
        body.add_link(a, b)

    legs: dict[int, list[list[int]]] = {}
    for station in (spine[1], spine[4]):
        x = body.chunk(station).pos[0]
        legs[station] = []
        for side in (0.3, -0.3):
            hip = body.add_chunk(f"h{x:.1f}{side:+.1f}", (x, 0.9, side), radius=0.12, mass=1.0)
            knee = body.add_chunk(f"k{x:.1f}{side:+.1f}", (x, 0.49, side), radius=0.1, mass=0.4)
            foot = body.add_chunk(f"f{x:.1f}{side:+.1f}", (x, 0.08, side), radius=0.08, mass=0.4)
            body.add_link(station, hip)
            body.add_link(hip, knee)
            body.add_link(knee, foot)
            legs[station].append([hip, knee, foot])
    return body, spine, legs


class TestFitLegs:
    def test_a_leg_becomes_a_reacted_limb(self):
        from mesh_xml.anim.controllers.limb import FootholdPicker, LimbIK

        body, _spine, legs = legged_worm()
        fitted = autorig.fit_legs(body, ground_y=0.0)
        assert fitted is not None

        iks = [c for c in fitted.controllers if isinstance(c, LimbIK)]
        assert len(iks) == 4
        fitted_limbs = {(ik.root, ik.mid, ik.tip) for ik in iks}
        authored = {
            tuple(leg) for station in legs.values() for leg in station
        }
        assert fitted_limbs == authored
        for ik in iks:
            assert ik.reaction == pytest.approx(1.0)
            assert isinstance(ik.target, FootholdPicker)
            assert ik.target.ground_y == pytest.approx(0.08)
            assert ik.target.stance[1] == pytest.approx(-0.82)

    def test_the_gait_is_a_trot(self):
        """Same station: opposite phase. Diagonal pair: same phase."""
        from mesh_xml.anim.controllers.limb import LimbIK

        body, spine, legs = legged_worm()
        fitted = autorig.fit_legs(body, ground_y=0.0)
        phase_of = {
            ik.root: ik.target.phase
            for ik in fitted.controllers
            if isinstance(ik, LimbIK)
        }
        (front_l, front_r) = [leg[0] for leg in legs[spine[1]]]
        (rear_l, rear_r) = [leg[0] for leg in legs[spine[4]]]

        assert phase_of[front_l] != phase_of[front_r]
        assert phase_of[rear_l] != phase_of[rear_r]
        assert phase_of[front_l] == phase_of[rear_r]
        assert phase_of[front_r] == phase_of[rear_l]

    def test_poles_split_fore_and_aft(self):
        """Knees fold against the direction of travel in front, with it behind.

        Heading-relative on purpose: the trunk's orientation is analyse's to
        choose, so the walk may come out +x or -x — what must hold is that
        the two stations' poles oppose each other along whatever drive the
        fitter chose, front folding back the way a quadruped's do.
        """
        from mesh_xml.anim.controllers.limb import LimbIK

        body, spine, legs = legged_worm()
        fitted = autorig.fit_legs(body, ground_y=0.0)
        iks = {
            ik.root: ik for ik in fitted.controllers if isinstance(ik, LimbIK)
        }
        drive = np.asarray(next(iter(iks.values())).target.drive, dtype=float)
        drive_hat = drive / np.linalg.norm(drive)

        def station_along(station: int) -> float:
            return float(np.asarray(body.chunk(station).pos) @ drive_hat)

        front, rear = sorted(
            (spine[1], spine[4]), key=station_along, reverse=True
        )
        for hip in (leg[0] for leg in legs[front]):
            assert float(np.asarray(iks[hip].pole) @ drive_hat) < 0.0
        for hip in (leg[0] for leg in legs[rear]):
            assert float(np.asarray(iks[hip].pole) @ drive_hat) > 0.0

    def test_posture_presses_on_the_feet_and_level_spans_the_hips(self):
        from mesh_xml.anim.controllers.balance import Level, Posture

        body, spine, legs = legged_worm()
        fitted = autorig.fit_legs(body, ground_y=0.0)

        postures = [c for c in fitted.controllers if isinstance(c, Posture)]
        assert len(postures) == 2
        feet = {leg[2] for station in legs.values() for leg in station}
        for posture in postures:
            assert set(posture.support) == feet
        assert set(postures[0].chunks) == set(spine)
        assert postures[0].height == pytest.approx(1.05, abs=1e-5)

        levels = [c for c in fitted.controllers if isinstance(c, Level)]
        assert len(levels) == 1
        assert set(levels[0].chain) == {
            leg[0] for station in legs.values() for leg in station
        }

    def test_the_truss_holds_hips_off_the_centreline(self):
        body, spine, legs = legged_worm()
        fitted = autorig.fit_legs(body, ground_y=0.0)

        pairs = {tuple(sorted(pair)) for pair in fitted.truss}
        front_hips = [leg[0] for leg in legs[spine[1]]]
        # Each hip is triangulated against its station's trunk neighbours...
        for hip in front_hips:
            assert tuple(sorted((hip, spine[0]))) in pairs
            assert tuple(sorted((hip, spine[2]))) in pairs
        # ... and the station's pair brace each other.
        assert tuple(sorted((front_hips[0], front_hips[1]))) in pairs

    def test_pins_are_listed_for_removal(self):
        body, spine, _legs = legged_worm()
        body.set_pinned(spine[0], True)
        fitted = autorig.fit_legs(body, ground_y=0.0)
        assert fitted.unpin == [spine[0]]

    def test_a_leggless_body_fits_nothing(self):
        body = chain(6)
        assert autorig.fit_legs(body, ground_y=0.0) is None


class TestRiggedWalkActuallyWalks:
    def test_the_fitted_worm_walks_its_own_heading(self):
        """The whole point: grow legs in the editor's shape, fit, and it goes.

        The proven walker travels 19.8 units in 25 s; this body was fitted
        automatically rather than tuned by hand, so the floor here is
        deliberately humbler — but it must genuinely *walk*: metres of travel
        along its own spine axis, trunk held up, every value finite.
        """
        from mesh_xml.anim import ControllerStack

        body, spine, _legs = legged_worm()
        fitted = autorig.fit_legs(body, ground_y=0.0)
        for a, b in fitted.truss:
            if not body.has_link(a, b):
                body.add_link(a, b, render=False)
        for cid in fitted.unpin:
            body.set_pinned(cid, False)
        stack = ControllerStack(body, seed=7)
        for controller in fitted.controllers:
            stack.add(controller)

        from mesh_xml.anim.controllers.limb import LimbIK

        drive = np.asarray(
            next(
                c for c in fitted.controllers if isinstance(c, LimbIK)
            ).target.drive,
            dtype=float,
        )
        drive_hat = drive / np.linalg.norm(drive)

        start = body.pos.mean(axis=0).copy()
        config = SolverConfig(ground_y=0.0, iterations=8)
        for _ in range(1000):
            stack.tick(body, 1.0 / 40.0, config)

        assert np.isfinite(body.pos).all()
        moved = np.asarray(body.pos.mean(axis=0) - start, dtype=float)
        travelled = float(moved[: drive_hat.size] @ drive_hat)
        spine_height = float(
            np.mean([body.chunk(cid).pos[1] for cid in spine])
        )
        assert travelled > 3.0, f"only travelled {travelled:.2f} along its drive"
        assert spine_height > 0.7, f"trunk sagged to {spine_height:.2f}"
