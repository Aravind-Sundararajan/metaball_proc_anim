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
        from mesh_xml.io import xml_legacy

        for name in ("chubby_worm", "character_armature", "simple_armature"):
            body = xml_legacy.read_legacy(f"examples/{name}.xml").body
            rig = autorig.auto_controllers(body, ground_y=0.0)
            for _ in range(300):
                rig.stack.tick(body, 1 / 40, SolverConfig(ground_y=0.0, iterations=10))

            assert np.isfinite(body.pos).all(), name
            assert rig.topology.trunk is not None, name
