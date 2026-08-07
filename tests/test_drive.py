"""Tests for the driver controllers: kinematic leads on pinned chunks."""

import numpy as np
import pytest

from mesh_xml.anim import ControllerStack
from mesh_xml.anim.controllers.drive import WaveAnchor
from mesh_xml.anim.serialize import decode_controller, encode_controller
from mesh_xml.core import Body
from mesh_xml.core.solver import SolverConfig

TICK = 1.0 / 40.0


def anchored_pair() -> tuple[Body, int, int]:
    body = Body()
    head = body.add_chunk("head", (1.0, 2.0, 3.0), pinned=True)
    tail = body.add_chunk("tail", (2.0, 2.0, 3.0))
    body.add_link(head, tail)
    return body, head, tail


class TestWaveAnchor:
    def test_the_anchor_rides_origin_plus_swing(self):
        body, head, _tail = anchored_pair()
        stack = ControllerStack(body)
        wave = stack.add(WaveAnchor(head, (1.0, 2.0, 3.0)))

        for _ in range(37):
            stack.tick(body, TICK, SolverConfig())

        expected = np.asarray(wave.origin) + wave.swing(stack.time - TICK)
        assert body.chunk(head).pos == pytest.approx(tuple(expected), abs=1e-5)

    def test_a_held_anchor_is_left_to_the_hand(self):
        body, head, _tail = anchored_pair()
        stack = ControllerStack(body)
        stack.add(WaveAnchor(head, (1.0, 2.0, 3.0)))
        stack.held.add(head)

        body.set_pos(head, (9.0, 9.0, 9.0))
        for _ in range(20):
            stack.tick(body, TICK, SolverConfig())
        assert body.chunk(head).pos == (9.0, 9.0, 9.0)

        stack.held.discard(head)
        stack.tick(body, TICK, SolverConfig())
        assert body.chunk(head).pos != (9.0, 9.0, 9.0)

    def test_swing_is_sized_to_the_origin(self):
        flat = WaveAnchor(0, (0.0, 0.0))
        assert flat.swing(1.7).shape == (2,)

    def test_round_trips_through_the_registry(self):
        wave = WaveAnchor(4, (0.5, 1.5, -2.0), amplitude=1.1, rate=0.9)
        clone = decode_controller(encode_controller(wave))

        assert isinstance(clone, WaveAnchor)
        assert clone.anchor == 4
        assert tuple(clone.origin) == pytest.approx((0.5, 1.5, -2.0))
        assert clone.amplitude == pytest.approx(1.1)
        assert clone.rate == pytest.approx(0.9)
