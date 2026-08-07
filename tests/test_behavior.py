"""Tests for the behavior tier: the FSM, the GOAP planner, and the two brains.

The planner and the machine are tested as pure logic. The brains are tested
through a real ControllerStack on a real Body, because their whole contract —
think slowly, steer only your own pinned anchor, yield to the hand — is about
how they sit in the stack, not what they compute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from mesh_xml.anim import ControllerStack
from mesh_xml.anim.behavior import FED, HUNT, Fsm, GoapAction, Grazer, Hunter, plan
from mesh_xml.anim.serialize import decode_controller, encode_controller, fields_of
from mesh_xml.core import Body
from mesh_xml.core.solver import SolverConfig

TICK = 1.0 / 40.0


@dataclass
class _Beacon:
    """A peer another brain can sense: a species tag on an anchor, nothing else."""

    anchor: int
    species: str
    enabled: bool = True

    def update(self, ctx) -> None:  # pragma: no cover - deliberately inert
        pass


def arena(*positions: tuple[float, float, float]) -> tuple[Body, list[int]]:
    body = Body()
    ids = [
        body.add_chunk(f"a{i}", pos, radius=0.3, pinned=True)
        for i, pos in enumerate(positions)
    ]
    return body, ids


def think_once(stack: ControllerStack, body: Body) -> None:
    for controller in stack.controllers:
        next_think = getattr(controller, "_next_think", None)
        if next_think is not None:
            controller._next_think = 0.0
    stack.accumulate(body, TICK, SolverConfig())


class TestFsm:
    def test_a_transition_acts_in_the_same_step(self):
        """The state that was chosen runs immediately — no one-think stutter."""
        log = []
        machine = Fsm(
            {"a": lambda: (log.append("a"), "b")[1], "b": lambda: log.append("b")},
            "a",
        )
        assert machine.step() == "b"
        assert log == ["a", "b"]
        assert machine.step() == "b"
        assert log == ["a", "b", "b"]

    def test_none_means_stay(self):
        machine = Fsm({"only": lambda: None}, "only")
        assert machine.step() == "only"

    def test_a_ping_pong_within_one_step_raises(self):
        machine = Fsm({"a": lambda: "b", "b": lambda: "a"}, "a")
        with pytest.raises(RuntimeError, match="did not settle"):
            machine.step()

    def test_unknown_initial_state_raises(self):
        with pytest.raises(ValueError, match="unknown initial"):
            Fsm({"a": lambda: None}, "b")

    def test_transition_to_unknown_state_raises(self):
        machine = Fsm({"a": lambda: "ghost"}, "a")
        with pytest.raises(ValueError, match="ghost"):
            machine.step()

    def test_an_empty_machine_raises(self):
        with pytest.raises(ValueError):
            Fsm({}, "a")


class TestPlanner:
    def test_a_satisfied_goal_needs_no_plan(self):
        assert plan({"fed": True}, FED, HUNT) == []

    def test_plans_the_whole_hunt_from_nothing(self):
        assert plan({}, FED, HUNT) == ["prowl", "close_in", "strike"]

    def test_plans_from_the_middle_when_the_world_is_ahead(self):
        assert plan({"prey_visible": True}, FED, HUNT) == ["close_in", "strike"]
        assert plan({"in_reach": True}, FED, HUNT) == ["strike"]

    def test_an_impossible_goal_is_an_answer(self):
        assert plan({}, {"airborne": True}, HUNT) is None

    def test_prefers_the_cheaper_route(self):
        actions = (
            GoapAction("door", pre={}, eff={"inside": True}, cost=1.0),
            GoapAction("window", pre={}, eff={"inside": True}, cost=5.0),
        )
        assert plan({}, {"inside": True}, actions) == ["door"]


class TestSteering:
    def test_moves_toward_the_destination_at_speed(self):
        body, (anchor,) = arena((0.0, 1.0, 0.0))
        stack = ControllerStack(body)
        brain = stack.add(Grazer(anchor, home=(0.0, 1.0, 0.0), speed=2.0, sway=0.0))
        brain._destination = np.array([10.0, 1.0, 0.0])
        brain._next_think = float("inf")

        stack.accumulate(body, TICK, SolverConfig())

        moved = np.asarray(body.chunk(anchor).pos)
        assert moved[0] == pytest.approx(2.0 * TICK, abs=1e-5)
        assert moved[1] == pytest.approx(1.0)

    def test_a_held_anchor_belongs_to_the_hand(self):
        body, (anchor,) = arena((0.0, 1.0, 0.0))
        stack = ControllerStack(body)
        brain = stack.add(Grazer(anchor, home=(0.0, 1.0, 0.0)))
        brain._destination = np.array([10.0, 1.0, 0.0])
        brain._next_think = float("inf")
        stack.held.add(anchor)

        stack.accumulate(body, TICK, SolverConfig())

        assert body.chunk(anchor).pos == (0.0, 1.0, 0.0)

    def test_an_arrived_brain_stays_put(self):
        body, (anchor,) = arena((3.0, 1.0, 0.0))
        stack = ControllerStack(body)
        brain = stack.add(Grazer(anchor, home=(3.0, 1.0, 0.0)))
        brain._destination = np.array([3.0, 1.0, 0.0])
        brain._next_think = float("inf")

        stack.accumulate(body, TICK, SolverConfig())

        assert body.chunk(anchor).pos == (3.0, 1.0, 0.0)


class TestGrazer:
    def test_wanders_its_home_range_when_safe(self):
        body, (anchor,) = arena((0.0, 1.0, 0.0))
        stack = ControllerStack(body, seed=3)
        brain = stack.add(Grazer(anchor, home=(0.0, 1.0, 0.0), wander_range=2.0))

        think_once(stack, body)

        assert brain.state == "wander"
        assert brain._destination is not None
        reach = np.linalg.norm(brain._destination - np.array([0.0, 1.0, 0.0]))
        assert 0.0 < float(reach) <= 2.0 + 1e-6

    def test_flees_a_hunter_inside_sense_range(self):
        body, (anchor, threat) = arena((0.0, 1.0, 0.0), (2.0, 1.0, 0.0))
        stack = ControllerStack(body, seed=3)
        brain = stack.add(Grazer(anchor, home=(0.0, 1.0, 0.0), sense_range=4.0))
        stack.add(_Beacon(threat, "hunter"))

        think_once(stack, body)

        assert brain.state == "flee"
        away = brain._destination - np.array([0.0, 1.0, 0.0])
        assert away[0] < 0.0  # threat is at +x; flight is -x

    def test_returns_to_wander_once_clear(self):
        body, (anchor, threat) = arena((0.0, 1.0, 0.0), (2.0, 1.0, 0.0))
        stack = ControllerStack(body, seed=3)
        brain = stack.add(Grazer(anchor, home=(0.0, 1.0, 0.0), sense_range=4.0))
        stack.add(_Beacon(threat, "hunter"))

        think_once(stack, body)
        assert brain.state == "flee"

        body.set_pos(threat, (50.0, 1.0, 0.0))
        think_once(stack, body)
        assert brain.state == "wander"

    def test_ignores_species_it_does_not_fear(self):
        body, (anchor, other) = arena((0.0, 1.0, 0.0), (1.0, 1.0, 0.0))
        stack = ControllerStack(body, seed=3)
        brain = stack.add(Grazer(anchor, home=(0.0, 1.0, 0.0)))
        stack.add(_Beacon(other, "grazer"))

        think_once(stack, body)

        assert brain.state == "wander"


class TestHunter:
    def test_plans_the_whole_hunt_when_no_prey_is_visible(self):
        body, (anchor,) = arena((0.0, 1.0, 0.0))
        stack = ControllerStack(body, seed=5)
        brain = stack.add(Hunter(anchor, home=(0.0, 1.0, 0.0)))

        think_once(stack, body)

        assert brain._plan == ["prowl", "close_in", "strike"]
        assert brain.intent == "prowl"

    def test_visible_prey_shortens_the_plan_and_aims_at_it(self):
        body, (anchor, prey) = arena((0.0, 1.0, 0.0), (4.0, 1.0, 0.0))
        stack = ControllerStack(body, seed=5)
        brain = stack.add(Hunter(anchor, home=(0.0, 1.0, 0.0), sense_range=6.0))
        stack.add(_Beacon(prey, "grazer"))

        think_once(stack, body)

        assert brain._plan == ["close_in", "strike"]
        assert brain._destination == pytest.approx((4.0, 1.0, 0.0))

    def test_prey_in_reach_is_struck_and_digested(self):
        body, (anchor, prey) = arena((0.0, 1.0, 0.0), (0.6, 1.0, 0.0))
        stack = ControllerStack(body, seed=5)
        brain = stack.add(Hunter(anchor, home=(0.0, 1.0, 0.0), catch_range=1.0))
        stack.add(_Beacon(prey, "grazer"))

        think_once(stack, body)

        assert brain.intent == "digest"
        assert brain._fed_until > 0.0

    def test_escaped_prey_forces_a_replan(self):
        body, (anchor, prey) = arena((0.0, 1.0, 0.0), (4.0, 1.0, 0.0))
        stack = ControllerStack(body, seed=5)
        brain = stack.add(Hunter(anchor, home=(0.0, 1.0, 0.0), sense_range=6.0))
        stack.add(_Beacon(prey, "grazer"))

        think_once(stack, body)
        assert brain._plan == ["close_in", "strike"]

        body.set_pos(prey, (60.0, 1.0, 0.0))
        think_once(stack, body)
        assert brain._plan == ["prowl", "close_in", "strike"]

    def test_hunger_returns_after_the_feast(self):
        """With prey still in reach when hunger returns, the plan is one
        action long and executes within the same think — straight back to
        digesting, with the feast clock pushed out."""
        body, (anchor, prey) = arena((0.0, 1.0, 0.0), (0.6, 1.0, 0.0))
        stack = ControllerStack(body, seed=5)
        brain = stack.add(Hunter(anchor, home=(0.0, 1.0, 0.0), feast_time=1.0))
        stack.add(_Beacon(prey, "grazer"))

        think_once(stack, body)
        assert brain.intent == "digest"
        first_feast = brain._fed_until

        stack.time = first_feast + 0.1
        think_once(stack, body)
        assert brain.intent == "digest"
        assert brain._fed_until > first_feast

        body.set_pos(prey, (4.0, 1.0, 0.0))
        stack.time = brain._fed_until + 0.1
        think_once(stack, body)
        assert brain.intent == "close_in"  # hungry again, prey visible but out of reach


class TestPersistence:
    def test_both_brains_round_trip_through_the_registry(self):
        grazer = Grazer(3, home=(1.0, 2.0, 3.0), sense_range=5.5, state="flee")
        hunter = Hunter(7, home=(0.0, 1.0, 0.0), hunts="grazer", catch_range=0.9)

        grazer_clone = decode_controller(encode_controller(grazer))
        hunter_clone = decode_controller(encode_controller(hunter))

        assert isinstance(grazer_clone, Grazer)
        assert grazer_clone.state == "flee"
        assert grazer_clone.sense_range == pytest.approx(5.5)
        assert isinstance(hunter_clone, Hunter)
        assert hunter_clone.catch_range == pytest.approx(0.9)

    def test_the_panel_will_see_the_fsm_state(self):
        assert "state" in fields_of(Grazer)


class TestPredatorAndPrey:
    def test_the_faster_hunter_eventually_feeds(self):
        """The integration claim: two brains, one body, a chase that ends.

        The hunter is faster, so however the seeded wander points fall, the
        gap must close to a strike within thirty simulated seconds.
        """
        body = Body()
        hunter_ids = [
            body.add_chunk(f"h{i}", (-3.0 - i * 0.6, 0.5, 0.0), radius=0.3)
            for i in range(3)
        ]
        grazer_ids = [
            body.add_chunk(f"g{i}", (3.0 + i * 0.6, 0.5, 0.2), radius=0.3)
            for i in range(3)
        ]
        for ids in (hunter_ids, grazer_ids):
            for a, b in zip(ids, ids[1:]):
                body.add_link(a, b)
            body.set_pinned(ids[0], True)

        stack = ControllerStack(body, seed=9)
        hunter = stack.add(
            Hunter(hunter_ids[0], home=(-3.0, 0.5, 0.0), speed=2.2, sense_range=12.0)
        )
        stack.add(
            Grazer(grazer_ids[0], home=(3.0, 0.5, 0.0), speed=1.2, sense_range=4.0)
        )

        config = SolverConfig(ground_y=0.0)
        fed_at = None
        for tick in range(1200):
            stack.tick(body, TICK, config)
            if hunter.intent == "digest":
                fed_at = tick
                break

        assert np.isfinite(body.pos).all()
        assert fed_at is not None, "the hunter never caught the grazer"
