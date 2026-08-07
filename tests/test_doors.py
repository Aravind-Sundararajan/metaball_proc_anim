"""Tests for doors, the simulation half: the record, the section, the goal.

Package 5's claims, short of the crossing. A door is inert data that
refuses to be authored half-empty. Its ``<doors>`` section is additive
exactly as ``<terrain>`` is — absent means none — so every document written
before doors existed keeps meaning what it meant. The stack carries the
room's exits into each tick's context as perception, on ``held``'s terms.
And "leave via X" is the one goal louder than living: it overrides wander
and fear alike, because the author said leave and a grazer that dawdled to
flee would never cross a dangerous room. The crossing itself — arrival
detection, the transfer — is the scene's to wire next; these tests end at
the doorway on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from mesh_xml.anim import ControllerStack
from mesh_xml.anim.behavior import Grazer, Hunter, Scavenger
from mesh_xml.anim.serialize import decode_controller, encode_controller, fields_of
from mesh_xml.core import Body
from mesh_xml.core.solver import SolverConfig
from mesh_xml.io import xml_v2
from mesh_xml.scene import Door

TICK = 1.0 / 40.0
HUNT_XML = Path(__file__).resolve().parent.parent / "examples" / "hunt.xml"

#: The doorway most of these tests walk to.
EAST = (6.0, 0.5, 0.0)


@dataclass
class _Beacon:
    """A peer another brain can sense: a species tag on an anchor, nothing else."""

    anchor: int
    species: str
    enabled: bool = True

    def update(self, ctx) -> None:  # pragma: no cover - deliberately inert
        pass


class _Compass:
    """A recording controller: keeps whatever doors each tick's context showed."""

    enabled = True

    def __init__(self) -> None:
        self.seen: list[dict[str, tuple[float, ...]]] = []

    def update(self, ctx) -> None:
        self.seen.append(dict(ctx.doors))


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


def stub_rig() -> tuple[Body, ControllerStack]:
    """One pinned creature with a brain that intends to leave."""
    body = Body(dim=3, name="den")
    anchor = body.add_chunk("a", (0.0, 0.5, 0.0), radius=0.3, pinned=True)
    stack = ControllerStack(body, seed=3)
    stack.add(Grazer(anchor, home=(0.0, 0.5, 0.0), exit="east"))
    return body, stack


class TestTheDoorRefusesHalfAuthoring:
    def test_a_door_needs_a_name(self):
        with pytest.raises(ValueError, match="name"):
            Door("", (0.0, 0.5, 0.0), "meadow", "west")

    def test_a_door_needs_a_destination_room(self):
        with pytest.raises(ValueError, match="destination room"):
            Door("east", (0.0, 0.5, 0.0), "", "west")

    def test_a_door_needs_a_door_to_arrive_at(self):
        with pytest.raises(ValueError, match="arrive"):
            Door("east", (0.0, 0.5, 0.0), "meadow", "")

    def test_an_arrival_radius_must_be_positive(self):
        for radius in (0.0, -0.9):
            with pytest.raises(ValueError, match="radius"):
                Door("east", (0.0, 0.5, 0.0), "meadow", "west", radius=radius)


# ----------------------------------------------------------------------
# the <doors> section: additive in v3, absent means none
# ----------------------------------------------------------------------


class TestTheDoorsSection:
    def test_two_doors_round_trip_whole_and_exact(self, tmp_path: Path):
        # Values chosen to be moved by a float32 round trip, so full-precision
        # persistence is what this test actually exercises.
        doors = (
            Door("east", (6.0, 0.5, 1.0 / 3.0), "meadow", "west"),
            Door("burrow", (-2.25, 0.1 + 0.2, 0.0), "den", "mouth", radius=1.5),
        )
        body, stack = stub_rig()
        path = tmp_path / "doored.xml"
        xml_v2.save(body, path, stack, doors=doors)

        rig = xml_v2.read(path)

        assert len(rig.doors) == 2
        for loaded, original in zip(rig.doors, doors):
            assert loaded.name == original.name
            assert loaded.at == pytest.approx(original.at)
            assert loaded.to_room == original.to_room
            assert loaded.to_door == original.to_door
            assert loaded.radius == pytest.approx(original.radius)
        # Exact as well as close: door floats are written at full float64
        # precision, so a format that quietly rounded them would fail here.
        assert rig.doors == doors

    def test_the_goal_of_leaving_rides_in_the_same_document(self, tmp_path: Path):
        body, stack = stub_rig()
        path = tmp_path / "leaver.xml"
        xml_v2.save(body, path, stack, doors=(Door("east", EAST, "meadow", "west"),))

        brain = xml_v2.read(path).controllers.controllers[0]

        assert isinstance(brain, Grazer)
        assert brain.exit == "east"

    def test_a_doorless_document_has_no_section_and_reads_back_empty(
        self, tmp_path: Path
    ):
        body, stack = stub_rig()
        path = tmp_path / "doorless.xml"
        xml_v2.save(body, path, stack)

        assert "<doors" not in path.read_text(encoding="utf-8")
        assert xml_v2.read(path).doors == ()

    def test_the_shipped_hunt_still_reads_and_honestly_has_no_doors(self):
        rig = xml_v2.read(HUNT_XML)

        assert rig.body.name == "hunt"
        assert rig.doors == ()


def door_document(door: str) -> str:
    return (
        '<body version="3" dim="3"><chunks/><links/>'
        f"<doors>{door}</doors></body>"
    )


class TestMalformedDoors:
    def test_a_door_without_a_point_is_refused_naming_at(self, tmp_path: Path):
        path = tmp_path / "adrift.xml"
        path.write_text(
            door_document('<door name="east" to_room="meadow" to_door="west"/>'),
            encoding="utf-8",
        )

        with pytest.raises(xml_v2.SchemaError, match="'at'"):
            xml_v2.read(path)

    def test_a_point_that_is_not_numbers_is_refused_naming_at(self, tmp_path: Path):
        path = tmp_path / "gibberish.xml"
        path.write_text(
            door_document(
                '<door name="east" at="6.0 wide 0.0" to_room="meadow" to_door="west"/>'
            ),
            encoding="utf-8",
        )

        with pytest.raises(xml_v2.SchemaError, match="at="):
            xml_v2.read(path)

    def test_a_door_to_nowhere_is_refused_naming_to_room(self, tmp_path: Path):
        path = tmp_path / "nowhere.xml"
        path.write_text(
            door_document('<door name="east" at="6.0 0.5 0.0" to_door="west"/>'),
            encoding="utf-8",
        )

        with pytest.raises(xml_v2.SchemaError, match="to_room"):
            xml_v2.read(path)

    def test_a_radius_that_is_not_a_number_is_refused_naming_it(
        self, tmp_path: Path
    ):
        path = tmp_path / "wide.xml"
        path.write_text(
            door_document(
                '<door name="east" at="6.0 0.5 0.0" to_room="meadow" '
                'to_door="west" radius="wide"/>'
            ),
            encoding="utf-8",
        )

        with pytest.raises(xml_v2.SchemaError, match="radius"):
            xml_v2.read(path)

    def test_a_hand_edited_negative_radius_is_refused_on_load(self, tmp_path: Path):
        """The record's own validation guards the file path too, like terrain."""
        path = tmp_path / "inverted.xml"
        path.write_text(
            door_document(
                '<door name="east" at="6.0 0.5 0.0" to_room="meadow" '
                'to_door="west" radius="-0.9"/>'
            ),
            encoding="utf-8",
        )

        with pytest.raises(xml_v2.SchemaError, match="radius"):
            xml_v2.read(path)

    def test_an_unknown_element_inside_doors_is_refused(self, tmp_path: Path):
        path = tmp_path / "hatched.xml"
        path.write_text(door_document('<hatch to="attic"/>'), encoding="utf-8")

        with pytest.raises(xml_v2.SchemaError, match="hatch"):
            xml_v2.read(path)


# ----------------------------------------------------------------------
# plumbing: the stack declares the exits, the context reports them
# ----------------------------------------------------------------------


class TestPlumbing:
    def test_the_stacks_doors_flow_into_the_context(self):
        body = Body()
        body.add_chunk("a", (0.0, 0.0, 0.0))
        stack = ControllerStack(body)
        compass = stack.add(_Compass())

        stack.accumulate(body, TICK, SolverConfig())
        stack.doors["east"] = EAST
        stack.accumulate(body, TICK, SolverConfig())

        assert compass.seen == [{}, {"east": EAST}]

    def test_what_a_controller_sees_is_a_copy_not_the_room_itself(self):
        """Read your peers, write only what you own — the doors map included."""
        body = Body()
        body.add_chunk("a", (0.0, 0.0, 0.0))
        stack = ControllerStack(body)
        stack.doors["east"] = EAST

        class Vandal:
            enabled = True

            def update(self, ctx) -> None:
                ctx.doors["forged"] = (0.0, 0.0, 0.0)

        stack.add(Vandal())
        stack.accumulate(body, TICK, SolverConfig())

        assert stack.doors == {"east": EAST}


# ----------------------------------------------------------------------
# the goal: leaving is louder than living
# ----------------------------------------------------------------------


class TestLeaving:
    def test_the_named_door_becomes_the_destination_and_is_reached(self):
        body, (anchor,) = arena((0.0, 0.5, 0.0))
        stack = ControllerStack(body, seed=3)
        brain = stack.add(Grazer(anchor, home=(0.0, 0.5, 0.0), exit="east"))
        stack.doors["east"] = EAST

        think_once(stack, body)
        assert brain._destination == pytest.approx(EAST)

        config = SolverConfig(ground_y=0.0)
        for _ in range(400):
            stack.tick(body, TICK, config)
        gap = np.asarray(body.chunk(anchor).pos) - np.asarray(EAST)
        assert float(np.linalg.norm(gap)) <= 0.5, "ten seconds is a stroll to the door"

    def test_a_door_this_room_does_not_have_is_inert(self):
        """A goal is not an error: the brain lives normally, wandering home."""
        body, (anchor,) = arena((0.0, 0.5, 0.0))
        stack = ControllerStack(body, seed=3)
        brain = stack.add(Grazer(anchor, home=(0.0, 0.5, 0.0), exit="attic"))
        stack.doors["east"] = EAST

        think_once(stack, body)

        assert brain.state == "wander"
        assert brain._destination is not None
        assert not np.allclose(brain._destination, np.asarray(EAST))
        reach = float(np.linalg.norm(brain._destination - np.array((0.0, 0.5, 0.0))))
        assert reach <= brain.wander_range + 1e-6

    def test_an_empty_exit_is_the_old_behaviour_exactly(self):
        """Twin arenas, same seed: doors on offer change nothing unnamed."""
        doored_body, (doored_anchor,) = arena((0.0, 0.5, 0.0))
        doored = ControllerStack(doored_body, seed=3)
        stays = doored.add(Grazer(doored_anchor, home=(0.0, 0.5, 0.0)))
        doored.doors["east"] = EAST

        plain_body, (plain_anchor,) = arena((0.0, 0.5, 0.0))
        plain = ControllerStack(plain_body, seed=3)
        twin = plain.add(Grazer(plain_anchor, home=(0.0, 0.5, 0.0)))

        think_once(doored, doored_body)
        think_once(plain, plain_body)

        assert stays.exit == ""
        assert stays._destination == pytest.approx(twin._destination)

    def test_fear_does_not_outrank_leaving(self):
        """The hunter stands between the grazer and its door, and loses."""
        body, (anchor, threat) = arena((0.0, 0.5, 0.0), (2.0, 0.5, 0.0))
        stack = ControllerStack(body, seed=3)
        brain = stack.add(
            Grazer(anchor, home=(0.0, 0.5, 0.0), sense_range=4.0, exit="east")
        )
        stack.add(_Beacon(threat, "hunter"))
        stack.doors["east"] = EAST

        think_once(stack, body)

        assert brain.state == "flee", "the fear was felt"
        assert brain._destination == pytest.approx(EAST), (
            "and overruled: the author said leave"
        )


# ----------------------------------------------------------------------
# the goal is authored, so it persists
# ----------------------------------------------------------------------


class TestTheGoalIsAuthored:
    def test_exit_is_a_persisted_field_on_every_brain(self):
        for brain in (Grazer, Hunter, Scavenger):
            assert "exit" in fields_of(brain), brain.__name__

    @pytest.mark.parametrize("species", [Grazer, Hunter, Scavenger])
    def test_the_goal_survives_the_registry(self, species):
        brain = species(3, home=(1.0, 2.0, 3.0), exit="east")

        clone = decode_controller(encode_controller(brain))

        assert isinstance(clone, species)
        assert clone.exit == "east"
