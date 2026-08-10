"""Tests for the scene and the transfer: package 3's acceptance.

Two claims carry it. The hunt split across two rooms round-trips whole — a
creature moved out and back is the same creature, authored field for
authored field, because transfer is the save/load journey run in memory.
And a transferred hunter hunts on arrival — the peers list is per-room, so
crossing is not teleportation but genuinely entering a world where the prey
now exists and the old room genuinely forgetting you.
"""

from __future__ import annotations

import numpy as np
import pytest

from mesh_xml.anim.behavior import Grazer, Hunter
from mesh_xml.anim.controllers.drive import WaveAnchor
from mesh_xml.anim.controllers.limb import FootholdPicker, LimbIK, LimbMode
from mesh_xml.anim.controllers.spine import SpineFollow
from mesh_xml.core import Body, LinkType
from mesh_xml.scene import Door, Room, Scene, move_creature

TICK = 1.0 / 40.0


def chain(room: Room, species: str, head: tuple, step: tuple, count: int = 3):
    ids = [
        room.body.add_chunk(
            f"{species}{i}",
            tuple(h + i * s for h, s in zip(head, step)),
            radius=0.3,
        )
        for i in range(count)
    ]
    for a, b in zip(ids, ids[1:]):
        room.body.add_link(a, b)
    room.body.set_pinned(ids[0], True)
    return ids


class TestScene:
    def test_rooms_share_the_scenes_clock_by_identity(self):
        scene = Scene()
        den = scene.add_room(Room(name="den"))
        meadow = scene.add_room(Room(name="meadow"))
        assert den.clock is scene.clock and meadow.clock is scene.clock

    def test_room_names_are_the_addressing_scheme(self):
        scene = Scene()
        scene.add_room(Room(name="den"))
        assert scene.room("den").name == "den"
        with pytest.raises(ValueError):
            scene.add_room(Room(name="den"))
        with pytest.raises(KeyError):
            scene.room("attic")

    def test_every_room_ticks_and_time_cannot_skew(self):
        scene = Scene()
        den = scene.add_room(Room(name="den"))
        meadow = scene.add_room(Room(name="meadow"))
        chain(den, "a", (0.0, 0.5, 0.0), (0.6, 0.0, 0.0))
        chain(meadow, "b", (0.0, 0.5, 0.0), (0.6, 0.0, 0.0))

        total = 0
        for _ in range(120):
            before = scene.clock.ticks
            scene.advance(1.0 / 60.0)
            total += scene.clock.ticks - before

        assert total > 0
        assert den.controllers.time == pytest.approx(total * scene.clock.dt)
        assert den.controllers.time == pytest.approx(meadow.controllers.time)

    def test_sensing_stops_at_the_room_boundary(self):
        """Overlapping coordinates, different rooms: the grazer never sees it.

        ``ctx.peers`` is the stack's own list, and the stack is the room's —
        so the boundary needs no distance check anywhere. This is the room
        doing for perception what it does for physics.
        """
        scene = Scene()
        den = scene.add_room(Room(name="den"))
        meadow = scene.add_room(Room(name="meadow"))
        hunter_ids = chain(den, "hunter", (3.1, 0.5, 0.0), (-0.6, 0.0, 0.0))
        grazer_ids = chain(meadow, "grazer", (3.0, 0.5, 0.0), (0.6, 0.0, 0.0))
        den.controllers.add(
            Hunter(hunter_ids[0], home=(3.1, 0.5, 0.0), sense_range=12.0)
        )
        grazer = meadow.controllers.add(
            Grazer(grazer_ids[0], home=(3.0, 0.5, 0.0), sense_range=4.0)
        )

        for _ in range(80):
            scene.tick()
        assert grazer.state == "wander", "a hunter one room over is no hunter at all"


class TestTransfer:
    def test_the_creature_crosses_whole_and_returns_whole(self):
        """Out and back, then compared field by field against how it started.

        Ids are per-body and legitimately change with each crossing; what
        must survive is everything authored — geometry, masses, pins, meta,
        link character, controller settings — with runtime re-derived, the
        same contract a save already makes.
        """
        scene = Scene()
        den = scene.add_room(Room(name="den"))
        meadow = scene.add_room(Room(name="meadow"))

        ids = chain(den, "newt", (0.0, 0.5, 0.0), (0.6, 0.0, 0.0))
        den.body.add_link(
            ids[0], ids[2], rest=1.0, type=LinkType.PUSH, elasticity=0.4, render=False
        )
        for cid in ids:
            den.body.meta_of(cid).update(
                {"creature": "newt", "colour": "0.2 0.7 0.3", "tempo": "2.0"}
            )
        picker = FootholdPicker(period=0.8, drive=(1.5, 0.0, 0.0))
        den.controllers.add(WaveAnchor(anchor=ids[0], origin=(0.0, 0.5, 0.0), rate=0.7))
        den.controllers.add(SpineFollow(chain=list(ids), strength=17.0))
        limb = den.controllers.add(
            LimbIK(root=ids[0], mid=ids[1], tip=ids[2], target=picker, reaction=1.0)
        )
        limb.target.mode = LimbMode.PLANTED
        before = {cid: den.body.chunk(cid) for cid in ids}

        out = move_creature(den, meadow, ids[1])
        assert not den.body.chunk_ids() and not den.controllers.controllers
        back = move_creature(meadow, den, out[ids[1]])
        home = {old: back[out[old]] for old in ids}

        for old, chunk in before.items():
            returned = den.body.chunk(home[old])
            assert returned.name == chunk.name
            assert returned.pos == pytest.approx(chunk.pos)
            assert returned.radius == pytest.approx(chunk.radius)
            assert returned.pinned == chunk.pinned
            assert returned.meta == chunk.meta

        brace = den.body.link(home[ids[0]], home[ids[2]])
        assert brace.rest == pytest.approx(1.0)
        assert brace.type is LinkType.PUSH
        assert brace.elasticity == pytest.approx(0.4)
        assert brace.render is False, "a brace crosses as a brace"

        wave, spine, returned_limb = den.controllers.controllers
        assert isinstance(wave, WaveAnchor)
        assert wave.anchor == home[ids[0]]
        assert wave.origin == pytest.approx((0.0, 0.5, 0.0))
        assert wave.rate == pytest.approx(0.7)
        assert spine.chain == [home[cid] for cid in ids]
        assert spine.strength == pytest.approx(17.0)
        assert returned_limb.target.period == pytest.approx(0.8)
        assert returned_limb.target.mode is LimbMode.DANGLING, (
            "a door is a serialization boundary: runtime re-derives"
        )

    def test_a_transferred_hunter_hunts_on_arrival(self):
        scene = Scene()
        den = scene.add_room(Room(name="den"))
        meadow = scene.add_room(Room(name="meadow"))
        hunter_ids = chain(den, "hunter", (-3.0, 0.5, 0.0), (-0.6, 0.0, 0.0))
        grazer_ids = chain(meadow, "grazer", (3.0, 0.5, 0.2), (0.6, 0.0, 0.0))
        hunter = den.controllers.add(
            Hunter(hunter_ids[0], home=(-3.0, 0.5, 0.0), speed=2.2, sense_range=12.0)
        )
        meadow.controllers.add(
            Grazer(grazer_ids[0], home=(3.0, 0.5, 0.0), speed=1.2, sense_range=4.0)
        )

        for _ in range(200):
            scene.tick()
        assert hunter.intent != "digest", "nothing to eat in an empty den"

        move_creature(den, meadow, hunter_ids[0])
        assert not den.controllers.controllers, "the den forgot its hunter"
        assert hunter not in meadow.controllers.controllers, (
            "what crosses is a rebuild, not the object"
        )

        arrived = next(
            c for c in meadow.controllers.controllers if isinstance(c, Hunter)
        )
        fed_at = None
        for tick in range(1200):
            scene.tick()
            if arrived.intent == "digest":
                fed_at = tick
                break

        assert np.isfinite(meadow.body.pos).all()
        assert fed_at is not None, "the hunter never hunted in its new room"

    def test_an_offset_arrival_moves_flesh_and_goals_together(self):
        den = Room(name="den")
        meadow = Room(name="meadow")
        ids = chain(den, "worm", (0.0, 0.5, 0.0), (0.6, 0.0, 0.0))
        den.controllers.add(WaveAnchor(anchor=ids[0], origin=(0.0, 0.5, 0.0)))
        den.controllers.add(Grazer(ids[0], home=(0.0, 0.5, 0.0)))

        mapping = move_creature(den, meadow, ids[0], offset=(10.0, 0.0, -2.0))

        arrived = meadow.body.chunk(mapping[ids[0]])
        assert arrived.pos == pytest.approx((10.0, 0.5, -2.0))
        wave = next(
            c for c in meadow.controllers.controllers if isinstance(c, WaveAnchor)
        )
        brain = next(
            c for c in meadow.controllers.controllers if isinstance(c, Grazer)
        )
        assert wave.origin == pytest.approx((10.0, 0.5, -2.0))
        assert brain.home == pytest.approx((10.0, 0.5, -2.0))

    def test_the_other_creature_and_its_gait_stay_behind(self):
        den = Room(name="den")
        meadow = Room(name="meadow")
        stays = chain(den, "stays", (0.0, 0.5, 0.0), (0.6, 0.0, 0.0))
        goes = chain(den, "goes", (5.0, 0.5, 0.0), (0.6, 0.0, 0.0))
        den.controllers.add(WaveAnchor(anchor=stays[0], origin=(0.0, 0.5, 0.0)))
        den.controllers.add(WaveAnchor(anchor=goes[0], origin=(5.0, 0.5, 0.0)))

        move_creature(den, meadow, goes[0])

        assert sorted(den.body.chunk_ids()) == sorted(stays)
        remaining = den.controllers.controllers
        assert len(remaining) == 1 and remaining[0].anchor == stays[0]
        arrived = meadow.controllers.controllers
        assert len(arrived) == 1 and arrived[0].origin[0] == pytest.approx(5.0)

    def test_a_grazer_walks_out_of_the_den_into_the_meadow(self):
        """The whole door story: goal, walk, crossing, and a spent exit.

        The brain owns the walk (``exit`` steering through ``ctx.doors``),
        the scene owns the crossing, and the paired door's offset keeps the
        stride continuous — the creature steps out of the far door, home
        and all, and the goal clears because one door is one journey.
        """
        scene = Scene()
        den = scene.add_room(Room(name="den"))
        meadow = scene.add_room(Room(name="meadow"))
        ids = chain(den, "grazer", (0.0, 0.5, 0.0), (0.6, 0.0, 0.0))
        den.controllers.add(Grazer(ids[0], home=(0.0, 0.5, 0.0), exit="east"))
        den.doors.append(
            Door("east", (5.0, 0.5, 0.0), to_room="meadow", to_door="west")
        )
        meadow.doors.append(
            Door("west", (-5.0, 0.5, 0.0), to_room="den", to_door="east")
        )

        crossed_at = None
        for tick in range(1200):
            scene.tick()
            if meadow.body.chunk_ids():
                crossed_at = tick
                break

        assert crossed_at is not None, "the grazer never reached its door"
        assert not den.body.chunk_ids() and not den.controllers.controllers
        assert len(meadow.body.chunk_ids()) == 3

        arrived = next(
            c for c in meadow.controllers.controllers if isinstance(c, Grazer)
        )
        assert arrived.exit == "", "the goal is spent by crossing"
        assert arrived.home == pytest.approx((-10.0, 0.5, 0.0)), (
            "home crossed with the creature: authored point plus door offset"
        )
        anchor = np.asarray(meadow.body.chunk(arrived.anchor).pos)
        west = np.asarray(meadow.doors[0].at)
        assert float(np.linalg.norm(anchor - west)) <= meadow.doors[0].radius + 0.5

    def test_a_door_to_nowhere_is_waited_at(self):
        scene = Scene()
        den = scene.add_room(Room(name="den"))
        scene.add_room(Room(name="meadow"))
        ids = chain(den, "grazer", (0.0, 0.5, 0.0), (0.6, 0.0, 0.0))
        brain = den.controllers.add(
            Grazer(ids[0], home=(0.0, 0.5, 0.0), exit="east")
        )
        den.doors.append(
            Door("east", (5.0, 0.5, 0.0), to_room="attic", to_door="hatch")
        )

        for _ in range(600):
            scene.tick()

        assert sorted(den.body.chunk_ids()) == sorted(ids), "a closed door holds"
        assert brain.exit == "east", "the goal is not spent on a door to nowhere"
        anchor = np.asarray(den.body.chunk(ids[0]).pos)
        door = np.asarray(den.doors[0].at)
        assert float(np.linalg.norm(anchor - door)) <= den.doors[0].radius, (
            "the creature waits in the doorway"
        )

    def test_a_parked_room_holds_its_travellers_until_it_realizes(self):
        scene = Scene()
        den = scene.add_room(Room(name="den"))
        meadow = scene.add_room(Room(name="meadow"))
        ids = chain(den, "grazer", (0.0, 0.5, 0.0), (0.6, 0.0, 0.0))
        den.controllers.add(Grazer(ids[0], home=(0.0, 0.5, 0.0), exit="east"))
        den.doors.append(
            Door("east", (5.0, 0.5, 0.0), to_room="meadow", to_door="west")
        )
        meadow.doors.append(
            Door("west", (-5.0, 0.5, 0.0), to_room="den", to_door="east")
        )
        den.abstract()

        for _ in range(600):
            scene.tick()
        assert den.body.chunk_ids(), "an abstract room's creatures stay put"
        assert not meadow.body.chunk_ids()

        den.realize()
        crossed = False
        for _ in range(200):
            scene.tick()
            if meadow.body.chunk_ids():
                crossed = True
                break
        assert crossed, "realization opens the door"

    def test_refusals_leave_both_rooms_untouched(self):
        den = Room(name="den")
        meadow = Room(name="meadow")
        ids = chain(den, "worm", (0.0, 0.5, 0.0), (0.6, 0.0, 0.0))

        with pytest.raises(ValueError):
            move_creature(den, den, ids[0])
        with pytest.raises(KeyError):
            move_creature(den, meadow, 999)

        flat = Room(name="flat")
        flat.body = Body(dim=2)
        with pytest.raises(ValueError):
            move_creature(den, flat, ids[0])

        assert sorted(den.body.chunk_ids()) == sorted(ids)
        assert not meadow.body.chunk_ids()


class TestRemoveRoom:
    def test_removal_releases_the_room_and_its_clock(self):
        scene = Scene()
        den = scene.add_room(Room(name="den"))
        meadow = scene.add_room(Room(name="meadow"))

        scene.remove_room(meadow)
        assert scene.rooms == [den]
        assert meadow.clock is not scene.clock, "a lone room owns its time again"
        with pytest.raises(ValueError):
            scene.remove_room(meadow)

    def test_a_door_into_a_removed_room_is_closed(self):
        """Removal breaks no invariant: the promise just cannot be kept."""
        scene = Scene()
        den = scene.add_room(Room(name="den"))
        meadow = scene.add_room(Room(name="meadow"))
        meadow.doors.append(
            Door("west", (-5.0, 0.5, 0.0), to_room="den", to_door="east")
        )
        ids = chain(den, "grazer", (0.0, 0.5, 0.0), (0.6, 0.0, 0.0))
        brain = den.controllers.add(
            Grazer(ids[0], home=(0.0, 0.5, 0.0), exit="east")
        )
        den.doors.append(
            Door("east", (5.0, 0.5, 0.0), to_room="meadow", to_door="west")
        )

        scene.remove_room(meadow)
        for _ in range(600):
            scene.tick()

        assert sorted(den.body.chunk_ids()) == sorted(ids), "a closed door holds"
        assert brain.exit == "east", "the goal is not spent on a vanished room"