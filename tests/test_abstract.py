"""Tests for abstraction: package 4's acceptance, the cheap tier behind the room.

Four claims carry it. A parked creature wakes whole — chunks, links, meta,
controllers, stack order — where its anchor now stands, because parking is
the save/load contract turned inward. An abstracted room is anchors-only:
one pinned chunk per creature and nothing else in the body at all. The
brain never notices: it keeps thinking and walking its anchor through the
cheap room, on the trajectory it would have walked realized, to within
float32 of nothing. And the boundaries hold — the flag refuses a double
park and a double wake, activation moves the expensive tier with the
camera, and a door refuses a parked room on either side.
"""

from __future__ import annotations

import numpy as np
import pytest

from mesh_xml.anim.behavior import Grazer
from mesh_xml.anim.controllers.drive import WaveAnchor
from mesh_xml.anim.controllers.limb import FootholdPicker, LimbIK, LimbMode
from mesh_xml.anim.controllers.spine import SpineFollow
from mesh_xml.core import LinkType
from mesh_xml.scene import Room, Scene, move_creature


def chain(
    room: Room, species: str, head: tuple, step: tuple, pin: int | None = 0
) -> list[int]:
    """Three linked chunks; ``pin`` says which of them is the anchor, if any."""
    ids = [
        room.body.add_chunk(
            f"{species}{i}",
            tuple(h + i * s for h, s in zip(head, step)),
            radius=0.3,
        )
        for i in range(3)
    ]
    for a, b in zip(ids, ids[1:]):
        room.body.add_link(a, b)
    if pin is not None:
        room.body.set_pinned(ids[pin], True)
    return ids


class TestTheRoundTrip:
    def test_a_parked_creature_wakes_whole_where_it_stood(self):
        """Park and wake with the anchor unmoved, then compare field by field.

        Everything authored comes back — ids, names, positions, radii, pins,
        meta, the brace as a brace, controllers at their old stack indexes —
        and everything runtime re-derives, because a parked controller is
        rebuilt fresh rather than thawed: the picker wakes DANGLING no matter
        how firmly it was planted when the room went cheap.
        """
        room = Room(name="den")
        ids = [
            room.body.add_chunk(
                f"newt{i}",
                (i * 0.6, 0.5, 0.0),
                radius=0.25 + 0.05 * i,
                pinned=(i == 0),
                meta={"creature": "newt", "colour": "0.2 0.7 0.3", "tempo": "2.0"},
            )
            for i in range(4)
        ]
        for a, b in zip(ids, ids[1:]):
            room.body.add_link(a, b)
        room.body.add_link(
            ids[0], ids[2], rest=1.0, type=LinkType.PUSH, elasticity=0.4, render=False
        )
        picker = FootholdPicker(period=0.8, drive=(1.5, 0.0, 0.0))
        wave = room.controllers.add(
            WaveAnchor(anchor=ids[0], origin=(0.0, 0.5, 0.0), rate=0.7)
        )
        spine = room.controllers.add(SpineFollow(chain=list(ids), strength=17.0))
        brain = room.controllers.add(Grazer(ids[0], home=(0.0, 0.5, 0.0)))
        limb = room.controllers.add(
            LimbIK(root=ids[1], mid=ids[2], tip=ids[3], target=picker, reaction=1.0)
        )
        picker.mode = LimbMode.PLANTED
        before = {cid: room.body.chunk(cid) for cid in ids}

        room.abstract()
        assert room.abstracted
        assert room.controllers.controllers == [brain], "only the brain stays awake"

        room.realize()
        assert not room.abstracted
        assert sorted(room.body.chunk_ids()) == ids
        for cid, was in before.items():
            now = room.body.chunk(cid)
            assert now.name == was.name
            assert now.pos == pytest.approx(was.pos, abs=1e-5)
            assert now.radius == pytest.approx(was.radius)
            assert now.pinned == was.pinned
            assert now.meta == was.meta
        for a, b in zip(ids, ids[1:]):
            assert room.body.has_link(a, b)
        brace = room.body.link(ids[0], ids[2])
        assert brace.rest == pytest.approx(1.0)
        assert brace.type is LinkType.PUSH
        assert brace.elasticity == pytest.approx(0.4)
        assert brace.render is False, "a brace wakes as a brace"

        woken = room.controllers.controllers
        assert [type(c) for c in woken] == [WaveAnchor, SpineFollow, Grazer, LimbIK]
        wave2, spine2, brain2, limb2 = woken
        assert brain2 is brain, "the brain never left the stack"
        assert wave2 is not wave and spine2 is not spine and limb2 is not limb, (
            "what wakes is a rebuild, not the parked object"
        )
        assert wave2.anchor == ids[0] and wave2.rate == pytest.approx(0.7)
        assert wave2.origin == pytest.approx((0.0, 0.5, 0.0))
        assert spine2.chain == ids and spine2.strength == pytest.approx(17.0)
        assert limb2.target is not picker
        assert limb2.target.period == pytest.approx(0.8)
        assert limb2.target.mode is LimbMode.DANGLING, (
            "abstraction is a serialization boundary: runtime re-derives"
        )

    def test_an_abstracted_room_is_anchors_only(self):
        """One pinned chunk per creature: the brain's anchor, or the min id.

        The walker's brain sits on the middle chunk, so that chunk — not the
        component's minimum id — is what the room keeps; the brainless moss
        has no opinion to consult and keeps its minimum.
        """
        room = Room(name="gorge")
        walker = chain(room, "walker", (0.0, 0.5, 0.0), (0.6, 0.0, 0.0), pin=1)
        room.controllers.add(Grazer(walker[1], home=(0.6, 0.5, 0.0)))
        moss = chain(room, "moss", (5.0, 0.5, 0.0), (0.6, 0.0, 0.0), pin=None)

        room.abstract()

        assert sorted(room.body.chunk_ids()) == sorted([walker[1], moss[0]])
        for anchor in (walker[1], moss[0]):
            assert room.body.chunk(anchor).pinned, "an anchor parks pinned"

    def test_a_pinless_creature_is_pinned_while_parked_and_freed_on_waking(self):
        room = Room(name="pond")
        ids = [
            room.body.add_chunk(f"eel{i}", (i * 0.5, 2.0, 0.0), radius=0.2, mass=2.5)
            for i in range(3)
        ]
        for a, b in zip(ids, ids[1:]):
            room.body.add_link(a, b)

        room.abstract()
        assert room.body.chunk_ids() == [ids[0]], "no brain: the minimum id anchors"
        assert room.body.chunk(ids[0]).pinned

        rest = room.body.chunk(ids[0]).pos
        for _ in range(40):
            room.tick()
        assert room.body.chunk(ids[0]).pos == pytest.approx(rest), (
            "pinned while parked: gravity cannot drop what the room owes back"
        )

        room.realize()
        anchor = room.body.chunk(ids[0])
        assert not anchor.pinned, "waking hands the pinless truth back"
        assert anchor.mass == pytest.approx(2.5)
        assert room.body.n_links == 2


class TestTheAbstractBrain:
    def test_the_brain_keeps_living_while_the_flesh_is_parked(self):
        room = Room(name="meadow")
        ids = chain(room, "grazer", (0.0, 0.5, 0.0), (0.6, 0.0, 0.0))
        brain = room.controllers.add(Grazer(ids[0], home=(0.0, 0.5, 0.0)))
        parked_at = {cid: np.asarray(room.body.chunk(cid).pos) for cid in ids}

        room.abstract()
        for _ in range(200):
            room.tick()

        walked = np.asarray(room.body.chunk(ids[0]).pos) - parked_at[ids[0]]
        assert float(np.linalg.norm(walked)) > 0.05, (
            "an abstract creature still wanders"
        )

        room.realize()
        assert brain.home == pytest.approx((0.0, 0.5, 0.0)), (
            "home is authored; only the flesh follows the anchor"
        )
        for cid in ids:
            assert room.body.chunk(cid).pos == pytest.approx(
                tuple(parked_at[cid] + walked), abs=1e-4
            ), "the body wakes at its old shape plus the anchor's walk"

    def test_abstraction_does_not_change_where_a_creature_decides_to_go(self):
        """Tick a twin realized and abstracted; the anchors travel together.

        This holds because the anchor is kinematic and the rng stream is the
        stack's: with the brain as the only controller, both rooms draw the
        same numbers at the same times, and the cheap tier changes what a
        tick costs without changing anything the brain can perceive.
        """

        def meadow(name: str) -> tuple[Room, int]:
            room = Room(name=name)
            ids = chain(room, "grazer", (0.0, 0.5, 0.0), (0.6, 0.0, 0.0))
            room.controllers.add(Grazer(ids[0], home=(0.0, 0.5, 0.0)))
            return room, ids[0]

        seen, seen_anchor = meadow("seen")
        unseen, unseen_anchor = meadow("unseen")
        unseen.abstract()

        for _ in range(400):
            seen.tick()
            unseen.tick()

        assert np.allclose(
            np.asarray(seen.body.chunk(seen_anchor).pos),
            np.asarray(unseen.body.chunk(unseen_anchor).pos),
            atol=1e-5,
        ), "abstraction must not change where a creature decides to go"


class TestTheRealizationBoundary:
    def test_the_flag_refuses_a_double_park_and_a_double_wake(self):
        room = Room(name="den")
        chain(room, "newt", (0.0, 0.5, 0.0), (0.6, 0.0, 0.0))
        room.abstract()
        with pytest.raises(ValueError):
            room.abstract()
        room.realize()
        with pytest.raises(ValueError):
            room.realize()

        void = Room(name="void")
        void.abstract()
        assert void.abstracted, "the flag, not the parked list, is the witness"
        with pytest.raises(ValueError):
            void.abstract()
        void.realize()
        assert not void.abstracted

    def test_activation_realizes_one_room_and_parks_the_rest(self):
        scene = Scene()
        homes = {}
        for name in ("den", "meadow", "gorge"):
            room = scene.add_room(Room(name=name))
            homes[name] = chain(room, name, (0.0, 0.5, 0.0), (0.6, 0.0, 0.0))

        scene.activate("meadow")
        assert [r.abstracted for r in scene.rooms] == [True, False, True]

        scene.activate("den")
        assert [r.abstracted for r in scene.rooms] == [False, True, True]
        assert sorted(scene.room("den").body.chunk_ids()) == sorted(homes["den"]), (
            "the camera's room is flesh again"
        )

    def test_a_crossing_refuses_a_parked_room_on_either_side(self):
        den, meadow = Room(name="den"), Room(name="meadow")
        newt = chain(den, "newt", (0.0, 0.5, 0.0), (0.6, 0.0, 0.0))
        moss = chain(meadow, "moss", (4.0, 0.5, 0.0), (0.6, 0.0, 0.0))

        meadow.abstract()
        with pytest.raises(ValueError):
            move_creature(den, meadow, newt[0])
        assert sorted(den.body.chunk_ids()) == sorted(newt), "nothing left the den"
        assert meadow.body.chunk_ids() == [moss[0]], (
            "nothing landed in the parked meadow"
        )

        meadow.realize()
        den.abstract()
        with pytest.raises(ValueError):
            move_creature(den, meadow, newt[0])
        assert sorted(meadow.body.chunk_ids()) == sorted(moss)
        assert den.body.chunk_ids() == [newt[0]]
