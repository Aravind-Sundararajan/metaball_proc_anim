"""Tests for the editor scene binding: package 6's identity rework.

The null result — a one-room scene is bit-for-bit the old editor — is the
whole of ``test_document.py``, ``test_panels.py`` and ``test_viewport.py``,
which run against the reworked document unchanged. What is tested here is
only what the plural adds: the cut, the per-room state it stashes and
restores, and the scene ticking whole through the document's own loop.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from mesh_xml.anim import ControllerStack, tempo
from mesh_xml.anim.behavior import Grazer
from mesh_xml.core import Body
from mesh_xml.editor.document import Document, SetRadius
from mesh_xml.scene import Door, Room


@pytest.fixture
def document() -> Document:
    doc = Document()
    body = Body()
    ids = [body.add_chunk(f"c{i}", [float(i), 0.5, 0.0], radius=0.3) for i in range(5)]
    for a, b in zip(ids, ids[1:]):
        body.add_link(a, b)
    doc._replace(body, None)
    return doc


def ids_of(document: Document) -> list[int]:
    return document.body.chunk_ids()


def room_with_chain(document: Document, name: str, count: int = 3):
    room = document.add_room(Room(name=name))
    ids = [
        room.body.add_chunk(f"{name}{i}", (float(i) * 0.6, 0.5, 0.0), radius=0.3)
        for i in range(count)
    ]
    for a, b in zip(ids, ids[1:]):
        room.body.add_link(a, b)
    room.body.set_pinned(ids[0], True)
    return room, ids


def flat_room(document: Document, name: str = "flat") -> Room:
    """A 2-D room — the dimension a creature is refused a crossing into."""
    room = Room(name=name)
    room.body = Body(dim=2)
    room.controllers = ControllerStack(room.body)
    room.body.add_chunk("f0", (0.0, 0.5), radius=0.3)
    return document.add_room(room)


def snapshot(room: Room) -> tuple:
    """Everything a refusal must leave exactly as it found it."""
    return (
        [
            (cid, tuple(room.body.chunk(cid).pos))
            for cid in room.body.chunk_ids()
        ],
        [(link.a, link.b) for link in room.body.links()],
        len(room.controllers.controllers),
    )


class _Unreadable:
    """A controller the serializer never met — the pre-flight's whole reason.

    :func:`mesh_xml.anim.tempo.chunk_refs` answers an empty set for a class
    the registry does not know, so the transfer's own gate never counts one
    as travelling and never refuses it: left to the mechanism, the creature
    would move and this would stay behind holding an id that the source's
    next Add hands to someone else.
    """

    def __init__(self, anchor: int) -> None:
        self.anchor = anchor

    def update(self, ctx) -> None:  # pragma: no cover - never ticked
        pass


class TestTheCut:
    def test_a_fresh_document_is_a_scene_of_one_room(self, document):
        assert document.scene.rooms == [document.room]
        assert document.clock is document.scene.clock

    def test_every_room_shares_the_scenes_clock_by_identity(self, document):
        room, _ = room_with_chain(document, "meadow")
        assert room.clock is document.clock is document.scene.clock

    def test_switching_repoints_the_documents_vocabulary(self, document):
        room, _ = room_with_chain(document, "meadow")
        room.solver_config.iterations = 9

        document.set_active_room(room)
        assert document.body is room.body
        assert document.controllers is room.controllers
        assert document.solver_config.iterations == 9
        assert document.room is room

    def test_switching_to_a_foreign_room_is_refused(self, document):
        with pytest.raises(ValueError):
            document.set_active_room(Room(name="stray"))

    def test_a_switch_emits_the_full_suite(self, document):
        """File > Open's emissions plus the three open resets rather than
        restores: settings, terrain, and the incoming room's dirty flag."""
        room, _ = room_with_chain(document, "meadow")
        suite = [
            "body_replaced",
            "topology_changed",
            "selection_changed",
            "controllers_changed",
            "history_changed",
            "settings_changed",
            "terrain_changed",
            "dirty_changed",
        ]
        fired: list[str] = []
        for name in suite:
            getattr(document, name).connect(lambda *args, n=name: fired.append(n))
        document.set_active_room(room)
        assert fired == suite


class TestPerRoomState:
    def test_undo_is_per_room(self, document):
        first = document.room
        chunk = ids_of(document)[0]
        document.do(SetRadius(chunk, 0.7))
        room, _ = room_with_chain(document, "meadow")

        document.set_active_room(room)
        assert not document.can_undo, "another room's history is not ours"

        document.set_active_room(first)
        assert document.can_undo
        document.undo()
        assert document.body.chunk(chunk).radius == pytest.approx(0.3)

    def test_dirty_is_per_room(self, document):
        first = document.room
        document.do(SetRadius(ids_of(document)[0], 0.7))
        assert document.dirty

        room, _ = room_with_chain(document, "meadow")
        document.set_active_room(room)
        assert not document.dirty
        assert document.any_dirty, "the first room still has unsaved changes"
        document.set_active_room(first)
        assert document.dirty

    def test_selection_is_stashed_per_room_and_pruned(self, document):
        first = document.room
        kept, lost = ids_of(document)[0], ids_of(document)[1]
        document.set_selection((kept, lost))
        room, _ = room_with_chain(document, "meadow")

        document.set_active_room(room)
        assert document.selection == set()
        first.body.remove_chunk(lost)  # the room changed while unwatched

        document.set_active_room(first)
        assert document.selection == {kept}, "restored, minus what departed"

    def test_draw_tuning_survives_a_room_round_trip(self, document):
        first = document.room
        chunk = ids_of(document)[0]
        document.draw.set_stiffness(document.body, chunk, 0.05)
        room, _ = room_with_chain(document, "meadow")

        document.set_active_room(room)
        document.draw.set_stiffness(document.body, ids_of(document)[0], 0.9)
        document.set_active_room(first)

        assert document.draw.stiffness[document.body.slot(chunk)] == pytest.approx(
            0.05
        )

    def test_pause_captures_every_realized_rooms_pose(self, document):
        room, _ = room_with_chain(document, "meadow")
        drop = room.body.add_chunk("drop", (0.0, 5.0, 0.0), radius=0.3)
        document.set_running(True)
        for _ in range(20):
            document.step_once()
        document.set_running(False)  # captures the unwatched meadow's pose too

        fallen = room.body.chunk(drop).pos
        document.set_active_room(room)
        for _ in range(10):
            document.step_once()
        document.reset_pose()
        assert room.body.chunk(drop).pos == pytest.approx(fallen)


class TestTheSceneTicksWhole:
    def test_play_ticks_every_room(self, document):
        room, _ = room_with_chain(document, "meadow")
        drop = room.body.add_chunk("drop", (0.0, 5.0, 0.0), radius=0.3)
        # The document stays pointed at its first room; the meadow ticks anyway.
        for _ in range(20):
            document.step_once()
        assert room.body.chunk(drop).pos[1] < 5.0, "an unwatched room simulates"

    def test_a_crossing_arrives_through_the_documents_own_tick(self, document):
        """The package-5 story, driven entirely through the document.

        The grazer leaves while the document is pointed at its room and
        arrives in a meadow nobody is drawing — then the cut goes to the
        meadow and the roster shows the immigrant.
        """
        den = document.room
        document.new()
        ids = [
            den.body.add_chunk(f"grazer{i}", (float(i) * 0.6, 0.5, 0.0), radius=0.3)
            for i in range(3)
        ]
        for a, b in zip(ids, ids[1:]):
            den.body.add_link(a, b)
        den.body.set_pinned(ids[0], True)
        den.controllers.add(Grazer(ids[0], home=(0.0, 0.5, 0.0), exit="east"))
        den.doors.append(
            Door("east", (5.0, 0.5, 0.0), to_room="meadow", to_door="west")
        )
        meadow = document.add_room(Room(name="meadow"))
        meadow.doors.append(
            Door("west", (-5.0, 0.5, 0.0), to_room="den", to_door="east")
        )

        for _ in range(1200):
            document.step_once()
            if meadow.body.chunk_ids():
                break
        assert meadow.body.chunk_ids(), "the grazer never crossed"
        assert not den.body.chunk_ids()

        document.set_active_room(meadow)
        roster = document.creatures()
        assert len(roster) == 1 and len(roster[0].chunks) == 3


class TestManifest:
    """Package 5: the assembly survives a cold start.

    Room files persist everything substantive; the manifest records only
    which of them were one world, by which names. The acceptance is the
    round trip from nothing: a two-room scene with an open door pair,
    saved, reopened by a fresh document, still one world.
    """

    def _authored_scene(self, document):
        den = document.room
        document.new()
        document.rename_room(den, "den")
        ids = [
            den.body.add_chunk(f"grazer{i}", (float(i) * 0.6, 0.5, 0.0), radius=0.3)
            for i in range(3)
        ]
        for a, b in zip(ids, ids[1:]):
            den.body.add_link(a, b)
        den.body.set_pinned(ids[0], True)
        den.controllers.add(Grazer(ids[0], home=(0.0, 0.5, 0.0), exit="east"))
        den.doors.append(
            Door("east", (5.0, 0.5, 0.0), to_room="meadow", to_door="west")
        )
        meadow = document.add_room(Room(name="meadow"))
        meadow.doors.append(
            Door("west", (-5.0, 0.5, 0.0), to_room="den", to_door="east")
        )
        return den, meadow

    def test_a_scene_round_trips_whole_from_a_cold_start(self, document, tmp_path):
        den, meadow = self._authored_scene(document)
        manifest = tmp_path / "world.xml"
        document.save_scene_as(manifest)
        assert not document.any_dirty, "Save Scene leaves nothing owed"
        assert (tmp_path / "den.xml").is_file()
        assert (tmp_path / "meadow.xml").is_file()

        cold = Document()
        cold.open_scene(manifest)
        assert [room.name for room in cold.scene.rooms] == ["den", "meadow"]
        assert cold.room.name == "den"
        assert cold.room_path(cold.scene.rooms[1]) == tmp_path / "meadow.xml"

        den2 = cold.scene.rooms[0]
        assert den2.doors == [
            Door("east", (5.0, 0.5, 0.0), to_room="meadow", to_door="west")
        ]
        assert cold.door_status(den2, den2.doors[0]) == "", "the pair is open"
        assert len(cold.creatures()) == 1, "the grazer made the trip"

        for _ in range(1200):
            cold.step_once()
            if cold.scene.rooms[1].body.chunk_ids():
                break
        assert cold.scene.rooms[1].body.chunk_ids(), (
            "a reopened world still carries its traveller"
        )

    def test_a_missing_room_file_fails_naming_the_room(self, document, tmp_path):
        from mesh_xml.io import scene_xml

        scene_xml.save_scene(
            tmp_path / "world.xml", "world", [("meadow", tmp_path / "meadow.xml")]
        )
        with pytest.raises(ValueError, match="meadow"):
            document.open_scene(tmp_path / "world.xml")

    def test_a_failed_open_leaves_the_scene_standing(self, document, tmp_path):
        from mesh_xml.io import scene_xml

        before = list(document.scene.rooms)
        chunks = document.body.n_chunks
        scene_xml.save_scene(
            tmp_path / "world.xml", "world", [("gone", tmp_path / "gone.xml")]
        )
        with pytest.raises(ValueError):
            document.open_scene(tmp_path / "world.xml")
        assert document.scene.rooms == before
        assert document.body.n_chunks == chunks

    def test_the_manifests_names_win(self, document, tmp_path):
        path = tmp_path / "meadow.xml"
        document.save(path)

        from mesh_xml.io import scene_xml

        scene_xml.save_scene(tmp_path / "world.xml", "world", [("pasture", path)])
        document.open_scene(tmp_path / "world.xml")
        assert document.room.name == "pasture"
        assert document.room.body.name == "pasture"

    def test_a_rig_reader_refuses_a_manifest_by_tag(self, document, tmp_path):
        from mesh_xml.io import scene_xml, xml_v2

        manifest = tmp_path / "world.xml"
        scene_xml.save_scene(manifest, "world", [])
        with pytest.raises(xml_v2.SchemaError, match="manifest"):
            xml_v2.read(manifest)
        with pytest.raises(xml_v2.SchemaError, match="manifest"):
            document.open(manifest)

    def test_is_manifest_answers_only_that(self, document, tmp_path):
        from mesh_xml.io import scene_xml

        manifest = tmp_path / "world.xml"
        scene_xml.save_scene(manifest, "world", [])
        rig = tmp_path / "rig.xml"
        document.save(rig)

        assert scene_xml.is_manifest(manifest)
        assert not scene_xml.is_manifest(rig)
        assert not scene_xml.is_manifest(tmp_path / "absent.xml")

    def test_an_empty_manifest_is_refused(self, document, tmp_path):
        from mesh_xml.io import scene_xml

        manifest = tmp_path / "world.xml"
        scene_xml.save_scene(manifest, "world", [])
        with pytest.raises(ValueError, match="no rooms"):
            document.open_scene(manifest)

    def test_duplicate_manifest_names_are_refused_at_the_read(self, tmp_path):
        from mesh_xml.io import scene_xml

        manifest = tmp_path / "world.xml"
        scene_xml.save_scene(
            manifest, "world", [("den", "den.xml"), ("den", "den2.xml")]
        )
        with pytest.raises(scene_xml.SchemaError, match="twice"):
            scene_xml.read_scene(manifest)

    def test_save_scene_saves_only_what_is_owed(self, document, tmp_path):
        den, meadow = self._authored_scene(document)
        manifest = tmp_path / "world.xml"
        document.save_scene_as(manifest)

        stamp = (tmp_path / "meadow.xml").stat().st_mtime_ns
        document.set_active_room(den)
        document.do(SetRadius(ids_of(document)[0], 0.5))
        document.save_scene_as(manifest)
        assert (tmp_path / "meadow.xml").stat().st_mtime_ns == stamp, (
            "a clean room is listed, not rewritten"
        )


class TestAbstractionMeetsEditing:
    def test_an_abstracted_room_refuses_edits_and_sleeps_its_history(
        self, document
    ):
        chunk = ids_of(document)[0]
        document.do(SetRadius(chunk, 0.7))
        document.room.abstract()

        assert not document.can_undo, "a parked history sleeps"
        depth = len(document._undo)
        document.do(SetRadius(chunk, 0.9))
        assert len(document._undo) == depth, "edits are refused, not queued"

        document.room.realize()
        assert document.can_undo, "ids survive the round trip; history wakes"
        document.undo()
        assert document.body.chunk(chunk).radius == pytest.approx(0.3)

    def test_parking_an_inactive_room_leaves_the_active_view_alone(
        self, document
    ):
        room, _ = room_with_chain(document, "meadow")
        fired: list[str] = []
        document.body_replaced.connect(lambda: fired.append("body"))
        document.topology_changed.connect(lambda: fired.append("topology"))

        document.set_room_abstracted(room, True)
        assert room.abstracted
        assert fired == [], "another room's tier is not the active view's news"

    def test_realizing_recaptures_the_rest_pose(self, document):
        room, ids = room_with_chain(document, "meadow")
        document.set_room_abstracted(room, True)
        anchor = room.body.chunk_ids()[0]
        room.body.set_pos(anchor, (3.0, 0.5, 0.0))  # the brain walked

        document.set_room_abstracted(room, False)
        document.set_active_room(room)
        moved = {cid: room.body.chunk(cid).pos for cid in room.body.chunk_ids()}
        document.reset_pose()
        after = {cid: room.body.chunk(cid).pos for cid in room.body.chunk_ids()}
        assert after == moved, "the realized pose is the pose Reset returns to"


class TestRoomManagement:
    def test_remove_refuses_the_last_room(self, document):
        with pytest.raises(ValueError):
            document.remove_room(document.room)

    def test_removing_the_active_room_activates_a_neighbour(self, document):
        first = document.room
        room, _ = room_with_chain(document, "meadow")
        document.set_active_room(room)

        document.remove_room(room)
        assert document.room is first
        assert document.scene.rooms == [first]

    def test_removing_a_dirty_room_clears_its_debt(self, document):
        room, _ = room_with_chain(document, "meadow")
        document.set_active_room(room)
        document.do(SetRadius(ids_of(document)[0], 0.9))
        assert document.any_dirty

        document.remove_room(room)
        assert not document.any_dirty

    def test_rename_writes_room_and_body_as_one_value(self, document):
        document.rename_room(document.room, "den")
        assert document.room.name == "den"
        assert document.room.body.name == "den"
        assert document.dirty, "the body's name is what saves"

    def test_rename_refuses_a_taken_name(self, document):
        room, _ = room_with_chain(document, "meadow")
        document.rename_room(room, "document")
        assert room.name == "meadow"

    def test_rename_counts_the_doors_left_addressing_the_old_name(self, document):
        room, _ = room_with_chain(document, "meadow")
        document.room.doors.append(
            Door("east", (5.0, 0.5, 0.0), to_room="meadow", to_door="west")
        )
        messages: list[str] = []
        document.status.connect(messages.append)

        document.rename_room(room, "field")
        assert room.name == "field"
        assert any("1 door(s)" in message for message in messages)

    def test_unique_room_name_counts_up(self, document):
        assert document.unique_room_name() == "room"
        document.add_room(Room(name="room"))
        assert document.unique_room_name() == "room 2"

    def test_open_as_room_names_from_the_file_and_suffixes_collisions(
        self, document, tmp_path
    ):
        path = tmp_path / "meadow.xml"
        document.save(path)  # a default-named body falls back to the stem

        room = document.open_as_room(path)
        assert room.name == "meadow"
        assert document.room is room
        assert document.room_path(room) == path
        assert len(document.scene.rooms) == 2

        again = document.open_as_room(path)
        assert again.name == "meadow 2"
        assert again.body.name == "meadow 2", "room and body stay one value"

    def test_open_as_room_carries_the_doors(self, document, tmp_path):
        door = Door("east", (4.0, 0.5, 0.0), to_room="meadow", to_door="west")
        document.room.doors = [door]
        path = tmp_path / "den.xml"
        document.save(path)

        room = document.open_as_room(path)
        assert room.doors == [door]

    def test_save_room_reaches_an_inactive_room(self, document, tmp_path):
        room, _ = room_with_chain(document, "meadow")
        document.set_active_room(room)
        document.do(SetRadius(ids_of(document)[0], 0.9))
        document.set_active_room(document.scene.rooms[0])
        assert document.any_dirty and not document.dirty

        path = tmp_path / "meadow.xml"
        document.save_room(room, path)
        assert path.is_file()
        assert not document.room_dirty(room)
        assert not document.any_dirty

    def test_save_room_without_a_path_is_refused_by_name(self, document):
        room, _ = room_with_chain(document, "meadow")
        with pytest.raises(ValueError, match="meadow"):
            document.save_room(room)

    def test_open_adopts_the_files_name_for_the_room(self, document, tmp_path):
        path = tmp_path / "meadow.xml"
        document.save(path)

        document.open(path)
        assert document.room.name == "meadow"
        assert document.room.body.name == "meadow"
        assert not document.dirty, "a load is not an edit"

    def test_open_suffixes_against_sibling_rooms(self, document, tmp_path):
        path = tmp_path / "meadow.xml"
        document.save(path)
        document.open(path)  # the active room now answers to "meadow"

        spare = document.add_room(Room(name="spare"))
        document.set_active_room(spare)
        document.open(path)
        assert document.room.name == "meadow 2"


class TestDoors:
    """Door authoring through the document: file-grade, dirty-tracked, and
    surfaced as open or closed by the scene's own rules."""

    def test_add_door_defaults_aim_at_the_first_other_room(self, document):
        document.add_room(Room(name="meadow"))

        door = document.add_door(document.room)
        assert door.name == "door"
        assert door.to_room == "meadow"
        assert door.to_door == "door", "pairing convention: same name far side"
        assert door.at == (0.0, 0.0, 0.0)
        assert document.dirty, "a doorway is unsaved until saved"

        second = document.add_door(document.room)
        assert second.name == "door 2"

    def test_add_door_lands_on_the_selection(self, document):
        document.set_selection(ids_of(document)[:2])
        door = document.add_door(document.room)
        assert door.at == pytest.approx((0.5, 0.5, 0.0))

    def test_a_lone_rooms_door_aims_home_and_reads_closed(self, document):
        door = document.add_door(document.room)
        assert door.to_room == document.room.name
        assert "its own room" in document.door_status(document.room, door)

    def test_set_door_refuses_a_duplicate_name(self, document):
        document.add_door(document.room)
        second = document.add_door(document.room)

        document.set_door(document.room, 1, replace(second, name="door"))
        assert document.room.doors[1].name == "door 2", "the pair would collide"

    def test_remove_door_empties_the_list_and_dirties(self, document):
        document.add_door(document.room)
        document.remove_door(document.room, 0)
        assert document.room.doors == []
        assert document.dirty

    def test_door_status_names_each_closed_reason(self, document):
        den = document.room
        document.rename_room(den, "den")
        meadow = document.add_room(Room(name="meadow"))

        document.add_door(den)
        assert "no door" in document.door_status(den, den.doors[0])

        document.add_door(meadow)  # meadow's own "door" aims back at den
        assert document.door_status(den, den.doors[0]) == ""

        document.set_room_abstracted(meadow, True)
        assert "parked" in document.door_status(den, den.doors[0])
        document.set_room_abstracted(meadow, False)

        document.set_door(
            den, 0, replace(den.doors[0], to_room="attic")
        )
        assert "no room" in document.door_status(den, den.doors[0])

    def test_the_money_path_a_dock_authored_door_pair_carries_a_grazer(
        self, document
    ):
        """SCENE_BINDING.md's money test, short of the mouse.

        Every step is the editor's own API — rename, add door, re-aim,
        play — and the grazer walks out of a room nobody re-plumbed by
        hand and into the roster of the other.
        """
        den = document.room
        document.new()
        document.rename_room(den, "den")
        ids = [
            den.body.add_chunk(f"grazer{i}", (float(i) * 0.6, 0.5, 0.0), radius=0.3)
            for i in range(3)
        ]
        for a, b in zip(ids, ids[1:]):
            den.body.add_link(a, b)
        den.body.set_pinned(ids[0], True)
        den.controllers.add(Grazer(ids[0], home=(0.0, 0.5, 0.0), exit="east"))
        meadow = document.add_room(Room(name="meadow"))

        document.add_door(den)
        document.set_door(
            den,
            0,
            Door("east", (5.0, 0.5, 0.0), to_room="meadow", to_door="west"),
        )
        document.add_door(meadow)
        document.set_door(
            meadow,
            0,
            Door("west", (-5.0, 0.5, 0.0), to_room="den", to_door="east"),
        )
        assert document.door_status(den, den.doors[0]) == "", "the pair is open"

        for _ in range(1200):
            document.step_once()
            if meadow.body.chunk_ids():
                break
        assert meadow.body.chunk_ids(), "the grazer never crossed"

        document.set_active_room(meadow)
        roster = document.creatures()
        assert len(roster) == 1 and len(roster[0].chunks) == 3


class TestMoveCreature:
    """The transfer's editor face: one verb, two rooms, no undo stack.

    ``docs/MOVE_CREATURE.md``. The mechanism's own five proofs are in
    ``test_scene.py`` and this design adds no core motion; what is tested
    here is everything the editor wraps around it â€” the pre-flight that
    makes the wholesale promise keepable, the bookkeeping either side of
    the motion, and the cut that lands on the arrival.
    """

    def test_the_creature_arrives_whole_at_its_own_coordinates(self, document):
        source = document.room
        meadow, incumbents = room_with_chain(document, "meadow")
        ids = ids_of(document)
        source.controllers.add(Grazer(ids[0], home=(0.0, 0.5, 0.0)))
        was = [tuple(source.body.chunk(cid).pos) for cid in ids]
        document.set_selection([ids[2]])  # one chunk arms the whole creature

        document.move_creature(meadow)

        assert document.room is meadow, "the cut lands on the arrival"
        assert source.body.chunk_ids() == [], "the source kept nothing"
        assert source.controllers.controllers == []
        arrivals = sorted(document.selection)
        assert len(arrivals) == len(ids)
        assert sorted(meadow.body.chunk_ids()) == sorted(incumbents + arrivals)
        assert [tuple(meadow.body.chunk(cid).pos) for cid in arrivals] == was
        assert len(list(meadow.body.links())) == 2 + 4, "its links came too"
        assert len(meadow.controllers.controllers) == 1, "and its gait"

    def test_one_chunk_moves_its_component_and_the_neighbour_stays(
        self, document
    ):
        source = document.room
        meadow, _ = room_with_chain(document, "meadow")
        other = source.body.add_chunk("other", (9.0, 0.5, 0.0), radius=0.3)
        tail = source.body.add_chunk("other1", (9.6, 0.5, 0.0), radius=0.3)
        source.body.add_link(other, tail)
        document.set_selection([ids_of(document)[0]])

        document.move_creature(meadow)

        assert sorted(source.body.chunk_ids()) == sorted([other, tail])
        assert len(document.creatures()) == 2, "the meadow's own, plus the guest"

    def test_the_move_dirties_both_rooms(self, document):
        source = document.room
        meadow, _ = room_with_chain(document, "meadow")
        assert not document.any_dirty
        document.set_selection([ids_of(document)[0]])

        document.move_creature(meadow)

        assert document.room_dirty(source), "a room that lost a creature differs"
        assert document.room_dirty(meadow), "so does one that gained one"

    def test_a_successful_move_clears_both_histories(self, document):
        source = document.room
        meadow, incumbents = room_with_chain(document, "meadow")
        document.do(SetRadius(ids_of(document)[0], 0.7))
        document.set_active_room(meadow)
        document.do(SetRadius(incumbents[1], 0.7))
        document.undo()  # a redo entry, armed to squat on an arrival's id
        assert document.can_redo
        document.set_active_room(source)
        document.set_selection([ids_of(document)[0]])

        document.move_creature(meadow)

        assert not document.can_undo and not document.can_redo
        assert document._states[source].undo == []
        assert document._states[source].redo == []
        assert document._states[meadow].undo == []
        assert document._states[meadow].redo == []

    def test_the_arrivals_replace_the_targets_selection(self, document):
        source = document.room
        meadow, incumbents = room_with_chain(document, "meadow")
        document.set_active_room(meadow)
        document.set_selection([incumbents[0]])
        document.set_active_room(source)
        document.set_selection([ids_of(document)[0]])

        document.move_creature(meadow)

        assert document.selection
        assert incumbents[0] not in document.selection, "replacement, not fold"
        assert all(meadow.body.has_chunk(cid) for cid in document.selection)

    def test_the_target_rest_pose_gains_the_arrivals_and_keeps_its_own(
        self, document
    ):
        meadow, _ = room_with_chain(document, "meadow")
        before = dict(document._states[meadow].rest_pose)
        document.set_selection([ids_of(document)[0]])

        document.move_creature(meadow)

        pose = document._states[meadow].rest_pose
        arrivals = set(document.selection)
        assert set(pose) == set(before) | arrivals
        assert all(pose[cid] == before[cid] for cid in before), "incumbents stand"
        assert all(
            pose[cid] == meadow.body.chunk(cid).pos for cid in arrivals
        ), "folded in where they arrived"

    def test_the_source_rest_pose_forgets_the_departed(self, document):
        """The unkind proof: ids reuse, so a stale entry binds wrong.

        Move everything out, Add a chunk â€” which takes the departed
        creature's id back, max-plus-one over an empty room â€” and Reset. A
        surviving pose entry would pass ``has_chunk`` and teleport the
        newcomer to where the departed creature once stood.
        """
        source = document.room
        meadow, _ = room_with_chain(document, "meadow")
        gone = ids_of(document)[0]
        document.set_selection([gone])
        document.move_creature(meadow)

        document.set_active_room(source)
        newcomer = document.create_chunk()
        assert newcomer == gone, "the id came back around"
        where = tuple(source.body.chunk(newcomer).pos)

        document.reset_pose()

        assert tuple(source.body.chunk(newcomer).pos) == where

    def test_the_only_creature_leaves_a_legal_empty_room(self, document):
        source = document.room
        meadow, _ = room_with_chain(document, "meadow")
        document.set_selection([ids_of(document)[0]])

        document.move_creature(meadow)
        document.set_active_room(source)

        assert document.creatures() == []
        assert document.selection == set()
        document.step_once()  # an empty room is a legal room, and it ticks

    def test_a_surviving_exit_goal_is_said_out_loud(self, document):
        meadow, _ = room_with_chain(document, "meadow")
        ids = ids_of(document)
        document.room.controllers.add(
            Grazer(ids[0], home=(0.0, 0.5, 0.0), exit="east")
        )
        document.set_selection([ids[0]])
        said: list[str] = []
        document.status.connect(said.append)

        document.move_creature(meadow)

        assert "still wants out via 'east'" in said[-1]
        goal = meadow.controllers.controllers[-1]
        assert goal.exit == "east", "the editor spends no goal the brain earned"

    def test_a_foreign_room_is_a_programmer_error(self, document):
        document.set_selection([ids_of(document)[0]])
        with pytest.raises(ValueError, match="not in this document's scene"):
            document.move_creature(Room(name="elsewhere"))

    def test_an_empty_selection_is_refused_by_name(self, document):
        meadow, _ = room_with_chain(document, "meadow")
        said: list[str] = []
        document.status.connect(said.append)

        document.move_creature(meadow)

        assert "Select a chunk" in said[-1]
        assert document.room is not meadow


class TestMoveCreatureRefusals:
    """Every refusal: a status line, no motion, and both histories standing.

    The pre-flight runs before anything moves, so a refused move leaves a
    room pair bit-identical to the one the gesture found â€” and clean,
    because a dirty flag over an unmutated room is the mutated-but-clean
    lie in mirror image.
    """

    def _armed(self, document):
        """A selected creature, a target, both rooms clean, both armed."""
        source = document.room
        meadow, incumbents = room_with_chain(document, "meadow")
        document.do(SetRadius(ids_of(document)[0], 0.7))
        document.set_active_room(meadow)
        document.do(SetRadius(incumbents[1], 0.7))
        document.set_active_room(source)
        document._states[source].dirty = False
        document._states[meadow].dirty = False
        document.set_selection([ids_of(document)[0]])
        return source, meadow

    def _refused(self, document, source, meadow, before, said, phrase):
        assert phrase in said[-1], said[-1]
        assert document.room is source, "no cut"
        assert snapshot(source) == before[0]
        assert snapshot(meadow) == before[1]
        assert not document.room_dirty(source)
        assert not document.room_dirty(meadow)
        assert document._states[source].undo, "a refusal clears no history"
        assert document._states[meadow].undo

    def test_a_parked_source_disarms_with_every_other_edit(self, document):
        source, meadow = self._armed(document)
        document.set_room_abstracted(source, True)
        before = (snapshot(source), snapshot(meadow))
        said: list[str] = []
        document.status.connect(said.append)

        document.move_creature(meadow)

        self._refused(
            document, source, meadow, before, said, "Realize 'document'"
        )

    def test_a_parked_target_is_refused_in_the_doors_vocabulary(self, document):
        source, meadow = self._armed(document)
        document.set_room_abstracted(meadow, True)
        before = (snapshot(source), snapshot(meadow))
        said: list[str] = []
        document.status.connect(said.append)

        document.move_creature(meadow)

        self._refused(document, source, meadow, before, said, "is parked")

    def test_rooms_of_different_dimension_are_refused(self, document):
        source, _ = self._armed(document)
        flat = flat_room(document)
        before = (snapshot(source), snapshot(flat))
        said: list[str] = []
        document.status.connect(said.append)

        document.move_creature(flat)

        assert "dimension" in said[-1]
        assert document.room is source
        assert snapshot(source) == before[0]
        assert snapshot(flat) == before[1]

    def test_an_unregistered_controller_refuses_the_whole_gesture(self, document):
        """The test the transfer alone cannot pass.

        Its gate asks ``chunk_refs`` who is travelling, and an unreadable
        class answers nobody â€” so the flesh would go and the controller
        would stay. The editor asks the registry instead, before any motion.
        """
        source, meadow = self._armed(document)
        stray = _Unreadable(ids_of(document)[0])
        assert tempo.chunk_refs(stray) == frozenset(), "invisible to the gate"
        source.controllers.add(stray)
        before = (snapshot(source), snapshot(meadow))
        said: list[str] = []
        document.status.connect(said.append)

        document.move_creature(meadow)

        self._refused(document, source, meadow, before, said, "_Unreadable")
        assert stray in source.controllers.controllers

    def test_one_bad_creature_refuses_the_others_too(self, document):
        """A gesture that half-happens is the lie the pre-flight forbids."""
        source, meadow = self._armed(document)
        other = source.body.add_chunk("other", (9.0, 0.5, 0.0), radius=0.3)
        source.controllers.add(_Unreadable(other))
        document.set_selection([ids_of(document)[0], other])
        before = (snapshot(source), snapshot(meadow))
        said: list[str] = []
        document.status.connect(said.append)

        document.move_creature(meadow)

        assert "_Unreadable" in said[-1]
        assert snapshot(source) == before[0]
        assert snapshot(meadow) == before[1]
