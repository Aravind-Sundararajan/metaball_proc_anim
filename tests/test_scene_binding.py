"""Tests for the editor scene binding: package 6's identity rework.

The null result — a one-room scene is bit-for-bit the old editor — is the
whole of ``test_document.py``, ``test_panels.py`` and ``test_viewport.py``,
which run against the reworked document unchanged. What is tested here is
only what the plural adds: the cut, the per-room state it stashes and
restores, and the scene ticking whole through the document's own loop.
"""

from __future__ import annotations

import pytest

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
