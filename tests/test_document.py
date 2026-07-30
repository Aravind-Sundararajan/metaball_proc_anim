"""Tests for the editor document and its command stack.

The commands worth testing hardest are the structural ones. A command that
restores a chunk but forgets the links that died with it leaves a rig that looks
correct in the table and is wrong in the simulation, and nothing about that
failure is loud.
"""

import numpy as np
import pytest

from mesh_xml.core import Body, LinkType
from mesh_xml.editor.document import (
    AddChunk,
    AddLink,
    Document,
    Macro,
    MoveChunk,
    RemoveChunk,
    RemoveLink,
    Rename,
    SetPinned,
    SetRadius,
)


@pytest.fixture
def document() -> Document:
    doc = Document()
    body = Body()
    ids = [body.add_chunk(f"c{i}", [float(i), 0.0, 0.0], radius=0.3) for i in range(5)]
    for a, b in zip(ids, ids[1:]):
        body.add_link(a, b)
    doc._replace(body, None)
    return doc


def ids_of(document: Document) -> list[int]:
    return document.body.chunk_ids()


class TestValueCommands:
    def test_set_radius_round_trips(self, document):
        target = ids_of(document)[0]
        document.do(SetRadius(target, 0.9))
        assert document.body.chunk(target).radius == pytest.approx(0.9)

        document.undo()
        assert document.body.chunk(target).radius == pytest.approx(0.3)

        document.redo()
        assert document.body.chunk(target).radius == pytest.approx(0.9)

    def test_rename_round_trips(self, document):
        target = ids_of(document)[2]
        document.do(Rename(target, "spine"))
        assert document.body.name_of(target) == "spine"
        document.undo()
        assert document.body.name_of(target) == "c2"

    def test_move_round_trips(self, document):
        target = ids_of(document)[1]
        document.do(MoveChunk(target, (9.0, 9.0, 9.0)))
        assert document.body.chunk(target).pos == pytest.approx((9.0, 9.0, 9.0))
        document.undo()
        assert document.body.chunk(target).pos == pytest.approx((1.0, 0.0, 0.0))

    def test_pinning_restores_the_original_mass(self, document):
        target = ids_of(document)[0]
        document.body.set_pinned(target, False, mass=4.0)

        document.do(SetPinned(target, True))
        assert document.body.inv_mass[document.body.slot(target)] == 0.0

        document.undo()
        assert document.body.inv_mass[document.body.slot(target)] == pytest.approx(0.25)


class TestStructuralCommands:
    def test_removing_a_chunk_restores_its_links_on_undo(self, document):
        """A restored chunk with no connections is silently wrong, not obviously."""
        target = ids_of(document)[2]
        links_before = document.body.n_links
        neighbours_before = sorted(document.body.neighbors(target))

        document.do(RemoveChunk(target))
        assert document.body.n_links == links_before - 2

        document.undo()
        assert document.body.n_links == links_before
        assert sorted(document.body.neighbors(target)) == neighbours_before

    def test_removing_a_chunk_preserves_link_properties(self, document):
        a, b = ids_of(document)[0], ids_of(document)[1]
        document.body.set_link(a, b, rest=2.5, type=LinkType.PULL, elasticity=0.4)

        document.do(RemoveChunk(a))
        document.undo()

        link = document.body.link(a, b)
        assert link.rest == pytest.approx(2.5)
        assert link.type is LinkType.PULL
        assert link.elasticity == pytest.approx(0.4)

    def test_add_chunk_reuses_its_id_on_redo(self, document):
        """A redo that mints a fresh id orphans every link recorded against it."""
        command = AddChunk("new", (7.0, 0.0, 0.0), 0.2)
        document.do(command)
        first_id = command.chunk_id

        document.undo()
        document.redo()
        assert command.chunk_id == first_id
        assert document.body.has_chunk(first_id)

    def test_a_link_survives_undo_redo_of_its_chunk(self, document):
        tail = ids_of(document)[-1]
        add = AddChunk("new", (7.0, 0.0, 0.0), 0.2)
        document.do(add)
        document.do(AddLink(add.chunk_id, tail))

        document.undo()
        document.undo()
        document.redo()
        document.redo()

        assert document.body.has_link(add.chunk_id, tail)

    def test_remove_link_restores_properties(self, document):
        a, b = ids_of(document)[0], ids_of(document)[1]
        document.body.set_link(a, b, rest=3.0, type=LinkType.PUSH, elasticity=0.6)

        document.do(RemoveLink(a, b))
        assert not document.body.has_link(a, b)

        document.undo()
        link = document.body.link(a, b)
        assert link.rest == pytest.approx(3.0)
        assert link.type is LinkType.PUSH

    def test_duplicate_link_undo_does_not_remove_the_original(self, document):
        """add_link returns False for an existing edge; undo must respect that."""
        a, b = ids_of(document)[0], ids_of(document)[1]
        document.do(AddLink(a, b))
        document.undo()
        assert document.body.has_link(a, b)


class TestMacro:
    def test_macro_reverts_in_reverse_order(self, document):
        """Reverting forwards would re-link a chunk that has not been restored yet."""
        a, b = ids_of(document)[0], ids_of(document)[1]
        document.do(Macro([RemoveLink(a, b), RemoveChunk(a)], text="Dissolve"))
        assert not document.body.has_chunk(a)

        document.undo()

        assert document.body.has_chunk(a)
        assert document.body.has_link(a, b)

    def test_macro_is_one_undo_entry(self, document):
        target = ids_of(document)[0]
        document.do(Macro([SetRadius(target, 0.7), Rename(target, "head")], text="Retune"))

        document.undo()
        assert document.body.chunk(target).radius == pytest.approx(0.3)
        assert document.body.name_of(target) == "c0"
        assert not document.can_undo


class TestHistory:
    def test_a_new_command_clears_the_redo_stack(self, document):
        target = ids_of(document)[0]
        document.do(SetRadius(target, 0.9))
        document.undo()
        assert document.can_redo

        document.do(SetRadius(target, 0.5))
        assert not document.can_redo

    def test_undo_on_an_empty_stack_is_a_noop(self, document):
        document.undo()
        document.redo()
        assert document.body.n_chunks == 5

    def test_history_text_follows_the_stack(self, document):
        document.do(SetRadius(ids_of(document)[0], 0.9))
        assert document.undo_text == "Set Radius"
        document.undo()
        assert document.redo_text == "Set Radius"

    def test_opening_a_file_clears_history(self, document):
        document.do(SetRadius(ids_of(document)[0], 0.9))
        document.open("examples/chubby_worm.xml")
        assert not document.can_undo
        assert not document.can_redo


class TestSelection:
    def test_selection_is_pruned_when_a_chunk_is_deleted(self, document):
        target = ids_of(document)[2]
        document.set_selection([target])
        document.do(RemoveChunk(target))
        assert target not in document.selection

    def test_selection_rejects_unknown_ids(self, document):
        document.set_selection([999])
        assert document.selection == set()

    def test_toggle_adds_and_removes(self, document):
        target = ids_of(document)[0]
        document.toggle_selection(target)
        assert target in document.selection
        document.toggle_selection(target)
        assert target not in document.selection


class TestSignals:
    def test_structural_edits_emit_topology_changed(self, document):
        seen = []
        document.topology_changed.connect(lambda: seen.append(True))
        document.do(RemoveChunk(ids_of(document)[0]))
        assert seen

    def test_value_edits_emit_chunk_changed_with_the_id(self, document):
        seen = []
        target = ids_of(document)[1]
        document.chunk_changed.connect(seen.append)
        document.do(SetRadius(target, 0.4))
        assert seen == [target]

    def test_editing_marks_the_document_dirty(self, document):
        assert not document.dirty
        document.do(SetRadius(ids_of(document)[0], 0.4))
        assert document.dirty

    def test_draw_points_resync_after_a_structural_edit(self, document):
        document.do(RemoveChunk(ids_of(document)[0]))
        assert len(document.draw.pos) == document.body.n_chunks


class TestSimulation:
    def test_paused_advance_does_not_tick_the_solver(self, document):
        document.set_running(False)
        document.advance(1.0)
        assert document.clock.ticks == 0

    def test_running_advance_ticks(self, document):
        document.set_running(True)
        document.advance(0.5)
        assert document.clock.ticks > 0

    def test_starting_playback_drops_banked_time(self, document):
        """Otherwise the first frame spends the pause as a burst of catch-up."""
        document.set_running(True)
        document.advance(0.02)
        document.set_running(False)
        document.set_running(True)
        assert document.clock.alpha == pytest.approx(0.0)

    def test_reset_pose_restores_captured_positions(self, document):
        target = ids_of(document)[3]
        original = document.body.chunk(target).pos
        document.set_running(True)
        document.advance(1.0)
        assert document.body.chunk(target).pos != pytest.approx(original)

        document.reset_pose()
        assert document.body.chunk(target).pos == pytest.approx(original)

    def test_primitives_track_the_draw_positions(self, document):
        document.set_running(True)
        document.advance(0.5)
        primitives = document.primitives()
        assert len(primitives) == document.body.n_links
        assert np.isfinite(primitives).all()


class TestFiles:
    def test_v2_round_trip(self, document, tmp_path):
        path = tmp_path / "rig.xml"
        document.do(Rename(ids_of(document)[0], "head"))
        document.save(path)
        assert not document.dirty

        reopened = Document()
        reopened.open(path)
        assert reopened.body.n_chunks == 5
        assert reopened.body.name_of(reopened.body.chunk_ids()[0]) == "head"

    def test_opening_a_legacy_file_falls_back(self, document):
        document.open("examples/chubby_worm.xml")
        assert document.body.n_chunks == 10
        assert document.body.n_links == 9

    def test_opening_the_absolute_era_fixture_uses_the_legacy_reader(self, document):
        document.open("tests/data/legacy_v1_absolute_spline.xml")
        xs = sorted(chunk.pos[0] for chunk in document.body.chunks())
        assert xs[-1] == pytest.approx(4.444, abs=1e-2)

    def test_new_clears_everything(self, document):
        document.new()
        assert document.body.n_chunks == 0
        assert not document.can_undo
        assert not document.dirty

    def test_title_marks_unsaved_changes(self, document):
        document.do(SetRadius(ids_of(document)[0], 0.4))
        assert document.title.startswith("*")
