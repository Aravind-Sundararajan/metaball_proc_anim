"""Tests for the editor document and its command stack.

The commands worth testing hardest are the structural ones. A command that
restores a chunk but forgets the links that died with it leaves a rig that looks
correct in the table and is wrong in the simulation, and nothing about that
failure is loud.
"""

import numpy as np
import pytest

from mesh_xml.core import Body, LinkType
from mesh_xml.scene.door import Door
from mesh_xml.editor.document import (
    AddChunk,
    AddLink,
    Document,
    Macro,
    MoveChunk,
    RemoveChunk,
    RemoveController,
    RemoveLink,
    Rename,
    SetControllerField,
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


class TestCreateChunk:
    """Add grows from the selection — the plan's replacement for create-at-origin.

    The legacy button dropped every new sphere at the origin and left linking
    as a second gesture; nothing about it composed. Here one call places, links,
    and moves the selection to the new chunk, so repeated calls extrude a limb.
    """

    def test_an_empty_document_starts_at_the_origin(self):
        document = Document()
        new = document.create_chunk()

        assert document.body.n_chunks == 1
        assert document.body.chunk(new).pos == pytest.approx((0.0, 0.0, 0.0))
        assert document.body.chunk(new).radius == pytest.approx(0.5)
        assert document.selection == {new}

    def test_with_nothing_selected_the_chunk_is_free_and_clear_of_the_body(
        self, document
    ):
        links = document.body.n_links
        document.set_selection(())
        new = document.create_chunk()

        assert document.body.n_links == links
        assert document.body.chunk(new).pos[0] > 4.0
        assert document.selection == {new}

    def test_repeated_free_adds_do_not_stack(self, document):
        """Stacking duplicates at one coordinate was a legacy defect (§2.5)."""
        document.set_selection(())
        first = document.create_chunk()
        document.set_selection(())
        second = document.create_chunk()

        apart = np.linalg.norm(
            np.subtract(
                document.body.chunk(first).pos, document.body.chunk(second).pos
            )
        )
        assert float(apart) > 0.5

    def test_growing_a_chain_end_continues_its_rhythm(self, document):
        end = ids_of(document)[-1]
        document.set_selection((end,))
        new = document.create_chunk()

        assert document.body.has_link(new, end)
        assert document.body.chunk(new).pos == pytest.approx((5.0, 0.0, 0.0))
        assert document.body.chunk(new).radius == pytest.approx(0.3)
        assert document.body.link(new, end).rest == pytest.approx(1.0)
        assert document.selection == {new}

    def test_repeated_add_extrudes_a_limb(self, document):
        end = ids_of(document)[-1]
        document.set_selection((end,))
        first = document.create_chunk()
        second = document.create_chunk()

        assert document.body.has_link(first, end)
        assert document.body.has_link(second, first)
        assert not document.body.has_link(second, end)
        assert document.body.chunk(second).pos == pytest.approx((6.0, 0.0, 0.0))

    def test_a_lone_chunk_grows_along_x(self):
        document = Document()
        first = document.create_chunk()
        second = document.create_chunk()

        assert document.body.has_link(first, second)
        assert document.body.chunk(second).pos == pytest.approx((0.8, 0.0, 0.0))

    def test_a_corner_grows_away_from_its_neighbours(self):
        document = Document()
        body = Body()
        hub = body.add_chunk("hub", (0.0, 0.0, 0.0), radius=0.3)
        body.add_link(hub, body.add_chunk("nx", (1.0, 0.0, 0.0), radius=0.3))
        body.add_link(hub, body.add_chunk("ny", (0.0, 1.0, 0.0), radius=0.3))
        document._replace(body, None)

        document.set_selection((hub,))
        new = document.create_chunk()

        pos = document.body.chunk(new).pos
        assert document.body.has_link(new, hub)
        assert pos[0] < 0.0 and pos[1] < 0.0

    def test_multi_selection_links_to_every_member(self, document):
        a, b = ids_of(document)[1], ids_of(document)[3]
        document.set_selection((a, b))
        new = document.create_chunk()

        assert document.body.has_link(new, a)
        assert document.body.has_link(new, b)
        assert document.body.chunk(new).radius == pytest.approx(0.3)
        assert document.selection == {new}

    def test_one_add_is_one_undo_step(self, document):
        chunks, links = document.body.n_chunks, document.body.n_links
        end = ids_of(document)[-1]
        document.set_selection((end,))
        new = document.create_chunk()
        assert (document.body.n_chunks, document.body.n_links) == (
            chunks + 1,
            links + 1,
        )

        document.undo()
        assert (document.body.n_chunks, document.body.n_links) == (chunks, links)
        assert not document.body.has_chunk(new)

        document.redo()
        assert document.body.has_chunk(new)
        assert document.body.has_link(new, end)


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
        document.open("tests/data/legacy_v1_chubby_worm.xml")
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

    def test_doors_survive_a_save(self, document, tmp_path):
        """The room's exits ride the file; an editor session must not strip them."""
        door = Door("east", (4.0, 0.5, 0.0), to_room="meadow", to_door="west")
        document.room.doors = [door]
        path = tmp_path / "den.xml"
        document.save(path)

        reopened = Document()
        reopened.open(path)
        assert reopened.room.doors == [door]

    def test_a_doorless_file_clears_previous_doors(self, document, tmp_path):
        """An exit into the old file's world is a promise the new one never made."""
        path = tmp_path / "plain.xml"
        document.save(path)
        document.room.doors = [
            Door("east", (0.0, 0.0, 0.0), to_room="meadow", to_door="west")
        ]
        document.open(path)
        assert document.room.doors == []

    def test_new_clears_doors(self, document):
        document.room.doors = [
            Door("east", (0.0, 0.0, 0.0), to_room="meadow", to_door="west")
        ]
        document.new()
        assert document.room.doors == []


class TestControllerCommands:
    """The gait is document state: edits arrive as commands and undo whole."""

    @staticmethod
    def _with_wave(document: Document) -> tuple[int, object]:
        from mesh_xml.anim.controllers.drive import WaveAnchor

        anchor = ids_of(document)[0]
        document.body.set_pinned(anchor, True)
        origin = document.body.chunk(anchor).pos
        wave = document.controllers.add(WaveAnchor(anchor, origin))
        return anchor, wave

    def test_set_controller_field_round_trips(self, document):
        _anchor, wave = self._with_wave(document)
        heard: list[int] = []
        document.controllers_changed.connect(lambda: heard.append(1))

        document.do(SetControllerField(0, "amplitude", 2.5))
        assert wave.amplitude == pytest.approx(2.5)
        assert heard

        document.undo()
        assert wave.amplitude == pytest.approx(0.8)
        document.redo()
        assert wave.amplitude == pytest.approx(2.5)

    def test_remove_controller_restores_the_same_object(self, document):
        _anchor, wave = self._with_wave(document)

        document.do(RemoveController(0))
        assert document.controllers.controllers == []

        document.undo()
        assert document.controllers.controllers[0] is wave

    def test_commit_drag_on_a_plain_chunk_is_a_move(self, document):
        target = ids_of(document)[1]

        document.commit_drag(target, (9.0, 8.0, 7.0))
        assert document.body.chunk(target).pos == pytest.approx((9.0, 8.0, 7.0))

        document.undo()
        assert document.body.chunk(target).pos == pytest.approx((1.0, 0.0, 0.0))

    def test_commit_drag_on_a_driven_anchor_reauthors_the_origin(self, document):
        """The gesture means "the motion's home is now here" — and the next
        tick must keep the chunk exactly under where the hand let go."""
        anchor, wave = self._with_wave(document)
        document.controllers.time = 3.7
        old_origin = tuple(wave.origin)

        final = (4.0, 1.0, -2.0)
        document.commit_drag(anchor, final)

        assert document.body.chunk(anchor).pos == pytest.approx(final)
        expected = np.asarray(final, dtype=np.float64) - wave.swing(3.7)
        assert wave.origin == pytest.approx(tuple(expected))

        document.step_once()
        assert document.body.chunk(anchor).pos == pytest.approx(final, abs=1e-5)

        document.undo()
        assert tuple(wave.origin) == pytest.approx(old_origin)

    def test_fit_autorig_is_one_undo_step(self, document):
        from mesh_xml.anim.controllers.drive import WaveAnchor

        anchor_mass = 4.0
        first = ids_of(document)[0]
        document.body.set_pinned(first, False, mass=anchor_mass)

        document.fit_autorig()
        names = [type(c).__name__ for c in document.controllers.controllers]
        assert names[0] == "WaveAnchor"
        assert "SpineFollow" in names
        wave = document.controllers.controllers[0]
        assert isinstance(wave, WaveAnchor)
        assert document.body.inv_mass[document.body.slot(wave.anchor)] == 0.0

        document.undo()
        assert document.controllers.controllers == []
        for chunk in document.body.chunks():
            assert not chunk.pinned
        if wave.anchor == first:
            assert document.body.chunk(first).mass == pytest.approx(anchor_mass)
        assert not document.can_undo


class TestRigLegs:
    """rig_legs turns leg-shaped appendages into the walking stack, undoably."""

    @staticmethod
    def _legged(document: Document) -> None:
        body = Body(name="legged")
        spine = [
            body.add_chunk(f"s{i}", (i * 0.42, 1.05, 0.0), radius=0.2, mass=1.5)
            for i in range(6)
        ]
        for a, b in zip(spine, spine[1:]):
            body.add_link(a, b)
        for station in (spine[1], spine[4]):
            x = body.chunk(station).pos[0]
            for side in (0.3, -0.3):
                hip = body.add_chunk(f"h{station}{side:+.1f}", (x, 0.9, side), radius=0.12)
                knee = body.add_chunk(f"k{station}{side:+.1f}", (x, 0.49, side), radius=0.1)
                foot = body.add_chunk(f"f{station}{side:+.1f}", (x, 0.08, side), radius=0.08)
                body.add_link(station, hip)
                body.add_link(hip, knee)
                body.add_link(knee, foot)
        body.set_pinned(spine[0], True)
        document._replace(body, None)

    def test_one_click_one_undo(self, document):
        self._legged(document)
        links_before = document.body.n_links
        pinned = document.body.chunk_ids()[0]

        document.rig_legs()

        names = sorted(type(c).__name__ for c in document.controllers.controllers)
        assert names == ["Level", "LimbIK", "LimbIK", "LimbIK", "LimbIK", "Posture", "Posture"]
        assert document.body.n_links > links_before, "the hip truss should land"
        assert document.body.inv_mass[document.body.slot(pinned)] > 0.0
        for link in document.body.links():
            if not link.render:
                break
        else:
            pytest.fail("truss links should be mechanism, not flesh")

        document.undo()
        assert document.controllers.controllers == []
        assert document.body.n_links == links_before
        assert document.body.inv_mass[document.body.slot(pinned)] == 0.0
        assert not document.can_undo

    def test_a_legless_body_is_a_polite_no_op(self, document):
        document.rig_legs()
        assert document.controllers.controllers == []
        assert not document.can_undo

    def test_the_rigged_walker_round_trips(self, document, tmp_path):
        """Posture and Level are registered now, so a walker saves whole."""
        self._legged(document)
        document.rig_legs()
        path = tmp_path / "walker.xml"
        document.save(path)

        reopened = Document()
        reopened.open(path)
        names = sorted(type(c).__name__ for c in reopened.controllers.controllers)
        assert names == ["Level", "LimbIK", "LimbIK", "LimbIK", "LimbIK", "Posture", "Posture"]
        posture = next(
            c for c in reopened.controllers.controllers
            if type(c).__name__ == "Posture"
        )
        assert posture.support, "the feet must survive the trip"


class TestCreatureColour:
    def test_paints_the_whole_component_and_undoes_whole(self, document):
        target = ids_of(document)[2]
        document.set_selection([target])

        document.set_creature_colour((0.9, 0.1, 0.2))
        for cid in ids_of(document):
            assert document.body.meta_of(cid)["colour"] == "0.9000 0.1000 0.2000"

        document.undo()
        for cid in ids_of(document):
            assert "colour" not in document.body.meta_of(cid)
        assert not document.can_undo

    def test_only_the_selected_creature_is_painted(self, document):
        lone = document.body.add_chunk("lone", (9.0, 0.0, 0.0))
        document.set_selection([ids_of(document)[0]])

        document.set_creature_colour((0.0, 1.0, 0.0))
        assert "colour" not in document.body.meta_of(lone)

    def test_no_selection_is_a_polite_no_op(self, document):
        document.set_selection(())
        document.set_creature_colour((1.0, 0.0, 0.0))
        assert not document.can_undo

    def test_the_colour_survives_a_save(self, document, tmp_path):
        document.set_selection([ids_of(document)[0]])
        document.set_creature_colour((0.3, 0.6, 0.9))
        path = tmp_path / "painted.xml"
        document.save(path)

        reopened = Document()
        reopened.open(path)
        first = reopened.body.chunk_ids()[0]
        assert reopened.body.meta_of(first)["colour"] == "0.3000 0.6000 0.9000"

    def test_paint_selection_paints_exactly_the_selection(self, document):
        """The per-part gesture: a statement about the chunks in hand,
        not their component — unlike the creature-wide button above."""
        ids = ids_of(document)
        document.set_selection(ids[:2])
        document.set_chunk_colour((1.0, 0.0, 0.0))

        assert document.body.meta_of(ids[0])["colour"] == "1.0000 0.0000 0.0000"
        assert document.body.meta_of(ids[1])["colour"] == "1.0000 0.0000 0.0000"
        assert "colour" not in document.body.meta_of(ids[2])

        document.undo()
        assert "colour" not in document.body.meta_of(ids[0]), "one undo step"

    def test_paint_selection_without_a_selection_is_a_polite_no_op(self, document):
        document.set_selection(())
        document.set_chunk_colour((1.0, 0.0, 0.0))
        assert not document.can_undo

    def test_pattern_paints_the_component_and_undoes_whole(self, document):
        ids = ids_of(document)
        document.set_selection(ids[:1])
        document.set_creature_pattern("stripes", 3.0)

        for cid in ids:
            assert document.body.meta_of(cid)["pattern"] == "stripes"
            assert document.body.meta_of(cid)["pattern_param"] == "3.0"
        assert document.creature_pattern() == ("stripes", 3.0)
        assert len(document.attribute_rows()) > 0, "a pattern arms the skin"

        document.undo()
        assert "pattern" not in document.body.meta_of(ids[0]), "one undo step"

    def test_none_clears_the_pattern(self, document):
        ids = ids_of(document)
        document.set_selection(ids[:1])
        document.set_creature_pattern("stripes")
        assert document.creature_pattern()[0] == "stripes"

        document.set_creature_pattern("none")
        assert document.creature_pattern() == ("none", 0.0)
        assert "pattern" not in document.body.meta_of(ids[0])

    def test_pattern_default_param_when_unspecified(self, document):
        document.set_selection(ids_of(document)[:1])
        document.set_creature_pattern("countershade")
        assert document.creature_pattern() == ("countershade", 0.7)

    def test_a_painted_head_arms_the_skin_and_survives_a_save(
        self, document, tmp_path
    ):
        """The package-2 money path: paint the head, save, reopen — still
        red-headed, and the weighted skin stays armed on the far side."""
        ids = ids_of(document)
        document.set_selection(ids[:1])
        document.set_chunk_colour((1.0, 0.0, 0.0))
        assert len(document.attribute_rows()) > 0, "a painted head arms the table"

        path = tmp_path / "painted.xml"
        document.save(path)
        reopened = Document()
        reopened.open(path)

        painted = [
            cid
            for cid in reopened.body.chunk_ids()
            if "colour" in reopened.body.meta_of(cid)
        ]
        assert len(painted) == 1
        assert len(reopened.attribute_rows()) > 0


class TestExportMesh:
    def test_exports_a_closed_obj(self, document, tmp_path):
        out = document.export_mesh(tmp_path / "rig.obj", resolution=48)
        text = out.read_text(encoding="utf-8")
        assert text.count("\nf ") > 100
        assert text.startswith("#") or "v " in text

    def test_a_herd_exports_unwelded(self, document, tmp_path):
        """Two creatures inside blend range must export as two skins: the
        grouped field is a hard union, so the meshes near the gap differ
        from the welded single-group surface."""
        body = Body()
        for creature, y in enumerate((0.0, 1.05)):
            ids = [
                body.add_chunk(f"w{creature}{i}", (i * 0.8, y, 0.0), radius=0.45)
                for i in range(3)
            ]
            for a, b in zip(ids, ids[1:]):
                body.add_link(a, b)
        document._replace(body, None)
        document.metaball.blend = 0.4

        out = document.export_mesh(tmp_path / "herd.obj", resolution=48)
        vertices = [
            line for line in out.read_text(encoding="utf-8").splitlines()
            if line.startswith("v ")
        ]
        assert len(vertices) > 200

        from mesh_xml.render.meshing import polygonise

        welded = polygonise(document.primitives(), blend=0.4, resolution=48)
        assert len(vertices) != len(welded.vertices)
