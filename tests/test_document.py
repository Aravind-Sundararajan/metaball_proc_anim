"""Tests for the editor document and its command stack.

The commands worth testing hardest are the structural ones. A command that
restores a chunk but forgets the links that died with it leaves a rig that looks
correct in the table and is wrong in the simulation, and nothing about that
failure is loud.
"""

import numpy as np
import pytest

from mesh_xml.core import Body, LinkType
from mesh_xml.core.solver import SolverConfig
from mesh_xml.core.terrain import Terrain
from mesh_xml.io import xml_v2
from mesh_xml.render.gl.metaball import MetaballSettings
from mesh_xml.render.look import Look
from mesh_xml.scene.door import Door
from mesh_xml.scene.room import Room
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
    SetTerrain,
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
        """Every field, ``render`` included.

        It is the one that decides membership in the surface, and
        ``add_link`` defaults it True — so a mechanism link restored without
        it comes back welded into flesh it was never part of, which reads as
        the skin growing a limb out of an undo.
        """
        a, b = ids_of(document)[0], ids_of(document)[1]
        document.body.set_link(
            a, b, rest=2.5, type=LinkType.PULL, elasticity=0.4, render=False
        )

        document.do(RemoveChunk(a))
        document.undo()

        link = document.body.link(a, b)
        assert link.rest == pytest.approx(2.5)
        assert link.type is LinkType.PULL
        assert link.elasticity == pytest.approx(0.4)
        assert link.render is False

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
        """``render`` included, for the reason its sibling above gives."""
        a, b = ids_of(document)[0], ids_of(document)[1]
        document.body.set_link(
            a, b, rest=3.0, type=LinkType.PUSH, elasticity=0.6, render=False
        )

        document.do(RemoveLink(a, b))
        assert not document.body.has_link(a, b)

        document.undo()
        link = document.body.link(a, b)
        assert link.rest == pytest.approx(3.0)
        assert link.type is LinkType.PUSH
        assert link.render is False

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


class TestCreateChunkAt:
    """Add with a stated position — the surface pick's half of the rule.

    The click states where: the position lands verbatim,
    the selection stands aside, and the hit chunk — when there is one —
    earns the link and lends its radius. Same command shapes, same undo
    grain, and the placed chunk still takes the selection, so a click then
    repeated Adds extrudes from where you pointed.
    """

    def test_a_flesh_hit_links_to_it_and_inherits_its_radius(self, document):
        links = document.body.n_links
        anchor = ids_of(document)[2]
        new = document.create_chunk(at=(2.0, 1.5, 0.5), attach_to=anchor)

        assert document.body.chunk(new).pos == pytest.approx((2.0, 1.5, 0.5))
        assert document.body.chunk(new).radius == pytest.approx(0.3)
        assert document.body.has_link(new, anchor)
        assert document.body.n_links == links + 1

    def test_a_world_hit_is_a_free_default_chunk_exactly_there(self, document):
        """The selection does not participate — the click is the statement."""
        links = document.body.n_links
        document.set_selection((ids_of(document)[0],))
        new = document.create_chunk(at=(-1.0, 0.25, 2.0))

        assert document.body.chunk(new).pos == pytest.approx((-1.0, 0.25, 2.0))
        assert document.body.chunk(new).radius == pytest.approx(0.5)
        assert document.body.n_links == links

    def test_an_explicit_radius_beats_the_inherited_one(self, document):
        anchor = ids_of(document)[0]
        new = document.create_chunk(
            at=(0.0, 1.0, 0.0), attach_to=anchor, radius=0.8
        )
        assert document.body.chunk(new).radius == pytest.approx(0.8)

    def test_a_placed_attachment_is_one_undo_step(self, document):
        chunks, links = document.body.n_chunks, document.body.n_links
        anchor = ids_of(document)[4]
        new = document.create_chunk(at=(4.0, 1.0, 0.0), attach_to=anchor)

        document.undo()
        assert (document.body.n_chunks, document.body.n_links) == (chunks, links)
        assert not document.body.has_chunk(new)

        document.redo()
        assert document.body.has_chunk(new)
        assert document.body.has_link(new, anchor)

    def test_the_placed_chunk_takes_the_selection(self, document):
        document.set_selection(ids_of(document)[:2])
        new = document.create_chunk(at=(0.0, 2.0, 0.0))
        assert document.selection == {new}

    def test_overrides_without_a_position_are_refused(self, document):
        with pytest.raises(ValueError):
            document.create_chunk(attach_to=ids_of(document)[0])
        with pytest.raises(ValueError):
            document.create_chunk(radius=0.8)
        assert not document.can_undo

    def test_a_stale_hit_is_refused_before_anything_applies(self, document):
        chunks, links = document.body.n_chunks, document.body.n_links
        with pytest.raises(ValueError):
            document.create_chunk(at=(0.0, 0.0, 0.0), attach_to=99)
        assert (document.body.n_chunks, document.body.n_links) == (chunks, links)
        assert not document.can_undo

    def test_no_overrides_still_grow_from_the_selection(self, document):
        end = ids_of(document)[-1]
        document.set_selection((end,))
        new = document.create_chunk()

        assert document.body.has_link(new, end)
        assert document.body.chunk(new).pos == pytest.approx((5.0, 0.0, 0.0))
        assert document.body.chunk(new).radius == pytest.approx(0.3)


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
        # On the chunk the autorig actually pinned, not on the one the test
        # happened to tune: the anchor is chosen by shape, and guarding this
        # on `wave.anchor == first` made it dead code — measured, the guard
        # was False every run, so undo's mass restoration had no coverage.
        assert document.body.chunk(wave.anchor).mass == pytest.approx(
            anchor_mass if wave.anchor == first else 1.0
        )
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

    def test_an_unpainted_export_keeps_the_classic_format(self, document, tmp_path):
        out = document.export_mesh(tmp_path / "plain.obj", resolution=32)
        v_lines = [
            line
            for line in out.read_text(encoding="utf-8").splitlines()
            if line.startswith("v ")
        ]
        assert v_lines and all(len(line.split()) == 4 for line in v_lines)

    def test_a_painted_creature_exports_its_skin(self, document, tmp_path):
        """The arc's closing acceptance: a two-tone creature arrives in
        Blender two-toned — vertex colours from the same albedo fold the
        viewport shades by."""
        ids = ids_of(document)
        document.set_selection(ids[:1])
        document.set_chunk_colour((1.0, 0.0, 0.0))

        out = document.export_mesh(tmp_path / "painted.obj", resolution=32)
        v_lines = [
            line
            for line in out.read_text(encoding="utf-8").splitlines()
            if line.startswith("v ")
        ]
        assert v_lines and all(len(line.split()) == 7 for line in v_lines)
        red = np.array([[float(x) for x in line.split()[4:]] for line in v_lines])
        assert red[:, 0].max() > 0.9, "the painted head reaches the mesh"
        assert red[:, 0].min() < 0.9, "the unpainted tail keeps its own colour"

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


class TestSettingsPersistence:
    """Solver tuning and blend ride the room's file; the file is the truth."""

    def test_a_tuned_room_reopens_as_tuned(self, document, tmp_path):
        document.solver_config.iterations = 8
        document.solver_config.gravity = (0.0, -3.0, 0.0)
        document.solver_config.self_contact = True
        document.metaball.blend = 0.5
        document.set_terrain(Terrain.flat(0.25))
        path = tmp_path / "tuned.xml"
        document.save(path)

        reopened = Document()
        reopened.open(path)
        config = reopened.solver_config
        assert config.iterations == 8
        assert tuple(config.gravity) == pytest.approx((0.0, -3.0, 0.0))
        assert config.self_contact is True
        assert reopened.metaball.blend == pytest.approx(0.5)
        assert reopened.terrain is not None, "the fold lands in the loaded config"
        assert reopened.terrain.planes == pytest.approx(document.terrain.planes)

    def test_a_sectionless_file_resets_the_sessions_tuning(
        self, document, tmp_path
    ):
        """Absent sections mean what every pre-arc document meant: defaults."""
        path = tmp_path / "plain.xml"
        xml_v2.save(document.body, path)

        document.solver_config.iterations = 16
        document.metaball.blend = 0.7
        document.settings_edited()
        assert document.dirty

        document.open(path)
        founding = SolverConfig(ground_y=0.0)
        config = document.solver_config
        assert config.iterations == founding.iterations
        assert tuple(config.gravity) == pytest.approx(founding.gravity)
        assert config.ground_y == pytest.approx(0.0)
        assert document.metaball.blend == pytest.approx(MetaballSettings().blend)
        assert not document.dirty

    def test_settings_edited_dirties_and_a_raw_emit_does_not(self, document):
        heard: list[bool] = []
        document.dirty_changed.connect(heard.append)

        document.settings_changed.emit()
        assert not document.dirty, "a view-only refresh is not an edit"

        document.settings_edited()
        assert document.dirty
        assert heard == [True]

    def test_a_two_room_scene_reopens_each_rooms_own_tuning(
        self, document, tmp_path
    ):
        meadow = document.add_room(Room(name="meadow"))
        meadow.body.add_chunk("m0", (0.0, 0.5, 0.0), radius=0.3)

        document.solver_config.iterations = 8
        document.metaball.blend = 0.5
        document.set_active_room(meadow)
        document.solver_config.iterations = 12
        document.metaball.blend = 0.1

        manifest = tmp_path / "scene.xml"
        document.save_scene_as(manifest)

        reopened = Document()
        reopened.open_scene(manifest)
        rooms = {room.name: room for room in reopened.scene.rooms}
        assert rooms["document"].solver_config.iterations == 8
        assert rooms["meadow"].solver_config.iterations == 12

        assert reopened.room is rooms["document"]
        assert reopened.metaball.blend == pytest.approx(0.5)
        reopened.set_active_room(rooms["meadow"])
        assert reopened.metaball.blend == pytest.approx(0.1)

    def test_save_room_carries_an_inactive_rooms_tuning(self, document, tmp_path):
        meadow = document.add_room(Room(name="meadow"))
        meadow.body.add_chunk("m0", (0.0, 0.5, 0.0), radius=0.3)
        document.set_active_room(meadow)
        document.solver_config.damping = 0.5
        document.metaball.blend = 0.4
        document.metaball.look.ao_strength = 0.85
        document.settings_edited()
        document.set_active_room(document.scene.rooms[0])
        assert document.room_dirty(meadow), "the tuning is owed to disk"

        target = document.save_room(meadow, tmp_path / "meadow.xml")

        rig = xml_v2.read(target)
        assert rig.solver is not None
        assert rig.solver.damping == pytest.approx(0.5)
        assert rig.blend == pytest.approx(0.4)
        assert rig.look is not None, "the editor writes the look record in full"
        assert rig.look.ao_strength == pytest.approx(0.85)
        assert not document.room_dirty(meadow)

    def test_a_tuned_look_survives_the_editor_round_trip(self, document, tmp_path):
        look = document.metaball.look
        look.light_direction = (0.0, 0.6, 0.8)
        look.light_strength = 1.3
        look.ao_strength = 0.9
        look.background_top = (0.3, 0.1, 0.2)
        document.settings_edited()
        path = tmp_path / "lit.xml"
        document.save(path)

        reopened = Document()
        reopened.open(path)
        restored = reopened.metaball.look
        assert restored.light_direction == pytest.approx((0.0, 0.6, 0.8))
        assert restored.light_strength == pytest.approx(1.3)
        assert restored.ao_strength == pytest.approx(0.9)
        assert restored.background_top == pytest.approx((0.3, 0.1, 0.2))
        assert not reopened.dirty

    def test_a_sectionless_file_resets_the_sessions_look(self, document, tmp_path):
        """The look is the file's to state; absence states the defaults."""
        path = tmp_path / "plain.xml"
        xml_v2.save(document.body, path)

        document.metaball.look.glow_strength = 0.9
        document.settings_edited()
        assert document.dirty

        document.open(path)
        assert document.metaball.look == Look()
        assert not document.dirty

    def test_adopt_without_a_look_resets_a_tuned_one(self, document):
        """None means the founding record — the file is the truth at every
        door, and New must not carry a session's lighting into a rig that
        never stated it."""
        document.metaball.look.glow_strength = 0.9
        document.adopt(Body())
        assert document.metaball.look == Look()

    def test_a_two_room_scene_reopens_each_rooms_own_look(
        self, document, tmp_path
    ):
        meadow = document.add_room(Room(name="meadow"))
        meadow.body.add_chunk("m0", (0.0, 0.5, 0.0), radius=0.3)

        document.metaball.look.ambient_strength = 0.6
        document.set_active_room(meadow)
        document.metaball.look.ambient_strength = 0.1

        manifest = tmp_path / "scene.xml"
        document.save_scene_as(manifest)

        reopened = Document()
        reopened.open_scene(manifest)
        rooms = {room.name: room for room in reopened.scene.rooms}
        assert reopened.room is rooms["document"]
        assert reopened.metaball.look.ambient_strength == pytest.approx(0.6)
        reopened.set_active_room(rooms["meadow"])
        assert reopened.metaball.look.ambient_strength == pytest.approx(0.1)


class TestPlacementRefusals:
    """create_chunk refuses rather than guesses, and declines without lying."""

    def test_a_wrong_arity_at_is_refused_by_name(self, document):
        with pytest.raises(ValueError, match="components"):
            document.create_chunk(at=(1.0, 2.0))
        assert document.body.n_chunks == 5
        assert not document.can_undo

    def test_an_abstracted_room_declines_without_lying(self, document):
        document.set_room_abstracted(document.room, True)
        kept = set(document.selection)
        count = document.body.n_chunks

        result = document.create_chunk()

        assert result is None, "no id is invented for a chunk that never existed"
        assert document.body.n_chunks == count
        assert set(document.selection) == kept, "the refusal must not clear it"
        assert not document.can_undo


class TestNewIsAdoptsNullCase:
    def test_new_resets_solver_and_blend(self, document):
        document.solver_config = SolverConfig(gravity=(0.0, -3.0, 0.0), iterations=8)
        document.metaball.blend = 0.5
        heard = []
        document.settings_changed.connect(lambda: heard.append(True))

        document.new()

        assert document.solver_config == Room.founding_config()
        assert document.metaball.blend == MetaballSettings().blend
        assert not document.dirty
        assert heard, "panels must re-read after the reset"


class TestRefusalsSurface:
    def test_a_solver_typo_surfaces_its_own_refusal(self, document, tmp_path):
        """The legacy ladder is for eras, not for a v3 file with a typo: the
        crafted SchemaError must reach the author, not an armature complaint."""
        path = tmp_path / "typo.xml"
        path.write_text(
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<body version="3" name="stub" dim="3">\n'
            '    <chunks>\n'
            '        <chunk id="0" name="a" pos="0.0 0.0 0.0" radius="0.25" mass="1.0"/>\n'
            "    </chunks>\n"
            "    <links/>\n"
            '    <solver damping="nan"/>\n'
            "</body>\n",
            encoding="utf-8",
        )

        with pytest.raises(xml_v2.SchemaError, match="finite"):
            document.open(path)


class TestTerrainAuthoring:
    """The world is flesh that stopped moving.

    The identity the arc stands on is that a chunk graph *is* a cone list â€”
    the same eight numbers ``sdf.pack`` writes, folded by the same
    ``round_cone`` â€” so most of what is tested here is that the boundary
    between the two is lossless in both directions and honest about what it
    drops at it.
    """

    def promote_all(self, document):
        """The body promoted whole; returns the world it produced."""
        document.set_selection(document.body.chunk_ids())
        document.make_terrain()
        return document.terrain

    def said(self, document) -> list[str]:
        messages: list[str] = []
        document.status.connect(messages.append)
        return messages

    def test_set_terrain_round_trips_by_identity(self, document):
        world = Terrain.flat(0.0)
        document.do(SetTerrain(world))
        assert document.terrain is world

        document.undo()
        assert document.terrain is None
        document.redo()
        assert document.terrain is world

    def test_a_terrain_command_dirties_where_the_load_path_does_not(self, document):
        """The seam the arc exists to close, from both sides."""
        document.set_terrain(Terrain.flat(0.0))
        assert not document.dirty, "a loaded world is not an unsaved change"

        document.do(SetTerrain(Terrain.flat(1.0)))
        assert document.dirty, "an authored one is"

    def test_a_promoted_chain_leaves_the_body_as_cones_in_order(self, document):
        chain = ids_of(document)
        positions = [document.body.chunk(cid).pos for cid in chain]

        world = self.promote_all(document)

        assert document.body.chunk_ids() == []
        assert len(world.cones) == len(chain) - 1, "one cone per rendered link"
        assert world.cones[0][0:3] == pytest.approx(positions[0])
        assert world.cones[0][4:7] == pytest.approx(positions[1])
        assert world.cones[-1][4:7] == pytest.approx(positions[-1])

    def test_a_mechanism_link_promotes_two_spheres_not_a_cone(self, document):
        """``link_render`` is the promotion rule: what you saw is what promotes."""
        a, b = ids_of(document)[0], ids_of(document)[1]
        for chunk in ids_of(document)[2:]:
            document.body.remove_chunk(chunk)
        document.body.set_link(a, b, render=False)

        world = self.promote_all(document)

        assert len(world.cones) == 2, "two lone chunks, no fused span"
        for cone in world.cones:
            assert cone[0:3] == pytest.approx(cone[4:7]), "a sphere is a == b"

    def test_the_promoted_records_are_the_packers_own_rows(self, document):
        """An agreement test where construction was not available.

        ``sdf.pack`` is float32, over draw positions, sorted group-major for
        the renderer's own reasons; a promotion is float64 over body truth.
        The numbers still have to be the same numbers, and this is what says
        so if either side ever drifts.
        """
        from mesh_xml.render import sdf

        rows = sdf.pack(document.body)
        world = self.promote_all(document)

        assert len(world.cones) == len(rows)
        for cone, row in zip(world.cones, rows):
            assert cone == pytest.approx(np.asarray(row, dtype=np.float64))

    def test_one_undo_restores_body_and_world_together(self, document):
        chain = ids_of(document)
        links = document.body.n_links

        self.promote_all(document)
        assert len(document._undo) == 1, "one gesture, one step"

        document.undo()
        assert sorted(document.body.chunk_ids()) == sorted(chain)
        assert document.body.n_links == links
        assert document.terrain is None

    def test_the_round_trip_is_the_identity_on_the_cone_list(self, document):
        """Â§9's invariant, and why demote may weld on exact equality."""
        before = self.promote_all(document).cones.copy()

        document.edit_terrain_as_chunks()
        self.promote_all(document)

        assert np.array_equal(document.terrain.cones, before)

    def test_demoted_chunks_come_back_pinned_and_welded(self, document):
        chain = ids_of(document)
        self.promote_all(document)

        document.edit_terrain_as_chunks()

        back = document.body.chunk_ids()
        assert len(back) == len(chain), "shared endpoints welded, not doubled"
        assert all(document.body.chunk(cid).pinned for cid in back)
        assert document.body.n_links == len(chain) - 1
        assert document.selection == set(back), "the product takes the selection"

    def test_demote_leaves_the_planes_standing(self, document):
        document.set_terrain(
            Terrain.flat(0.0).add_cone((0, 0, 0), 0.3, (1, 0, 0), 0.3)
        )

        document.edit_terrain_as_chunks()

        assert len(document.terrain.planes) == 1
        assert len(document.terrain.cones) == 0

    def test_demote_refuses_a_world_with_no_cones(self, document):
        document.set_terrain(Terrain.flat(0.0))
        said = self.said(document)
        before = document.body.chunk_ids()

        document.edit_terrain_as_chunks()

        assert "No terrain cones" in said[-1]
        assert document.body.chunk_ids() == before
        assert len(document.terrain.planes) == 1

    def test_promote_refuses_a_driven_creature_whole(self, document):
        """Terrain does not walk â€” and refusing beats stripping the rig."""
        from mesh_xml.anim.behavior import Grazer

        document.controllers.add(Grazer(ids_of(document)[0], home=(0.0, 0.0, 0.0)))
        said = self.said(document)
        before = ids_of(document)

        self.promote_all(document)

        assert "terrain does not walk" in said[-1]
        assert document.body.chunk_ids() == before
        assert document.terrain is None
        assert not document.can_undo

    def test_promote_refuses_an_empty_selection(self, document):
        said = self.said(document)
        document.make_terrain()
        assert "Select a chunk" in said[-1]
        assert document.terrain is None

    def test_the_first_cone_names_the_floor_it_repeals(self, document):
        """The solver picks one world per tick, so the first cone stands the
        reference floor down. Not compensated â€” said."""
        document.solver_config.ground_y = 0.5
        said = self.said(document)

        self.promote_all(document)

        assert "reference floor at 0.5 no longer acts" in said[-1]

    def test_a_later_promotion_says_nothing_about_the_floor(self, document):
        self.promote_all(document)
        said = self.said(document)

        document.do(AddChunk("lone", (9.0, 0.0, 0.0), 0.4))
        self.promote_all(document)

        assert "reference floor" not in said[-1], "it was already repealed"

    def test_add_floor_promotes_the_reference_height(self, document):
        document.solver_config.ground_y = 0.25

        document.add_floor_plane()

        plane = document.terrain.planes[0]
        assert plane[:3] == pytest.approx((0.0, 1.0, 0.0))
        assert plane[3] == pytest.approx(0.25)
        assert document.dirty and len(document._undo) == 1

    def test_add_floor_answers_the_promotion_that_vetoed_the_grid(self, document):
        """The verb reads the reference, never the drawn grid.

        Reading the grid would make the promotion's own advice impossible:
        the cone that repealed the floor is exactly what stands the grid down.
        """
        self.promote_all(document)
        assert not document.floor_state()[0], "the world vetoed the grid"

        document.add_floor_plane()

        assert len(document.terrain.planes) == 1

    def test_add_floor_refuses_what_the_author_turned_off(self, document):
        document.metaball.show_floor = False
        said = self.said(document)

        document.add_floor_plane()

        assert "No reference floor" in said[-1]
        assert document.terrain is None

    def test_add_floor_refuses_a_second_plane(self, document):
        document.add_floor_plane()
        said = self.said(document)

        document.add_floor_plane()

        assert "already stands on 1 plane" in said[-1]
        assert len(document.terrain.planes) == 1

    def test_the_blend_is_a_setting_not_a_step(self, document):
        self.promote_all(document)
        depth = len(document._undo)

        document.set_terrain_blend(0.5)

        assert document.terrain.blend == pytest.approx(0.5)
        assert len(document._undo) == depth, "a dragged number is not a move"
        assert document.dirty
        assert len(document.terrain.cones) == 4, "the world kept its shape"

    def test_promotion_drops_what_the_record_cannot_carry(self, document):
        """A cone is eight floats: paint, names and tempo stop at the boundary."""
        from mesh_xml.editor.document import CREATURE_META, SetMeta

        for cid in ids_of(document):
            document.do(SetMeta(cid, CREATURE_META, "ridge"))

        self.promote_all(document)
        document.edit_terrain_as_chunks()

        assert all(
            document.body.meta_of(cid).get(CREATURE_META) is None
            for cid in document.body.chunk_ids()
        )

    def test_a_promoted_world_saves_and_reopens(self, document, tmp_path):
        """The arc's whole point: a world that survives the session."""
        document.add_floor_plane()
        self.promote_all(document)
        path = tmp_path / "world.xml"
        document.save(path)

        reopened = Document()
        reopened.open(path)

        assert np.allclose(reopened.terrain.planes, document.terrain.planes)
        assert np.allclose(reopened.terrain.cones, document.terrain.cones)
        assert not reopened.dirty, "a loaded world is not an unsaved change"
