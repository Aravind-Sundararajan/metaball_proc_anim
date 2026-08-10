"""Tests for the dock panels.

Two things are worth testing here and one of them is not obvious.

The obvious half is the round trip: a control edit has to reach the document and
a document change has to reach the control. That is most of the file and it is
mechanical.

The half that matters is the feedback loop. Writing a document value into a
control makes Qt emit that control's change signal, and a panel that reads it as
a user edit writes straight back. The symptom is never a hang — it is a spurious
command on the undo stack, or a selection that empties itself whenever the
topology changes. So the loop tests are written as a comparison: the same panel
class with its latch defeated is constructed alongside the real one, and the
assertion is that the real one does nothing while the defeated one misbehaves.
Without that contrast a passing test proves only that nothing happened, which is
also what a panel that was never wired up would prove.
"""

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QWidget,
)

from mesh_xml.core import Body
from mesh_xml.editor.document import (
    AddChunk,
    Document,
    Macro,
    RemoveChunk,
    SetRadius,
)

context = pytest.importorskip("mesh_xml.render.gl.context")

try:
    APPLICATION = context.ensure_application()
except Exception as exc:  # pragma: no cover - headless box with no platform plugin
    pytest.skip(f"no Qt platform available: {exc}", allow_module_level=True)

from mesh_xml.editor import panels  # noqa: E402 - needs the QApplication above

_NAME, _RADIUS, _PINNED = 0, 1, 2

LEGACY_RIG = "tests/data/legacy_v1_chubby_worm.xml"


class _UnguardedChunkPanel(panels.ChunkPanel):
    """:class:`ChunkPanel` with the latch defeated, to prove the latch is load bearing."""

    @property
    def syncing(self) -> bool:
        return False


class _UnguardedSurfacePanel(panels.SurfacePanel):
    """:class:`SurfacePanel` with the latch defeated."""

    @property
    def syncing(self) -> bool:
        return False


class _UnguardedSmoothingPanel(panels.SmoothingPanel):
    """:class:`SmoothingPanel` with the latch defeated."""

    @property
    def syncing(self) -> bool:
        return False


@pytest.fixture
def make_document():
    """Build a chain rig.

    The radii are deliberately finer than the table's three-decimal display. A
    value that survives the round trip through its own rendering hides every
    echo bug in this file, because the panel then reads back exactly what it
    wrote and has no reason to submit anything.
    """

    def build(n: int = 4) -> Document:
        document = Document()
        body = Body()
        ids = [
            body.add_chunk(f"c{i}", [float(i), 0.0, 0.0], radius=0.3001 + 0.0001 * i)
            for i in range(n)
        ]
        for a, b in zip(ids, ids[1:]):
            body.add_link(a, b)
        document._replace(body, None)
        return document

    return build


@pytest.fixture
def document(make_document) -> Document:
    return make_document()


def undo_depth(document: Document) -> int:
    return len(document._undo)


def label_texts(widget: QWidget) -> list[str]:
    return [label.text() for label in widget.findChildren(QLabel)]


class TestBuildPanels:
    def test_returns_one_panel_per_dock_title(self, document):
        built = panels.build_panels(document)
        assert list(built) == [
            "Playback",
            "Rooms",
            "Chunks",
            "Creatures",
            "Controllers",
            "Surface",
            "Solver",
            "Smoothing",
            "Curve",
        ]

    def test_every_panel_is_a_widget_bound_to_the_document(self, document):
        for panel in panels.build_panels(document).values():
            assert isinstance(panel, QWidget)
            assert panel.document is document

    def test_assembling_the_panels_edits_nothing(self, document):
        built = panels.build_panels(document)
        assert built
        assert undo_depth(document) == 0
        assert not document.dirty


class TestPlaybackPanel:
    def test_the_play_button_drives_the_document(self, document):
        panel = panels.PlaybackPanel(document)
        panel._play.click()
        assert document.running

    def test_the_document_drives_the_play_button(self, document):
        panel = panels.PlaybackPanel(document)
        document.set_running(True)
        assert panel._play.isChecked()
        assert panel._play.text() == "Pause"

    def test_rest_pose_restores_the_captured_positions(self, document):
        panel = panels.PlaybackPanel(document)
        target = document.body.chunk_ids()[3]
        original = document.body.chunk(target).pos

        document.set_running(True)
        document.advance(1.0)
        assert document.body.chunk(target).pos != pytest.approx(original)

        panel._rest.click()
        assert document.body.chunk(target).pos == pytest.approx(original)

    def test_a_single_step_advances_a_paused_body(self, document):
        panel = panels.PlaybackPanel(document)
        seen = []
        panel.stepped.connect(lambda: seen.append(1))

        before = document.body.pos.copy()
        panel.single_step()

        assert not np.array_equal(before, document.body.pos)
        assert seen == [1]

    def test_a_single_step_is_inert_while_running(self, document):
        """Stepping a running simulation would double-tick it, invisibly."""
        panel = panels.PlaybackPanel(document)
        document.set_running(True)
        before = document.body.pos.copy()

        panel.single_step()
        assert np.array_equal(before, document.body.pos)
        assert not panel._step.isEnabled()

    def test_the_readout_follows_the_clock(self, document):
        panel = panels.PlaybackPanel(document)
        assert panel._ticks.text() == "0"

        document.set_running(True)
        document.advance(0.5)
        panel.refresh()

        assert document.clock.ticks > 0
        assert panel._ticks.text() == str(document.clock.ticks)
        assert panel._dropped.text() == str(document.clock.dropped)
        assert panel._alpha.text() == f"{document.clock.alpha:.3f}"


class TestSurfacePanel:
    def test_the_blend_spin_reaches_the_document(self, document):
        panel = panels.SurfacePanel(document)
        panel._blend.setValue(0.42)
        assert document.metaball.blend == pytest.approx(0.42)

    def test_the_blend_slider_reaches_the_document(self, document):
        panel = panels.SurfacePanel(document)
        panel._blend_slider.setValue(600)
        assert document.metaball.blend == pytest.approx(0.6)

    def test_a_document_change_moves_both_blend_controls(self, document):
        panel = panels.SurfacePanel(document)
        document.metaball.blend = 0.75
        document.settings_changed.emit()

        assert panel._blend.value() == pytest.approx(0.75)
        assert panel._blend_slider.value() == 750

    def test_the_debug_combo_offers_every_mode_and_selects_one(self, document):
        panel = panels.SurfacePanel(document)
        assert panel._debug.count() == 5

        panel._debug.setCurrentIndex(2)
        assert document.metaball.debug_mode == 2

    def test_max_steps_reaches_the_document(self, document):
        panel = panels.SurfacePanel(document)
        panel._max_steps.setValue(256)
        assert document.metaball.max_steps == 256

    def test_colours_reach_the_document_and_the_swatches(self, document):
        panel = panels.SurfacePanel(document)
        panel.set_surface_colour((1.0, 0.0, 0.0))
        panel.set_background_colour((0.0, 0.0, 1.0))

        assert document.metaball.surface_colour == pytest.approx((1.0, 0.0, 0.0))
        assert document.metaball.background_colour == pytest.approx((0.0, 0.0, 1.0))
        assert panel._surface_colour.text() == "#ff0000"
        assert panel._background_colour.text() == "#0000ff"

    def test_the_paint_buttons_route_to_their_gestures(self, document, monkeypatch):
        """Two buttons, two scopes: the selection's chunks, or their whole
        component. The dialog is stubbed; the routing is what is tested."""
        panel = panels.SurfacePanel(document)
        monkeypatch.setattr(panel, "_pick", lambda current, title: (1.0, 0.0, 0.0))
        ids = document.body.chunk_ids()
        document.set_selection(ids[:1])

        panel._chunk_colour.click()
        assert document.body.meta_of(ids[0])["colour"] == "1.0000 0.0000 0.0000"
        assert "colour" not in document.body.meta_of(ids[1]), "selection only"

        panel._creature_colour.click()
        assert document.body.meta_of(ids[1])["colour"] == "1.0000 0.0000 0.0000", (
            "the creature-wide button reaches the whole component"
        )

    def test_the_pattern_row_explains_its_own_greyness(self, document):
        """A disabled combo reads as an empty one — found by a user asking
        exactly that. The row says why it is grey, and stops saying it the
        moment the reason is gone."""
        panel = panels.SurfacePanel(document)
        assert "selected creature" in panel._pattern_scope.text()

        document.set_selection(document.body.chunk_ids()[:1])
        assert panel._pattern_scope.text() == ""

    def test_the_pattern_combo_routes_and_reads_back(self, document):
        panel = panels.SurfacePanel(document)
        ids = document.body.chunk_ids()
        assert not panel._pattern.isEnabled(), "no selection, no pattern edits"

        document.set_selection(ids[:1])
        assert panel._pattern.isEnabled()

        index = panel._pattern.findText("stripes")
        panel._pattern.setCurrentIndex(index)
        assert document.creature_pattern() == ("stripes", 4.0), (
            "picking a pattern applies it with its default parameter"
        )
        assert panel._pattern_param.value() == pytest.approx(4.0)

        panel._pattern_param.setValue(7.0)
        assert document.creature_pattern() == ("stripes", 7.0)

        panel._pattern.setCurrentIndex(panel._pattern.findText("none"))
        assert document.creature_pattern() == ("none", 0.0)

    def test_the_overlay_is_panel_state_and_announces_itself(self, document):
        panel = panels.SurfacePanel(document)
        seen = []
        panel.overlay_changed.connect(seen.append)

        panel._overlay.setChecked(True)
        assert seen == [True]
        assert panel.show_rig_overlay

    def test_the_overlay_can_be_set_from_outside(self, document):
        panel = panels.SurfacePanel(document)
        panel.set_show_rig_overlay(True)
        assert panel._overlay.isChecked()

    def test_the_projection_is_panel_state_and_announces_itself(self, document):
        """The camera belongs to the viewport, so the panel can only say so."""
        panel = panels.SurfacePanel(document)
        seen = []
        panel.orthographic_changed.connect(seen.append)

        panel._ortho.setChecked(True)
        assert seen == [True]
        assert panel.orthographic

        panel._ortho.setChecked(False)
        assert seen == [True, False]
        assert not panel.orthographic

    def test_the_projection_can_be_set_from_outside(self, document):
        """The main window adopts the camera's own state on startup this way."""
        panel = panels.SurfacePanel(document)
        panel.set_orthographic(True)
        assert panel._ortho.isChecked()
        assert panel.orthographic

    def test_setting_the_projection_to_what_it_already_is_says_nothing(self, document):
        panel = panels.SurfacePanel(document)
        seen = []
        panel.orthographic_changed.connect(seen.append)
        panel.set_orthographic(False)
        assert seen == []

    def test_the_projection_survives_a_document_refresh(self, document):
        """It is not in MetaballSettings, so a settings refresh must not clear it."""
        panel = panels.SurfacePanel(document)
        panel.set_orthographic(True)
        document.settings_changed.emit()
        assert panel.orthographic
        assert panel._ortho.isChecked()

    def test_the_projection_note_explains_the_framing(self, document):
        panel = panels.SurfacePanel(document)
        assert any("ortho height" in text for text in label_texts(panel))

    def test_the_blend_note_warns_about_authored_radii(self, document):
        """Nobody tuning radii finds this in sdf.py before they lose an hour to it."""
        panel = panels.SurfacePanel(document)
        assert any("k/4" in text for text in label_texts(panel))

    def test_show_floor_reaches_the_settings(self, document):
        panel = panels.SurfacePanel(document)
        assert document.metaball.show_floor
        assert panel._floor.isChecked()

        panel._floor.setChecked(False)
        assert not document.metaball.show_floor


class TestSolverPanel:
    def test_iterations_reach_the_config(self, document):
        panel = panels.SolverPanel(document)
        panel._iterations.setValue(8)
        assert document.solver_config.iterations == 8

    def test_damping_and_max_velocity_reach_the_config(self, document):
        panel = panels.SolverPanel(document)
        panel._damping.setValue(0.9)
        panel._max_velocity.setValue(50.0)

        assert document.solver_config.damping == pytest.approx(0.9)
        assert document.solver_config.max_velocity == pytest.approx(50.0)

    def test_gravity_axes_reach_the_config(self, document):
        panel = panels.SolverPanel(document)
        panel._gravity[0].setValue(1.5)
        panel._gravity[2].setValue(-2.5)

        assert document.solver_config.gravity == pytest.approx((1.5, -9.81, -2.5))

    def test_zero_g_zeroes_gravity_and_gives_it_back(self, document):
        panel = panels.SolverPanel(document)
        panel._zero_g.setChecked(True)
        assert document.solver_config.gravity == pytest.approx((0.0, 0.0, 0.0))
        assert not panel._gravity[1].isEnabled()

        panel._zero_g.setChecked(False)
        assert document.solver_config.gravity == pytest.approx((0.0, -9.81, 0.0))
        assert panel._gravity[1].isEnabled()

    def test_self_contact_reaches_the_config_and_gates_its_knob(self, document):
        panel = panels.SolverPanel(document)
        assert not document.solver_config.self_contact
        assert not panel._contact_elasticity.isEnabled()

        panel._self_contact.setChecked(True)
        assert document.solver_config.self_contact
        assert panel._contact_elasticity.isEnabled()

        panel._contact_elasticity.setValue(0.8)
        assert document.solver_config.self_contact_elasticity == pytest.approx(0.8)

    def test_a_config_change_reflects_into_the_controls(self, document):
        panel = panels.SolverPanel(document)
        document.solver_config.iterations = 16
        document.solver_config.gravity = (0.0, -3.0, 0.0)
        document.settings_changed.emit()

        assert panel._iterations.value() == 16
        assert panel._gravity[1].value() == pytest.approx(-3.0)

    def test_the_iterations_tooltip_states_the_measured_give(self, document):
        panel = panels.SolverPanel(document)
        tooltip = panel._iterations.toolTip()
        assert "8%" in tooltip
        assert "softness" in tooltip


class TestSmoothingPanel:
    def test_stiffness_is_written_to_every_draw_point(self, document):
        panel = panels.SmoothingPanel(document)
        panel._stiffness.setValue(0.9)

        assert float(document.draw.stiffness.min()) == pytest.approx(0.9)
        assert float(document.draw.stiffness.max()) == pytest.approx(0.9)

    def test_damping_and_squash_reach_the_draw_points(self, document):
        panel = panels.SmoothingPanel(document)
        panel._damping.setValue(0.5)
        panel._squash.setValue(0.25)

        assert float(document.draw.damping.max()) == pytest.approx(0.5)
        assert document.draw.squash == pytest.approx(0.25)

    def test_a_new_chunk_picks_up_the_uniform_value(self, document):
        """DrawPoints.sync hands a new point the constructor default, not this one."""
        panel = panels.SmoothingPanel(document)
        panel._stiffness.setValue(0.9)
        document.do(AddChunk("tip", (9.0, 0.0, 0.0)))

        assert len(document.draw.stiffness) == document.body.n_chunks
        assert float(document.draw.stiffness.min()) == pytest.approx(0.9)

    def test_a_replaced_body_is_adopted_rather_than_overwritten(self, document):
        panel = panels.SmoothingPanel(document)
        panel._stiffness.setValue(0.9)
        document.open(LEGACY_RIG)

        assert panel._stiffness.value() != pytest.approx(0.9)
        assert panel._stiffness.value() == pytest.approx(float(document.draw.stiffness[0]))

    def test_the_notes_state_the_per_tick_convention(self, document):
        panel = panels.SmoothingPanel(document)
        texts = " ".join(label_texts(panel))
        assert "Per-tick" in texts
        assert "per point" in texts


class TestPerPointSmoothing:
    """A tight head over a loose gut: the reason DrawPoints stores these per point."""

    def slot_of(self, document, chunk_id: int) -> int:
        return document.body.chunk_ids().index(chunk_id)

    def test_an_edit_reaches_only_the_selected_chunks(self, document):
        panel = panels.SmoothingPanel(document)
        ids = document.body.chunk_ids()
        panel._stiffness.setValue(0.5)

        document.set_selection([ids[0]])
        panel._stiffness.setValue(0.95)

        stiffness = document.draw.stiffness
        assert float(stiffness[self.slot_of(document, ids[0])]) == pytest.approx(0.95)
        assert float(stiffness[self.slot_of(document, ids[1])]) == pytest.approx(0.5)
        assert float(stiffness[self.slot_of(document, ids[3])]) == pytest.approx(0.5)

    def test_damping_is_scoped_the_same_way(self, document):
        panel = panels.SmoothingPanel(document)
        ids = document.body.chunk_ids()
        panel._damping.setValue(0.1)

        document.set_selection([ids[2]])
        panel._damping.setValue(0.99)

        damping = document.draw.damping
        assert float(damping[self.slot_of(document, ids[2])]) == pytest.approx(0.99)
        assert float(damping[self.slot_of(document, ids[0])]) == pytest.approx(0.1)

    def test_the_scope_is_the_selection_in_table_order(self, document):
        panel = panels.SmoothingPanel(document)
        ids = document.body.chunk_ids()
        assert panel.scope_ids() == ids

        document.set_selection([ids[3], ids[1]])
        assert panel.scope_ids() == [ids[1], ids[3]]

    def test_the_boxes_follow_the_selection(self, document):
        panel = panels.SmoothingPanel(document)
        ids = document.body.chunk_ids()
        document.set_selection([ids[1]])
        panel._stiffness.setValue(0.2)

        document.set_selection([ids[2]])
        assert panel._stiffness.value() != pytest.approx(0.2)

        document.set_selection([ids[1]])
        assert panel._stiffness.value() == pytest.approx(0.2)

    def test_a_mixed_scope_says_so(self, document):
        panel = panels.SmoothingPanel(document)
        ids = document.body.chunk_ids()
        assert "differ" not in " ".join(label_texts(panel))

        document.set_selection([ids[1]])
        panel._stiffness.setValue(0.2)
        document.set_selection([])

        assert "differ" in " ".join(label_texts(panel))

    def test_squash_stays_rig_wide(self, document):
        """DrawPoints stores one scalar; a per-point squash would be a field nothing reads."""
        panel = panels.SmoothingPanel(document)
        document.set_selection([document.body.chunk_ids()[0]])
        panel._squash.setValue(0.4)
        assert document.draw.squash == pytest.approx(0.4)

    def test_a_per_chunk_edit_does_not_become_the_rig_default(self, document):
        panel = panels.SmoothingPanel(document)
        ids = document.body.chunk_ids()
        panel._stiffness.setValue(0.5)
        document.set_selection([ids[0]])
        panel._stiffness.setValue(0.95)

        document.do(AddChunk("tip", (9.0, 0.0, 0.0)))

        added = document.body.chunk_ids()[-1]
        assert float(
            document.draw.stiffness[self.slot_of(document, added)]
        ) == pytest.approx(0.5)

    def test_a_structural_edit_does_not_flatten_per_chunk_work(self, document):
        """A radius-curve apply arrives as a Macro, and Macro is structural."""
        panel = panels.SmoothingPanel(document)
        ids = document.body.chunk_ids()
        panel._stiffness.setValue(0.5)
        document.set_selection([ids[1]])
        panel._stiffness.setValue(0.2)
        document.set_selection([])

        document.do(Macro([SetRadius(ids[0], 0.44)], text="Radius Curve"))

        assert float(
            document.draw.stiffness[self.slot_of(document, ids[1])]
        ) == pytest.approx(0.2)

    def test_per_chunk_values_survive_a_deletion_elsewhere(self, document):
        """remove_chunk swaps the last chunk into the freed slot; sync remaps by id."""
        panel = panels.SmoothingPanel(document)
        ids = document.body.chunk_ids()
        panel._stiffness.setValue(0.5)
        document.set_selection([ids[3]])
        panel._stiffness.setValue(0.2)
        document.set_selection([])

        document.do(RemoveChunk(ids[0]))

        assert float(
            document.draw.stiffness[self.slot_of(document, ids[3])]
        ) == pytest.approx(0.2)

    def test_a_replaced_body_forgets_the_rig_default(self, document):
        panel = panels.SmoothingPanel(document)
        panel._stiffness.setValue(0.9)
        document.open(LEGACY_RIG)
        document.do(AddChunk("tip", (9.0, 0.0, 0.0)))

        assert float(document.draw.stiffness.max()) != pytest.approx(0.9)
        assert float(document.draw.stiffness.min()) == pytest.approx(
            float(document.draw.stiffness.max())
        )

    def test_the_note_describes_per_point_authoring(self, document):
        panel = panels.SmoothingPanel(document)
        texts = " ".join(label_texts(panel))
        assert "selected chunks only" in texts
        assert "loose gut" in texts


class TestChunkPanel:
    def test_the_table_lists_every_chunk(self, document):
        panel = panels.ChunkPanel(document)
        assert panel.table.rowCount() == document.body.n_chunks
        assert panel._rows == document.body.chunk_ids()
        assert panel.table.item(0, _NAME).text() == "c0"
        assert panel.table.item(0, _PINNED).checkState() == Qt.CheckState.Unchecked

    def test_add_button_grows_the_body_and_the_table(self, document):
        panel = panels.ChunkPanel(document)
        chunks = document.body.n_chunks
        end = document.body.chunk_ids()[-1]
        document.set_selection((end,))

        panel._add.click()

        assert document.body.n_chunks == chunks + 1
        assert panel.table.rowCount() == chunks + 1
        new = next(iter(document.selection))
        assert document.body.has_link(new, end)

    def test_editing_a_radius_submits_set_radius(self, document):
        panel = panels.ChunkPanel(document)
        panel.table.item(0, _RADIUS).setText("0.750")

        assert document.body.chunk(panel._rows[0]).radius == pytest.approx(0.75)
        assert undo_depth(document) == 1

    def test_an_invalid_radius_reverts_and_submits_nothing(self, document):
        panel = panels.ChunkPanel(document)
        radius = document.body.chunk(panel._rows[0]).radius

        panel.table.item(0, _RADIUS).setText("nonsense")

        assert undo_depth(document) == 0
        assert panel.table.item(0, _RADIUS).text() == format(radius, ".3f")

    def test_a_non_positive_radius_is_refused(self, document):
        panel = panels.ChunkPanel(document)
        panel.table.item(0, _RADIUS).setText("0")
        assert undo_depth(document) == 0

    def test_toggling_pinned_submits_set_pinned(self, document):
        panel = panels.ChunkPanel(document)
        panel.table.item(0, _PINNED).setCheckState(Qt.CheckState.Checked)

        assert document.body.chunk(panel._rows[0]).pinned
        assert document.undo_text == "Pin Chunk"

    def test_selecting_rows_sets_the_document_selection(self, document):
        panel = panels.ChunkPanel(document)
        panel.table.selectRow(1)
        assert document.selection == {panel._rows[1]}

    def test_the_document_selection_reaches_the_table(self, document):
        panel = panels.ChunkPanel(document)
        ids = document.body.chunk_ids()
        document.set_selection([ids[0], ids[2]])

        rows = {index.row() for index in panel.table.selectionModel().selectedRows()}
        assert rows == {0, 2}

    def test_connect_chains_the_selection_in_table_order(self, document):
        panel = panels.ChunkPanel(document)
        ids = document.body.chunk_ids()
        document.set_selection([ids[0], ids[3]])

        panel._connect.click()
        assert document.body.has_link(ids[0], ids[3])
        assert document.undo_text == "Connect"

    def test_disconnect_drops_links_between_selected_chunks(self, document):
        panel = panels.ChunkPanel(document)
        ids = document.body.chunk_ids()
        document.set_selection([ids[0], ids[1]])

        panel._disconnect.click()
        assert not document.body.has_link(ids[0], ids[1])

    def test_delete_removes_every_selected_chunk(self, document):
        panel = panels.ChunkPanel(document)
        ids = document.body.chunk_ids()
        document.set_selection([ids[0], ids[1]])

        panel._delete.click()
        assert document.body.n_chunks == 2
        assert panel.table.rowCount() == 2

    def test_the_buttons_track_what_the_selection_can_support(self, document):
        panel = panels.ChunkPanel(document)
        ids = document.body.chunk_ids()

        assert not panel._delete.isEnabled()
        document.set_selection([ids[0]])
        assert panel._delete.isEnabled()
        assert not panel._connect.isEnabled()

        document.set_selection([ids[0], ids[1]])
        assert panel._connect.isEnabled()
        assert panel._disconnect.isEnabled()

    def test_no_dangling_rows_survive_a_deletion(self, document):
        """remove_chunk swaps the last chunk into the freed slot; rows must follow."""
        panel = panels.ChunkPanel(document)
        document.do(RemoveChunk(document.body.chunk_ids()[0]))

        assert panel.table.rowCount() == document.body.n_chunks
        assert panel._rows == document.body.chunk_ids()
        for row, chunk_id in enumerate(panel._rows):
            assert document.body.has_chunk(chunk_id)
            assert panel.table.item(row, _NAME).text() == document.body.name_of(chunk_id)

    def test_the_table_survives_a_body_replacement(self, document):
        panel = panels.ChunkPanel(document)
        document.open(LEGACY_RIG)

        assert panel.table.rowCount() == document.body.n_chunks
        assert panel._rows == document.body.chunk_ids()
        for row, chunk_id in enumerate(panel._rows):
            assert panel.table.item(row, _NAME).text() == document.body.name_of(chunk_id)

    def test_an_empty_body_leaves_an_empty_table_and_no_edits(self):
        """The action handlers are reached directly; a disabled button proves nothing."""
        document = Document()
        panel = panels.ChunkPanel(document)
        assert panel.table.rowCount() == 0

        panel._on_connect()
        panel._on_disconnect()
        panel._on_delete()

        assert undo_depth(document) == 0
        assert not document.dirty

    def test_an_undone_deletion_comes_back_into_the_table(self, document):
        panel = panels.ChunkPanel(document)
        document.do(RemoveChunk(document.body.chunk_ids()[2]))
        document.undo()

        assert panel.table.rowCount() == document.body.n_chunks
        assert panel._rows == document.body.chunk_ids()


class TestReentrancy:
    """The panel writes the document, the document emits, the panel rebuilds."""

    def test_the_latch_is_held_across_a_refresh_and_released_after(self, document):
        panel = panels.SolverPanel(document)
        seen: list[bool] = []
        original = panel._refresh

        def spy() -> None:
            seen.append(panel.syncing)
            original()

        panel._refresh = spy
        panel.refresh()

        assert seen == [True]
        assert not panel.syncing

    def test_building_the_chunk_table_pushes_no_commands(self, make_document):
        guarded, unguarded = make_document(), make_document()

        alive = [panels.ChunkPanel(guarded), _UnguardedChunkPanel(unguarded)]
        assert alive

        assert undo_depth(guarded) == 0
        assert undo_depth(unguarded) > 0, "the comparison is not exercising the latch"

    def test_a_structural_edit_does_not_clear_the_selection(self, make_document):
        """The rebuild clears the table's selection on its way to restoring it."""
        results = []
        alive = []
        for panel_class in (panels.ChunkPanel, _UnguardedChunkPanel):
            document = make_document()
            # An unparented widget with no reference is collected immediately,
            # taking its connections with it, and every assertion below then
            # passes for the wrong reason.
            alive.append(panel_class(document))
            ids = document.body.chunk_ids()
            document.set_selection([ids[0], ids[1]])
            document.do(RemoveChunk(ids[3]))
            results.append(set(document.selection))

        guarded, unguarded = results
        assert guarded == {0, 1}
        assert unguarded == set(), "the comparison is not exercising the latch"

    def test_a_settings_change_is_not_echoed_back(self, make_document):
        counts = []
        alive = []
        for panel_class in (panels.SurfacePanel, _UnguardedSurfacePanel):
            document = make_document()
            alive.append(panel_class(document))
            seen: list[int] = []
            document.settings_changed.connect(lambda: seen.append(1))

            document.metaball.blend = 0.7
            document.settings_changed.emit()
            counts.append(len(seen))

        guarded, unguarded = counts
        assert guarded == 1
        assert unguarded > 1, "the comparison is not exercising the latch"

    def test_a_selection_change_does_not_repaint_the_rig_with_one_value(
        self, make_document
    ):
        """The scope moves between the write and the echo, so the echo lands wrong.

        Selecting a chunk refreshes the spin boxes from *that* chunk. Unguarded,
        Qt reports the write as a user edit, the handler applies it to the scope
        the selection has just become, and clicking a second chunk silently
        flattens everything the author gave the first one.
        """
        results = []
        alive = []
        for panel_class in (panels.SmoothingPanel, _UnguardedSmoothingPanel):
            document = make_document()
            panel = panel_class(document)
            alive.append(panel)
            ids = document.body.chunk_ids()

            document.set_selection([ids[1]])
            panel._stiffness.setValue(0.2)
            document.set_selection([])

            results.append(float(document.draw.stiffness[1]))

        guarded, unguarded = results
        assert guarded == pytest.approx(0.2)
        assert unguarded != pytest.approx(0.2), (
            "the comparison is not exercising the latch"
        )

    def test_editing_a_name_submits_exactly_one_command(self, document):
        panel = panels.ChunkPanel(document)
        panel.table.item(0, _NAME).setText("spine")

        assert undo_depth(document) == 1
        assert document.undo_text == "Rename Chunk"
        assert document.body.name_of(panel._rows[0]) == "spine"

    def test_a_value_edit_pushed_by_the_document_adds_nothing(self, document):
        panel = panels.ChunkPanel(document)
        chunk_id = document.body.chunk_ids()[0]

        document.do(SetRadius(chunk_id, 0.9))

        assert undo_depth(document) == 1
        assert panel.table.item(0, _RADIUS).text() == "0.900"

    def test_every_panel_leaves_the_latch_released_after_a_document_edit(
        self, document
    ):
        built = panels.build_panels(document)
        document.do(SetRadius(document.body.chunk_ids()[0], 0.6))
        document.settings_changed.emit()
        document.set_running(True)

        for panel in built.values():
            assert not panel.syncing


class TestControllerPanel:
    def test_lists_the_shipped_worms_gait(self, document):
        document.open("examples/chubby_worm.xml")
        panel = panels.ControllerPanel(document)

        titles = [box.title() for box in panel.findChildren(QGroupBox)]
        assert titles == ["WaveAnchor", "SpineFollow", "Breathe"]

    def test_editing_a_float_field_is_undoable(self, document):
        document.open("examples/chubby_worm.xml")
        panel = panels.ControllerPanel(document)
        wave = document.controllers.controllers[0]
        box = panel.findChildren(QGroupBox)[0]
        amplitude = box.findChildren(QDoubleSpinBox)[0]
        assert amplitude.value() == pytest.approx(wave.amplitude)

        amplitude.setValue(1.5)
        assert wave.amplitude == pytest.approx(1.5)

        document.undo()
        assert wave.amplitude == pytest.approx(0.8)

    def test_the_enabled_toggle_reaches_the_controller(self, document):
        document.open("examples/chubby_worm.xml")
        panel = panels.ControllerPanel(document)
        breathe = document.controllers.controllers[2]
        box = panel.findChildren(QGroupBox)[2]
        enabled = box.findChildren(QCheckBox)[0]

        enabled.setChecked(False)
        assert breathe.enabled is False

        document.undo()
        assert breathe.enabled is True

    def test_remove_is_undoable(self, document):
        document.open("examples/chubby_worm.xml")
        panel = panels.ControllerPanel(document)
        wave = document.controllers.controllers[0]
        box = panel.findChildren(QGroupBox)[0]
        remove = next(
            b for b in box.findChildren(QPushButton) if b.text() == "Remove"
        )

        remove.click()
        assert len(document.controllers.controllers) == 2

        document.undo()
        assert document.controllers.controllers[0] is wave

    def test_autorig_fits_a_bare_rig(self, document):
        panel = panels.ControllerPanel(document)
        assert not document.controllers.controllers

        panel._autorig.click()
        names = [type(c).__name__ for c in document.controllers.controllers]
        assert names[0] == "WaveAnchor"

        document.undo()
        assert not document.controllers.controllers

    def test_the_tempo_spin_retimes_the_selected_creature(self, document):
        document.open("examples/chubby_worm.xml")
        panel = panels.ControllerPanel(document)
        wave = document.controllers.controllers[0]
        rate = wave.rate

        assert not panel._tempo.isEnabled(), "no selection, nothing to retime"
        document.set_selection([document.body.chunk_ids()[0]])
        assert panel._tempo.isEnabled()

        panel._tempo.setValue(2.0)
        assert wave.rate == pytest.approx(rate * 2.0)
        assert undo_depth(document) == 1
        assert document.undo_text == "Creature Tempo"

        document.undo()
        assert wave.rate == pytest.approx(rate)
        assert panel._tempo.value() == pytest.approx(1.0)

    def test_a_tempo_pushed_by_the_document_adds_nothing(self, document):
        document.open("examples/chubby_worm.xml")
        document.set_selection([document.body.chunk_ids()[0]])
        panel = panels.ControllerPanel(document)

        document.set_creature_tempo(2.0)

        assert undo_depth(document) == 1
        assert panel._tempo.value() == pytest.approx(2.0)


class TestCreaturePanel:
    def test_lists_creatures_and_picking_selects(self, document):
        document.open("examples/hunt.xml")
        panel = panels.CreaturePanel(document)

        assert panel._roster.count() == 2
        panel._on_pick(1)
        grazer = document.creatures()[1]
        assert document.selection == set(grazer.chunks)

    def test_renaming_is_one_undo_step(self, document):
        document.open("examples/hunt.xml")
        panel = panels.CreaturePanel(document)
        document.set_selection(set(document.creatures()[0].chunks))

        panel._name.setText("apex")
        panel._on_rename()

        assert undo_depth(document) == 1
        assert document.undo_text == "Rename Creature"
        assert document.creatures()[0].name == "apex"
        assert "apex" in panel._roster.itemText(0)

    def test_a_focus_out_without_change_is_not_an_edit(self, document):
        document.open("examples/hunt.xml")
        panel = panels.CreaturePanel(document)
        document.set_selection(set(document.creatures()[0].chunks))

        panel._on_rename()

        assert undo_depth(document) == 0

    def test_the_duplicate_button_herds(self, document):
        document.open("examples/chubby_worm.xml")
        panel = panels.CreaturePanel(document)
        document.set_selection({document.body.chunk_ids()[0]})

        panel._duplicate.click()

        assert len(document.creatures()) == 2
        assert undo_depth(document) == 1
        assert panel._roster.count() == 2


class TestRoomPanel:
    """The rooms dock: the row selection is the active room, and the cut
    goes through the document, never around it."""

    def test_rows_mirror_the_scene(self, document):
        from mesh_xml.scene import Room

        panel = panels.RoomPanel(document)
        document.add_room(Room(name="meadow"))

        names = [
            panel._table.item(row, 0).text()
            for row in range(panel._table.rowCount())
        ]
        assert names == ["document", "meadow"]

    def test_picking_a_row_activates_the_room(self, document):
        from mesh_xml.scene import Room

        panel = panels.RoomPanel(document)
        meadow = document.add_room(Room(name="meadow"))
        landed = []
        panel.activated.connect(lambda: landed.append(1))

        panel._table.selectRow(1)

        assert document.room is meadow
        assert landed == [1]

    def test_refresh_does_not_reactivate(self, document):
        from mesh_xml.scene import Room

        panel = panels.RoomPanel(document)
        document.add_room(Room(name="meadow"))
        landed = []
        panel.activated.connect(lambda: landed.append(1))

        panel.refresh()

        assert landed == [], "re-rendering the table is not a user pick"
        assert document.room is document.scene.rooms[0]

    def test_renaming_through_the_cell_renames_room_and_body(self, document):
        panel = panels.RoomPanel(document)

        panel._table.item(0, 0).setText("den")

        assert document.room.name == "den"
        assert document.room.body.name == "den"
        assert undo_depth(document) == 0, "renames are file-grade, not commands"

    def test_the_abstract_checkbox_parks_a_room(self, document):
        from mesh_xml.scene import Room

        panel = panels.RoomPanel(document)
        meadow = document.add_room(Room(name="meadow"))

        panel._table.item(1, 2).setCheckState(Qt.CheckState.Checked)
        assert meadow.abstracted

        panel._table.item(1, 2).setCheckState(Qt.CheckState.Unchecked)
        assert not meadow.abstracted

    def test_the_file_column_stars_unsaved_changes(self, document):
        panel = panels.RoomPanel(document)
        assert panel._table.item(0, 3).text() == "—"

        document.do(SetRadius(document.body.chunk_ids()[0], 0.5))
        assert panel._table.item(0, 3).text() == "*unsaved"

    def test_remove_stays_disabled_for_the_last_room(self, document):
        from mesh_xml.scene import Room

        panel = panels.RoomPanel(document)
        assert not panel._remove.isEnabled()

        document.add_room(Room(name="meadow"))
        assert panel._remove.isEnabled()

    def test_door_rows_follow_the_active_room(self, document):
        from mesh_xml.scene import Room

        panel = panels.RoomPanel(document)
        meadow = document.add_room(Room(name="meadow"))
        document.add_door(document.room)
        assert panel._doors.rowCount() == 1

        document.set_active_room(meadow)
        assert panel._doors.rowCount() == 0

    def test_editing_a_door_cell_rewrites_the_door(self, document):
        from mesh_xml.scene import Room

        panel = panels.RoomPanel(document)
        document.add_room(Room(name="meadow"))
        document.add_door(document.room)

        panel._doors.item(0, 3).setText("west")
        assert document.room.doors[0].to_door == "west"
        assert undo_depth(document) == 0, "door edits are file-grade"

    def test_a_bad_door_edit_snaps_back_to_the_truth(self, document):
        panel = panels.RoomPanel(document)
        document.add_door(document.room)

        panel._doors.item(0, 4).setText("not-a-number")
        assert document.room.doors[0].radius == pytest.approx(0.9)
        assert panel._doors.item(0, 4).text() == "0.9"

    def test_the_door_buttons_add_and_remove(self, document):
        panel = panels.RoomPanel(document)

        panel._add_door.click()
        assert len(document.room.doors) == 1

        panel._doors.setCurrentCell(0, 0)
        panel._remove_door.click()
        assert document.room.doors == []

    def test_the_status_column_surfaces_closed_doors(self, document):
        panel = panels.RoomPanel(document)
        document.add_door(document.room)  # a lone room's door aims home
        assert "closed" in panel._doors.item(0, 5).text()
