"""Tests for the radius-curve editor.

Four claims here are worth more than the rest of the file.

**The view is the only pixel conversion, and it is invertible.** The widget this
panel replaces had its pixel-to-radius mapping written out eight times and its
renderer and sampler consequently disagreed. A round trip through
``_CurveView`` is the cheapest possible proof that there is one map here and that
dragging does not creep.

**Handles are rigid with their key.** ``Curve1D`` guarantees it and tests it; what
is tested here is that the *panel* does not break it, which is a real risk
because a panel that re-derived handles after a move would look correct in the
plot and be wrong in the file.

**One apply is one undo entry.** A twenty-chunk chain must not leave twenty
commands on the stack, because undo is exactly the tool an author reaches for
after a curve apply.

**The latch is load bearing.** Written as a comparison, in the style of
``test_panels.py``: the real panel and one with its latch defeated are driven
identically, and the assertion is that the real one keeps the value the drag
authored while the defeated one rounds it away through its own spin box. Without
the contrast a passing test proves only that nothing happened.
"""

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QLabel, QWidget

from mesh_xml.core import Body
from mesh_xml.core.curve import Curve1D, Key
from mesh_xml.editor.document import AddChunk, Document

context = pytest.importorskip("mesh_xml.render.gl.context")

try:
    APPLICATION = context.ensure_application()
except Exception as exc:  # pragma: no cover - headless box with no platform plugin
    pytest.skip(f"no Qt platform available: {exc}", allow_module_level=True)

from mesh_xml.editor import curve_panel  # noqa: E402 - needs the QApplication above

LEGACY_RIG = "examples/chubby_worm.xml"

#: A y a three-decimal spin box cannot hold. Every echo test in this file turns
#: on the panel refusing to write its own rounded rendering back into the model.
FINE_Y = 0.612345


class _UnguardedCurvePanel(curve_panel.CurvePanel):
    """:class:`CurvePanel` with the latch defeated, to prove the latch is load bearing."""

    @property
    def syncing(self) -> bool:
        return False


@pytest.fixture
def make_document():
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


@pytest.fixture
def panel(document) -> curve_panel.CurvePanel:
    built = curve_panel.CurvePanel(document)
    built.resize(360, 620)
    return built


def undo_depth(document: Document) -> int:
    return len(document._undo)


def label_texts(widget: QWidget) -> list[str]:
    return [label.text() for label in widget.findChildren(QLabel)]


# ----------------------------------------------------------------------
# mouse plumbing, borrowed from tests/test_viewport.py
# ----------------------------------------------------------------------


def press(view, point, button=Qt.MouseButton.LeftButton):
    view.mousePressEvent(_event(QMouseEvent.Type.MouseButtonPress, point, button))


def move(view, point, button=Qt.MouseButton.LeftButton):
    view.mouseMoveEvent(
        _event(QMouseEvent.Type.MouseMove, point, Qt.MouseButton.NoButton, button)
    )


def release(view, point, button=Qt.MouseButton.LeftButton):
    view.mouseReleaseEvent(_event(QMouseEvent.Type.MouseButtonRelease, point, button))


def _event(kind, point, button, buttons=None):
    point = QPointF(point)
    return QMouseEvent(
        kind,
        point,
        point,
        button,
        buttons if buttons is not None else button,
        Qt.KeyboardModifier.NoModifier,
    )


def drag(canvas, start: QPointF, end: QPointF) -> None:
    press(canvas, start)
    move(canvas, end)
    release(canvas, end)


def key_pixel(canvas, index: int) -> QPointF:
    key = canvas.curve.keys[index]
    return canvas.view().to_pixel(key.x, key.y)


def drag_key_to(canvas, index: int, x: float, y: float) -> None:
    """Pick a key up by its own centre and drop it on a curve-space point."""
    drag(canvas, key_pixel(canvas, index), canvas.view().to_pixel(x, y))


# ----------------------------------------------------------------------
# the view: the one place curve space meets pixels
# ----------------------------------------------------------------------


class TestCurveView:
    def test_a_point_survives_a_round_trip_through_pixels(self):
        view = curve_panel._CurveView(400.0, 300.0, 0.0, 1.0, 0.1, 0.9)
        for x, y in ((0.0, 0.1), (0.25, 0.42), (1.0, 0.9)):
            pixel = view.to_pixel(x, y)
            assert view.to_curve(pixel.x(), pixel.y()) == pytest.approx((x, y))

    def test_y_grows_upward_on_screen(self):
        view = curve_panel._CurveView(400.0, 300.0, 0.0, 1.0, 0.0, 1.0)
        assert view.to_pixel(0.5, 1.0).y() < view.to_pixel(0.5, 0.0).y()

    def test_the_corners_land_on_the_plot_rectangle(self):
        view = curve_panel._CurveView(400.0, 300.0, 0.0, 1.0, 0.0, 2.0)
        plot = view.plot()
        assert view.to_pixel(0.0, 2.0).x() == pytest.approx(plot.left())
        assert view.to_pixel(0.0, 2.0).y() == pytest.approx(plot.top())
        assert view.to_pixel(1.0, 0.0).x() == pytest.approx(plot.right())
        assert view.to_pixel(1.0, 0.0).y() == pytest.approx(plot.bottom())

    def test_a_collapsed_widget_does_not_divide_by_zero(self):
        """Qt delivers zero-size widgets; a paint callback is a terrible place to learn."""
        view = curve_panel._CurveView(0.0, 0.0, 0.5, 0.5, 1.0, 1.0)
        assert np.isfinite(view.to_pixel(0.5, 1.0).x())
        assert all(np.isfinite(v) for v in view.to_curve(0.0, 0.0))


# ----------------------------------------------------------------------
# the canvas
# ----------------------------------------------------------------------


class TestCanvasDragging:
    def test_dragging_a_key_updates_the_model(self, panel):
        canvas = panel.canvas
        drag_key_to(canvas, 1, 0.8, 0.55)

        moved = canvas.curve.keys[1]
        assert (moved.x, moved.y) == pytest.approx((0.8, 0.55))

    def test_dragging_a_key_carries_its_handles_rigidly(self, panel):
        """Curve1D promises this; the panel is what could break it."""
        curve = Curve1D(
            [Key(0.0, 0.3, out_handle=(0.15, 0.5)), Key(1.0, 0.3)],
            y_range=curve_panel.Y_RANGE,
        )
        panel.set_curve(curve)
        canvas = panel.canvas
        before = curve.keys[0]

        drag_key_to(canvas, 0, 0.2, 0.7)

        after = curve.keys[0]
        assert (after.x, after.y) == pytest.approx((0.2, 0.7))
        assert after.out_offset == pytest.approx(before.out_offset)
        assert after.out_handle == pytest.approx((0.35, 0.9))

    def test_a_drag_announces_itself(self, panel):
        seen = []
        panel.curve_changed.connect(lambda: seen.append(1))
        drag_key_to(panel.canvas, 1, 0.7, 0.5)
        assert seen

    def test_clicking_a_key_selects_it(self, panel):
        canvas = panel.canvas
        assert canvas.selected == 0
        press(canvas, key_pixel(canvas, 1))
        release(canvas, key_pixel(canvas, 1))
        assert canvas.selected == 1

    def test_clicking_empty_space_grabs_nothing(self, panel):
        canvas = panel.canvas
        before = canvas.curve.keys
        far = canvas.view().to_pixel(0.5, curve_panel.MAX_RADIUS)

        drag(canvas, far, canvas.view().to_pixel(0.5, 0.5))
        assert canvas.curve.keys == before

    def test_dragging_a_key_past_a_neighbour_keeps_hold_of_it(self, panel):
        """move_key re-sorts; a drag holding the old index would swap victims."""
        panel._on_add()
        canvas = panel.canvas
        assert len(canvas.curve) == 3
        canvas.select(1)

        press(canvas, key_pixel(canvas, 1))
        move(canvas, canvas.view().to_pixel(1.5, 0.9))
        move(canvas, canvas.view().to_pixel(1.6, 0.95))
        release(canvas, canvas.view().to_pixel(1.6, 0.95))

        assert canvas.selected == 2
        last = canvas.curve.keys[-1]
        assert (last.x, last.y) == pytest.approx((1.6, 0.95))
        assert [k.x for k in canvas.curve.keys] == sorted(
            k.x for k in canvas.curve.keys
        )

    def test_a_drag_is_clamped_into_the_codomain(self, panel):
        """A spin box cannot show a value its own range forbids."""
        canvas = panel.canvas
        drag_key_to(canvas, 1, 0.5, -5.0)
        assert canvas.curve.keys[1].y == pytest.approx(curve_panel.MIN_RADIUS)

        drag_key_to(canvas, 1, 40.0, 500.0)
        moved = canvas.curve.keys[-1]
        assert moved.x == pytest.approx(curve_panel.X_LIMIT)
        assert moved.y == pytest.approx(curve_panel.MAX_RADIUS)

    def test_a_handle_belongs_to_the_selected_key_and_drags_alone(self, panel):
        canvas = panel.canvas
        canvas.select(0)
        handle = canvas.handles(0)[curve_panel._Part.OUT]
        before = canvas.curve.keys[0]

        drag(canvas, canvas.view().to_pixel(*handle), canvas.view().to_pixel(0.2, 0.8))

        after = canvas.curve.keys[0]
        assert (after.x, after.y) == pytest.approx((before.x, before.y))
        assert after.out_handle == pytest.approx((0.2, 0.8))


class TestCanvasHandles:
    def test_handles_come_from_the_curve_s_own_control_points(self, panel):
        canvas = panel.canvas
        p0, p1, p2, p3 = canvas.curve.segment_control_points(0)
        assert canvas.handles(0)[curve_panel._Part.OUT] == pytest.approx(p1)
        assert canvas.handles(1)[curve_panel._Part.IN] == pytest.approx(p2)

    def test_the_outer_handle_of_a_terminal_key_is_absent(self, panel):
        """An auto handle there sits on the key, where it has no effect and no target."""
        canvas = panel.canvas
        assert curve_panel._Part.IN not in canvas.handles(0)
        assert curve_panel._Part.OUT not in canvas.handles(len(canvas.curve) - 1)


class TestCanvasPainting:
    def test_the_canvas_paints(self, panel):
        assert not panel.canvas.grab().isNull()

    def test_a_one_key_curve_paints(self, panel):
        panel.set_curve(Curve1D.constant(0.4, y_range=curve_panel.Y_RANGE))
        assert not panel.canvas.grab().isNull()

    def test_an_empty_curve_paints(self, panel):
        panel.set_curve(Curve1D(y_range=curve_panel.Y_RANGE))
        assert not panel.canvas.grab().isNull()

    def test_fitting_frames_every_key_and_handle(self, panel):
        panel.set_curve(
            Curve1D(
                [Key(0.0, 0.3, out_handle=(0.1, 2.5)), Key(1.0, 0.3)],
                y_range=curve_panel.Y_RANGE,
            )
        )
        _, _, y0, y1 = panel.canvas.window
        assert y0 < 0.3
        assert y1 > 2.5


# ----------------------------------------------------------------------
# the panel
# ----------------------------------------------------------------------


class TestChainAndSampling:
    def test_the_chain_is_the_whole_rig_when_nothing_is_selected(self, panel, document):
        assert panel.target_ids() == document.body.chunk_ids()

    def test_the_chain_is_the_selection_in_table_order(self, panel, document):
        ids = document.body.chunk_ids()
        document.set_selection([ids[3], ids[1]])
        assert panel.target_ids() == [ids[1], ids[3]]

    def test_positions_are_normalised_index_not_arc_length(self, panel):
        assert panel.sample_positions() == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])

    def test_a_one_chunk_chain_samples_at_the_start(self, panel, document):
        document.set_selection([document.body.chunk_ids()[2]])
        assert panel.sample_positions() == pytest.approx([0.0])

    def test_an_empty_rig_samples_nowhere(self):
        empty = curve_panel.CurvePanel(Document())
        assert len(empty.sample_positions()) == 0
        assert len(empty.sampled_radii()) == 0

    def test_sampling_matches_curve1d_exactly(self, panel):
        panel.canvas.select(1)
        drag_key_to(panel.canvas, 1, 1.0, 0.9)

        positions = panel.sample_positions()
        assert panel.sampled_radii().tolist() == panel.curve.sample_many(
            positions
        ).tolist()
        # The scalar path too: Curve1D promises the two agree bit for bit, and
        # the panel must not be the thing that introduces a second solve.
        assert panel.sampled_radii().tolist() == [
            panel.curve.sample(float(x)) for x in positions
        ]


class TestApply:
    def test_applying_writes_the_curve_into_the_chain(self, panel, document):
        drag_key_to(panel.canvas, 1, 1.0, 0.9)
        panel._apply.click()

        ids = document.body.chunk_ids()
        expected = panel.curve.sample_many(np.linspace(0.0, 1.0, len(ids)))
        for chunk_id, radius in zip(ids, expected):
            assert document.body.chunk(chunk_id).radius == pytest.approx(
                float(radius), abs=1e-6
            )

    def test_applying_is_exactly_one_undo_entry(self, panel, document):
        drag_key_to(panel.canvas, 1, 1.0, 0.9)
        panel._apply.click()

        assert undo_depth(document) == 1
        assert document.undo_text == "Radius Curve"

    def test_undo_puts_every_radius_back_in_one_step(self, panel, document):
        ids = document.body.chunk_ids()
        before = [document.body.chunk(c).radius for c in ids]

        drag_key_to(panel.canvas, 1, 1.0, 0.9)
        panel._apply.click()
        document.undo()

        assert [document.body.chunk(c).radius for c in ids] == pytest.approx(before)
        assert undo_depth(document) == 0

    def test_applying_only_the_selection_leaves_the_rest_alone(self, panel, document):
        ids = document.body.chunk_ids()
        untouched = document.body.chunk(ids[3]).radius
        document.set_selection([ids[0], ids[1]])

        drag_key_to(panel.canvas, 1, 1.0, 0.9)
        panel._apply.click()

        assert document.body.chunk(ids[3]).radius == pytest.approx(untouched)
        assert document.body.chunk(ids[1]).radius == pytest.approx(0.9, abs=1e-6)

    def test_a_no_op_apply_pushes_nothing(self, panel, document):
        """Re-applying an unchanged curve must not bury the stack in self-undos."""
        panel._apply.click()
        depth = undo_depth(document)
        panel._apply.click()
        assert undo_depth(document) == depth

    def test_applying_to_an_empty_rig_does_nothing(self):
        empty_document = Document()
        empty = curve_panel.CurvePanel(empty_document)
        assert not empty.apply_to_chain()
        assert undo_depth(empty_document) == 0

    def test_the_curve_never_writes_a_non_positive_radius(self, panel, document):
        """Body.set_radius refuses one outright, so the clamp is load bearing."""
        drag_key_to(panel.canvas, 0, 0.0, -10.0)
        drag_key_to(panel.canvas, 1, 1.0, -10.0)
        panel._apply.click()

        for chunk_id in document.body.chunk_ids():
            assert document.body.chunk(chunk_id).radius > 0.0


class TestKeys:
    def test_adding_a_key_splits_without_moving_the_shape(self, panel):
        drag_key_to(panel.canvas, 1, 1.0, 0.9)
        before = panel.curve.sample_many(np.linspace(0.0, 1.0, 17))

        panel._add.click()

        assert len(panel.curve) == 3
        assert panel.curve.sample_many(np.linspace(0.0, 1.0, 17)) == pytest.approx(
            before
        )

    def test_a_new_key_becomes_the_selected_one(self, panel):
        panel._add.click()
        assert panel.canvas.selected == 1

    def test_removing_a_key_drops_it(self, panel):
        panel._add.click()
        panel._remove.click()
        assert len(panel.curve) == 2

    def test_the_curve_cannot_be_reduced_below_two_keys(self, panel):
        assert not panel._remove.isEnabled()
        panel._on_remove()
        assert len(panel.curve) == curve_panel.MIN_KEYS

    def test_the_spin_boxes_move_the_selected_key(self, panel):
        panel.canvas.select(1)
        panel._key_y.setValue(0.75)
        assert panel.curve.keys[1].y == pytest.approx(0.75)

    def test_editing_one_axis_does_not_round_the_other(self, panel):
        """The spin shows three decimals; a key does not, and only one was touched."""
        drag_key_to(panel.canvas, 1, 1.0, FINE_Y)
        panel.canvas.select(1)

        panel._key_x.setValue(0.75)

        assert panel.curve.keys[1].x == pytest.approx(0.75)
        assert panel.curve.keys[1].y == FINE_Y


class TestReadout:
    def test_the_readout_is_the_curve_s_own_sample(self, panel):
        drag_key_to(panel.canvas, 1, 1.0, 0.9)
        panel._sample_x.setValue(0.5)
        assert panel._sample_y.text() == format(panel.curve.sample(0.5), ".4f")

    def test_the_readout_follows_a_drag(self, panel):
        panel._sample_x.setValue(1.0)
        drag_key_to(panel.canvas, 1, 1.0, 0.9)
        assert panel._sample_y.text() == format(0.9, ".4f")

    def test_the_target_count_follows_the_selection(self, panel, document):
        assert "4 chunk(s), whole rig" in panel._target.text()
        document.set_selection(document.body.chunk_ids()[:2])
        assert "2 chunk(s), selected" in panel._target.text()

    def test_the_note_states_the_x_domain(self, panel):
        texts = " ".join(label_texts(panel))
        assert "normalised position" in texts
        assert "arc length" in texts


class TestDocumentChanges:
    def test_a_new_chunk_reaches_the_chain(self, panel, document):
        document.do(AddChunk("tip", (9.0, 0.0, 0.0)))
        assert len(panel.sample_positions()) == document.body.n_chunks

    def test_a_replaced_body_relevels_the_curve(self, panel, document):
        drag_key_to(panel.canvas, 1, 1.0, 0.9)
        document.open(LEGACY_RIG)

        levels = {key.y for key in panel.curve.keys}
        assert len(levels) == 1
        assert 0.9 not in levels

    def test_the_panel_survives_a_body_replacement(self, panel, document):
        document.open(LEGACY_RIG)
        assert panel.target_ids() == document.body.chunk_ids()
        assert len(panel.sampled_radii()) == document.body.n_chunks

    def test_resetting_levels_at_the_chain_mean(self, panel, document):
        drag_key_to(panel.canvas, 1, 1.0, 0.9)
        panel._reset.click()

        ids = document.body.chunk_ids()
        mean = float(np.mean([document.body.chunk(c).radius for c in ids]))
        assert panel.curve.sample(0.5) == pytest.approx(mean)


# ----------------------------------------------------------------------
# the latch
# ----------------------------------------------------------------------


class TestReentrancy:
    def test_the_latch_is_held_across_a_refresh_and_released_after(self, panel):
        seen: list[bool] = []
        original = panel._refresh

        def spy() -> None:
            seen.append(panel.syncing)
            original()

        panel._refresh = spy
        panel.refresh()

        assert seen == [True]
        assert not panel.syncing

    def test_a_dragged_value_is_not_rounded_away_by_its_own_spin_box(
        self, make_document
    ):
        """Refresh writes the key into a three-decimal box; the echo would write back."""
        results = []
        alive = []
        for panel_class in (curve_panel.CurvePanel, _UnguardedCurvePanel):
            built = panel_class(make_document())
            alive.append(built)
            built.resize(360, 620)
            built.canvas.select(1)
            drag_key_to(built.canvas, 1, 1.0, FINE_Y)
            results.append(built.curve.keys[1].y)

        guarded, unguarded = results
        assert guarded == FINE_Y
        assert unguarded != FINE_Y, "the comparison is not exercising the latch"
        assert unguarded == pytest.approx(round(FINE_Y, 3))

    def test_a_selection_change_does_not_reshape_the_curve(self, make_document):
        """Selecting a chunk refreshes the readout; nothing about it is a curve edit."""
        results = []
        alive = []
        for panel_class in (curve_panel.CurvePanel, _UnguardedCurvePanel):
            document = make_document()
            built = panel_class(document)
            alive.append(built)
            built.resize(360, 620)
            built.canvas.select(1)
            drag_key_to(built.canvas, 1, 1.0, FINE_Y)
            document.set_selection(document.body.chunk_ids()[:2])
            results.append(built.curve.keys[1].y)

        guarded, unguarded = results
        assert guarded == FINE_Y
        assert unguarded != FINE_Y, "the comparison is not exercising the latch"

    def test_constructing_the_panel_edits_nothing(self, document):
        built = curve_panel.CurvePanel(document)
        assert built.curve is not None
        assert undo_depth(document) == 0
        assert not document.dirty

    def test_the_panel_leaves_the_latch_released_after_an_apply(self, panel):
        drag_key_to(panel.canvas, 1, 1.0, 0.9)
        panel._apply.click()
        assert not panel.syncing
