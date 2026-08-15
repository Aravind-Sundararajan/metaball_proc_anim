"""Tests for the main window's shared actions and toolbar.

The window's own docstring calls it the worst place for logic, and the
toolbar keeps that promise: every action delegates to a document method
tested elsewhere. What is worth testing here is the *sharing* — one
QAction per gesture, owned once — because the failure mode of duplicated
actions is two surfaces drifting: a Play button that says Play while the
menu says Pause, a shortcut registered twice and firing ambiguously.
"""

from __future__ import annotations

import numpy as np
import pytest

from mesh_xml.core import Body
from mesh_xml.editor.document import Document

context = pytest.importorskip("mesh_xml.render.gl.context")

try:
    APPLICATION = context.ensure_application()
except Exception as exc:  # pragma: no cover - headless box with no platform plugin
    pytest.skip(f"no Qt platform available: {exc}", allow_module_level=True)

from mesh_xml.editor.app import MainWindow  # noqa: E402


@pytest.fixture
def window() -> MainWindow:
    document = Document()
    body = Body()
    ids = [
        body.add_chunk(f"c{i}", [float(i), 1.0, 0.0], radius=0.3) for i in range(5)
    ]
    for a, b in zip(ids, ids[1:]):
        body.add_link(a, b)
    document._replace(body, None)
    return MainWindow(document)


class TestToolbar:
    def test_the_toolbar_carries_the_core_gestures(self, window):
        texts = [action.text() for action in window.toolbar.actions()]
        for gesture in (
            "Play",
            "Step",
            "Rest Pose",
            "Add Chunk",
            "Fit Autorig",
            "Rig Legs",
            "Frame All",
        ):
            assert gesture in texts

    def test_menu_and_toolbar_share_one_play_action(self, window):
        """The point of the refactor: one owner, so the surfaces cannot drift."""
        assert window.play_action in window.menus["view"].actions()
        assert window.play_action in window.toolbar.actions()
        assert window.autorig_action in window.menus["edit"].actions()

    def test_play_action_and_document_stay_in_step(self, window):
        window.play_action.trigger()
        assert window.document.running
        assert window.play_action.text() == "Pause"
        assert not window.step_action.isEnabled(), "no stepping mid-run"

        window.document.set_running(False)
        assert not window.play_action.isChecked()
        assert window.play_action.text() == "Play"
        assert window.step_action.isEnabled()

    def test_step_ticks_a_paused_document_once(self, window):
        before = window.document.body.pos.copy()
        window.step_action.trigger()
        assert not np.array_equal(before, window.document.body.pos)

    def test_rest_action_restores_the_captured_pose(self, window):
        document = window.document
        target = document.body.chunk_ids()[3]
        original = document.body.chunk(target).pos

        # Deliberately not paused first: pausing captures a *new* rest pose
        # at the fallen position, which is the panel test's lesson too.
        document.set_running(True)
        document.advance(1.0)
        assert document.body.chunk(target).pos != pytest.approx(original)

        window.rest_action.trigger()
        assert document.body.chunk(target).pos == pytest.approx(original)

    def test_the_rig_actions_reach_the_document(self, window):
        assert not window.document.controllers.controllers
        window.autorig_action.trigger()
        assert window.document.controllers.controllers, (
            "Fit Autorig fits a stack from the menu surface too"
        )


class TestClockReadout:
    """The clock moved to the status bar when its panel stopped earning a slot.

    The Playback panel repeated Play, Step and Rest Pose as three buttons over
    a transport the window already owned as shared actions, and spent an
    inspector slot on it. The three numbers underneath were the only thing in
    it that was not a duplicate, so they went where a number that changes ten
    times a second belongs.
    """

    def test_the_readout_follows_the_clock(self, window):
        readout = window.clock_readout
        assert "0 ticks" in readout.text()

        window.document.set_running(True)
        window.document.advance(0.5)
        readout.refresh()

        assert window.document.clock.ticks > 0
        assert f"{window.document.clock.ticks} ticks" in readout.text()
        assert f"{window.document.clock.dropped} dropped" in readout.text()
        assert f"{window.document.clock.alpha:.3f}" in readout.text()

    def test_it_is_a_permanent_fixture_of_the_status_bar(self, window):
        """Permanent, so a transient status message cannot cover the clock."""
        assert window.clock_readout.parent() is not None
        window.statusBar().showMessage("Opened something.xml")
        assert not window.clock_readout.isHidden()

    def test_dropped_ticks_colour_themselves(self, window):
        """Zero is the quiet case; anything above it is the news."""
        from mesh_xml.editor import theme

        window.clock_readout.refresh()
        assert theme.DIM in window.clock_readout.styleSheet()

        window.document.clock.dropped = 3
        window.clock_readout.refresh()
        assert theme.ACCENT in window.clock_readout.styleSheet()

    def test_stepping_while_running_is_refused(self, window):
        """The guard the Playback panel used to own, now the window's."""
        window.document.set_running(True)
        before = window.document.body.pos.copy()

        window.step_once()

        assert np.array_equal(before, window.document.body.pos)
        assert not window.step_action.isEnabled()


class TestHintsToggle:
    """One toggle, reaching both docks.

    The Scene dock sits outside the inspector, and it is the panel carrying the
    two longest paragraphs — a hint toggle that reached the ten inspector
    panels but not that one would have left the worst of it on screen.
    """

    def test_hints_start_folded_and_the_toggle_is_off(self, window):
        from PySide6.QtWidgets import QLabel

        assert not window.hints_action.isChecked()
        hints = window.findChildren(QLabel, "hint")
        assert hints, "no panel registered any prose as a hint"
        assert all(h.isHidden() for h in hints)

    def test_the_action_reaches_both_docks(self, window):
        from PySide6.QtWidgets import QLabel

        window.hints_action.trigger()
        try:
            rooms = window.panels["Rooms"].findChildren(QLabel, "hint")
            solver = window.panels["Solver"].findChildren(QLabel, "hint")
            assert rooms and solver, "one of the two docks carries no hints"
            assert not any(h.isHidden() for h in rooms + solver)
        finally:
            window.hints_action.setChecked(False)

    def test_the_header_button_and_the_menu_stay_in_step(self, window):
        """Two surfaces for one preference is the drift this repo keeps fixing."""
        window.inspector._hints.click()
        assert window.hints_action.isChecked()

        window.hints_action.trigger()
        assert not window.inspector._hints.isChecked()

    def test_the_toggle_reports_what_it_reached(self, window):
        """"nothing happened" and "nothing to happen to" look identical."""
        reached = window.set_hints_visible(True)
        try:
            from PySide6.QtWidgets import QLabel

            assert reached == len(window.findChildren(QLabel, "hint"))
            assert reached > 0
        finally:
            window.set_hints_visible(False)


class TestIcon:
    def test_the_icon_asset_ships_and_loads(self):
        from pathlib import Path

        from PySide6.QtGui import QIcon

        from mesh_xml.editor import app as editor_app

        icon = Path(editor_app.__file__).parent / "icon.png"
        assert icon.is_file(), "the app icon ships beside the entrypoint"
        assert not QIcon(str(icon)).isNull()


class TestRoomCamera:
    """The dock's cut carries the camera: restore where the author left a
    room, frame only a first entry. Projection is deliberately absent here —
    it is window state, untouched by cuts."""

    @staticmethod
    def add_meadow(document: Document):
        from mesh_xml.scene import Room

        room = document.add_room(Room(name="meadow"))
        ids = [
            room.body.add_chunk(f"m{i}", (9.0 + i * 0.9, 1.0, 0.0), radius=0.3)
            for i in range(3)
        ]
        for a, b in zip(ids, ids[1:]):
            room.body.add_link(a, b)
        return room

    def test_the_dock_cut_restores_rather_than_frames(self, window):
        self.add_meadow(window.document)
        rooms = window.panels["Rooms"]
        camera = window.viewport.camera

        camera.yaw = 123.0
        camera.distance = 9.5
        left = camera.pose()

        rooms._table.selectRow(1)
        assert camera.pose() != left, "a first entry frames"
        rooms._table.selectRow(0)
        assert camera.pose() == left, "a re-entry restores"
