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
