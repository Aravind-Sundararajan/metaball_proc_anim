"""The application entry point.

Assembly only: a window, menus, and the wiring between the document and the
views. Anything with behaviour worth testing lives in :mod:`document`,
:mod:`viewport` or :mod:`panels` instead, because a main window is the one place
in a Qt app that is genuinely awkward to test and therefore the worst place to
put logic.

The one thing that cannot move elsewhere is the surface-format call. It must run
before the ``QApplication`` is constructed or the context comes up as
compatibility profile, and the failure mode is a shader compile error blaming
the shader rather than the setup.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QToolBar,
)

from mesh_xml.editor import icons
from mesh_xml.editor.document import Document, RemoveChunk
from mesh_xml.editor.panels import build_panels
from mesh_xml.editor.rail import Inspector
from mesh_xml.editor.viewport import MetaballViewport
from mesh_xml.render.gl.context import configure_surface_format, ensure_application

__all__ = ["MainWindow", "main"]

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


class MainWindow(QMainWindow):
    def __init__(self, document: Document | None = None) -> None:
        super().__init__()
        self.document = document if document is not None else Document()
        self.viewport = MetaballViewport(self.document)
        self.setCentralWidget(self.viewport)

        self._docks: dict[str, QDockWidget] = {}
        self._build_docks()
        self._build_shared_actions()
        self._build_menus()
        self._build_toolbar()

        self.document.status.connect(self.statusBar().showMessage)
        self.document.dirty_changed.connect(lambda _: self._refresh_title())
        self.document.body_replaced.connect(self._refresh_title)
        self.document.history_changed.connect(self._refresh_history)

        self.resize(1500, 940)
        self._refresh_title()
        self._refresh_history()
        self.statusBar().showMessage("Ready")

    def _build_docks(self) -> None:
        """Two docks: the scene on the left, everything else in one inspector.

        The scene map and the thing being inspected are different questions —
        *which room am I in* against *what is this made of* — so the rooms keep
        their own edge, the way a hierarchy and an inspector sit apart in every
        editor that has both. Everything else goes into one stack behind an
        icon rail (:mod:`mesh_xml.editor.rail`), because eleven panels in a
        tabified dock is eleven panels you cannot find.
        """
        self.panels = build_panels(self.document)

        rooms = self.panels.get("Rooms")
        if rooms is not None:
            dock = QDockWidget("Scene", self)
            dock.setWidget(rooms)
            dock.setObjectName("dock_scene")
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
            self._docks["Rooms"] = dock

        self.inspector = Inspector(
            {t: w for t, w in self.panels.items() if t != "Rooms"}, self
        )
        dock = QDockWidget("Inspector", self)
        dock.setWidget(self.inspector)
        dock.setObjectName("dock_inspector")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._docks["Inspector"] = dock

        # The render is the product, so it gets the room. Left to itself Qt
        # hands each dock its widest child's hint, which put the viewport at
        # under half the window while a one-room table and a column of
        # four-character spin boxes took the rest. These are the floor of what
        # the docks stay legible at — narrower and the Scene table starts
        # dropping columns behind a scrollbar.
        self.resizeDocks(
            list(self._docks.values()),
            [280 if t == "Rooms" else 420 for t in self._docks],
            Qt.Orientation.Horizontal,
        )

        self._wire_panels()

    def _wire_panels(self) -> None:
        """Connect the panel signals that address the viewport rather than the document."""
        surface = self.panels.get("Surface")
        if surface is not None:
            surface.overlay_changed.connect(self._set_rig_overlay)
            surface.orthographic_changed.connect(self._set_orthographic)
            # The panel is the checkbox's owner, so the viewport follows it
            # rather than the two starting out disagreeing.
            surface.set_show_rig_overlay(self.viewport.show_rig)
            surface.set_orthographic(self.viewport.camera.orthographic)

        playback = self.panels.get("Playback")
        if playback is not None:
            playback.stepped.connect(self.viewport.update)

        rooms = self.panels.get("Rooms")
        if rooms is not None:
            # No frame wiring here: the viewport carries the camera across
            # the cut itself — restoring where the author left a room, and
            # framing only a first entry, which is never pointed at nothing.
            rooms.open_requested.connect(self.open_into_scene)
            rooms.remove_requested.connect(self.remove_active_room)

        creatures = self.panels.get("Creatures")
        if creatures is not None:
            # The one cut that does ask for a frame: a move's proof is the
            # arrival on screen, and the camera is the window's resource,
            # not the dock's.
            creatures.moved.connect(self.viewport.frame_all)

    def _set_rig_overlay(self, visible: bool) -> None:
        self.viewport.show_rig = visible

    def _set_orthographic(self, orthographic: bool) -> None:
        """Switch projection and mark the frame dirty.

        The emit is load-bearing. The camera uniforms are uploaded every frame,
        but nothing marks a frame dirty on a projection change, and the viewport
        idles when nothing has moved — so without this the toggle would sit
        invisible until something unrelated happened to invalidate a frame.
        """
        self.viewport.camera.orthographic = orthographic
        self.viewport.camera_changed.emit()

    def _build_shared_actions(self) -> None:
        """The gestures both the menus and the toolbar offer: one QAction
        each, so shortcut, enabled state and checked state have a single
        owner and the two surfaces cannot disagree.
        """
        document = self.document

        self.play_action = QAction(icons.icon("play"), "Play", self)
        self.play_action.setCheckable(True)
        self.play_action.setShortcut(QKeySequence(Qt.Key.Key_Space))
        self.play_action.setToolTip("Play or pause the scene (Space)")
        self.play_action.triggered.connect(
            lambda checked=False: document.set_running(bool(checked))
        )
        document.running_changed.connect(self._sync_play_action)

        self.step_action = QAction(icons.icon("step"), "Step", self)
        self.step_action.setToolTip(
            "Advance one fixed tick while paused; the clock is not consumed"
        )
        self.step_action.triggered.connect(lambda checked=False: self.step_once())
        document.running_changed.connect(
            lambda running: self.step_action.setEnabled(not running)
        )

        self.rest_action = QAction(icons.icon("rest"), "Rest Pose", self)
        self.rest_action.setToolTip(
            "Return every chunk to the pose captured when playback last stopped"
        )
        self.rest_action.triggered.connect(
            lambda checked=False: document.reset_pose()
        )

        self.add_action = QAction(icons.icon("add"), "Add Chunk", self)
        self.add_action.setShortcut(QKeySequence("A"))
        self.add_action.setToolTip("Grow a chunk out of the selection (A)")
        self.add_action.triggered.connect(lambda checked=False: self.add_chunk())

        self.autorig_action = QAction(icons.icon("autorig"), "Fit Autorig", self)
        self.autorig_action.setToolTip(
            "Fit a spine, an idle breath and a wave driver to the body — "
            "the shipped examples' recipe, one undo step"
        )
        self.autorig_action.triggered.connect(
            lambda checked=False: document.fit_autorig()
        )

        self.rig_legs_action = QAction(icons.icon("legs"), "Rig Legs", self)
        self.rig_legs_action.setToolTip(
            "Compile the measured walker onto every leg-shaped appendage — "
            "three chunks grown from the spine, ending near the floor"
        )
        self.rig_legs_action.triggered.connect(
            lambda checked=False: document.rig_legs()
        )

        self.frame_action = QAction(icons.icon("frame"), "Frame All", self)
        self.frame_action.setShortcut(QKeySequence("F"))
        self.frame_action.setToolTip("Fit the camera to the creature (F)")
        self.frame_action.triggered.connect(
            lambda checked=False: self.viewport.frame_all()
        )

        # Registered on the window as well, so the shortcuts fire while
        # focus sits in the viewport — where it lives most of the time.
        for action in (
            self.play_action,
            self.step_action,
            self.rest_action,
            self.add_action,
            self.autorig_action,
            self.rig_legs_action,
            self.frame_action,
        ):
            self.addAction(action)

    def _sync_play_action(self, running: bool) -> None:
        self.play_action.setChecked(running)
        # Word and glyph both, because a transport button that says one state
        # while drawing the other is the classic play/pause ambiguity.
        self.play_action.setText("Pause" if running else "Play")
        self.play_action.setIcon(icons.icon("pause" if running else "play"))

    def step_once(self) -> None:
        """One paused tick from the window's own surfaces.

        Delegates to the document like the Playback panel does, and buys
        the repaint itself: stepping emits no document signal, so without
        the update the tick happens invisibly.
        """
        if self.document.running:
            return
        self.document.step_once()
        self.viewport.update()

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Transport", self)
        toolbar.setObjectName("toolbar_main")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        # Icon *beside* text, not instead of it. These are domain verbs — "Fit
        # Autorig", "Rig Legs" — that no glyph teaches on its own; the icon
        # makes the row scannable and the word keeps it unambiguous. The rail
        # is where icons stand alone, because there the header names the one
        # that matters.
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toolbar.setIconSize(QSize(15, 15))
        for action in (self.play_action, self.step_action, self.rest_action):
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addAction(self.add_action)
        toolbar.addSeparator()
        for action in (self.autorig_action, self.rig_legs_action):
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addAction(self.frame_action)
        self.addToolBar(toolbar)
        self.toolbar = toolbar

    def _build_menus(self) -> None:
        #: Menus by name, the docks-dict precedent: assembly the window can
        #: answer questions about without a traversal of Qt internals.
        self.menus: dict[str, object] = {}
        file_menu = self.menuBar().addMenu("&File")
        self.menus["file"] = file_menu
        self._action(file_menu, "&New", QKeySequence.StandardKey.New, self.new)
        self._action(file_menu, "&Open…", QKeySequence.StandardKey.Open, self.open)
        self._action(file_menu, "Open &Into Scene…", None, self.open_into_scene)
        self._action(file_menu, "Open S&cene…", None, self.open_scene)
        self._build_examples_menu(file_menu)
        file_menu.addSeparator()
        self._action(file_menu, "&Save", QKeySequence.StandardKey.Save, self.save)
        self._action(file_menu, "Save &As…", QKeySequence.StandardKey.SaveAs, self.save_as)
        self._action(file_menu, "Save A&ll", None, self.save_all)
        self._action(file_menu, "Save Scene &As…", None, self.save_scene_as)
        self._action(file_menu, "Export &Mesh…", None, self.export_mesh)
        file_menu.addSeparator()
        self._action(file_menu, "E&xit", QKeySequence.StandardKey.Quit, self.close)

        edit_menu = self.menuBar().addMenu("&Edit")
        self.menus["edit"] = edit_menu
        self.undo_action = self._action(
            edit_menu, "&Undo", QKeySequence.StandardKey.Undo, self.document.undo
        )
        self.redo_action = self._action(
            edit_menu, "&Redo", QKeySequence.StandardKey.Redo, self.document.redo
        )
        edit_menu.addSeparator()
        edit_menu.addAction(self.add_action)
        self._action(edit_menu, "&Delete Selected", QKeySequence.StandardKey.Delete, self.delete_selected)
        edit_menu.addSeparator()
        edit_menu.addAction(self.autorig_action)
        edit_menu.addAction(self.rig_legs_action)

        view_menu = self.menuBar().addMenu("&View")
        self.menus["view"] = view_menu
        view_menu.addAction(self.frame_action)
        self._action(view_menu, "Toggle &Rig Overlay", QKeySequence("R"), self.toggle_rig)
        self._action(view_menu, "&Orthographic", QKeySequence("O"), self.toggle_orthographic)
        view_menu.addAction(self.play_action)
        view_menu.addSeparator()
        # Blender's numbers: 1/3/7 for front/right/top, Ctrl for the opposite.
        for label, key, view in (
            ("Front", "1", "front"),
            ("Back", "Ctrl+1", "back"),
            ("Right", "3", "right"),
            ("Left", "Ctrl+3", "left"),
            ("Top", "7", "top"),
            ("Bottom", "Ctrl+7", "bottom"),
        ):
            self._action(
                view_menu, label, QKeySequence(key), lambda v=view: self.snap_view(v)
            )
        view_menu.addSeparator()
        for title, dock in self._docks.items():
            view_menu.addAction(dock.toggleViewAction())
            dock.toggleViewAction().setText(title)

    def _action(self, menu, text: str, shortcut, slot) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(lambda checked=False: slot())
        menu.addAction(action)
        # Also registered on the window so the shortcut fires while focus is in
        # the viewport, which is where it will be nearly all of the time.
        self.addAction(action)
        return action

    def _build_examples_menu(self, file_menu) -> None:
        examples = sorted(EXAMPLES.glob("*.xml")) if EXAMPLES.is_dir() else []
        if not examples:
            return
        submenu = file_menu.addMenu("Open &Example")
        for path in examples:
            action = QAction(path.stem.replace("_", " "), self)
            action.triggered.connect(lambda checked=False, p=path: self.open_path(p))
            submenu.addAction(action)

    # ------------------------------------------------------------------

    def snap_view(self, view: str) -> None:
        self.viewport.camera.snap_view(view)
        self.viewport.camera_changed.emit()

    def toggle_rig(self) -> None:
        visible = not self.viewport.show_rig
        surface = self.panels.get("Surface")
        if surface is not None:
            surface.set_show_rig_overlay(visible)
        else:
            self.viewport.show_rig = visible

    def toggle_orthographic(self) -> None:
        surface = self.panels.get("Surface")
        if surface is not None:
            surface.set_orthographic(not surface.orthographic)
        else:
            self._set_orthographic(not self.viewport.camera.orthographic)

    def add_chunk(self) -> None:
        self.document.create_chunk()

    def delete_selected(self) -> None:
        for chunk_id in sorted(self.document.selection):
            self.document.do(RemoveChunk(chunk_id))

    def new(self) -> None:
        if self._confirm_discard():
            self.document.new()

    def open(self) -> None:
        """Pick the file first, then ask the right-sized discard question:
        a rig replaces the active room, a manifest replaces every room."""
        from mesh_xml.io import scene_xml

        path, _ = QFileDialog.getOpenFileName(
            self, "Open Rig", str(EXAMPLES), "Rig files (*.xml);;All files (*)"
        )
        if not path:
            return
        if scene_xml.is_manifest(Path(path)):
            if not self._confirm_discard_all():
                return
        elif not self._confirm_discard():
            return
        self.open_path(Path(path))

    def open_path(self, path: Path) -> None:
        """Open a file for what it is: a manifest opens as a scene, a rig
        as a rig — the user picked a file, not a parser."""
        from mesh_xml.io import scene_xml

        try:
            if scene_xml.is_manifest(path):
                self.document.open_scene(path)
            else:
                self.document.open(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open Failed", f"{path.name}\n\n{exc}")
            return
        self.viewport.frame_all()

    def open_scene(self) -> None:
        """Open a scene manifest, replacing every room after the discard prompt."""
        if not self._confirm_discard_all():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Scene", str(EXAMPLES), "Scene files (*.xml);;All files (*)"
        )
        if not path:
            return
        try:
            self.document.open_scene(Path(path))
        except Exception as exc:
            QMessageBox.critical(
                self, "Open Failed", f"{Path(path).name}\n\n{exc}"
            )
            return
        self.viewport.frame_all()

    def save_scene_as(self) -> None:
        """Write the manifest and every room that owes the disk something."""
        suggested = str(Path.cwd() / f"{self.document.scene.name}.xml")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Scene", suggested, "Scene files (*.xml);;All files (*)"
        )
        if not path:
            return
        try:
            self.document.save_scene_as(Path(path))
        except Exception as exc:
            QMessageBox.critical(
                self, "Save Failed", f"{Path(path).name}\n\n{exc}"
            )

    def open_into_scene(self) -> None:
        """Open a file as a new room beside the others, and cut to it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Into Scene", str(EXAMPLES), "Rig files (*.xml);;All files (*)"
        )
        if not path:
            return
        try:
            self.document.open_as_room(Path(path))
        except Exception as exc:
            QMessageBox.critical(
                self, "Open Failed", f"{Path(path).name}\n\n{exc}"
            )

    def remove_active_room(self) -> None:
        """Remove the room being looked at, after the dirty question."""
        document = self.document
        room = document.room
        if len(document.scene.rooms) == 1:
            self.statusBar().showMessage("A document is at least one room")
            return
        if document.dirty:
            answer = QMessageBox.question(
                self,
                "Unsaved Changes",
                f"Room {room.name!r} has unsaved changes. Remove it anyway?",
                QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Discard:
                return
        document.remove_room(room)

    def save_all(self) -> None:
        """Save every dirty room — to its own file, or through Save As for
        one that never had a file, which activates so the dialog names what
        it is asking about."""
        for room in list(self.document.dirty_rooms()):
            path = self.document.room_path(room)
            if path is None:
                self.document.set_active_room(room)
                self.save_as()
            else:
                try:
                    self.document.save_room(room, path)
                except Exception as exc:
                    QMessageBox.critical(
                        self, "Save Failed", f"{path.name}\n\n{exc}"
                    )
                    return

    def save(self) -> None:
        if self.document.path is None:
            self.save_as()
            return
        self._save_to(self.document.path)

    def save_as(self) -> None:
        suggested = str(self.document.path or Path.cwd() / "rig.xml")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Rig", suggested, "Rig files (*.xml);;All files (*)"
        )
        if path:
            self._save_to(Path(path))

    def export_mesh(self) -> None:
        stem = self.document.path.stem if self.document.path else "rig"
        suggested = str((self.document.path or Path.cwd() / "x").parent / f"{stem}.obj")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Mesh", suggested, "Meshes (*.obj *.stl)"
        )
        if not path:
            return
        try:
            self.document.export_mesh(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"{Path(path).name}\n\n{exc}")

    def _save_to(self, path: Path) -> None:
        try:
            self.document.save(path)
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", f"{path.name}\n\n{exc}")
        self._refresh_title()

    def _confirm_discard(self) -> bool:
        """The active room's question — what New and Open actually replace."""
        if not self.document.dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved Changes",
            "This rig has unsaved changes. Discard them?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Discard

    def _confirm_discard_all(self) -> bool:
        """The whole scene's question — closing loses every room at once."""
        dirty = self.document.dirty_rooms()
        if not dirty:
            return True
        names = ", ".join(room.name for room in dirty)
        answer = QMessageBox.question(
            self,
            "Unsaved Changes",
            f"Unsaved changes in: {names}. Discard them?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Discard

    def closeEvent(self, event) -> None:
        if self._confirm_discard_all():
            event.accept()
        else:
            event.ignore()

    def _refresh_title(self) -> None:
        self.setWindowTitle(self.document.title)

    def _refresh_history(self) -> None:
        self.undo_action.setEnabled(self.document.can_undo)
        self.redo_action.setEnabled(self.document.can_redo)
        self.undo_action.setText(f"&Undo {self.document.undo_text}".rstrip())
        self.redo_action.setText(f"&Redo {self.document.redo_text}".rstrip())
        self._refresh_title()


def main(argv: list[str] | None = None) -> int:
    """Launch the editor. ``configure_surface_format`` runs before QApplication."""
    configure_surface_format()
    app = ensure_application(argv if argv is not None else sys.argv)
    # The entrypoint dresses the app; the library never does, so tests and
    # embedders meet stock widgets unless they opt in.
    from PySide6.QtGui import QIcon

    from mesh_xml.editor import theme

    theme.apply(app)
    # Two metaballs mid-blend, drawn by the CPU field itself — the one-glyph
    # summary of what the app computes. Regenerate with scripts/make_icon.py
    # if the accent ever moves.
    icon = Path(__file__).parent / "icon.png"
    if icon.is_file():
        app.setWindowIcon(QIcon(str(icon)))

    window = MainWindow()
    args = (argv if argv is not None else sys.argv)[1:]
    if args:
        window.open_path(Path(args[0]))
    else:
        default = EXAMPLES / "chubby_worm.xml"
        if default.is_file():
            window.open_path(default)

    window.show()
    window.viewport.frame_all()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
