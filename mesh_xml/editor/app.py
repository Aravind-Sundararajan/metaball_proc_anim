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

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QDockWidget, QFileDialog, QMainWindow, QMessageBox

from mesh_xml.editor.document import Document, RemoveChunk
from mesh_xml.editor.panels import build_panels
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
        self._build_menus()

        self.document.status.connect(self.statusBar().showMessage)
        self.document.dirty_changed.connect(lambda _: self._refresh_title())
        self.document.body_replaced.connect(self._refresh_title)
        self.document.history_changed.connect(self._refresh_history)

        self.resize(1500, 940)
        self._refresh_title()
        self._refresh_history()
        self.statusBar().showMessage("Ready")

    def _build_docks(self) -> None:
        self.panels = build_panels(self.document)
        for title, widget in self.panels.items():
            dock = QDockWidget(title, self)
            dock.setWidget(widget)
            dock.setObjectName(f"dock_{title.lower().replace(' ', '_')}")
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
            self._docks[title] = dock

        # Playback stays visible; the four editing panels share one tab stack,
        # because five panels stretched down a single column leaves every one of
        # them too short to use.
        tabbed = [
            self._docks[t]
            for t in ("Chunks", "Controllers", "Surface", "Solver", "Smoothing", "Curve")
            if t in self._docks
        ]
        for previous, dock in zip(tabbed, tabbed[1:]):
            self.tabifyDockWidget(previous, dock)
        if tabbed:
            tabbed[0].raise_()

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

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self._action(file_menu, "&New", QKeySequence.StandardKey.New, self.new)
        self._action(file_menu, "&Open…", QKeySequence.StandardKey.Open, self.open)
        self._build_examples_menu(file_menu)
        file_menu.addSeparator()
        self._action(file_menu, "&Save", QKeySequence.StandardKey.Save, self.save)
        self._action(file_menu, "Save &As…", QKeySequence.StandardKey.SaveAs, self.save_as)
        self._action(file_menu, "Export &Mesh…", None, self.export_mesh)
        file_menu.addSeparator()
        self._action(file_menu, "E&xit", QKeySequence.StandardKey.Quit, self.close)

        edit_menu = self.menuBar().addMenu("&Edit")
        self.undo_action = self._action(
            edit_menu, "&Undo", QKeySequence.StandardKey.Undo, self.document.undo
        )
        self.redo_action = self._action(
            edit_menu, "&Redo", QKeySequence.StandardKey.Redo, self.document.redo
        )
        edit_menu.addSeparator()
        self._action(edit_menu, "&Add Chunk", QKeySequence("A"), self.add_chunk)
        self._action(edit_menu, "&Delete Selected", QKeySequence.StandardKey.Delete, self.delete_selected)

        view_menu = self.menuBar().addMenu("&View")
        self._action(view_menu, "&Frame All", QKeySequence("F"), self.viewport.frame_all)
        self._action(view_menu, "Toggle &Rig Overlay", QKeySequence("R"), self.toggle_rig)
        self._action(view_menu, "&Orthographic", QKeySequence("O"), self.toggle_orthographic)
        self._action(view_menu, "&Play / Pause", QKeySequence(Qt.Key.Key_Space), self.toggle_running)
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

    def toggle_running(self) -> None:
        self.document.set_running(not self.document.running)

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
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Rig", str(EXAMPLES), "Rig files (*.xml);;All files (*)"
        )
        if path:
            self.open_path(Path(path))

    def open_path(self, path: Path) -> None:
        try:
            self.document.open(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open Failed", f"{path.name}\n\n{exc}")
            return
        self.viewport.frame_all()

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

    def closeEvent(self, event) -> None:
        if self._confirm_discard():
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
