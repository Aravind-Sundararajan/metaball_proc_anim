---
name: run
description: Launch mesh_xml's editor, run its tests, or screenshot the viewport with the correct environment (the poetry venv, not any PATH python). Trigger on run/launch/test/screenshot requests for this project.
---

# Running mesh_xml

## The environment — the only fact that matters cold

This is a poetry project. The **only** Python with the dependencies is the
poetry venv; every `python`/`pytest` on PATH or in pyenv lacks them, and a
cold shell that reaches for those gets `ModuleNotFoundError`, not a hint.
Rediscover the venv rather than trusting a remembered path:

```powershell
poetry env info --path   # currently E:\poetry_venvs\mesh-xml-oNgf88MM-py3.11
```

Call that `$venv` below. The GUI stack is **PySide6** (not PyQt6) +
PyOpenGL + numpy + pyqtgraph; dev deps add pytest, ruff, imageio. The
package is installed editable (`mesh_xml.pth` → this repo), so imports
resolve from the working tree from any cwd inside the venv.

## Tests

```powershell
& "$venv\Scripts\python.exe" -m pytest tests -q
```

Full suite: 1513 tests, ~60 s, including real-GL widget tests that pass
from a normal shell (they gate on `gl_available()` and skip only when no
core-profile 3.3 context exists). `pyproject.toml` scopes `testpaths` to
`tests/` because two legacy root-level `test_*.py` scripts open Qt windows
on import — pass `tests`, never name root-level test files.

## Launch the editor

Two equivalent invocations (the poetry script is a `.cmd` shim on Windows,
not an `.exe`):

```powershell
& "$venv\Scripts\mesh-xml.cmd" [file.xml]
& "$venv\Scripts\python.exe" -m mesh_xml.editor.app [file.xml]
```

No argument opens `examples/chubby_worm.xml`. This blocks on the Qt event
loop — run it in the background if the session needs to keep working.
Entry point: `mesh_xml.editor.app:main`, which calls
`configure_surface_format()` before constructing the QApplication; any
hand-rolled launcher must keep that ordering or shaders fail to compile
blaming the shader.

## Screenshot / live-render check

Render through `MetaballViewport`, never the bare GL widget. The house
lesson (`render/gl/metaball.py` ~101): a texture-unit collision once left
the whole suite green while the editor drew wrong, because only the editor
runs the coarse prepass — `MetaballViewport` switches it on
(`prepass_divisor = 4`, `editor/viewport.py` ~557) and the bare widget
defaults it off (`render/gl/context.py` ~264). A screenshot from the bare
widget verifies a code path the editor does not use.

Copy the canonical setup from `tests/test_viewport.py::TestViewport`:
`viewport()` builds the widget (`gl_context.ensure_application()` first,
then `MetaballViewport(document)`, resize, set `camera.distance`) and
`image()` grabs it — `view.grabFramebuffer()` converted to
`Format_RGB888`, reshaped into numpy. `grabFramebuffer` works on a
never-shown widget (`context.py` `_device_size`, ~308–322, exists to keep
that path honest); briefly showing a window is also fine. Save with
`QImage.save` or imageio (both in the venv). Load a document the way the
tests do (`document_with(worm())`) or via `Document.open` on an example.

## The creature to put on screen

`examples/quadruped.xml` is the shipped walkable example — load it when
you need a real creature in the viewport, not a synthetic worm.
