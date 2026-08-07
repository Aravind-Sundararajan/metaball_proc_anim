"""Animate any rig file without knowing anything about it in advance.

    poetry run python examples/animate_rig.py examples/chubby_worm.xml
    poetry run python examples/animate_rig.py examples/chubby_worm.xml --gui

The quadruped example wires controllers to chunk ids it chose itself, which only
works because it built the body. A file made by someone else is just a graph, so
this asks :mod:`mesh_xml.anim.autorig` what shape it is and fits controllers to
the answer.

The rig is driven by moving its anchor — a single pinned chunk — and letting the
rest follow. That is the smallest possible input that produces whole-body motion,
and it is what makes one script work for a worm, a stalk, and a tentacle without
branching on which it got.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from mesh_xml.anim import autorig
from mesh_xml.anim.controllers.drive import WaveAnchor
from mesh_xml.core.solver import SolverConfig
from mesh_xml.io import xml_legacy, xml_v2
from mesh_xml.render import DrawPoints

GROUND = 0.0


def prepare(path: Path):
    """A body and the stack driving it — the file's own gait when it has one.

    A v3 document carries its controllers, so re-rigging it here would stack a
    second spine on top of the shipped one and the two would fight. Only a
    gait-less file (v2, or a v1 legacy armature) gets the autorig treatment.
    """
    try:
        rig = xml_v2.read(path)
        if rig.controllers.controllers:
            count = len(rig.controllers.controllers)
            return rig.body, rig.controllers, f"shipped gait ({count} controller(s))"
        body = rig.body
    except Exception:
        body = xml_legacy.read_legacy(path).body

    fitted = autorig.auto_controllers(body, ground_y=GROUND, seed=5)
    if fitted.anchor is not None:
        origin = body.chunk(fitted.anchor).pos
        # First, so the followers chase an anchor already at this tick's position.
        fitted.stack.controllers.insert(0, WaveAnchor(fitted.anchor, origin))
    return body, fitted.stack, f"autorigged: {fitted.description}"


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else Path("examples/chubby_worm.xml")
    body, stack, description = prepare(path)

    print(f"{path.name}: {body.n_chunks} chunks, {body.n_links} links")
    print(f"  {description}")

    if "--gui" in argv:
        from mesh_xml.editor.app import MainWindow
        from mesh_xml.editor.document import Document
        from mesh_xml.render.gl.context import configure_surface_format, ensure_application

        configure_surface_format()
        app = ensure_application(argv)
        document = Document()
        document.adopt(body, controllers=stack, path=path)
        document.solver_config = SolverConfig(ground_y=GROUND, iterations=10)
        document.metaball.blend = 0.18
        window = MainWindow(document)
        window.resize(1500, 940)
        window.show()
        window.viewport.frame_all()
        document.set_running(True)
        return app.exec()

    config = SolverConfig(ground_y=GROUND, iterations=10)
    draw = DrawPoints(body, stiffness=0.30, damping=0.82)
    start = body.pos.copy()
    for _ in range(600):
        stack.tick(body, 1 / 40, config)
        draw.update(body)

    print(f"  moved {float(np.abs(body.pos - start).max()):.3f} units over 15s")
    print(f"  all finite: {bool(np.isfinite(body.pos).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
