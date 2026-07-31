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

from mesh_xml.anim import ControlContext, autorig
from mesh_xml.core.solver import SolverConfig
from mesh_xml.io import xml_legacy, xml_v2
from mesh_xml.render import DrawPoints

GROUND = 0.0


class WaveAnchor:
    """Moves the anchor on a slow lissajous so the body has something to chase.

    Writes a position, which controllers otherwise must not do. Legitimate here
    for the same reason as in the quadruped: the anchor is pinned, so the solver
    has no authority over it and there is nothing to fight.
    """

    def __init__(self, anchor: int, origin, amplitude: float = 0.8, rate: float = 0.6) -> None:
        self.enabled = True
        self.anchor = anchor
        self.origin = np.asarray(origin, dtype=np.float64)
        self.amplitude = amplitude
        self.rate = rate

    def update(self, ctx: ControlContext) -> None:
        if not ctx.has(self.anchor):
            return
        t = ctx.time * self.rate
        offset = np.array(
            [np.sin(t * 2.0) * self.amplitude, np.sin(t * 1.3) * self.amplitude * 0.35, np.cos(t) * self.amplitude],
            dtype=np.float64,
        )
        ctx.body.set_pos(self.anchor, self.origin + offset)


def load(path: Path):
    try:
        return xml_v2.load(path)
    except Exception:
        return xml_legacy.read_legacy(path).body


def prepare(path: Path):
    body = load(path)
    rig = autorig.auto_controllers(body, ground_y=GROUND, seed=5)
    if rig.anchor is not None:
        origin = body.chunk(rig.anchor).pos
        # First, so the followers chase an anchor already at this tick's position.
        rig.stack.controllers.insert(0, WaveAnchor(rig.anchor, origin))
    return body, rig


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else Path("examples/chubby_worm.xml")
    body, rig = prepare(path)

    print(f"{path.name}: {body.n_chunks} chunks, {body.n_links} links")
    print(f"  topology: {rig.description}")
    if rig.anchor is not None:
        print(f"  anchored at {body.name_of(rig.anchor)!r}")

    if "--gui" in argv:
        from mesh_xml.editor.app import MainWindow
        from mesh_xml.editor.document import Document
        from mesh_xml.render.gl.context import configure_surface_format, ensure_application

        configure_surface_format()
        app = ensure_application(argv)
        document = Document()
        document.adopt(body, controllers=rig.stack, path=path)
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
        rig.stack.tick(body, 1 / 40, config)
        draw.update(body)

    print(f"  moved {float(np.abs(body.pos - start).max()):.3f} units over 15s")
    print(f"  all finite: {bool(np.isfinite(body.pos).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
