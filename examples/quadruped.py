"""Author a controller-driven quadruped and watch it walk.

The whole stack in one file: a chunk graph, a gait built from four
:class:`FootholdPicker` / :class:`LimbIK` pairs, a free-swinging tail on
:class:`SpineFollow`, the visual smoothing layer, and the SDF raymarcher.

    poetry run python examples/quadruped.py          # headless, prints what it did
    poetry run python examples/quadruped.py --gui    # opens the editor on it

Three things here are easy to get wrong and worth copying rather than
rediscovering.

**A picker's ``stance`` is an offset from the limb root, not a world position.**
The root carries it as the body moves, so it is set once and never touched again.
Rewriting it every tick — the obvious thing to try — drags the neutral point away
from the limb, and the leg either dangles or never re-plants.

**``ground_y`` for a picker is the plane plus the foot's radius.** Ground
collision holds a chunk one radius clear of the plane, so a foot aimed at the
plane itself is aimed one radius underground and seeks into the floor forever.
:meth:`ControlContext.ground_level` computes this at run time; it is spelled out
here because the picker is built before any tick has run.

**Force applied to a pinned chunk goes nowhere.** Pinned means zero inverse mass,
so :class:`Breathe` and :class:`SpineFollow` are silently absorbed by a pinned
spine. They act on the free tail here, which is the part that can actually move.

**A distance link cannot hold a limb out to the side.** It is isotropic: a rest
length of 0.3 says "0.3 away", not "0.3 away laterally". Hanging a leg root off
the spine by one link lets gravity and the IK drag it inward until it sits
directly under the centreline, satisfying the constraint the whole way down, and
the legs collapse into the body. So the hip is part of the *body* here — carried,
pinned, held at its offset — and the limb is the two bones below it. That is also
what two-bone IK assumes: a fixed root, solving for mid and tip.
"""

from __future__ import annotations

import sys

import numpy as np

from mesh_xml.anim import ControlContext, ControllerStack
from mesh_xml.anim.controllers.limb import FootholdPicker, LimbIK
from mesh_xml.anim.controllers.spine import Breathe, SpineFollow
from mesh_xml.core import Body
from mesh_xml.core.solver import SolverConfig
from mesh_xml.render import DrawPoints, sdf

GROUND = 0.0
HIP_Y = 1.15
UPPER = LOWER = 0.62
FOOT_R = 0.13
SPEED = 0.62
SEGMENT = 0.42

#: name, spine anchor, gait phase, side, knee direction along x.
#:
#: Both legs of a pair share an anchor, or they are not a pair — they are two
#: legs at different points down the body, and the feet read as a zigzag rather
#: than a stance.
#:
#: Diagonal pairs share a phase, which is a trot. The knee direction is the IK
#: pole: front knees fold back and rear knees fold forward, as a quadruped's do.
#: Poling every leg along z instead bends all four sideways, which is a splay.
GAIT = (
    ("FL", 1, 0.0, 1, -1.0),
    ("FR", 1, 0.5, -1, -1.0),
    ("RL", 3, 0.5, 1, 1.0),
    ("RR", 3, 0.0, -1, 1.0),
)

#: Front of the spine is carried; the rest trails.
DRIVEN = 4


class KinematicPath:
    """Carries the driven chunks along a path, and is the one place positions are written.

    Everything else in this framework contributes force and lets the solver
    reconcile. This does not, and the exception is narrow enough to be worth
    stating: these chunks are **pinned**, so the solver never moves them and
    there is nothing to fight. Writing an unpinned chunk here would be the
    failure the contract warns about.

    A real creature would propel itself from its own footfalls. This stands in
    for locomotion so the gait has something to walk against.
    """

    def __init__(self, carried: list[tuple[int, tuple[float, float, float]]], speed: float = SPEED) -> None:
        self.enabled = True
        self.carried = carried
        self.speed = speed

    def update(self, ctx: ControlContext) -> None:
        x = self.speed * ctx.time
        for index, (chunk, offset) in enumerate(self.carried):
            if not ctx.has(chunk):
                continue
            sway = 0.03 * np.sin(ctx.time * 4.0 + index * 0.5)
            ctx.body.set_pos(chunk, (x + offset[0], offset[1] + sway, offset[2]))


def build() -> tuple[Body, ControllerStack]:
    body = Body(name="quadruped")

    spine = []
    for i, radius in enumerate((0.13, 0.19, 0.23, 0.24, 0.22, 0.17, 0.11)):
        spine.append(
            body.add_chunk(
                f"spine{i}",
                [-i * SEGMENT, HIP_Y, 0.0],
                radius=radius,
                pinned=i < DRIVEN,
            )
        )
    for a, b in zip(spine, spine[1:]):
        body.add_link(a, b)

    head = body.add_chunk("head", [0.5, HIP_Y + 0.12, 0.0], radius=0.24, pinned=True)
    body.add_link(head, spine[0])

    carried: list[tuple[int, tuple[float, float, float]]] = [
        (chunk, (-i * SEGMENT, HIP_Y, 0.0)) for i, chunk in enumerate(spine[:DRIVEN])
    ]
    carried.append((head, (0.5, HIP_Y + 0.12, 0.0)))

    stack = ControllerStack(body, seed=11)
    for name, anchor_index, phase, side, knee_x in GAIT:
        anchor = spine[anchor_index]
        z = 0.34 * side
        x = -anchor_index * SEGMENT

        # Pinned and carried: the hip is body, not limb.
        root = body.add_chunk(f"{name}_root", [x, HIP_Y, z], radius=0.15, pinned=True)
        knee = body.add_chunk(f"{name}_knee", [x, HIP_Y - UPPER, z], radius=0.13)
        foot = body.add_chunk(f"{name}_foot", [x, GROUND + FOOT_R, z], radius=FOOT_R)
        carried.append((root, (x, HIP_Y, z)))
        body.add_link(anchor, root, rest=abs(z))
        body.add_link(root, knee, rest=UPPER)
        body.add_link(knee, foot, rest=LOWER)

        picker = FootholdPicker(
            stance=(0.0, -(HIP_Y - GROUND - FOOT_R), 0.0),
            ground_y=GROUND + FOOT_R,
            step_threshold=0.26,
            step_height=0.20,
            step_duration=0.26,
            lead_time=0.22,
            period=1.0,
            phase=phase,
            duty=0.44,
        )
        stack.add(LimbIK(root, knee, foot, target=picker, pole=(knee_x, 0.0, 0.0)))

    # The tail is free, so these have something to act on. Chain is head-first,
    # led by the last driven segment.
    tail = spine[DRIVEN - 1 :]
    stack.add(SpineFollow(tail, strength=18.0, max_bend_degrees=32.0))
    stack.add(Breathe(tail[1:], amplitude=2.5, frequency=0.5))
    # First, so the limbs solve against hips already at this tick's position.
    stack.controllers.insert(0, KinematicPath(carried))
    return body, stack


def config() -> SolverConfig:
    return SolverConfig(ground_y=GROUND, iterations=12)


def headless(ticks: int = 900) -> int:
    body, stack = build()
    settings = config()
    draw = DrawPoints(body, stiffness=0.32, damping=0.80)
    plants: dict[int, set] = {}

    for _ in range(ticks):
        stack.tick(body, 1 / 40, settings)
        draw.update(body)
        for index, controller in enumerate(stack.controllers):
            picker = getattr(controller, "target", None)
            foothold = getattr(picker, "foothold", None)
            if foothold is not None:
                plants.setdefault(index, set()).add(tuple(np.round(foothold, 2)))

    primitives = sdf.pack(body, positions=draw.render_pos(1.0))
    head = body.chunk_ids()[body.n_chunks - 1]
    print(f"{body.n_chunks} chunks, {body.n_links} links -> {len(primitives)} primitives")
    print(f"distinct footholds per leg: {sorted(len(v) for v in plants.values())}")
    print(f"travelled {SPEED * stack.time:.2f} units")
    print(f"all finite: {bool(np.isfinite(body.pos).all())}")
    assert head is not None
    return 0


def gui() -> int:
    from mesh_xml.editor.app import MainWindow
    from mesh_xml.editor.document import Document
    from mesh_xml.render.gl.context import configure_surface_format, ensure_application

    configure_surface_format()
    app = ensure_application(sys.argv)

    body, stack = build()
    document = Document()
    document.adopt(body, controllers=stack)
    document.solver_config = config()
    document.metaball.blend = 0.20

    window = MainWindow(document)
    window.resize(1500, 940)
    window.show()
    window.viewport.frame_all()
    document.set_running(True)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(gui() if "--gui" in sys.argv else headless())
