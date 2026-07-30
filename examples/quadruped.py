"""Author a controller-driven quadruped and render it as metaballs.

The whole stack in one file: a chunk graph, a gait built from four
:class:`FootholdPicker` / :class:`LimbIK` pairs, the visual smoothing layer, and
the SDF raymarcher.

Two things here are easy to get wrong and worth copying rather than rediscovering.

**A picker's ``stance`` is an offset from the limb root, not a world position.**
The root carries it as the body moves, so it is set once and never touched again.
Rewriting it every tick — the obvious thing to try — drags the neutral point away
from the limb and the leg either dangles or never re-plants.

**``ground_y`` for a picker is the plane plus the foot's radius.** Ground
collision holds a chunk one radius clear of the plane, so a foot aimed at the
plane itself is aimed one radius underground and seeks into the floor forever.
:meth:`ControlContext.ground_level` computes this; it is spelled out here because
the picker is constructed before any tick has run.

Run with ``poetry run python examples/quadruped.py``.
"""

from __future__ import annotations


import numpy as np

from mesh_xml.anim import ControllerStack
from mesh_xml.anim.controllers.limb import FootholdPicker, LimbIK
from mesh_xml.anim.controllers.spine import Breathe
from mesh_xml.core import Body
from mesh_xml.core.solver import SolverConfig
from mesh_xml.render import DrawPoints, sdf

GROUND = 0.0
HIP_Y = 0.95
UPPER = LOWER = 0.52
FOOT_R = 0.11
SPEED = 0.62

#: Diagonal pairs share a phase, which is a trot.
GAIT = (("FL", 1, 0.0), ("FR", 2, 0.5), ("RL", 4, 0.5), ("RR", 5, 0.0))


def build() -> tuple[Body, ControllerStack, list[int], int]:
    body = Body(name="quadruped")

    spine = []
    for i, radius in enumerate((0.15, 0.22, 0.27, 0.28, 0.26, 0.21, 0.14)):
        # Pinned: the gait drives the body, so the spine is kinematic here rather
        # than something the solver is asked to hold up.
        spine.append(body.add_chunk(f"spine{i}", [-i * 0.42, HIP_Y, 0.0], radius=radius, pinned=True))
    for a, b in zip(spine, spine[1:]):
        body.add_link(a, b)

    head = body.add_chunk("head", [0.5, HIP_Y + 0.12, 0.0], radius=0.24, pinned=True)
    body.add_link(head, spine[0])

    stack = ControllerStack(body, seed=11)
    for name, anchor_index, phase in GAIT:
        side = 1 if name.endswith("L") else -1
        anchor = spine[anchor_index]
        z = 0.30 * side
        x = float(body.chunk(anchor).pos[0])

        root = body.add_chunk(f"{name}_root", [x, HIP_Y, z], radius=0.13)
        knee = body.add_chunk(f"{name}_knee", [x, HIP_Y - UPPER, z], radius=0.11)
        foot = body.add_chunk(f"{name}_foot", [x, GROUND + FOOT_R, z], radius=FOOT_R)
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
        stack.add(LimbIK(root, knee, foot, target=picker, pole=(0.0, 0.0, 1.0), flip=float(side)))

    stack.add(Breathe(spine, amplitude=2.0, frequency=0.4))
    return body, stack, spine, head


def walk(body: Body, stack: ControllerStack, spine: list[int], head: int, ticks: int = 900):
    config = SolverConfig(ground_y=GROUND, iterations=12)
    draw = DrawPoints(body, stiffness=0.32, damping=0.80)

    for _ in range(ticks):
        x = SPEED * stack.time
        for i, chunk in enumerate(spine):
            sway = 0.03 * np.sin(stack.time * 4.0 + i * 0.5)
            body.set_pos(chunk, (x - i * 0.42, HIP_Y + sway, 0.0))
        body.set_pos(head, (x + 0.5, HIP_Y + 0.12, 0.0))
        stack.tick(body, 1 / 40, config)
        draw.update(body)

    return draw


def main() -> int:
    body, stack, spine, head = build()
    draw = walk(body, stack, spine, head)

    primitives = sdf.pack(body, positions=draw.render_pos(1.0))
    print(f"{body.n_chunks} chunks, {body.n_links} links -> {len(primitives)} primitives")
    print(f"travelled {float(body.chunk(head).pos[0]):.2f} units")
    print(f"all finite: {bool(np.isfinite(body.pos).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
