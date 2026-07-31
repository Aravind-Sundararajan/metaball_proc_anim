"""Infer a rig's structure and fit controllers to it.

Controllers are configured with explicit chunk ids, which is fine when a
creature is authored in Python and useless for a file somebody else made. An
imported body is just a graph — nothing in it says which chain is a spine and
which is a leg. This works that out from the topology.

The analysis is deliberately about *shape*, not names. Names in these files are
whatever the original tool emitted (``Sphere_0``, ``Spline_7``), so anything
keying off them would work on one rig and silently mis-rig the next.

The trunk is the longest path through the graph, found by the usual two-pass
walk — reach the farthest node from anywhere, then the farthest node from there
— which is the diameter for a tree, and every rig seen so far is a tree.

**Leg-shaped appendages are stripped before that measurement**, and this is the
step that makes the answer usable rather than merely true. Measured with the
legs in, the longest path through a quadruped runs down one leg, along part of
the spine, and back down another leg. That path really is the longest, and it is
useless as a spine — it also picks a foot as the chunk to anchor the creature by.

Everything not on the trunk hangs off it as an appendage, oriented outward from
its attachment, and gets spine behaviour.

**Legs are detected and reported, not fitted.** A leg needs its hip held out to
the side, and a distance link cannot do that — it is isotropic, so the hip drifts
inward until it hangs under the centreline. The working quadruped solves this by
making the hip part of the carried body, which requires knowing how the creature
is driven. That is a decision this module has no way to infer, so it names the
candidates in :attr:`Topology.leg_candidates` and leaves them to the caller
rather than fitting something that looks plausible and behaves badly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mesh_xml.anim.controller import ControllerStack
from mesh_xml.anim.controllers.spine import Breathe, SpineFollow
from mesh_xml.core.body import Body

__all__ = ["Chain", "Topology", "AutoRig", "analyse", "auto_controllers"]

#: An appendage tip within this multiple of its own length from the floor is a leg.
LEG_GROUND_TOLERANCE = 0.6


@dataclass(frozen=True)
class Chain:
    """An ordered run of chunks between two joints, interior nodes all degree two."""

    chunks: tuple[int, ...]
    length: float

    @property
    def base(self) -> int:
        return self.chunks[0]

    @property
    def tip(self) -> int:
        return self.chunks[-1]

    def __len__(self) -> int:
        return len(self.chunks)

    def reversed(self) -> Chain:
        return Chain(tuple(reversed(self.chunks)), self.length)


@dataclass(frozen=True)
class Topology:
    trunk: Chain | None
    appendages: tuple[Chain, ...]
    leg_candidates: tuple[Chain, ...]
    joints: frozenset[int]
    endpoints: frozenset[int]

    @property
    def is_single_chain(self) -> bool:
        return self.trunk is not None and not self.appendages


def analyse(body: Body, ground_y: float | None = None) -> Topology:
    """Decompose a body into a trunk path and the appendages hanging off it."""
    ids = body.chunk_ids()
    if not ids:
        return Topology(None, (), (), frozenset(), frozenset())

    adjacency = {cid: body.neighbors(cid) for cid in ids}
    degree = {cid: len(n) for cid, n in adjacency.items()}
    joints = frozenset(cid for cid, d in degree.items() if d != 2)
    endpoints = frozenset(cid for cid, d in degree.items() if d == 1)

    def span(a: int, b: int) -> float:
        return float(np.linalg.norm(body.pos[body.slot(a)] - body.pos[body.slot(b)]))

    def farthest(source: int) -> tuple[int, dict[int, int | None], dict[int, float]]:
        """Arclength from ``source`` to every reachable chunk, visiting each once.

        A relaxation that accepts any longer candidate does not terminate here:
        stepping back toward the parent always looks like an improvement, so the
        walk oscillates along one edge forever. A tree has exactly one path to
        each node, so a single visit is both correct and finite. On a rig with a
        cycle this measures the walk's own tree rather than the true longest
        path, which is NP-hard and not worth it for a spine heuristic.
        """
        distance = {source: 0.0}
        parent: dict[int, int | None] = {source: None}
        frontier = [source]
        seen = {source}
        while frontier:
            node = frontier.pop()
            for neighbour in adjacency[node]:
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                distance[neighbour] = distance[node] + span(node, neighbour)
                parent[neighbour] = node
                frontier.append(neighbour)
        target = max(distance, key=lambda cid: (distance[cid], -cid))
        return target, parent, distance

    def pendant(endpoint: int) -> Chain:
        """Walk inward from a free end until the graph branches."""
        run = [endpoint]
        length = 0.0
        previous, current = None, endpoint
        while True:
            nxt = [n for n in adjacency[current] if n != previous]
            if len(nxt) != 1:
                break
            previous, current = current, nxt[0]
            length += span(previous, current)
            run.append(current)
            if degree[current] != 2:
                break
        return Chain(tuple(reversed(run)), length)

    pendants = [pendant(e) for e in sorted(endpoints)]

    # Legs are stripped before the trunk is measured. Left in, the longest path
    # through a quadruped runs down one leg, along part of the spine and down
    # another — geometrically true and useless as a spine, and it pins a foot.
    legs = tuple(p for p in pendants if _is_leg(body, p, ground_y))
    leg_chunks = {c for leg in legs for c in leg.chunks[1:]}

    remaining = [cid for cid in ids if cid not in leg_chunks]
    if not remaining:
        remaining = ids
        leg_chunks = set()
    for cid in list(adjacency):
        if cid in leg_chunks:
            del adjacency[cid]
        else:
            adjacency[cid] = [n for n in adjacency[cid] if n not in leg_chunks]

    start, _, _ = farthest(remaining[0])
    end, parent, distance = farthest(start)

    path: list[int] = []
    node: int | None = end
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    trunk = Chain(tuple(path), distance[end])
    on_trunk = set(path)

    appendages: list[Chain] = []
    for attachment in path:
        for neighbour in adjacency[attachment]:
            if neighbour in on_trunk:
                continue
            run = [attachment, neighbour]
            length = span(attachment, neighbour)
            previous, current = attachment, neighbour
            seen = {attachment, neighbour}
            while True:
                nxt = [n for n in adjacency[current] if n != previous and n not in seen]
                if len(nxt) != 1:
                    break
                previous, current = current, nxt[0]
                seen.add(current)
                length += span(previous, current)
                run.append(current)
            appendages.append(Chain(tuple(run), length))

    return Topology(trunk, tuple(appendages), legs, joints, endpoints)


@dataclass
class AutoRig:
    """What :func:`auto_controllers` fitted, and the handle to drive it by."""

    stack: ControllerStack
    topology: Topology
    anchor: int | None
    appendages: tuple[tuple[int, ...], ...]

    @property
    def description(self) -> str:
        parts = []
        if self.topology.trunk:
            parts.append(f"trunk of {len(self.topology.trunk)}")
        if self.appendages:
            parts.append(f"{len(self.appendages)} appendage(s)")
        if self.topology.leg_candidates:
            parts.append(f"{len(self.topology.leg_candidates)} leg-shaped")
        return ", ".join(parts) if parts else "nothing to drive"


def auto_controllers(
    body: Body,
    ground_y: float | None = None,
    anchor: int | None = None,
    seed: int = 0,
    breathe: bool = True,
) -> AutoRig:
    """Fit controllers to whatever shape ``body`` turns out to be.

    ``anchor`` is the chunk the caller intends to drive; it is pinned, because
    something has to lead or the whole rig simply falls. Left unset, the trunk
    end that sits highest is used — a hanging rig reads better than one lying on
    the floor, and for a rig authored standing up it is the base anyway.

    Everything fitted here is an ordinary controller on an ordinary stack, so a
    caller that disagrees can drop the stack and build its own.
    """
    topology = analyse(body, ground_y=ground_y)
    stack = ControllerStack(body, seed=seed)
    if topology.trunk is None:
        return AutoRig(stack, topology, None, ())

    trunk = topology.trunk
    if anchor is None or not body.has_chunk(anchor):
        head, tail = trunk.base, trunk.tip
        anchor = head if body.pos[body.slot(head)][1] >= body.pos[body.slot(tail)][1] else tail
    if anchor == trunk.tip:
        trunk = trunk.reversed()

    body.set_pinned(anchor, True)
    if len(trunk) > 1:
        stack.add(SpineFollow(list(trunk.chunks), strength=16.0, max_bend_degrees=38.0))

    driven: list[tuple[int, ...]] = []
    for appendage in topology.appendages:
        if len(appendage) > 1:
            stack.add(SpineFollow(list(appendage.chunks), strength=14.0, max_bend_degrees=45.0))
            driven.append(appendage.chunks)

    if breathe:
        free = [cid for cid in body.chunk_ids() if body.inv_mass[body.slot(cid)] > 0.0]
        if free:
            stack.add(Breathe(free, amplitude=1.8, frequency=0.35))

    return AutoRig(stack, topology, anchor, tuple(driven))


def _is_leg(body: Body, chain: Chain, ground_y: float | None) -> bool:
    """Three chunks past a branch point, ending near the floor.

    Wrong sometimes by construction — a three-chunk neck on a low creature reads
    as a leg. Stated rather than patched, because the fix is for the caller to
    override, not for the heuristic to grow special cases until nobody can
    predict it.
    """
    if ground_y is None or len(chain) != 4:
        return False
    tip_y = float(body.pos[body.slot(chain.tip)][1])
    return (tip_y - ground_y) <= chain.length * LEG_GROUND_TOLERANCE
