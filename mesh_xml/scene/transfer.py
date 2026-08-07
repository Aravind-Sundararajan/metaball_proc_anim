"""Creature transfer: serialize-out, re-id, serialize-in — without the file.

Moving a creature between rooms is the door's mechanism ahead of the door
(``docs/ROOMS.md`` package 3). It is deliberately the save/load journey run
in memory: the chunks cross with their masses, pins and meta, the links
with their rests, types and render flags, and the controllers are rebuilt
through :func:`mesh_xml.anim.serialize.remap_ids` — the decode path — so
runtime state (a plan mid-hunt, a foothold mid-swing, a scrap site) resets
exactly as it resets through a file. A door is a serialization boundary;
that ruling was taken in the plan and this module is where it executes.

Ids are per-body identity, so the creature takes fresh ids in the target
and the returned mapping says which became which. The registry's untyped
caveat applies here the way it applies everywhere ids travel: an int field
is presumed a chunk id, and a controller that straddles two creatures
travels with the one it is asked about, its foreign references passing
through unmapped — a straddling controller was already a design smell, and
the stack's ``has_chunk`` guards turn its dangling half into silence, not a
crash. Refusal is wholesale, the format's own ruling: if any travelling
controller cannot be rebuilt, nothing has moved yet and nothing moves.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from mesh_xml.anim import place, serialize, tempo
from mesh_xml.scene.room import Room

__all__ = ["move_creature"]


def move_creature(
    source: Room, target: Room, chunk_id: int, offset=None
) -> dict[int, int]:
    """Move the creature containing ``chunk_id``; return old id to new id.

    ``offset`` displaces the arrival — chunk positions and the controllers'
    placed world points together (:mod:`mesh_xml.anim.place`), so a moved
    creature ranges around where it arrived, not where it left. ``None``
    keeps coordinates, which is the honest default while doors do not exist
    to say where "here" is.
    """
    if source is target:
        raise ValueError("a creature cannot move to the room it is in")
    if source.body.dim != target.body.dim:
        raise ValueError(
            f"rooms disagree on dimension: {source.body.dim} vs {target.body.dim}"
        )
    body = source.body
    body.slot(chunk_id)  # unknown ids fail here, before anything moves

    component: set[int] = set()
    frontier = [chunk_id]
    while frontier:
        current = frontier.pop()
        if current in component:
            continue
        component.add(current)
        frontier.extend(body.neighbors(current))

    shift = (
        np.zeros(body.dim, dtype=np.float64)
        if offset is None
        else np.asarray(offset, dtype=np.float64)
    )
    displaced = bool(np.any(shift))

    next_id = max(target.body.chunk_ids(), default=-1) + 1
    mapping: dict[int, int] = {}
    for cid in sorted(component):
        mapping[cid] = next_id
        next_id += 1

    # Rebuild every travelling controller before touching either room: a
    # creature that cannot be carried whole is not carried at all.
    travelling = []
    for controller in source.controllers.controllers:
        if not (tempo.chunk_refs(controller) & component):
            continue
        rebuilt = serialize.remap_ids(controller, mapping)
        if displaced:
            moved = place.translate(rebuilt, shift)
            if moved:
                rebuilt = replace(rebuilt, **moved)
        travelling.append((controller, rebuilt))

    for cid in sorted(component):
        chunk = body.chunk(cid)
        target.body.add_chunk(
            chunk.name,
            tuple(float(p) + float(o) for p, o in zip(chunk.pos, shift)),
            radius=chunk.radius,
            mass=1.0 if chunk.pinned else chunk.mass,
            pinned=chunk.pinned,
            id=mapping[cid],
            meta=dict(chunk.meta),
        )
    for link in body.links():
        if link.a in component and link.b in component:
            target.body.add_link(
                mapping[link.a],
                mapping[link.b],
                rest=link.rest,
                type=link.type,
                elasticity=link.elasticity,
                render=link.render,
            )

    for original, rebuilt in travelling:
        target.controllers.add(rebuilt)
        source.controllers.remove(original)
    for cid in sorted(component):
        body.remove_chunk(cid)

    return mapping
