# Rooms: the scene-level plan

**Status:** landed — all six packages: 1 (creature roster), 2 (headless
room), 3 (scene + transfer), 4 (abstraction), 5 (doors), and 6 (the editor
scene binding — designed and landed in `docs/SCENE_BINDING.md`).
**Date:** 2026-08-07
**North star:** Rain World's world model — `Room` as the unit of simulation,
`AbstractRoom`/`AbstractCreature` as the cheap tier behind it, creatures that
exist whether or not anyone is looking at them.

---

## 1. Where the stack already is

Everything below the scene level is done and pulls in this direction:

* Multiple creatures already share one `Body` as connected components. Blend
  groups keep their surfaces apart, `ctx.peers` is the room object list,
  per-creature facts (colour, tempo) ride chunk meta, and the shipped
  `hunt.xml` / `scavenge.xml` are two- and three-creature ecosystems in one
  document.
* The document already owns exactly what a room is: a body, a stack, a world
  (`SolverConfig.ground_y` or a `Terrain`), a clock. What it lacks is a
  *plural*.
* Brains steer only their own pinned anchor. That single-point authority is
  the whole abstraction tier waiting to be noticed: a creature reduced to its
  anchor is Rain World's `AbstractCreature`, and the brain code would not
  know the difference.

The missing nouns are **creature** (as something you can list, name, and
move — today it is a derived fact about link connectivity) and **room** (as
something a scene can hold more than one of).

## 2. The design, in the creature's vocabulary

A **Room** is a body, a stack, a world, and a name. A **Scene** is an ordered
set of rooms, one clock, and one active room in the viewport. **Doors** are
named points in a room paired with a destination room; crossing one moves a
creature between bodies. **Abstraction** is a room ticking its creatures as
anchors-only — brains think and steer at full authority, but no solver, no
links, no surface — with realization rebuilding the full creature from its
saved form on entry.

The law that carries over unchanged: brains write goals, controllers write
forces, and nobody reaches across a room. `ctx.peers` stays per-room — a
creature in another room is unsensed by construction, which is not a limit
but the point of rooms.

## 3. Work packages, ordered by leverage

1. **The creature roster (lands value alone).** A `creatures()` view on the
   document — one entry per connected component, named from chunk meta the
   way colour and tempo already ride there — plus `duplicate_creature()`
   (offset copy, fresh ids, one undo step: the herd-authoring gesture that
   never landed) and a roster dropdown in the panel. No new stance on
   identity: the roster is a *view* of the graph, recomputed like blend
   groups, and the name is meta that dies with its chunks.
2. **The headless room (no behaviour change).** Extract what `Document` owns
   into a Qt-free `Room` (body, stack, solver config, terrain); `Document`
   becomes the editor binding of one. Acceptance is the null result: every
   editor test unchanged, plus a game-loop smoke test that ticks a room with
   no QApplication anywhere.
3. **The scene and the transfer.** `Scene` holds rooms and the clock;
   creature transfer is serialize-out, re-id, serialize-in — the v3
   round-trip is already the one tested way to lift a creature whole
   (chunks, links, meta, controllers). Re-iding is the honest hard part:
   ids are per-body identity, and an int field cannot prove it is an id
   (the registry's own caveat, same as `tempo.chunk_refs`). The remap walks
   registered fields and rewrites ints that match moved ids; the caveat is
   named where it lives. Acceptance: the hunt split across two rooms
   round-trips whole, and a transferred hunter hunts on arrival.
4. **Abstraction.** An inactive room drops to anchors-only: each creature one
   pinned chunk, its brain ticking at think cadence, links and non-anchor
   controllers parked as data. Realize on entry by rebuilding from the parked
   form at the anchor's current position. Acceptance: a grazer crosses an
   abstracted room in about the time it would have taken realized (±think
   quantization), and a scene of twenty abstracted rooms ticks for less than
   one realized room's cost.
5. **Doors.** Doors as named world points with a destination (a v3 `<door>`
   beside `<terrain>`, additive, `scene/door.py`); a brain-legible goal
   ("leave via X") that walks the anchor there through `ctx.doors`; and the
   scene's crossing — arrival inside the door's radius, `move_creature`
   with the paired door's offset, the goal spent on arrival. A door whose
   promise cannot be kept (unknown room or door, an abstracted side) is a
   closed door: the creature waits in the doorway, and nothing raises
   mid-tick.
6. **The editor scene binding.** The rooms dock, per-room world settings,
   and Play that ticks the whole scene while the viewport draws the active
   room. Split from package 5 when the doors landed, because it is a
   `Document`-identity rework, not a feature: a document is one room, so
   several rooms means either several documents behind one window or one
   document re-pointed live, and every panel holds a document reference
   today. The constraints a design must answer before code: where the undo
   stack lives when edits span rooms (per-room histories, or one history
   of room-tagged commands); how panels follow a room switch (rebind,
   rebuild, or a document that re-points its `room` and re-emits
   everything); draw points and rest poses per room, not per window; and
   what Play drives (the scene) versus what the transport shows (the
   active room's clock, which is the scene's). `Document.open/save` also
   grow doors here — the format is ready (`Rig.doors`), the editor simply
   does not read it yet.

Packages 1 and 2 are file-disjoint and can land in either order; 3 needs 2;
4 and 5 need 3; 6 needs the rest.

## 4. Decisions taken in advance

* **One clock per scene, not per room.** Rooms tick from the scene's
  accumulator so cross-room time never skews; abstraction changes what a
  tick costs, never how often it happens.
* **Transfer moves authored state only.** In-flight runtime (plans, scrap
  sites, foothold modes) resets on the way through a door, exactly as it
  resets through a save — a door is a serialization boundary, and the
  serializer's "rest pose plus gait definition" ruling already decided this.
* **The viewport draws one room.** No cross-room rendering, no seamless
  camera handoff; the editor switches rooms the way Rain World's camera
  does — a cut.

## 5. Not in scope

Streaming or large-world paging. Cross-room physics (a body straddling a
door). Multiplayer. Mid-simulation snapshots. Procedural room generation.
