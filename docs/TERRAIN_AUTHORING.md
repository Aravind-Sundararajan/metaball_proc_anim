# Terrain authoring: the world is flesh that stopped moving

**Status:** design — not yet reviewed.
**Arc:** `docs/TERRAIN.md` §5 named "terrain editing UI" out of scope with
a promise attached — "the format and rendering land first; authoring stays
in Python this round" — and every other round has since landed on top of
it. This is that round.
**North star:** shape a ridge with the tools that shape a creature, say
"Make Terrain", and stand on it.

---

## 1. The gap, measured

Terrain is the most thoroughly finished feature in this codebase that no
author can reach. The field is landed (`core/terrain.py`), the solver
collides with it and grips at its μ, the foothold picker projects onto it,
`Dangling` means a hole in it, the raymarcher draws it with its own colour
through a second primitive texture, the cast-shadow arc lands shadows on
it, and the world-contact arc drapes spans across it. The format carries
it: `<terrain>` is an additive v3 section, planes and cones at `repr`
precision, absent meaning none.

**And not one shipped file has a `<terrain>` in it.** Not an example, not a
test fixture — the grep is empty across `examples/` and `tests/data/`. The
only way a world exists is a Python session:

```python
document.set_terrain(Terrain.flat(0.0, blend=0.35).add_cone(...))
```

which is how every screenshot in every arc that photographed terrain was
made. So the deliverable here is not "an editing UI" in the abstract. It is
**terrain becoming reachable**, and the acceptance is an author opening a
file with a world in it — which means this arc ships one (§11).

Two smaller truths measured on the way in, both of which the design has to
answer rather than inherit:

* `Document.set_terrain` is **not a dirty-flag edit**, and says why in its
  own docstring: "terrain authoring stays in Python this round." The moment
  authoring lands that becomes a lie — an authored world that leaves the
  room clean is a world `Save All` skips.
* `ground_y` — the reference floor's height, and the physics floor whenever
  no terrain is set — **has no control anywhere in the editor**. The Solver
  panel carries iterations, damping, max velocity, friction, self-contact,
  contact elasticity, self-friction and gravity, and not the floor. It was
  never a solver knob; it is the world's, and the world had no dock.

## 2. The identity this whole design stands on

`render/sdf.py`'s `pack` writes one row per **rendered** link:

```python
out[:n, 0:3] = pos[a];  out[:n, 3] = rad[a]
out[:n, 4:7] = pos[b];  out[:n, 7] = rad[b]
```

and one row per chunk no rendered link claims, with the centre written into
both ends. `Terrain.add_cone(a, ra, b, rb)` builds *that record*, and
`Terrain.distance` folds it with `core.distance.round_cone` — the same
function, from the same module, that the body's own field and `sdf.glsl`
are pinned to line-for-line.

So a chunk graph **is** a cone list. Not a metaphor, not a conversion with
a tolerance: the same eight numbers, folded by the same code. Everything
below follows from that one sentence:

* Authoring terrain with the body's tools is not an approximation of a
  terrain editor. Add (A), drag, Ctrl+click placement on a surface, the
  radius spin, the radius curve, the chunk table — all of them already
  author cone records, and have since before terrain existed.
* The round trip is exact, so terrain can be edited a second time
  (§9) rather than authored once and frozen.
* **`link_render` is the promotion rule.** A mechanism link (`render=False`)
  is not part of the skin and does not become a cone; its chunks promote as
  spheres, exactly as `pack` treats them. What you saw is what you
  promoted.

The one thing the identity does not cover is planes: a half-space is a
normal and an offset, with no position, no extent, and no chunk that could
stand for it. §5 answers planes separately, and the asymmetry is honest —
they are a different kind of thing, which is why the vocabulary has two.

## 3. The gesture: promote the selection, demote the world

**Make Terrain** is selection-grown, the sixth verb in the line that runs
colour, tempo, rename, duplicate, move: every creature the selection
touches leaves the body and its cones append to the room's terrain. Any
chunk arms the whole component, because the component is what a creature is
and what a feature is.

**Edit as Chunks** is total: the terrain's whole cone list comes back as
pinned chunk components; planes and blend stay where they are. Total
rather than selection-grown because **terrain has no components** — it is
one field folded from one list, and two hills that touch are one surface
with no seam to cut. There is nothing to select half of. (A surface pick
could name a *cone* — `surface_hit` already answers `kind="terrain"` — but
a cone is not a feature either; the ridge you want back is the four cones
that blend into it, and the field does not know they are four.)

The loop is therefore: demote, edit as flesh, promote. Round-tripping the
whole world to move one hill is the cost, and it is small because the world
is small — the vocabulary tops out at the handful of primitives a room's
landscape is made of, not a mesh.

Rejected by name:

- **A modal terrain tool in the viewport** — a toolbar toggle that repoints
  the existing gestures at the terrain. It rebuilds the body editor's whole
  gesture set against a second data model, and there is no chord left to
  build it from: Ctrl+Left places (PICK §4, consuming hit and miss alike),
  bare Left resolves gizmo-then-pick-then-drag-then-clear, Shift+Left
  toggles, Middle and Right drive the camera. Every gesture would have to
  learn a mode it does not have today, for capability the promote path
  already gives away.
- **A table of primitives in a dock** (the doors precedent: one row per
  cone, six spin boxes across). Honest about the format and useless as a
  world editor — authoring a hill by typing eight floats is the Python
  authoring with worse ergonomics. The dock keeps a *ledger* of what has
  no geometry to grab (§5), which is the part of the table idea that earns
  its place.
- **A heightfield brush.** The representation is not a heightfield and the
  reason is `TERRAIN.md` §1's first argument: overhangs, walls and ceilings
  are free in a field and unsayable in a height map, and the reference is a
  game about climbing.

## 4. Pinned on return, and what promotion actually changes

Demoted chunks come back **pinned**. Terrain does not move, and an unpinned
hill collapses the first time the author presses Play — the pin is what
makes "edit the world as flesh" survive contact with the transport. It is
the author's to drop from the chunk table if they want to watch a landslide,
and promotion ignores pin state entirely (the record has no mass), so the
pinning costs nothing on the way back.

What that admits, stated once and loudly, because a half-truth here is a
bug report waiting: **while a feature is chunks it is not the world.**

* It blends with creatures instead of hard-mining against them. Terrain
  composites by hard minimum precisely so a foot rests *on* the ground
  rather than becoming it (`TERRAIN.md` §1); flesh does not.
* Nothing stands on it. Creature-vs-creature collision is out of scope and
  has been since `TERRAIN.md` §5, so a walker passes straight through a
  hill that is still chunks.
* It wears its blend group's palette hue, not `look.terrain_colour`.

That last one is the design's favourite accident: **the colour change is
the receipt.** The moment a promotion lands, the shape stops being a
creature-coloured blob and turns terrain-coloured, in the same frame, with
no notification to write. The move-creature arc's live check photographed
the same mechanism from the other side — an unpainted guest arriving in a
new room wearing a different group's hue — and here it is doing useful work.

## 5. The first promotion repeals the floor, and the floor answers back

`core/solver.py` picks exactly one world per tick:

> A set, non-empty terrain is the world and `ground_y` stands down; an empty
> terrain stands down itself, because "no primitives yet" is an authoring
> state, not a request to repeal a floor the config still asks for.

So **the first cone an author promotes silently repeals the floor they were
standing on.** Promote a ridge into a room whose ground is `ground_y = 0`
and the creature falls past everything except the ridge. This is not a
defect in the solver — one world per tick is the correct rule and the
comment argues it well — but it is a cliff edge sitting directly under the
arc's first gesture.

The answer is **not** compensation. Folding the floor in automatically
would write a plane the author did not author, which is the stance PICK
took when it refused to invent an origin for a missed click and the stance
`TERRAIN.md` took when it declined to project arrivals onto the ground. The
answer is the one `rename_room` already uses when it counts the doors it
just left addressing a dead name: **say the consequence**.

> Promoted 'ridge' — 3 cones; the reference floor at 0.0 no longer acts.

And beside that sentence, the gesture that answers it: **Add Floor** takes
the reference grid — the one being drawn, at the height it is drawn at —
and promotes it into a real half-space plane. The grid's own veto then
stands it down, so the gesture reads exactly as what it is: *the reference
becomes the world.* It refuses when the grid is not drawn (a floorless
room, a debug view, a world already set), because promoting a plane the
author cannot see is inventing one.

`ground_y` gets its spin here too, in the Terrain dock, which is where it
always belonged: it is the world's height, not a solver setting, and the
Solver panel's omission was right by accident. Editing it is
`settings_edited` — file-grade, dirtying, on no undo stack — because that
is what every other `SolverConfig` field is (`docs/SETTINGS.md` §5, whose
site ledger grows by one).

## 6. Edit grade: one undo step, and the dirty seam closes by construction

The whole gesture is **one command**, not a file-grade act.

`SetTerrain` is the `ReplaceControllers` shape exactly: hold the previous
object, swap the new one in, put the old one back on revert. That is legal
without a deep copy for a reason already written into `set_terrain`'s
docstring — *identity is the change signal*, so a `Terrain` is already
treated as immutable by every consumer in the codebase. Then:

* **Make Terrain** = `Macro([RemoveChunk × N, SetTerrain(grown)])`
* **Edit as Chunks** = `Macro([SetTerrain(planes only), AddChunk × N, AddLink × M])`
* **Add Floor** = `SetTerrain(with the plane)`

One undo step each, redo included, and the chunk half of the gesture is
made of commands that have been reverting correctly for the entire life of
the editor.

**The dirty seam closes without touching `set_terrain`.** It stays the
*load* path — `adopt` routes through it after `_replace` has declared the
document clean, and a loaded world must not reopen as an unsaved change —
while `SetTerrain` is the *edit* path and dirties through `_after_edit`
like every command does. That is the settings arc's own split (view-only
emits raw and stays clean; file-grade edits dirty) arriving at the one
setting that never got it.

`_after_edit` grows a `_TERRAIN` tuple beside `_STRUCTURAL` and
`_CONTROLLER`, emitting `terrain_changed`. One line, the shape
`_CONTROLLER` already models.

Rejected: **file-grade, like rooms, doors and terrain-today.** That grade
exists for edits a per-room history structurally cannot hold — a room list,
a two-room move — and this is one room's body and one room's world. Half of
it is already command-shaped, and a gesture that is half-undoable is worse
than either grade taken whole. The abstraction refusal comes free with the
choice: `do()` already turns away every command while a room is parked, and
says "Realize 'den' to edit it" while doing it.

## 7. A prerequisite this arc walks straight into

`RemoveChunk` records its dying chunk's incident links as
`(other, rest, type, elasticity)` — and **not `render`**. `Body.add_link`
defaults `render=True`, so undo brings a mechanism link back welded into
the skin. Measured, not inferred:

```
before: [(0, 1, False)]     # a mechanism link
after : [(1, 0, True)]      # after do(RemoveChunk) + undo
```

A pre-existing defect, latent since the command was written, and this arc
drives a truck through it: promote is a `Macro` of `RemoveChunk`s and
`link_render` is the rule deciding what promotes, so the very links this
gesture is most careful about are the ones undo would corrupt. One field on
`_links`, one argument at the `add_link` call, one regression test. It
lands **first**, on its own commit, because it is a bug fix that stands
without the arc.

## 8. Refusals, each a status line and no motion

- **An empty selection** — "Select a chunk of the feature to make terrain".
- **A component any controller references** — `tempo.chunk_refs`, the probe
  the move-creature pre-flight already leans on: *"'grazer' is driven by 2
  controllers; terrain does not walk."* Refusing beats stripping, because
  silently deleting an authored rig to make a hill is the mutated-but-clean
  lie in a new costume. The author deletes the gait themselves if they mean
  it, and the refusal names the count so they know what they are deleting.
- **Edit as Chunks with no cones** — an empty terrain or a planes-only one
  has nothing to hand back, and says so rather than emptying a body.
- **A parked room** — `do()`'s own refusal, inherited by being a command.

Wholesale, and for the move verb's reason: a multi-component selection
where one component carries a controller promotes nothing at all, so the
gesture never half-happens.

## 9. Order, welding, and the round trip

Fold order is semantic — `smooth_min` is not associative, planes fold first
and cones second, both in authoring order, and `Terrain.distance` says so.
An authoring surface that scrambled order would be a surface that reshapes
the ground when the author was not looking. Three rulings keep it honest:

1. **Promote appends.** New cones go on the end, so terrain written by this
   editor is always grouped by the promotion that made it.
2. **Demote emits in cone order,** welding endpoints that match exactly in
   position and radius. Exact matching is not a tolerance gamble here: every
   shared endpoint in editor-written terrain came from copying one chunk's
   own numbers into two records, so the comparison is between a float64 and
   itself.
3. Therefore the invariant, and the arc's sharpest test: **`promote(demote(T))
   is T`, cone for cone, for any terrain this editor wrote.** Hand-authored
   XML gets the weaker and stated promise — the same set of cones, regrouped
   by component, which changes the fold only where two features already
   touch, since that is the only place a smooth minimum can tell the
   difference.

## 10. What promotion costs

A cone record is eight floats and carries nothing else, so everything meta
is dropped at the boundary: **paint** (the terrain wears
`look.terrain_colour`, one colour for the whole world), the creature
**name**, **tempo**, and the **pin state**. Demotion returns unpainted,
unnamed, pinned chunks named for what they are.

This is the same boundary a door crossing is — a serialization boundary,
where runtime state resets and only what the format carries survives — and
it is stated in the same voice: the record is the truth, and anything not
in the record was never in the world.

## 11. Tests, churn, and the file this arc owes the repository

Core: **none.** `Terrain` grows nothing; `add_plane` and `add_cone` are the
whole mechanism and were tested when they landed.

Command layer (`tests/test_document.py`): `SetTerrain` applies and reverts
by identity; the `RemoveChunk` render-flag regression (§7) with the
mechanism link surviving a delete/undo round trip; `_after_edit` emitting
`terrain_changed` for it and not for its neighbours.

Document verbs (`tests/test_document.py`): a promoted component leaves the
body and its cones arrive in order; a mechanism-linked chunk promotes as a
sphere while its rendered neighbours promote as cones; **the promoted
records equal `sdf.pack`'s rows for the same component** (an agreement test
standing where construction was not available — `pack` is float32, over
draw positions, sorted group-major for the renderer's own reasons); one
undo restores body and world together; the round-trip identity of §9; every
refusal of §8 leaving body and terrain bit-identical; Add Floor writing the
drawn height and refusing when the grid is not drawn; the first promotion's
status line naming the repealed floor.

Dock face (`tests/test_panels.py`, behind the suite's Qt guard): the verbs
disarm without a selection and without cones respectively; the ledger counts
planes and cones; the `ground_y` spin round-trips and dirties.

**The file.** `examples/` gains a world — a floor plane, a ramp and a step,
sized to the shipped quadruped so that opening it and pressing Play walks
the acceptance test from `TERRAIN.md` §4 that only ever existed in Python.
That file is this arc's real acceptance: the grep that came back empty in
§1 comes back with a name.

Churn inventory: `mesh_xml/editor/document.py` (`SetTerrain`, `_TERRAIN`,
the two verbs, Add Floor, the `RemoveChunk` fix),
`mesh_xml/editor/panels.py` (a `TerrainPanel`), `mesh_xml/editor/app.py`
(the dock, **and the tab tuple — mandatory, or the dock stretches the
column**), `docs/TERRAIN.md` §5's "not in scope" line, `docs/SETTINGS.md`
§5's site ledger, the README row, this document, and the example.
`core/`, `scene/`, `io/`, `render/`: untouched — the format already carries
everything this arc can author, which is the strongest evidence that §2's
identity is the right spine.

## 12. Deliberately not done

- **Boxes.** `TERRAIN.md` §1 already holds the door open for them and
  states the price: the CPU/GPU pair grows together, on the day chunky-soft
  stops being the right look. Not a day this arc gets to declare.
- **Moving platforms.** Terrain is static in the solver's world phase; a
  moving one is a physics arc, not an authoring one.
- **Per-primitive terrain colour.** One world, one colour, by `LIGHTING.md`'s
  own record. A per-feature look is the same argument the per-creature look
  lost.
- **Dragging the terrain surface directly.** That is the modal tool of §3
  wearing a smaller hat, and it buys nothing the demote path does not
  already give.
- **A separate terrain history.** One room, one stack; the commands are
  ordinary commands and interleave with body edits exactly as they should.
- **Importing a mesh or a height map as terrain.** The vocabulary is
  deliberately small and the import would be a mesh-fitting arc wearing the
  authoring arc's name.
