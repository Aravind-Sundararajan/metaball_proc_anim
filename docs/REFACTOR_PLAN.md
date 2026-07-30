# Refactor Plan: from XML sphere editor to procedural animation + metaballs

**Status:** Phases 0–2 landed, Phase 3 in progress — see §7
**Date:** 2026-07-30
**North star:** Rain World's creature simulation model (`PhysicalObject` / `BodyChunk` /
`BodyChunkConnection` / `GraphicsModule` / `Limb`).

---

## 1. Diagnosis

There is no procedural animation and no metaball code in this repository. Zero matches
across all 3,608 lines for `metaball`, `marching`, `isosurface`, `sdf`, `smin`, `shader`,
`glsl`, `animate`, `keyframe`, `timeline`, or any fixed-timestep concept.

What exists is a static sphere-placement tool: it writes a transform tree to XML and draws
GLU spheres with fixed-function OpenGL.

Two load-bearing abstractions are wrong for the target:

### 1.1 The rig is a tree of transforms; it needs to be a graph of masses with constraints

Rain World's `PhysicalObject` owns `BodyChunk[] bodyChunks` and
`BodyChunkConnection[] bodyChunkConnections`. There is **no parent transform at simulation
time** — hierarchy exists only in template data and in the graphics module.

`armature_manager.py` instead:

- stores world positions but exports relative ones,
- re-derives hierarchy from an undirected edge list on every export,
- and actively **rejects cycles**.

Rain World body topologies are cyclic. A hip girdle, a ribcage, and a palm are all closed
loops. The current model can never express them.

### 1.2 The renderer is fixed-function

`glBegin`/`glEnd`, `gluNewQuadric` allocated and destroyed per sphere per frame,
`GL_LIGHTING`. An SDF fragment shader cannot be written against this pipeline. Porting to
core-profile GL is a **prerequisite for metaballs**, not a cleanup task.

---

## 2. Verified defects

Each item below was reproduced by executing the code, not inferred by reading it.

### 2.1 Silent total data loss — `armature_manager.py`

Connect `A→B`, then `B→A`. Both calls return `True`, hierarchy resolves to `{}`, and export
writes `<armature/>`. The entire rig vanishes with no error or warning.

Reachable from the UI in three shift-clicks. `_would_create_loop` skips the parent edge
during DFS and therefore never sees the reversed duplicate.

### 2.2 Merges are impossible

`A→B`, `A→C`, `B→D` all succeed; `C→D` is rejected as a closed loop. No closed chunk
topology can ever be authored.

### 2.3 The XML format silently broke

Export changed from absolute to relative coordinates with no version marker.
`test_export_spline.xml` is an old-format file still in the repository. Importing it today
places the last sphere at **x = 20.0 instead of 4.444** — cumulative garbage, no warning.

### 2.4 The importer is lossy

`character_armature.xml` uses `type="capsule"` with a `height` attribute. Both are
discarded; every element returns as a sphere. `Root` has no `<mesh>` child and silently
receives the default `radius = 0.5`.

### 2.5 The sampled curve is not the drawn curve

In `spline_generator_simple.py:392`, `handles[i+1]` is the **out**-handle of the next
point. `update_curve` (line 264) correctly draws using `next_point.in_handle`. Generated
radii therefore do not match the displayed curve.

Additional defects in the same widget:

| Location | Defect |
| --- | --- |
| `:388` | `t` computed as an x-fraction and fed as the Bézier parameter — wrong unless handles sit at exact x-thirds |
| `:80` | Handle-follow is a no-op: `setPos(pos + (handle - pos))` == `handle` |
| `:306` | `add_point()` always inserts at x=150, stacking duplicates at one coordinate |
| ×8 | The `2.0 - (y/180)*1.9` y-to-radius mapping is copy-pasted eight times |

### 2.6 Identity is a dict key

Two spline generations both emit `Spline_1..N` and silently overwrite each other.
`sphere_counter` resets to 0 on import, so newly created spheres collide with imported ones.

### 2.7 Roughly 1,200 lines are dead

| File | Lines | Status |
| --- | --- | --- |
| `spline_generator.py` | 407 | never imported |
| `opengl_widget_simple.py` | 355 | never imported |
| `curve_editor_pyqtgraph.py` | 297 | never imported |
| `curve_editor_simple.py` | 130 | never imported |

`main.py` aliases `SplineGeneratorSimple as SplineGenerator`, so the "simple" variant won.

### 2.8 Zero git commits

Every file was untracked. A baseline commit precedes all deletion work.

---

## 3. Plan

### Phase 0 — stop the bleeding

- Baseline commit before touching anything.
- Delete the four dead modules.
- Normalize link edges so `(A,B)` and `(B,A)` are the same edge.
- Stamp `version="2"` on exports; reject or convert v1 on import.
- Move stray generated XML into `examples/` and `tests/data/`.

### Phase 1 — rewrite the core (no Qt, no GL)

```
mesh_xml/core/
  chunk.py      Chunk(id, name, pos, prev_pos, radius, mass, pinned)
  link.py       Link(a, b, rest, type: NORMAL|PULL|PUSH, elasticity)
  body.py       the graph — cycles allowed, integer ids, name as a field
  solver.py     Verlet integrate + N iterations of distance projection
  curve.py      pure Curve1D, no QGraphicsItem
  serialize.py  v2 read/write + legacy v1 reader
```

`Link` is Rain World's `BodyChunkConnection`: a rest distance, an inverse-mass-weighted
correction, and a type mask where `PULL` corrects only when too far apart and `PUSH` only
when too close. That type mask is what makes a spine feel like a spine and a tentacle feel
like a rope.

Positions are stored struct-of-arrays (`pos: (N,3) float32`) so the solver vectorizes.
numpy is already a dependency and is currently unused.

This layer is pure math and genuinely unit-testable. The three reproductions in §2 are the
first three test cases.

### Phase 2 — the simulation loop

- Fixed timestep with an accumulator. Rain World ticks at 40 Hz; 60 is also fine.
- Render with `alpha` interpolation between `prev_pos` and `pos`.
- Replace the unconditional 16 ms `QTimer → update()` in `main.py:134`, which currently
  repaints continuously whether or not anything has moved.

Then the piece that produces the actual look: **draw positions are not chunk positions.**
Rain World's `GraphicsModule` keeps its own points that chase the physics chunks with their
own springs. That lag is where all the gelatinous secondary motion comes from. It is about
twenty lines and it is the highest visual-payoff item in the plan.

### Phase 3 — metaballs

- Port the viewport to core-profile GL 3.3 (`QOpenGLShaderProgram`, a VAO, a fullscreen
  quad).
- Raymarch an SDF built by polynomial smooth-min over a sphere per chunk **and a round
  capsule per link**.

The capsules matter. Spheres alone read as beads on a string; capsules along the links are
why `character_armature.xml` was authored with `type="capsule"` in the first place.

Upload chunk data as an SSBO each frame. At ≤128 primitives this is trivially 60 fps.

Expose the smooth-min blend radius `k` as an authored per-body parameter. That single knob
is the difference between "spheres" and "creature".

Marching cubes / surface nets comes later, on a background thread, for mesh export only —
never for the viewport.

### Phase 4 — procedural animation

`anim/controllers/`:

- `SpineFollow` — each chunk chases the one ahead with lag and a maximum bend angle.
- `LimbIK` — two-bone analytic IK with a flip/pole sign.
- `FootholdPicker` — Rain World's `Limb.mode` state machine: hunt an absolute world point,
  re-pick when it goes out of reach.
- `LookAt`, `Breathe`.

**Critical ordering constraint: controllers write target positions or forces, never final
positions.** The constraint solver reconciles afterward. Controllers writing positions
directly is the single most common way these systems go unstable.

Seed a per-creature RNG from the body id so individuals vary in proportion. It is cheap,
and it is most of why Rain World creatures read as animals rather than instances.

### Phase 5 — editor rework

- Document plus command stack, which makes the two `undo`/`redo` stubs at `main.py:749`
  real.
- Currently `main.py`, `opengl_widget.py`, and `spline_generator_simple.py` all mutate the
  manager directly and then each manually calls `bone_model.update_bones()` and
  `opengl_widget.update()`. Miss one and the UI goes stale. Move to document-emits-signals,
  views-subscribe.
- The app splits into **Rig mode** (author chunks, links, curves) and **Play mode** (run
  the sim, scrub, tweak live).
- The curve editor generalizes from "radius" to any named curve on the rig — taper,
  stiffness-along-chain, gait timing, squash response.

---

## 4. Target layout

```
mesh_xml/
  core/        chunk.py link.py body.py solver.py curve.py serialize.py   (no Qt, no GL)
  anim/        controllers/{spine.py, limb_ik.py, foothold.py} clock.py
  render/      gl/{context.py, shaders/*.glsl} sdf_raymarch.py meshing.py draw_points.py
  editor/      app.py document.py commands.py viewport.py panels/*.py
  io/          xml_v2.py xml_legacy.py obj_export.py
tests/         test_solver.py test_curve.py test_serialize.py data/*.xml
examples/      *.xml
```

---

## 5. Dimensionality decision

Rain World is 2D, and much of its cleverness is cheap specifically because of that: limb IK
reduces to a flip sign, footholds test against tile geometry, smoothing happens on sprites.
In 3D you inherit pole vectors, twist along chains, and volumetric marching.

**Decision: stay 3D, but keep `core/` dimension-agnostic.** The chunk/link solver is
identical in 2D and 3D when it operates on numpy arrays. Use SDF raymarching rather than
meshing for the viewport — raymarching is genuinely easier in 3D than marching cubes is.

Phases 1 and 2 are valid under either choice. Only Phase 4's foothold logic is affected.

---

## 6. Sequencing

Phases 0 and 1 carry the leverage — nothing else is buildable until the core model is a
constraint graph. Phase 3 is what makes the project the thing it was described as.

---

## 7. Progress

Branch `refactor/core-rewrite`. The legacy app (`main.py`, `opengl_widget.py`,
`armature_manager.py`) is deliberately untouched and still runs; the new stack is being
built alongside it rather than in place of it, so there is never a point with nothing
working.

| Phase | State | Notes |
| --- | --- | --- |
| 0 — stop the bleeding | done | Baseline commit first; the tree was entirely untracked. |
| 1 — core rewrite | done | `core/{body,solver,curve}.py`, `io/{xml_v2,xml_legacy}.py`. |
| 2 — sim loop | done | `anim/clock.py`, `render/draw_points.py`. |
| 3 — metaballs | in progress | Packing contract landed; field and GL scaffolding underway. |
| 4 — controllers | not started | |
| 5 — editor rework | not started | |

### Defects closed

* §2.1 reversed duplicate links — links are undirected and deduplicated; the graph can no
  longer be silently emptied.
* §2.2 rejected cycles — cycles are legal, and `spanning_forest()` reports non-tree edges
  rather than dropping chunks.
* §2.3 silent format break — v2 carries a version, and the legacy reader detects which era
  a v1 file came from. The absolute-era fixture reads 4.444 where it had read 20.0.
* §2.4 lossy import — mesh type, height and material survive as opaque chunk metadata; a
  missing `<mesh>` is reported rather than silently defaulted.
* §2.5 curve defects — one accessor supplies the control points for both sampling and
  drawing, the Bézier parameter is solved for rather than approximated, handles translate
  with their key.
* §2.6 name-as-identity — identity is a stable integer id; duplicate names are legal.
* §2.7 dead code — deleted.
* §2.8 no commits — baseline plus per-phase history.

### Decisions taken during implementation

* **Constraint corrections are averaged over each chunk's acting links, not summed.**
  Summing diverges to NaN within ~400 ticks at valence 4 and above, which any cyclic body
  reaches immediately. Measured, not assumed.
* **Legacy mesh fields stay opaque metadata rather than becoming `Body` fields.** A v1
  capsule's `height` describes a segment, which in this model is a link's rest length, not
  a property of either chunk it joins. Promoting it would reintroduce the per-bone mesh
  concept the rewrite exists to remove.
* **Every SDF primitive is a round cone; links carry the geometry.** A chunk emits a
  primitive only when nothing links it, because a cone already contains its end spheres and
  smooth-min over the redundancy would inflate the surface by `k/4`.
* **Solver iteration count is a per-creature softness knob, not a global constant.** At the
  default 4 iterations a hanging chain gives ~8% under load, halving with each doubling of
  iterations. For a Rain World body that give is desirable, not a defect.

### Deferred

The plan's Phase 2 item to replace the unconditional 16 ms `QTimer` in `main.py:134` has
been moved to Phase 3. That timer drives the *legacy* widget against the *legacy*
`ArmatureManager`; the clock that replaces it belongs to the new viewport, which Phase 3
builds. Fixing it in place would mean wiring the new simulation into the old widget only to
delete that wiring immediately afterward.
