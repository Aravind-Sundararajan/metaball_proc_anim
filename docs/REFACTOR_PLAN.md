# Refactor Plan: from XML sphere editor to procedural animation + metaballs

**Status:** all phases landed — see §7
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

- Document plus command stack, which makes the `undo`/`redo` stubs real. The legacy app
  had both menu actions wired to empty methods.
- The legacy app had `main.py`, `opengl_widget.py` and `spline_generator_simple.py` all
  mutating the manager directly and then each manually calling `bone_model.update_bones()`
  and `opengl_widget.update()`. Miss one and the UI goes stale. Move to
  document-emits-signals, views-subscribe.
- The app splits into **Rig mode** (author chunks, links, curves) and **Play mode** (run
  the sim, scrub, tweak live).
- The curve editor generalizes from "radius" to any named curve on the rig — taper,
  stiffness-along-chain, gait timing, squash response.

#### Interaction inventory carried forward from the deleted editor

The legacy app was never used in anger, but its UI is the only written record of the
intended interaction, so it is transcribed here and the code deleted rather than kept
around. Retrieve the original with `git show ec05cdf:main.py` if a detail is needed.

| Panel | Controls |
| --- | --- |
| Create Sphere | radius spin box; create-at-origin button |
| Edit Position | X / Y / Z spin boxes, live-bound to the selection |
| Armature | connection mode combo (Direct / Chain); editable bone-name table; connect, disconnect, clear-connections, clear-selection, delete |
| Spline Generator | sphere count, base radius, spline length, curve type (Linear / Sine / Spiral / Custom), radius-curve editor, generate |
| Export | export to XML |
| Menus | New / Open / Save / Exit, Undo / Redo, on standard shortcuts |

Viewport interactions, all of which the new viewport must eventually regain: left-drag to
orbit, wheel to zoom, click to select, shift-click to connect under the active mode,
right-click to create at the ground-plane intersection, Delete to remove the selection.

Two of these are worth rethinking rather than reproducing. **Chain Connect** existed
because the old model could not express a cycle, so building a loop meant sequencing edges
by hand; the constraint graph makes it far less necessary. And **right-click creation onto
the y=0 ground plane** is a 2-D affordance in a 3-D scene — it was clamped to ±10 and fell
back to the origin whenever the ray missed. Placement relative to the current selection, or
onto the surface itself, fits the tool better.

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
`armature_manager.py`) was deliberately kept runnable while the new stack was built
alongside it rather than in place of it, so there was never a point with nothing
working. It was retired (`6ab81c6`) once the editor covered it.

| Phase | State | Notes |
| --- | --- | --- |
| 0 — stop the bleeding | done | Baseline commit first; the tree was entirely untracked. |
| 1 — core rewrite | done | `core/{body,solver,curve}.py`, `io/{xml_v2,xml_legacy}.py`. |
| 2 — sim loop | done | `anim/clock.py`, `render/draw_points.py`. |
| 3 — metaballs | done | `render/{sdf,field}.py`, `render/gl/`. Validated against the CPU oracle. |
| 4 — controllers | done | `anim/controller.py`, `anim/controllers/`. |
| 5 — editor rework | done | `editor/`. Runnable via `poetry run mesh-xml`. |

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

### Beyond the original phases

Landed after Phase 5, in one fan-out:

* **Gait serialization (v3).** A creature saves whole — controllers beside
  chunks and links, registry keyed on constructor signatures, editor wired.
  The quadruped reopens to a bit-identical walk.
* **Mesh export.** Surface nets over the CPU field; closed, consistently wound
  meshes to OBJ/STL. The worm meshes at 96^3 in 0.13 s.
* **Self-propulsion.** Opt-in reaction force on the limb root plus a postural
  controller: an unpinned quadruped walks +13.4 units in 50 s from footfalls
  alone, with sign-reversal and frictionless-floor controls proving the
  mechanism. The reaction pairs unconditionally — gating it on the plant
  travels five times further on un-paired swing forces, which is a momentum
  source, not locomotion.
* **Editor authoring.** Radius-curve editor (Curve1D as the model, one apply =
  one undo), orthographic toggle, per-point smoothing on the selection.
* **Solver momentum conservation.** The overlap averaging pumped momentum into
  unevenly-connected pairs — a fifth of the first walker's thrust. Replaced by
  per-link symmetric weights: conserving for any masses, same hub bound.
* **Self-contact.** Opt-in collision between a body's own chunks when they
  are graph-strangers — three or more links apart (``docs/CONTACT_MODEL.md``
  §7). Discovered per tick, resolved as transient PUSH contacts under the
  link pass's momentum-conserving symmetric weights; an uneven free cluster's
  centre of mass holds to 1e-3 over 300 ticks. Neighbours and second
  neighbours stay exempt — authored overlap is the surface. Elasticity is the
  restoring-force knob: 1.0 is billiards, the 0.5 default is flesh. Runs
  before the world passes, so the floor keeps the last word: two overlapping
  strangers dropped under gravity stack at one and three radii.
* **Chunk creation in the editor.** The one §5 inventory item that had landed
  nowhere: the app could open, retune, link and simulate a creature but not
  begin one, so rigs were born in scripts. `Add` — Chunks-panel button, Edit
  menu, shortcut `A` — grows from the selection as §5 asked: linked to it,
  radius inherited, continuing the chain; several selected place outward and
  link to each, which closes a girdle; none makes a free chunk clear of the
  body. One add is one undo step (chunk and links arrive and leave together),
  and the new chunk takes the selection, so repeated Add extrudes a limb —
  the composing gesture the legacy create-at-origin button never had.
* **Viewport orientation.** A reference grid floor in the raymarcher — an
  analytic plane at the solver's floor height, screen-space anti-aliased
  lines, distance fade, and a penumbra-marched shadow of the creature, which
  is what visually grounds it. Never part of ``sd_scene``, so the CPU-oracle
  comparisons stay exact; stands down when a terrain is loaded, because two
  floors contradict each other. Plus a clickable axis gizmo in the overlay
  and named view snaps (``OrbitCamera.snap_view``, View menu, Blender's
  1/3/7 keys).
* **Examples carry their gaits.** The shipped armatures were bare v1 relics:
  opened in the editor, Play just dropped them, which read as a broken app.
  ``WaveAnchor`` — the drive that made ``animate_rig.py`` wriggle, previously
  script-local and unsaveable — is now a registered controller
  (``anim/controllers/drive.py``), and every ``examples/*.xml`` is a v3
  document with autorig plus that driver baked in: open the chubby worm,
  press Play, it swims. The v1 originals moved to ``tests/data/legacy_v1_*``
  where the legacy-reader tests always wanted them.
* **Blend groups: every creature its own skin.** The SDF was one global
  smooth minimum, so creatures within blend distance welded into shared goo —
  the identity wall the herd benchmark named. Now ``sdf.pack`` orders rows
  group-major by connected component (over *all* links, mechanism braces
  included — identity is physical, not fleshly) and ``sdf.group_table`` emits
  ``[start, count][r, g, b]`` rows that ride the existing stride-8 texture
  machinery on a third unit. Smooth minimum within a group, hard minimum
  across — associative, so cross-group order never becomes a contract — and
  the winning group colours the hit: group 0 wears the authored surface
  colour exactly, later groups rotate the hue by the golden-ratio step, so a
  herd colours itself. CPU oracle and shader changed in lockstep; a grouped
  parity test holds them to the same tolerances as the global fold, and a
  document with no groups uploaded renders bit-for-bit as before.
* **The bound cull: herd frames at less than half the cost, provably free.**
  Each blend group carries a bounding sphere (``sdf.group_bounds``, radius
  covering every end ball plus the smooth-min inflation ``k``, re-uploaded
  per frame through the same stride-8 machinery on a fourth unit), making
  ``|p - c| - r`` a true lower bound on the group's field. The fold then
  evaluates the nearest-bound group first — a tight running minimum
  immediately — and skips any group whose bound cannot beat it; folding in
  table order instead leaves ~ln(G) record-setters folding per step, which
  was most of the win left on the table. The floor's shadow march gets the
  same treatment: a group can only dim the penumbra within ``max_t/8`` of
  the light ray, so floor pixels no bound reaches skip the 48-step march
  outright. Every skip is exact — a skipped fold is never evaluated, and the
  survivors' arithmetic is untouched — enforced by byte-equality tests over
  culled and unculled frames, shaded and packed-depth, floor on and off.
  Measured at 1280x800: 40 worms 66.8 to 27.8 ms creature-only, and the
  editor's default floor-on frame 104.0 to 36.9 ms — 2.8x.
* **The tiled broad phase — the "GL 3.3 has no compute stage" blocker,
  dissolved.** The prepass never needed one: the CPU already owns every
  per-frame derived artefact, and ``render/tiles.py`` builds per-screen-tile
  group bitmasks (conservative sphere projection, ~0.4 ms at 40 worms) that
  ride an ``RGBA32UI`` texture — integer textures are core GL 3.0. The
  fragment shader arms the mask from ``gl_FragCoord`` and iterates only the
  groups whose bounds the tile's rays can approach; shadow rays stand the
  mask down, since their occluders are not the camera tile's business. This
  is the one optimisation that is *surface-exact rather than byte-exact* —
  field values far from surfaces are legitimately global, so empty-space
  step sizes drift — and the module docstring owns that contract: same
  surfaces hit, same colours, ``t`` within the march's tolerance, silhouette
  flips confined to the hit boundary, and the untiled path bit-identical.
  Measured win over the bound cull alone: 4-9%, growing with herd size
  (24.7 vs 27.6 ms at 40 worms) — modest because nearest-first already ate
  the meal, kept because the delta grows with G and the structure is what
  larger tile work hangs off. An adversarial review fleet (14 agents) then
  confirmed and fixed what single-pass work missed: ``resizeGL`` receives
  *logical* pixels (its old comment claimed the opposite) and never runs in
  the offscreen path, so the framebuffer size is now derived fresh every
  paint; the unset integer sampler aliased unit 0 against a float sampler —
  a draw-time GL error on conformant drivers this machine's happened to
  forgive; the mask's view transform used a differently-rounded float32
  path than the shader's rays, breaking conservatism beyond |coords| ~3e6
  (now built from the rays' own inputs, regression-tested at 1e8); a float32
  ulp hole in the cull's byte-exactness at k = 0 (bounds now round outward
  by 8 ulps); a one-frame group-table/primitive desync on topology edits
  (the table now snapshots beside the primitives it describes); and the
  tile slack now scales with the settings' own ``max_distance`` instead of
  assuming the default.
* **The cone prepass, and the plateau named.** A coarse pass at quarter
  resolution sphere-traces each pixel block's whole ray bundle — stepping
  only the slack beyond a cone wide enough to contain it — and writes the
  proven-empty depth to an R32F target that seeds every full-resolution
  ray. Same 1-Lipschitz safety argument as the bound cull, same
  surface-exact contract as the tiles, held by the same hit-set and drift
  tests in both projections; the oracle harness leaves it off and renders
  the single-pass path bit-for-bit. Measured honestly: ~15% on small
  scenes, 0-2% at herd scale — because the cull and tiles had already made
  empty-space steps nearly free, and what remains is near-surface marching
  and shading, which no seed can skip. Kept for the small-scene win and
  the second-pass machinery, and recorded as the point where the
  march-acceleration ladder converged: global fold to cull 2.4x, tiles
  +8%, prepass +2%. Further frame time lives in shading cost or resolution
  scaling, not in marching fewer steps.
* **Creature colour, honest meta, and unwelded exports.** Blend groups grew
  from render trick to authored data. First the prerequisite bug: v3 saves
  silently dropped chunk meta — the very fields the §2.4 fix preserved on
  import — and the legacy reader itself only kept them in a sidecar the
  editor discards, so a capsule's type and height never actually reached
  disk. Meta now rides ``<meta>`` children (additive, no version bump) and
  the legacy importer stamps the chunk too, closing §2.4's journey at last.
  On that foundation: an authored creature colour lives in chunk meta under
  ``sdf.COLOUR_META``, painted onto every chunk of a component (a creature
  has no object of its own; a colour on one chunk dies with it) by
  ``Document.set_creature_colour`` — one undo step, Surface-panel picker,
  outranks the golden-angle palette, garbage falls back rather than
  breaking the render. And ``File > Export Mesh`` finally exists: surface
  nets over the *grouped* field, so a herd arrives in Blender as separate
  animals — the hard minimum of exact fields is the exact union field, so
  the mesh stays as closed as the ungrouped case.
* **Rig Legs: the worm becomes a quadruped in one click.** The decision the
  autorig declined — legs need to know how the creature is driven — is now
  made from evidence: ``autorig.fit_legs`` compiles the *measured*
  self-propelled walker (the balance suite's, 19.8 units in 25 s) onto every
  ``leg_candidates`` chain. Each ``(spine, hip, knee, foot)`` becomes a
  ``LimbIK(reaction=1.0)`` fed by a trot-phased ``FootholdPicker`` whose
  stance is the leg's own authored droop and whose step lengths normalise by
  leg span; the trunk and hips get ``Posture`` pressing on the feet, the
  hips get ``Level`` and a mechanism-link truss (a lone link is isotropic
  and lets a hip swing under the centreline); every pin is removed, because
  a walker's weight must reach its feet. Heading, drive, trot pairing and
  pole vectors all derive from the trunk's own axis, so a creature walks
  the way it was authored to face. ``Document.rig_legs`` lands it as one
  undo step, the panel gets the button beside Fit Autorig, and ``Posture``
  and ``Level`` are registered at last — which also makes the balance
  suite's walker savable for the first time. Proven end to end: a
  gesture-shaped legged worm, fitted automatically, walks metres along its
  own heading with the trunk held high, and round-trips through v3 whole.
* **The behavior tier** (``anim/behavior.py``). Brains above controllers,
  obeying the same law one level up: a brain writes goals — destinations,
  choices — never forces, never positions it does not own. Two architectures
  ship as registered controllers: ``Grazer``, a finite state machine
  (``Fsm``: transitions settle within one think, so a creature acts on a
  decision the moment it makes it), and ``Hunter``, goal-oriented action
  planning (``plan``: A* over symbolic preconditions and effects, re-planned
  whenever the world stops matching the plan's assumptions). Both think on
  their own slow cadence, steer only their creature's pinned anchor — the
  same authority ``WaveAnchor`` holds and the editor's hand overrides — and
  sense each other through ``ctx.peers``, the stack as Rain World's room
  object list. ``examples/hunt.xml`` is the proof: a predator and prey saved
  as one ordinary v3 document, and the faster hunter feeds within thirty
  simulated seconds from any seeded start (the integration test's claim).
* **Creature tempo: one clock for the whole gait.** A creature's pace was
  smeared across its stack — trot ``period``, ``step_duration``, ``lead_time``
  and ``drive`` on every picker, the brain's travel ``speed`` above them — so
  speeding one up by hand meant five coherent edits or an incoherent creature.
  Each controller now declares its tempo-dimensioned fields in a ``TEMPO``
  attribute beside the fields themselves (rates and speeds multiply by the
  factor, periods and durations divide; ``anim/tempo.py`` intersects that with
  the registry's persisted list, so a knob the file cannot carry can never be
  silently rescaled, and recurses into the picker nested in a limb).
  ``Document.set_creature_tempo`` lands the rescale and the authored value —
  chunk meta, the colour mechanism — as one undo step, with the ratio taken
  per creature so a mixed-tempo herd lands together. Shape survives speed by
  construction: spatial triggers never move, and the shrunken lead times
  times the grown velocities predict the same distance ahead. Cognition
  (``think_period``) and muscle (strengths, dampings) deliberately keep their
  values — a fast creature is neither hasty nor propped up, and a gait pushed
  past its own muscles fails honestly in the solver. The Controllers panel
  carries the spin; the money test opens the shipped worm at tempo 2 and it
  sweeps the same anchor figure in half the ticks — a clock, not a new gait.
* **The third brain: a utility scavenger that eats what the story leaves.**
  ``choose`` — scores in, best out, a commitment bonus so boundaries do not
  flicker — joins ``Fsm`` and ``plan`` as the third architecture, one
  primitive per brain kind. ``Scavenger`` is the creature that earns it: it
  neither chases nor is chased, but watches every hunter/grazer pair through
  ``ctx.peers``, and when a chase concludes within its senses — the pair
  converging inside ``scrap_range`` — it remembers the midpoint as a scrap
  site that ages out over ``site_ttl``. That is perception the first two
  brains never needed: a relation between two *other* creatures, and a memory
  of where something happened rather than where anything is. The trade-offs
  fall out of scoring, not rules — a guarded kill offers no appetite, so it
  lurks at ``observe_range`` until the owner wanders off, and fear always
  outbids food, so a feed run aborts the moment the hunter turns back.
  ``examples/scavenge.xml`` ships the food chain as one v3 document (hunter
  feeds at 8 simulated seconds, scavenger robs the site at 33), and the
  integration test asserts one bit — the scavenger fed — which proves the
  chain end to end, because feeding is only reachable through witness,
  memory, patience and the dart.
* **The creature roster: the herd gesture, and creatures you can name**
  (``docs/ROOMS.md`` package 1). ``Body.components()`` makes the graph's own
  answer to "how many creatures live here" public, in the renderer's
  blend-group order — the roster's creature *n* wears surface group *n* —
  and ``Document.creatures()`` names each from chunk meta the way colour and
  tempo already ride there. The Creatures panel picks (which aims the
  selection every per-creature gesture fires from), renames, and duplicates.
  ``duplicate_creature`` copies a creature *whole*: chunks with masses, pins
  and meta; links with rests, types and render flags (a brace stays a
  brace); and its controllers rebuilt through ``serialize.remap_ids`` — the
  decode path without the document, fresh instance, runtime re-derived —
  with world points carried sideways by ``anim/place.py``, the third
  per-class field declaration (``PLACED``, after ``TEMPO``): a copy whose
  driver still pointed at its ancestor's ``origin`` would fight it for one
  soul, and a ``pole`` translated like a point would stop being a direction.
  Placement is the Add rule one creature up — past the body's +x extent,
  copies in a row — one undo step, the copy takes the selection, so
  duplicate-again herds. The money test herds the shipped worm: two waves,
  two anchors, both swim, and the herd round-trips through v3 whole.
* **Controllers in the GUI, and grabs that win.** A Controllers panel: one
  group per controller on the stack, an enabled switch, spins for every
  numeric field the v3 registry persists — the widget list *is* the format's
  field list (``serialize.fields_of``), so a knob the file cannot save never
  appears — plus per-controller Remove and one-click Fit Autorig, all
  undoable. And the grab standoff is settled the way Rain World settles it,
  by authority rather than force: a pinned anchor is infinitely massive, so
  the hand cannot out-pull its driver — instead ``ControllerStack.held``
  marks hand-owned chunks, drivers stand down for them, and releasing the
  drag re-authors the driver's ``origin`` minus its current swing, one undo
  step, so the worm keeps swimming from exactly where you dropped it.

**Terrain is landed** (`docs/TERRAIN.md`). The two contact-model limits that gated it are
closed — position-level Coulomb friction (grip scales with load, zero creep
below the threshold) and angle limits compiled to PUSH/PULL brace links (a
braced pair of legs holds a load with no postural spring). Design, corrections
and integration measurements in ``docs/CONTACT_MODEL.md``.

### Deferred

The plan's Phase 2 item to replace the unconditional 16 ms `QTimer` in `main.py:134` has
been moved to Phase 3. That timer drives the *legacy* widget against the *legacy*
`ArmatureManager`; the clock that replaces it belongs to the new viewport, which Phase 3
builds. Fixing it in place would mean wiring the new simulation into the old widget only to
delete that wiring immediately afterward.
