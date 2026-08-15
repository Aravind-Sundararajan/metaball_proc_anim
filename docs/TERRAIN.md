# Terrain: a world that is also a field

**Status:** landed — integration measurements in §5
**Unblocked by:** `docs/CONTACT_MODEL.md` — grip that scales with load, joints that stop

---

## 1. Representation: the world is an SDF, like the creature

The candidates were a heightfield, a triangle mesh, and a signed distance
field. The field wins on every axis that matters here:

* **The creature already is one.** Collision against an SDF is one line —
  `d = distance(pos) − radius` — and the contact normal is the gradient. The
  Coulomb formula generalises unchanged: the slide is the displacement minus
  its component along the normal, and grip is still `μ` times the depth.
* **Overhangs, walls and ceilings are free.** A heightfield can never say
  "ceiling", and the Rain World reference is a game about climbing. A field
  has no preferred axis.
* **The renderer needs nothing new in kind.** The raymarcher already walks
  primitive arrays; terrain is a second array and a hard minimum against the
  creature — hard, because a foot must rest *on* the ground, not smooth-blend
  into it and become it.
* **Footholds are projection.** The nearest surface point is
  `p − d(p)·normal(p)`, iterated twice. A foothold *on a slope* falls out of
  the same machinery that placed one on the plane.

**Vocabulary, deliberately small:** infinite half-space planes plus the same
stride-8 round cones the creature is made of, folded with terrain's own
smooth-min radius. Ramps are tilted buried capsules, steps are fat ones,
pillars stand on end; a plane blending into a ramp gains a fillet, which reads
as landscape rather than as level geometry. Boxes can join the vocabulary
later if chunky-soft stops being the right look — that day, the CPU/GPU pair
grows together.

## 2. The contract move this forces

`round_cone` and `smooth_min` live in `render/field.py`, and `core/` must not
import from `render/`. The pure distance math moves to `core/distance.py`;
`field.py` keeps its public API and delegates. The line-for-line pairing
invariant becomes **`core/distance.py` ↔ `shaders/sdf.glsl`** and is restated
in both files. `core/terrain.py` then builds on it:

```python
Terrain(blend=0.3)
  .add_plane(normal, offset)      # half-space, unit normal
  .add_cone(a, ra, b, rb)         # the stride-8 record
  .distance(points)               # smooth-min fold, (N,) from (N,3)
  .normal(points)                 # tetrahedron gradient, unit
  .project(points, clearance)     # nearest point at `clearance` off the surface
Terrain.flat(y)                   # the old world, spelled in the new vocabulary
```

`SolverConfig.terrain` exists from the contract commit (documented as landing
with collision); `ControlContext.terrain` passes it through to controllers.

## 3. Work packages

1. **Contract (single-owner, first):** the `core/distance.py` move, `Terrain`,
   the config field and context passthrough. No behaviour change anywhere.
2. **Collision** (`core/solver.py`): `_collide_terrain` — push out along the
   normal, Coulomb on the tangential plane; the `ground_y` fast path stays.
   Acceptance is physical: a chunk resting on a slope of angle θ stays put
   while `tan θ < μ` and slides above, bracketing the threshold.
3. **Footholds** (`anim/controllers/limb.py`): the picker learns to project
   candidates onto a `Terrain` instead of flattening them to a plane, offset
   by the tip's radius (`LimbFrame` grows what it needs). *Dangling* starts
   meaning something: a gap in the floor is a candidate the projection cannot
   bring within reach.
4. **Rendering + persistence** (`render/gl/`, `editor/document.py`,
   `io/xml_v2.py`): a second primitive texture, hard-min composite with its
   own colour in the shader, `<terrain>` as an additive v3 section (absent
   means none — no version bump), `Document.terrain` through save and open.

2–4 are file-disjoint and run in parallel. Unit tests live with each package;
the cross-cutting acceptance is integration, after they land:

* **The ramp walk** — the quadruped crosses a plane, climbs a gentle ramp,
  and its footholds sit on the incline, not at any global height.
* **The slope-μ law** — walking up ever-steeper ground until grip fails, at
  roughly the angle whose tangent is μ. If that comes out wildly off, the
  contact model is lying somewhere.

## 4. What landed, measured

Every package delivered on its acceptance. The slope law resolves to a tenth
of a degree on a resting chunk; ``Terrain.flat`` reproduces the plane world
bit-for-bit; an overhang stops an upward push. Footholds project to the
surface at 2e-9 residual, a retreat on a 20° incline slides ``drive·cos²θ``
because re-projection annihilates the normal component, and Dangling now means
a hole in the world. GPU and CPU agree on every pixel of a floor-and-mound
scene, and the fold order is pinned from both sides. ``<terrain>`` is additive
in v3 and the document keeps one world, never two copies.

The composed result (``tests/test_terrain_gait.py``): the walker crosses the
floor and climbs the ramp with every foothold on the terrain surface, climbs
at 5–10°, stalls at 12–14°, and slides outright at 25° — all below the static
ceiling of ``atan(μ) = 19.3°``, as it must be, since propulsion spends the
same friction budget standing still keeps. A guard test fails if any future
change lets walking beat the static limit: that would be the contact model
inventing grip.

Three field notes for authors. High friction is not free: at μ = 0.5 the walker
*wedges* on the steep fillet at the ramp's base where μ = 0.35 skates over it —
sticky feet trip on terrain features that looser feet slide past. And
``upload_terrain`` is the terrain upload, not the inherited raw ``upload``;
the raw path leaves the plane/cone split at zero and the world silently does
not draw.

The third was measured while building ``examples/ramp_world.xml``: **a fat
capsule laid across the walking line stops the quadruped at every height
tried** — 0.06 blocks it exactly as hard as 0.15, so this is not the feet
failing to lift but the gait meeting a face it has no answer for. Ramps it
climbs; steps it does not take. Two consequences for authoring: put a ramp
under anything you want walked up, and prefer a *plane* for that ramp — a
buried capsule is a ridge with sides, and a wandering walker strolls past a
ridge rather than onto it. Whether a gait should learn to step is a question
for the limb tier, not this one.

## 5. Not in scope

Moving platforms. Terrain editing UI (the format and rendering land first;
authoring stays in Python this round) — **since landed by
docs/TERRAIN_AUTHORING.md**, on the observation this round was already
sitting on without noticing: a chunk graph *is* a cone list, so the world
is shaped with the creature's own tools and promoted. Creature-vs-creature
collision. Boxes.
