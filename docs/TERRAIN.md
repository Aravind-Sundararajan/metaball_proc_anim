# Terrain: a world that is also a field

**Status:** approved, in progress
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

## 4. Not in scope

Moving platforms. Terrain editing UI (the format and rendering land first;
authoring stays in Python this round). Creature-vs-creature collision. Boxes.
