# The surface pick: a click that states a position

**Status:** landed — all three packages; the ortho repair rode package
1, and §2's floor paragraph was corrected against the shader mid-arc.
**Arc:** second half of the authoring-completeness branch, beside
docs/SETTINGS.md.
**North star:** clicking the thing you see puts a chunk on the thing you
see — flesh, ground, or floor — and a miss does nothing at all.

---

## 1. The gap, corrected

REFACTOR_PLAN §5 indicts the legacy editor's right-click creation onto
the y=0 plane — "a 2-D affordance in a 3-D scene", clamped to ±10,
falling back to the origin on a miss — and suggests placement "onto the
surface itself". The plan's own resolution already landed the other
half of that sentence: today's `Add` grows out of the selection
(`Document.create_chunk`'s three-branch rule), and no click-creation
exists at all. The viewport's rays serve selecting and dragging what
already exists; nothing serves *there, where I'm pointing*.

So this is not a replacement of a bad unproject. It is a new gesture
beside an `Add` that stays exactly as it is: the selection rule remains
the keyboard's grammar (`A` extrudes), and the click becomes the
mouse's — a position stated directly, against the rendered surface.

## 2. What the ray traces: what the author sees

The pick marches the composite the frag shader draws, because the
gesture's whole claim is "where you clicked": the creature field
(smooth-min within a blend group, hard min across — `field.field` over
`Document.primitives()` and the `group_rows()[:, :2]` adapter), hard-min
against the terrain fold (`Terrain.distance`), and — only when the
terrain is empty and the floor is shown — the analytic reference plane
at the shader's own height — `solver_config.ground_y`, or the zero the
shader substitutes when the solver has none, because the frag draws the
reference grid even for a floorless world. The pick follows the drawn
grid exactly, gates and substitution both, so what is on screen is
pickable and nothing else is; a chunk placed on that reference
furniture in a floorless world falls, which is what it reads as.

The creature∪terrain hard min deliberately does not exist as a library
function — "compositing against a creature is the consumer's job"
(`core/terrain.py`), and the only prior composite lives in the frag
shader and a test-local twin. That ruling stands: the composite loop is
**editor code**, a small float64 sphere-trace in `viewport.py` beside
the module that already owns the mouse, stepping `min(field.field,
terrain.distance)` with the session's march settings and the room's
authored blend. One ray per click — ~128 evaluations, nothing against a
frame budget whose remaining cost is "shading cost or resolution
scaling, not marching fewer steps".

The hit lands on the *draw* surface (primitives pack at
`draw.render_pos`), and the chunk is authored into the *body*. Paused,
these coincide; during playback the draw points lag by design, so a
place-click lands where the flesh was a spring-lag ago — the same
imperceptible gap every other affordance already accepts, and the same
house position: editing during playback behaves the way it reads.

## 3. What a hit means: position from the click, attachment from the hit

The click states position. The *kind* of hit states attachment:

- **Flesh.** The march reports which primitive won the fold; a sphere
  names its chunk, a cone the nearer of its two endpoints. The new
  chunk takes that chunk's radius — grown anatomy matches local
  anatomy, the single-selection rule's reasoning relocated to the
  cursor — and one `AddLink` to it. Clicking a creature says "more of
  this creature, here."
- **World** (terrain or the reference floor). A free chunk at the
  default radius 0.5, no links: clicking the ground says you meant a
  new beginning, which is precisely what `create_chunk` already says an
  empty selection means. It rests *on* the ground, not in it —

- **Placement offset, both kinds:** the centre sits at the hit point
  plus the new radius along the surface normal (`field.normal` for
  flesh, `Terrain.normal` for terrain, +y for the floor) — the
  `Terrain.project` clearance argument, generalized: "aiming at the
  surface itself is aiming underground." `field.normal` returns the
  zero vector at gradient sinks; the fallback direction is back along
  the ray, which is always defined and always faces the author.

The selection does not participate — the click is the statement — but
the placed chunk *takes* the selection afterward, exactly as
`create_chunk` does, so a click then `A A A` extrudes from where you
pointed. The command shape is `create_chunk`'s own: `AddChunk` alone
for a world hit, `Macro([AddChunk, AddLink])` for flesh, one undo step,
routed through `Document.do` with everything that implies (refused
while abstracted, `_STRUCTURAL` sync, dirty). Mechanically,
`create_chunk` grows optional `at=` / `attach_to=` / `radius=`
overrides rather than gaining a sibling — one Add rule with two
position sources, not two rules that drift.

## 4. The gesture, and what a miss does

**Ctrl+Left-click places.** The bare left ladder is fully spoken for
(gizmo, select, toggle, drag, clear+orbit), Shift is claimed on both
full and empty clicks, and right-click cannot distinguish click from
pan-drag without inventing a threshold. Ctrl is unclaimed, chords with
nothing, and reads as "apply here" everywhere else in the desktop
idiom. Exactly Ctrl — a chord carrying Shift is a selection gesture
and falls through to the ladder that owns it. On press with Ctrl: run
the pick; on a hit, place and consume the click — no drag begins, no
orbit; on a **miss, consume and do nothing**. Not the legacy origin fallback (indicted), not a fall-through
to orbit (a failed place that spins the camera punishes the author for
missing by a pixel). Sky means no.

The selection pick (`nearest_hit`, analytic spheres at body positions)
and the surface pick (cones at draw positions, blend-fattened) disagree
near silhouettes by construction. They never race: Ctrl consults only
the surface pick, bare click only the sphere pick. The two gestures
answer different questions, and the modifier is what keeps their
different answers from fighting over one button.

## 5. Orthographic rays, repaired on the way in

`viewport.ray_through` builds perspective rays unconditionally;
`camera.ray_directions` branches on `camera.orthographic`. The editor
has an ortho toggle on `O` — so today's *selection* picking already
casts the wrong rays in ortho, and the surface pick would inherit the
bug. Package 1 fixes `ray_through` to branch the way `ray_directions`
does (parallel rays from a moving origin), held to agreement by the
same invariant the overlay tests already enforce: the dot must sit
where a click on it would land — now in both projections.

## 6. Deliberately not done

- **No hover ghost.** A preview is a CPU march per mouse-move against a
  frame loop built to do nothing when nothing changed, and a QPainter
  overlay besides (`sdf.pack` row order is a surface contract — no
  injecting phantom primitives). The gesture is atomic; a misplaced
  chunk is one Ctrl+Z.
- **No curvature-derived radius, no `k/4` compensation.** The hit
  chunk's authored radius is the honest local answer; compensating for
  blend fattening would put an unauthored number in an authored field.
- **No place-drag sculpting, no context menu, no modal tool state.**
  One modifier, one click, one command.
- **The keyboard `Add` does not change**, and neither does its
  documentation: the selection rule stays "the Add rule", and the pick
  cites it rather than forking it.

## 7. Packages

**Package 1 — honest rays.** `ray_through` branches on
`camera.orthographic`. Tests: a pixel's ray equals
`ray_directions(w,h)[h-1-y, x]` in *both* projections (the stated
contract, currently only true in one); the overlay-agreement invariant
under ortho.

**Package 2 — the trace.** `viewport.surface_hit(x, y) -> hit | None`:
the editor-side composite march (creature fold via
`field.field`/`field.raymarch` machinery, terrain hard-min, floor plane
when visible), returning point, normal, kind, and the hit chunk id for
flesh. Tests against the CPU-oracle conventions: a ray at a lone
sphere hits at radius; between two blended chunks hits the fused flesh
where sphere-picking misses; terrain wins where it is nearer; the
floor answers only when terrain is empty and shown, at the shader's
substituted zero when `ground_y` is `None`; hidden floor means sky;
normal-zero fallback.

**Package 3 — the gesture.** `create_chunk` grows `at`/`attach_to`/
`radius` overrides (defaults preserve the three-branch rule
bit-for-bit); `mousePressEvent` claims Ctrl+Left ahead of the existing
ladder, consuming hit and miss alike. Tests: flesh click links to the
hit chunk with its radius; ground click is a free 0.5 at
clearance; miss changes nothing (selection, history, camera); undo
restores in one step; placed chunk takes the selection; refused while
abstracted.
