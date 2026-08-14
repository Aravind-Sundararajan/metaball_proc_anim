# World contact: the flesh meets the world between its chunks

**Status:** landed — one pass, three constants, thirteen test cases, and
one geometry lesson the landing taught back: `smooth_min` of half-space
fields is *concave along any line*, so no plane-only world can seat an
interior landing at all — §2's two-plane counterexample does not
penetrate, §4's chord silence is exact jurisdiction over every
plane-only world rather than a float courtesy, and the valley twin
carries a buried crease capsule because every feature this pass can
keep needs a cone. Measured at landing: a mid-span burial closes at
exactly half rate per tick (0.110 → 0.055 in one call, the linear
denominator's stated price); the draped span recovers the floor's load
law to the constant (holds at 0.6·μmg per chunk, slides at 1.4, doubled
mass doubles it); the free end beside a pin moves exactly `depth`; the
degradation pair discovers 0.11 at `K = 21` and 0.23 — one radius — at
the `K = 32` ceiling; four spans on one hub move it 0.037 against a
0.049 deepest depth, convex; and every span rest sits the sphere pass's
own `g·dt² ≈ 6.1e-3` proud-of-zero reground under it.
Written against the landed self-contact arc, whose §7 names this gap as
its first residual: "Terrain and floor still meet chunk spheres only,
so a span can still sag through a ridge between footfalls."
**Arc:** the world side of the contact fork. SELF_CONTACT.md taught the
flesh to collide with itself; this arc teaches the same flesh to
collide with the world it stands on.
**North star:** a shin's span cannot stand through a ledge both its
spheres clear, and flesh resting on the world grips it at the world's
own μ.

---

## 1. The gap, measured

`_collide_terrain` lifts chunk spheres — one field evaluation per
chunk, depth `radius − distance(pos)`, a clamp along the gradient with
Coulomb on the way (`core/solver.py`). The flesh *between* chunks
collides with nothing. The victims are not hypothetical: of the eight
shipped bodies, only two (`hunt`, `scavenge`) have every rendered span
tiled by its end spheres. The quadruped's worst rendered span runs 2.3
radii-sums, the legacy armature imports reach 3.3 — exactly the
population self-contact named, the long rest length whose cone is
mostly flesh no sphere owns. A shin crossing a narrow ledge stands
through it while both endpoint spheres clear it to either side; a body
draped over a ridge hangs by its chunks with the ridge inside its
flesh.

The self-contact arc deferred this deliberately, with two reasons:
"different broadphase economics and none of the graph-exclusion
machinery." Both were correct and both are answered here — the
broadphase is one Lipschitz bound per span (§8), and of the two landed
exclusion sources only sphere jurisdiction survives the transfer,
narrowed to what can be proved at each landing (§4).

## 2. The floor needs nothing: an equality, and exactly where it stops

The rendered truth of a span is `round_cone`'s shape — "the convex hull
of the two end balls," `core/distance.py`'s own words, exact down to
the degenerate case (one ball containing the other *is* the larger
ball). The contact model's linearly-worn ball is not an approximation
of that shape but a parametrization of it, by the **sweep identity**:

```
∪ₜ B(p(t), r(t))  =  conv(B_a ∪ B_b),        exactly
```

both inclusions elementary — the lerped ball is the Minkowski convex
combination `(1−t)·B_a ⊕ t·B_b`, inside the hull; and any hull point is
`(1−t)·u + t·v` with `u ∈ B_a, v ∈ B_b`, which lands in exactly that
ball. §3 stands on this; the floor proof falls out of it:

- The admissible region above a plane, `{p : n·p − y ≥ 0}`, is convex.
- The deepest point of the hull against a plane is a support point, and
  the support of a hull of balls in direction `−n` is `max(c·(−n) + r)`
  over the balls — attained on an *end* ball. The span's maximum
  penetration *equals* the deeper endpoint sphere's penetration, which
  the sphere pass already measures and clamps.
- After the clamp both end balls lie inside the admissible region, and
  a hull of two sets inside a convex set is inside it: the whole span
  clears the moment its ends do.

One premise, stated: the clamp masks `inv_mass > 0`, so the proof
speaks for free end balls only. A pinned chunk buried in the world
keeps its authored position — the landed pinned convention — and every
pass in this design keeps the same word (§4, §7). Named rejection on
the proof's strength: **`_collide_ground` stays sphere-only and no
capsule pass exists in the `ground_y` branch.** The same argument
covers a terrain of one plane, because a fold seeded with its only
primitive is that primitive exactly.

The proof stops at the second primitive. `smooth_min` never exceeds the
hard minimum and "only ever inflates the surface" — so every meeting of
primitives fattens into a fillet, the admissible outside stops being
convex twice over (union geometry, then inflation), and a span bridging
a V of two planes penetrates the crease mid-span while both spheres
clear. One cone is already enough: the outside of a pillar is
non-convex, and a span lying across it sinks in at mid-span. The design
adds no primitive-counting special case for the safe world: on a plane
— and on every flat stretch of a larger world, since the polynomial
smooth minimum is exactly `min` outside its blend band — discovery
silences itself *structurally*, by §4's chord rule, not by branch.

## 3. Discovery: one mechanism, tap grid to candidate

The pipeline, whole, before its parts: rendered spans prefilter on one
midpoint evaluation (§8), survivors compress, each lays a tap grid over
its parameter, a deterministic stencil turns taps into candidate
landings, interior candidates refine once, and §4's two deferrals — the
end-grid rule and the chord rule — decide which candidates are
contacts. Every stage is lane-selected over compressed arrays; nothing
iterates data-dependently.

The span's clearance is one scalar function of its parameter,

```
φ(t) = distance(p(t)) − r(t),    p(t) = (1−t)·a + t·b,
                                 r(t) = (1−t)·ra + t·rb
```

and by §2's sweep identity `min_t φ` is the **exact** field-convention
clearance of the true round-cone solid — not an under-opened
approximation. (Self-contact's honesty paragraph about lerped radii
does not transfer: its one-sided error came from taking `s, t` at
centreline closest points; a minimization over `t` has no such gap.)
The honesty owed here is the field's own: in blend bands
`smooth_min ≤ min`, the field under-reports clearance by up to `k/4`,
so contact triggers fractionally before the flesh reaches the rendered
fillet and rests that much proud of it — the sphere pass's landed
convention exactly, and now one convention shared by both world passes.

`φ` is Lipschitz along the parameter with constant `L + |ra − rb|`
(the field is 1-Lipschitz, `p` moves at rate `L`, `r` at `|Δr|`),
which is what every cost and guarantee below stands on.

**The grid.** Each surviving span takes `K_i = 1 + ceil(L_i /
max(ra_i, rb_i))` taps, clipped to module constants `_SPAN_TAPS_MIN =
8` and `_SPAN_TAPS_MAX = 32` — named the way `_MAX_CATCHUP_TICKS` is,
constants with a stated job, not tuning. `K` is **per span, from its
own geometry**: a shared per-call `K` would re-grid every resting
span's landing parameter the moment an unrelated span entered the
prefilter, and §6's frozen friction anchor would move with strangers.
The batch is still one rectangle at the survivors' largest `K_i`; a
span fills its first `K_i` columns with its own `linspace(0, 1, K_i)`
and padding repeats its final tap — inert by construction, because the
stencil below never fires without a strict drop from the left. Tap
spacing is at most one fat radius in world units, and the Lipschitz
bound turns that into the guarantee: **any penetration deeper than
`(L + |Δr|)/(2(K−1))` — about half the span's fat radius — is
discovered the tick it exists.** Shallower grazes may wait a tick for
the geometry to move, the same class of statement rediscovery already
makes everywhere. At the clip's ceiling the guarantee degrades
honestly: a 64-radius span holds it only down to about one full
radius, and §10 pins both ends of that pair.

**The stencil, with its tie rule.** A tap becomes a candidate landing
by a half-open comparison:

```
interior i :  φ[i] <  φ[i−1]   and   φ[i] ≤ φ[i+1]
tap 0      :  φ[0] ≤  φ[1]
tap K−1    :  φ[K−1] < φ[K−2]
```

— strict from the left, slack to the right, so a *plateau* yields
exactly one candidate, at its entry, deterministically. The tie rule is
load-bearing, not defensive: in the side-view world §9 calls everyday,
primitives have `n_z = 0` and the field is *bitwise* invariant along
`z` (`z·0 = 0` exactly), so a span lying along a groove, or level over
a flat-bottomed trench, yields bitwise-equal `φ` across many taps. A
both-sides-strict stencil finds no minimum there and buries the span
silently forever. Self-contact made precisely this a rule — "breaks
parallel ties deterministically, and that is a stated requirement, not
a hope" — and the stencil is that rule's world-side spelling, pinned by
§10's determinism test.

**Refinement, interior candidates only.** The parabola through the
candidate tap and its two neighbours is free (all three values exist);
its vertex is clamped to the bracket, `φ` is evaluated there, and the
deeper of tap and vertex is kept — one lane-selected step, no
iteration, guarded because `φ` is only piecewise-smooth and a crease
can make the parabola lie. A landing carries its parameter `t̂` and its
depth into §4; only the contacts §4 keeps sample their normal,
`Terrain.normal(p(t̂))` — four tetrahedron evaluations, the only
gradient the world offers (§11), and §8's dominant term. A zero normal —
deep in a blend pocket — discovers nothing; one asymmetry acknowledged:
the sphere pass *keeps* its zero-normal contact and stalls the chunk by
treating the whole displacement as slide, while a span landing with no
direction is dropped — its friction budget would be zero anyway (§6),
and every chunk in the pocket still belongs to the sphere pass.

Named rejection: **the sphere-tracing skip.** An adaptive march along
the segment bounds its cost by the same Lipschitz constant, but it
answers *first entry*, not *deepest point* — after piercing the surface
it must still search the inside — and its step count is data-dependent
per lane, which is precisely what the landed cascade style refuses. The
grid at radius spacing certifies the same bound the trace would, in one
rectangular batch.

Two grid points come with their meaning attached: `φ(0)` and `φ(1)`
are *exactly* the endpoint spheres' own world clearances —
`distance(a) − ra` is the sphere pass's `d` for that chunk — and §4
spends both, for free.

## 4. Jurisdiction: defer only what is provably already owned

Self-contact's exclusion had two sources: the graph rule and sphere
jurisdiction. The graph rule has no world analogue — terrain sits at no
graph distance — so it does not transfer, and saying so is the
transfer. Sphere jurisdiction transfers *narrowed*: a landing is
deferred only where the deferral can be proved, per landing, and there
are exactly two such proofs.

**The end-grid rule.** A landing whose `t̂` sits at grid point 0 or 1
*is* the sphere pass's contact — same point (`p(0) = a`), same
measurement (`φ(0)` is the clamp's own `d`), same normal
(`Terrain.normal(a)`) — and the clamp resolves it exactly, later this
tick (§7). Deferral here is duplicate elimination and nothing else. (At
a *pinned* buried end the clamp then drops it by the pinned convention,
and so does this rule — authored position wins, one word kept
everywhere.)

**The chord rule.** A landing is deferred when it fails to dip below
the line its own ends draw:

```
defer  iff  φ(t̂) ≥ (1−t̂)·φ(0) + t̂·φ(1) − _CHORD_SLACK
```

The license is §2's own geometry: along a single plane — and along any
flat stretch, where the smooth minimum is exactly `min` — `φ` is affine
in `t` and *equals* its chord, so plane silence is structural rather
than numerical. This is the clause that carries TERRAIN.md §4's
"``Terrain.flat`` reproduces the plane world" sentence and the landed
flat-vs-plane agreement test (`abs=1e-5`, its only divergence float
lanes) through the arc untouched: over a flat world the span pass keeps
zero contacts and touches nothing, bit-identity by construction.
Physically, an at-or-above-chord landing is one the ends' own
resolutions bound — lift `φ(0)` and `φ(1)` to zero and the chord goes
with them — while every feature this pass exists for dips strictly
below it: the V-crease, the ledge lip, the pillar, both ridges of a
saddle, and the end-zone shoulder bite all have the interior as the
binding geometry, not the ends. `_CHORD_SLACK` is a module constant,
absolute on distances (`_COINCIDENT_EPS`'s stated discipline), sized at
`1e-6`: an order above float-lane lerp jitter at world scale (~1e-7),
three orders under both the per-tick reground `g·dt² ≈ 6e-3` and a
default fillet's `k/4 = 0.075`.

**Everything else is kept — mid-span and interior end-zone landings
alike.** The rejected spelling, named because two reviews shot it
independently: *"defer an end-zone landing when the owner's sphere sees
the world."* The floor's own docstring is the refutation — a chunk at
rest regrounds by `g·dt²` *every tick*, so a grounded owner's end tap
is negative forever, and a lateral ledge-lip bite in its end zone would
be deferred every tick while the clamp corrected only the vertical
reground: a different feature, under a different normal, never this
landing. That is a permanent standing overlap — the exact class the
last arc's review existed to kill. "The owner touches the world
somewhere" implies "the owner's pass resolves this landing" only when
the world is locally one convex feature, which is the assumption §2
demolishes to justify this pass existing. Deferral must name the *same
contact*, and only the end-grid rule can.

**The composition, written out.** The three rules close each other's
gaps and none suffices alone:

- A strict interior feature — V, lip, pillar, saddle — fires the
  stencil, sits below its chord, and is kept. The mechanism's purpose.
- Constant `φ` across the whole span (the along-groove span): no
  interior tap has a strict left drop, the plateau's entry emits at
  tap 0, and the end-grid rule defers it — to the sphere pass, which is
  *correct* jurisdiction, because ends and middle are equally deep and
  the end contacts are the same contact. Stencil alone would emit
  nothing at all under a both-strict rule (the silent burial), or
  plateau entries with nowhere to send them without the end-grid rule.
- Constant `φ` across an interior stretch with clear ends (the
  flat-bottomed trench): the plateau's entry tap has the strict drop,
  fires once, dips below the chord, kept — real contact, one landing,
  deterministic.
- Float-jitter pseudo-minima on a plane — a resting level span
  re-buried `g·dt²`, `φ` affine in real arithmetic but ulp-dipped in
  float lanes: the stencil may fire an interior candidate and the
  end-grid rule keeps it; the chord rule removes it, jitter being
  orders below the slack. Without it, the landed flat-terrain agreement
  fails on noise.
- Together, over a single plane: **zero kept contacts, structurally.**

One narrowing stated plainly: silence is structural on *planes*. Over
creases even a well-chunked span may now legitimately fire — a fillet
bulges up into a bridging span's middle, below its chord — and that is
flesh meeting world, not regression; end-zone tiling was never a
world-contact exemption, only a self-contact one. The landed solver
terrain tests are plane-worlds or single chunks and stand bit-identical
(§10); the walker's gait suite re-verifies behaviorally at landing
(§12).

## 5. Response: the barycentric rule with half its terms at zero

A kept contact pushes out along its frozen normal and lands on the two
endpoint chunks barycentrically. The landed rule is restated for one
side rather than replaced: the denominator is still the **first-power
interpolated inverse mass over every participant**, and the world
participates at inverse mass zero, so its two terms vanish —

```
denom = (1−t̂)·w_a + t̂·w_b            # the world's terms are 0, not absent
share = depth / denom                  # movable iff denom > 0
Δa = w_a·(1−t̂)·(share·weight)·n̂,     Δb = w_b·t̂·(share·weight)·n̂
```

A span with both ends pinned has `denom == 0` and is skipped — the
world will not fight a pin. The first power is load-bearing here for
the same fling the pair case proved: with the near end pinned and a
landing at small `t̂`, the squared form's denominator `t̂²·w_b` hands
the free end `depth/t̂`, unbounded, while the first power hands it
`depth` — every correction bounded by its own error, the module's
oldest promise. The price transfers untouched: a mid-span landing
closes at half the rate of an endpoint one, under-closure only,
absorbed by rediscovery. One scalar per contact still: both endpoints
scale the *same* `share·weight`, so the two pushes are the barycentric
split of a single normal impulse — momentum leaves through the world
exactly as it leaves through a pinned participant in the landed pair
scheme, and per-endpoint normalization would manufacture an impulse no
contact warrants, the same refactoring mistake §4 of the last arc names.

**Full share, no elasticity knob** — named rejection. The world passes
have never owned one: the clamp closes everything it finds, and that
character is why "the floor keeps the last word" is a guarantee rather
than a tuning. A `world_contact_elasticity` would be io, panel, and
counted-list churn (SETTINGS.md §5) for a squash nobody asked of the
world, and the pass already under-closes mid-span by the linear
denominator, so flesh-like transients come free. If authored squash is
ever wanted, the settings rails are there; the default would still be
one.

**The incidence gather is this pass's own, not the self-contact
families'** — the shared-gather lesson answered on its own reasoning.
That gather exists to keep one *simultaneous* Jacobi scatter convex;
these passes are sequential and measure fresh positions, and the
double-correction it guards against is prevented here by jurisdiction
(§4) plus ordering (§7), not by counting. What does need counting is
this pass's own scatter: a hub chunk carrying several grounded spans —
a hip over a ridge — receives one full correction per contact and
overshoots exactly like the module's founding NaN story. So the world
contacts gather incidence among themselves, each contact counting one
full incidence against each endpoint, weight one over the larger
count — the landed spelling, one family wide. The sphere clamp that
follows needs no gather at all: one world, one contact per chunk,
resolved exactly.

## 6. Friction: the world's μ, budgeted by what was applied

`ground_friction` is the coefficient — the world has one law, and a
span and a foot resting on the same ledge grip by the same μ. A
separate span coefficient would be two frictions for one material;
named rejection, and one fewer field.

The budget is the landed lesson verbatim: **`d` is the normal closure
this contact actually applied this tick** — `share·weight·q`, with
`q = (1−t̂)²·w_a + t̂²·w_b` the squared barycentric form that measures
the contact *point's* displacement — never the discovered depth. The
sphere clamp may equate the two only because it closes everything it
finds, alone; a span share is weight- and denominator-scaled, so
budgeting the raw overlap would inflate grip whenever the scatter
under-delivers and μ would become a function of solver structure — the
exact corruption the lesson exists to forbid, live here even with no
elasticity knob in sight.

Measurement is the floor's own order: slide reads *after* the normal
scatter, from the settled positions against `prev`, with `t̂` and the
normal frozen from discovery — and §3's per-span grid is what keeps
that anchor from moving with strangers. The contact point's motion is
the barycentric blend of its endpoints' displacements; the world
contributes no motion of its own — the slide has no world term because
the world is static, and the day moving platforms exist (TERRAIN.md
rules them out today) this is the line that learns a velocity.
Projection along the frozen normal removes exactly the applied approach
and leaves exactly the slide, the lift-is-parallel proof one-sided.

Delivery is the landed two-regime split: the undo divides by the same
squared form `q` that measured the applied closure, so a saturated cap
*is* static grip at the contact point, mid-span included — the
first-power form delivers half there and the shortfall replays as
implicit velocity, the creep the last arc measured perfectly linear.
The squared form's pinned-end lever is capped once per contact,
`u = min(scale/q, 1/lever)` with `lever = max((1−t̂)·w_a, t̂·w_b)`:
grip near a pin degrades with the lever arm instead of flinging.
`_grip_scale` guards the zero-slide lanes, the undo acts on `pos` and
never `prev`, and it runs under the same per-contact incidence weight
as the normal share, because several grips can pile on one hub chunk
where the floor only ever lands one.

## 7. Cascade placement: the world phase learns a second word

The landed order is shape, separation, world. The world phase now
speaks twice, in this order:

```
_project → _collide_self → _collide_terrain_spans → _collide_terrain
```

Spans before the sphere clamp, and the order is load-bearing three
ways. First, the clamp is exact and terminal, so "the floor keeps the
last word" survives verbatim and **the tick still cannot end with a
chunk underground** — reversed, a span scatter on curved world could
re-bury a clamped chunk (lifting a span out of a V-valley pushes its
endpoints toward the valley's other wall) and the guarantee dies.
Second, §4's end-grid deferral hands a landing to a pass that still
lies *ahead* — handed to a re-measurement, not resolved by fiat:
whatever the clamp's own correction leaves of it belongs to
rediscovery, like every other under-closure in this module. One
deliberate exception, the pinned convention: the clamp masks
`inv_mass > 0`, so a deferred landing at a pinned buried owner is
*dropped*, this tick and every tick — authored position wins, §2's
premise kept as policy. Third, running after `_collide_self` keeps the
landed reasoning: the world corrects what shape and separation
produced, including a self-contact push that drove a span toward the
ground.

What the phase still cannot promise, stated rather than hidden: a
*mid-span* can end a tick slightly underground — the incidence weight
divides a pile-up and the linear denominator halves mid-span closure —
and rediscovery finishes the job over the following ticks, exactly as
self-contact's elasticity does. Chunks keep the hard guarantee; flesh
between them converges. The `ground_y` branch gains nothing at all,
by §2's proof.

## 8. Candidates: one evaluation is the broadphase

Only rendered links are candidates — `body.link_render`, the same mask
the surface and the self-contact manifold read; "mechanism links must
carry force without carrying geometry" extends to the world exactly as
it extended to self-contact, verbatim. A brace across a knee neither
grows flesh nor trips on a ledge.

The prefilter is one field evaluation per rendered span, at its
midpoint: contact anywhere along the span requires

```
distance(mid) < L/2 + max(ra, rb)
```

because the field is 1-Lipschitz and no centreline point is farther
than `L/2` from the midpoint — the landed midpoint-sweep prefilter with
the world as the other side, and the Lipschitz property is what makes a
single evaluation an *honest* cull rather than a heuristic. Survivors
compress through `nonzero` before any tap is laid; the tap rectangle,
the stencil, and the refinement run on survivor arrays alone. No dense
span-by-primitive table, no dense span-by-tap grid before compression:
the dense-then-mask idiom is refused here for the reason the last arc
wrote down — prefilter, compress, then cascade.

Terrain has no spatial structure — the fold is a flat pass over `P`
planes and `C` cones — and this design adds none: march-plateau
discipline, deferred until a measured world hurts, recorded not built.
The honest costs, at stated scales, with one field evaluation ≈
`17·P + 65·C` flops (≈ 0.8 Kflop at `P = 2, C = 12`) — and counted
against the *whole* mechanism, normals included, because
`Terrain.normal` is four field evaluations and the kept contacts, not
the taps, dominate the worst case:

- **Editor scale** — N ≈ 60 chunks, M ≈ 60 spans: 60 prefilter
  evaluations, a handful of survivors at ≤ 32 taps, a few kept contacts
  at 5 evaluations each (refinement plus normal) — well under 1 Mflop
  per tick, noise at 40 Hz.
- **N = 1000, M ≈ 1000** — the honesty scale, worst case (washboard
  world, every span surviving at the `K = 32` ceiling): 1 K prefilter
  + 32 K taps + up to `⌊(K−1)/2⌋ ≈ 15` interior minima per span
  → ~15 K refinement evaluations and ~60 K normal evaluations —
  ≈ 108 K field evaluations, ≈ 88 Mflop per tick, ≈ 3.5 Gflop/s
  at 40 Hz. The landed self-contact adjacency matmul alone is ~2e9
  flops per tick at that N: this pass sits roughly **23× under it —
  one and a half orders, not two** — and the conclusion stands on the
  honest count: the near-matmul is still what leaves the plateau
  first. Nothing here is quadratic in anything: survivors × taps ×
  primitives, every factor bounded, and the dominant term is the
  per-contact normals, which §4's deferrals are also busy shrinking.
- The one growth axis is the *world*: a hundred-cone terrain multiplies
  every evaluation by ~8, and the sphere pass rides the same axis (its
  N evaluations per tick grow identically). The day a measured terrain
  hurts, the spatial structure arrives for both passes together.

## 9. Two dimensions

The pass is dimension-blind the way the solver is: span arrays carry
`(…, dim)`, `Body` refuses `dim < 2`, and `Terrain` widens 2-D points
onto the `z = 0` plane and slices its answers back — the sphere path's
own landed note, inherited unchanged. No new guard, no new branch.

What changes in 2-D is the *frequency*, and the last arc's review is
the warning on record: its dim-2 blindspots were found only after the
fact, because configurations that are corner cases in 3-D are the
common case in the plane. Here the inversion is stronger still — a
side-view world is valleys, ledges, and pillars by construction, its
`n_z = 0` primitives make bitwise-flat `φ` ordinary rather than exotic
(§3's tie rule exists for it), and a span bridging a crease is the
everyday resting pose of every sparse 2-D body. §10 names its dim-2
twins outright.

## 10. Test plan

Everything lands in `tests/test_solver.py` against `core` imports
only — the suite must collect on a GL-less machine, the standing rule
for core and io tests.

1. **The shin on the ledge** (the headline, dim 3): a sparse two-chunk
   span over a narrow step both endpoint spheres clear sideways. Today
   it stands through the step; after, it rests on it at flesh distance.
2. **The valley twin** (dim 2 by construction): a span bridging two
   smooth-min'd planes penetrates the fillet mid-span today, rests on
   it after — §2's counterexample, pinned.
3. **Structural plane silence, at the unit level**: over
   `Terrain.flat`, discovery keeps **zero** contacts and the span pass
   changes nothing — asserted with `==`, not `approx`, because a no-op
   pass is bit-identity by construction. Three bodies, each a different
   silence: the landed `_drop_chain` chain (spans at 1.25 radii-sums —
   real mid-span stretch, resting); a sparse span *at rest* on the
   plane (the constant-`φ`, `g·dt²`-reground steady state, not only in
   flight); and the along-groove span. The untouched landed suite
   carries the trajectory burden — including the flat-vs-plane
   `abs=1e-5` agreement test, which §4's chord rule exists to keep.
4. **Determinism of the tie rule**, pinned the landed way (the
   `_segment_params` tie-break precedent): bitwise-equal `φ` across
   taps — a span along the groove of an `n_z = 0` world — yields
   exactly one candidate, at the plateau's entry, and its end-grid
   deferral hands the burial to the spheres that own it equally.
5. **The pinned-end probe**: one end pinned, mid-span landing — the
   free end moves no farther than the depth (the first-power
   regression, restated one-sided). Both ends pinned: no correction,
   no NaN, the contact skipped. A *pinned buried owner*: the deferred
   end landing is dropped, authored position wins.
6. **Friction on the world's terms**: a mid-span rest across a ridge
   holds a tangential load at high `ground_friction` and skates at
   zero; doubling the carried mass doubles the holdable load — grip
   scales with applied closure, not with tuning.
7. **Hub convexity**: several grounded spans sharing one chunk resolve
   finite and separated — the incidence gather's regression.
8. **Mechanism spans collide with nothing** — the `link_render` mask.
9. **The guarantee extended**:
   `test_contact_cannot_push_a_chunk_underground` grows the span case —
   spans landing in a V-valley, no chunk ends the tick under the field.
10. **The saddle** (dim 3 *and* dim 2): a span across two ridges lands
    on both the same tick; then the house late-time assertion — settle,
    run 50 further ticks, positions unchanged to tolerance (the
    `test_static_grip_holds_mid_span` pattern) — because "settles"
    without the late window would not pin lateral pumping between
    rediscovered minima.
11. **The standing-overlap kill** (dim 3 *and* dim 2): a lateral
    ledge-lip bite inside a *grounded* owner's end zone — the owner's
    end tap negative every tick from reground — is kept and corrects,
    the case the rejected owner-sees predicate deferred forever. And an
    end-grid landing defers: the tick's result equals the
    sphere-clamp-only result exactly, the two passes never both acting
    on one landing.
12. **The blend pocket**: a zero normal at the landing discovers
    nothing — no push, no NaN, chunks still owned (and stalled) by the
    sphere pass.
13. **The degradation pair**: a penetration just deeper than
    `(L + |Δr|)/(2(K−1))` is found on an ordinary span (`K` inside the
    clip) *and* on a ~64-radius span at the `K = 32` ceiling, where the
    bound has degraded to about one radius — the guarantee holds at
    both ends of the clip, and the clip's price is measured, not
    implied.

## 11. Deliberately not done

- **Creature-vs-creature.** Still its own arc if ever — every contact
  here is one body against the one world.
- **Terrain deformation and moving platforms.** The world is static,
  and §6's slide has no world term *because* of it; TERRAIN.md's
  exclusion stands, and the line that would learn a velocity is named.
- **No analytic gradients.** `Terrain.normal` is four tetrahedron
  evaluations and the pass takes it as offered. An analytic gradient
  would be a second line-for-line pairing burden against `sdf.glsl`
  for a cost §8 now counts honestly and still affords.
- **No blend-band compensation.** The geometry is exact (§2's sweep
  identity); the field's `k/4` clearance convention near creases is
  the sphere pass's landed convention, kept — flesh rests fractionally
  proud of a fillet rather than the solver second-guessing the field.
- **No new `SolverConfig` fields.** No toggle — the world's law was
  never optional for spheres and is not optional for flesh; no
  elasticity (§5); no second μ (§6). The evidence that this changes no
  shipped scene: **no XML in the repository carries a `<terrain>`
  section** — the quadruped's four "terrain" matches are
  `<none name="terrain"/>` controller arguments — so no shipped
  document even reaches the terrain branch. (Not claimed: end-zone
  tiling. Measured, six of eight shipped bodies carry untiled rendered
  spans — §1's point; hand a world to one of them and the pass *will*
  speak, which is the fix working.) `<solver>`'s attribute list, the
  reader, the writer, the record-completeness test, and SETTINGS.md's
  counted panel sites all stand exactly as landed — zero io churn,
  zero editor churn, by decision rather than omission.
- **No spatial structure for terrain.** Recorded, not built, with §8's
  measurements as the trigger condition.

## 12. Package and churn

One package: **`core/solver.py` only**, plus its tests.
`_collide_terrain_spans`, slotted between `_collide_self` and
`_collide_terrain` inside the terrain branch of `step()`: the midpoint
prefilter and compression (§8), the per-span tap rectangle and half-open
stencil with `_SPAN_TAPS_MIN`/`_SPAN_TAPS_MAX` (§3), the end-grid and
chord deferrals with `_CHORD_SLACK` (§4), the one-sided first-power
scatter under its own incidence gather (§5), the Coulomb undo on §6's
terms. The span-array gather (`la`, `lb`, `seg`, `length`, `r1`, `r2`,
`mid`, `half`, `rmax`) is the same one `_collide_self` builds; whether
it is factored or recomputed is the implementer's call — it is M-linear
either way.

Every file implementation touches:

- `mesh_xml/core/solver.py` — the pass and its three module constants;
  `step()`'s ordering comment gains its clause ("world last" becomes
  spans, then spheres, and the sphere clamp keeps the last word);
  `SolverConfig.terrain`'s docstring gains the sentence that spans now
  collide too.
- `tests/test_solver.py` — §10, including the extension of the landed
  underground guarantee.
- `docs/WORLD_CONTACT.md` — status flips to landed, with measurements.
- `docs/SELF_CONTACT.md` — §7's first residual gains its resolution
  pointer (the cross-reference rides with this design; the retirement
  rides with the landing — the docstring must not outlive the truth).
- `docs/CONTACT_MODEL.md` — §5's parenthetical ("the body-vs-itself
  part of that has since landed") gains its body-vs-world half, at
  landing.
- `README.md` — one row in the doc table, at landing.

Which fix carries which landed sentence, so review can check the deed
against the claim: §4's *chord rule* carries TERRAIN.md §4's flat-world
sentence and the `abs=1e-5` flat-vs-plane test; §4's *end-grid rule*
carries `test_a_pinned_chunk_ignores_the_terrain`'s convention into the
span era; §7's ordering carries
`test_contact_cannot_push_a_chunk_underground`. The landing's
regression evidence is three-fold: the untouched landed suite, the
structural-silence unit test (§10.3), and the no-shipped-`<terrain>`
observation above.

Nothing else: no `io/`, no `editor/`, no `render/`, no
`core/terrain.py` (the API as landed — `distance`, `normal` — is
sufficient), no `docs/SETTINGS.md`. The zero-field decision is what
holds the fan-out at six files.
