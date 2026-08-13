# Self-contact completed: the flesh collides, and it can grip

**Status:** landed — all three packages, with one rule sharpened by the
landing itself: end zones are the sphere family's jurisdiction outright
(§3), which is what keeps every landed test bit-exact, and the swallowed
span came off the residual list as a consequence.
**Arc:** the contact side of the fork, off the authoring-completeness
branch. Retires two of CONTACT_MODEL.md §7's three named non-features;
the third — the surface — stands by design (§7 here).
**North star:** a sparse span cannot pass through another, and a coiled
body holds itself by Coulomb friction rather than by luck.

---

## 1. The two non-features, measured

CONTACT_MODEL.md §7 names them plainly: "there is no capsule-capsule
test (chunk spheres only, so two long links can still cross between
sparsely-chunked spans)", and "there is no friction between a body's own
parts (strangers separate without grip — a creature holding onto its own
body is a controller feature)".

The first is a geometry gap whose victims are the *sparsely-chunked*,
exactly as the sentence says. An editor-grown chain at
`_GROW_SPACING = 1.6` radii is the least affected case: its spheres
overlap enough to cover the whole centreline, and the uncollidable
deficit is only the thin waist between sphere union and cone. The real
gap is the long rest length — a legacy capsule import carrying its
`height` as one link, a hand-authored span, an autorig bone — where most
of the cone's length is flesh that collides with nothing, and two such
spans pass through each other exactly where the surface says they
cannot.

The second gap is what the first one hides: even where spheres do meet,
they separate without grip, so a coiled body slides off itself under any
sustained load. §7's ruling about holding is kept for what it actually
covers (§5).

## 2. The manifold becomes the flesh

The sphere–sphere pass keeps its rule, its exclusions, and its tests;
its weight computation joins the one shared gather §4 describes, which
is the only way mixed pile-ups stay convex. Two discovery families join
it, both drawn from what the renderer already calls flesh:

- **capsule–capsule**, between rendered links: each link is a segment
  wearing its endpoints' radii, the same round cone `sdf.pack` emits.
- **sphere–capsule**, between a chunk and a rendered link's span — the
  head resting on the body's back mid-span.

Mechanism links carry no flesh and therefore no contact: §3's ruling —
"mechanism links must carry force without carrying geometry" — extends
to collision verbatim, on `body.link_render`, the same mask the surface
reads. A brace across a knee neither grows a membrane nor becomes a wall.

Honest approximations, stated rather than hidden:

- Closest points are solved on the centreline segments and the required
  separation is the radii interpolated at those parameters. The error is
  one-sided — it under-opens, never over-opens — and second-order in the
  taper for transverse crossings. For near-parallel tapered spans it is
  first-order, and the closest-point parameter can flap along the flat
  valley on numeric noise: bounded jitter, not divergence, and the fat
  ends are chunks, so the sphere families usually own the depth the
  centreline test misses. The segment routine breaks parallel ties
  deterministically, and that is a stated requirement, not a hope.
- The parallel guard is *relative* — a denominator is degenerate when
  `denom ≤ eps·a·e` — because `a` and `e` are squared lengths and an
  absolute epsilon misfires at both large and small scales.
  `_COINCIDENT_EPS` is absolute *on distances* and stays where it is;
  the landed `where=`-guard-then-overwrite idiom for the coincident
  sphere pair generalizes to every degenerate lane here, fallback axis
  included. Zero-length segments are legal inputs (coincident chunks
  exist transiently; `rest=0` links pass validation) and reduce to the
  point cases.
- In two dimensions, crossed centrelines are the *common* deep-contact
  configuration, not a corner: overlapping non-parallel segments in the
  plane generically intersect, the closest distance sits at zero, and a
  pair discovered after crossing corrects the wrong way until
  re-separated — the sphere pass's own accepted defect, which almost
  never fires in 3-D and routinely would in 2-D. The coincident guard
  owns the exact crossing; the crossing regression in §8 runs in both
  dimensionalities because the last arc's review found its dim-2
  blindspots only after the fact.

## 3. Exclusions: decided at the contact, not the pair

The chunk rule is graph distance three or more, because distance one is
the constraint set and distance two is the pair one folding joint drives
into contact, whose governed form is a brace. Extending that rule to
whole *span pairs* fails in both directions at once — review found a
two-fold "scissor" that crosses second-neighbour spans mid-length where
a pair-level exemption tests nothing, and a tested outer pair whose
end-zone contact re-fights the very fold the distance-two ruling
renounces. So the exemption moves down a level, onto the contact itself:

- **Pairs sharing a chunk are wholly exempt.** Their overlap is the
  fold's own geometry, and the fold's governed form is the brace.
- **Everywhere else, jurisdiction decides at the landing.** The stretch
  of cone an endpoint's own sphere dominates is that chunk's flesh — its
  end zone — and a contact landing with every side in an end zone is the
  *sphere family's* outright: exempt where the chunk rule exempts,
  discovered there where it does not, and never emitted by the capsule
  families, whose fatter duplicate would perturb the sphere family's
  exact arithmetic. Any side landing on a stranger's mid-span makes the
  contact real. On a well-chunked body the end zones tile every span and
  the capsule families go silent — the landed behaviour exactly, which
  is what keeps every pre-arc test bit-identical; they speak where the
  body is sparse, which is the gap they exist to close.
- **Sphere–capsule on the same terms:** a chunk against its own span's
  flesh is authored containment; an end-zone landing is a chunk pair and
  the sphere family's business; a mid-span landing is real, except where
  both endpoints sit within distance two of the chunk — then the whole
  span is the joint's own flesh, never fought.

This closes the scissor, stops the end-zone fight, and — the landing's
own gift — catches the swallowed span the draft had conceded: a short
span inside a long neighbour's flesh lands on the long side's mid-span,
and a mid-span landing is real whatever single fold produced it. The
brace still governs the fold's angle; contact now governs its overlap.
In a three-link hairpin the far-end contact is the distance-three sphere
pair the existing pass already resolves; the diagonals are the
second-neighbour pairs the rule protects. One tie the cascade breaks
deterministically: an exactly-parallel pair projects the *other*
segment's midpoint, landing inside the shared run — axis-aligned
authoring makes exact parallelism common rather than measure-zero, and a
tie broken off the end would defer a swallowed parallel span to spheres
that cannot see it.

The masks come from the adjacency and its square — the arrays the sphere
pass already computes per call — and exclusion stays a derived view. Two
scale statements, kept honest: at renderer scales (tens to low hundreds
of chunks) the dense idiom is the plateau the module already stands on;
at the backlog's thousand chunks even the *landed* adjacency matmul
leaves that plateau, which is precisely why the exclusion-caching note
exists and why it stays deferred rather than denied. And the candidate
kernel never runs dense: pairs prefilter on segment bounding spheres in
the upper triangle, compress through `nonzero`, and the closest-point
cascade runs on the survivor arrays alone — a twenty-operation cascade
over a dense L×L field is the one place the sphere pass's
dense-then-mask idiom must not be copied.

## 4. Resolution: the same convex Jacobi, four ends at a time

A capsule contact corrects along its axis at the closest points and
lands on chunks barycentrically. The share's denominator is the
**first-power interpolated inverse mass** —
`(1−s)·w_a1 + s·w_a2 + (1−t)·w_b1 + t·w_b2` — and the choice is
load-bearing: the textbook squared form is exact at the contact point
but flings the free end of a nearly-pinned span by `error/s`, unbounded,
and every correction this module has ever applied is bounded by its own
error. The linear form keeps that guarantee — no endpoint moves farther
than the depth — at a stated price: a mid-span contact closes up to
half as fast as an endpoint one, under-closure only, absorbed by the
same rediscovery elasticity already leans on. A pair is movable when
the four-term denominator is positive; both-ends-pinned on one side is
fine while the other side can move.

One scalar per contact, signs flipped between the sides: span A's
mass-weighted sum is exactly `−k·n̂` and span B's `+k·n̂` for arbitrary
masses, because `m·w = 1` and barycentric coefficients sum to one — so
momentum is conserved to algebra, not to tuning, and the closest-point
offset lies along the axis so torque cancels too. A pinned participant
absorbs momentum exactly as the landed pair scheme does. Per-side
normalization is the one spelling that breaks this; the design says so
because it is the natural refactoring mistake.

The convex weight generalizes as **one shared incidence gather across
all three families**: every contact counts one full incidence against
each of its chunks — a graze at a 0.02 barycentric share counts a whole
one, over-damped on purpose, because weighted counts would forfeit the
convexity proof — and the contact's weight is one over the largest
count among its chunks. Shared counting is also what makes duplicates
benign: a clamped capsule contact that re-finds a sphere pair sums with
it convexly instead of correcting twice. `np.add.at` accumulation and
the `free`-mask apply carry over verbatim. One Jacobi pass,
`self_contact_elasticity` scales the share, discovery repeats next
tick. Ordering is untouched: shape, separation, world — the floor keeps
the last word.

## 5. Self-friction: Coulomb between a body's own parts

§7's ruling splits under §2's own vocabulary. *Active* prehension —
choosing to hold, and letting go — is a controller feature and stays
one. *Passive* non-slip — a coil that does not slide off itself under
load — is Coulomb friction, which this solver already speaks at the
floor, position-level and load-scaled.

`self_friction` is the Coulomb `μ` between a body's own parts, default
`0.0` — today's behaviour, byte for byte. Its `d` is pinned precisely:
**the relative normal closure actually applied to that pair this tick**
— the elasticity- and weight-scaled share, not the discovered overlap.
The floor may equate the two only because it closes everything it
finds; here `μ·overlap` would inflate grip as elasticity drops, making
`μ` a function of solver tuning. With the applied closure, the standing
overlap self-adjusts until per-tick closure equals per-tick re-burial,
and the floor's law — grip proportional to the load actually crossing
the contact — comes back out exactly, elasticity moving only the
transient.

Measurement order is the floor's own, and it dictates package 1's pass
structure: the normal scatter lands first, then slide is read from the
settled positions against `prev`, with `s`, `t`, and the normal frozen
from discovery. The lift-is-parallel proof carries — all four endpoint
pushes are parallel to the contact normal, so the projection removes
exactly the approach and leaves exactly the slide. Corrections arriving
from *other* contacts at shared chunks read as slide; the floor already
tolerates precisely this from the passes ahead of it, and rediscovery
absorbs it. The tangential undo is split equal and opposite in
mass-weighted terms like the normal share, capped at `μ·d`, acting on
`pos` and never on `prev` — §2's whole fix, applied between parts —
and it runs under the same convex per-contact weight, because several
grips can pile on one chunk where the floor only ever lands one.
Zero-slide lanes stay out of the division, the floor's own guard.

## 6. Riding the landed rails

Persistence is one attribute: `self_friction` joins `<solver>` on the
section's own terms — one line in the full-record writer, one overlay
kwarg and one finiteness entry in the reader, absent meaning `0.0`,
which is the truth about every document written before it existed. The
record-completeness test, `tuned_config`, and the headline round trip
each extend by the one field they exist to extend by. The editor is one
spin box on the SolverPanel through `settings_edited()`, with its own
`_admit` + `setValue` line in `_refresh` and the same `setEnabled` gate
the other self-contact field rides. The surface pick is untouched — it
reads flesh, not contact.

## 7. Deliberately not done

- **Capsule-vs-world.** Terrain and floor still meet chunk spheres
  only, so a span can still sag through a ridge between footfalls. A
  real gap, named here, and its own arc: world contact has different
  broadphase economics and none of the graph-exclusion machinery.
- **No prehension.** A grip a brain commands — grab, hold, release —
  is the controller feature §7 always meant. This arc gives it the
  substrate: commanded holds can be honest contact instead of pinned
  fakery, but the commanding is not contact's job.
- **No exclusion caching, no spatial broadphase.** Recorded, not
  built: the per-call derived view and the bounding-sphere prefilter
  stand until a measured body hurts — march-plateau discipline, with
  §3's honest note about where the plateau actually ends.
- **The surface is untouched.** Nearby limbs still merge visually at
  blend distance before flesh touches; §7's third sentence stands —
  margin is authored in radii or blend groups, a render decision.

## 8. Packages

**Package 1 — the capsule manifold.** `core/solver.py` only:
point–segment and segment–segment closest points with relative-epsilon
degeneracy guards and deterministic parallel tie-breaks, vectorized
over prefiltered-and-compressed candidate pairs; parameter-aware
exclusion off the adjacency arrays; first-power barycentric resolution
under the unified incidence gather. Tests: the crossing regression (two
long-rest strangers dropped orthogonally pass through each other today,
stack after) in dim-3 *and* dim-2, where the pair is exactly parallel
and the tie-break is exercised too; the skew scissor (two folds crossing
second-neighbour spans mid-length — untestable under a pair-level
exemption, caught under this one); an authored fold's end-zone contact
is *not* fought; a mid-span head rest (sphere–capsule); mechanism links
collide with nothing; a contact beside a pinned endpoint moves the free
end no farther than the depth; sphere pairs and a mid-span contact
piling on the same chunks stay convex under the shared gather; centre
of mass stationary on a *fully free* tangle;
`test_contact_cannot_push_a_chunk_underground`,
`test_a_free_cluster_does_not_drift`,
`test_a_pile_stays_finite_and_resolves`, and
`test_off_by_default_overlapping_strangers_stay_put` unchanged.

**Package 2 — self-friction.** `SolverConfig.self_friction`, the
Coulomb undo at the contact points on §5's exact terms. Tests: `μ=0`
byte-identical to package 1; a two-turn coil under gravity holds its
wrap at high `μ` and sheds it at zero; holdable tangential load scales
with the applied closure; several grips piling on one chunk stay
convex; every ground and terrain friction test untouched. Retires
`_collide_self`'s closing no-grip paragraph in the same commit — the
docstring must not outlive the truth it states.

**Package 3 — the rails and the prose.** `<solver self_friction>`
through writer, reader, and the three extended io tests; the SolverPanel
spin through `settings_edited()`; then every sentence the arc falsifies:
CONTACT_MODEL.md §7's closing paragraph (two of its three non-features),
`SolverConfig.self_contact`'s docstring (sphere-only vocabulary), the
SolverPanel self-contact tooltip, and SETTINGS.md §5's "exactly nine
panel sites" count — a design that changes a counted list updates the
count where it is written.
