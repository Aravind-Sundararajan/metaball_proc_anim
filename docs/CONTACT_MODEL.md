# Contact model: the two blockers gating terrain

**Status:** landed — with the corrections in §6
**Found by:** the self-propulsion work, which audited its own thrust
**Gates:** terrain, and any creature that stands rather than hangs

---

## 1. The blockers, measured

**Friction cannot hold.** `ground_friction` is a fractional velocity bleed: each
contact tick removes a fraction of tangential velocity. Even at full bleed a
planted foot creeps, because bleeding velocity never undoes the *displacement*
the constraints already caused this tick — the foot has already slid by the time
friction runs, and the slide is kept. Measured: the walker holds 0.27 u/s
against a commanded drive of 0.5, and a "planted" foot moves continuously under
load. Grip is also independent of load, which is backwards.

**Nothing angular exists.** Every constraint is a distance. A two-bone leg has a
free knee, so no arrangement of links can hold a torso off the ground — which is
why `balance.py` must assert body height with a spring, honestly documented as
such. A knee that cannot lock means legs are dampers, not columns.

---

## 2. Design: Coulomb friction, position-level

The normal impulse in a position solver *is* the penetration correction: the
depth `d` a chunk had to be lifted this tick already encodes gravity plus every
constraint load transmitted into that foot (collision runs after projection, so
`d` includes the body's weight arriving through the leg). Coulomb then says the
ground may cancel tangential motion up to `μ·d`, and no more:

```
t     = (pos - prev)[tangential]          # this tick's slide; ground is static
scale = min(μ·d / |t|, 1.0)
pos[tangential] -= t · scale              # undo up to μ·d of it
```

Acting on **pos**, not `prev`, is the entire fix. Under Verlet, moving the
position back both erases the slide and zeroes the implicit tangential velocity
in one step; `scale == 1` *is* static grip (the foot ends the tick exactly where
it started it), and the saturated case degrades continuously into kinetic
sliding with no branch between regimes. A foot that grips leaves its leg link
stretched; next tick the projection pulls against the grip, and that tension
arriving at the hip is what locomotion is.

Semantics change, stated plainly: `ground_friction` becomes the Coulomb
coefficient `μ`. Nothing persists `SolverConfig`, so no file format is
affected; the propulsion tests swept 0.15–0.6 as a bleed and those values
remain sensible as `μ`.

**Acceptance**
- A loaded chunk under constant lateral force below `μ·N` does not creep —
  zero drift over 1000 ticks. The bleed model fails this at any setting.
- Holdable lateral force scales with normal load (double the weight, double the
  grip). The bleed model fails this too — its grip ignores load.
- `μ = 0` reproduces the frictionless-floor control from the propulsion suite.
- The walker's speed moves toward its commanded drive (was 0.27 of 0.5).
- The 2-D body path and every existing ground test keep passing.

## 3. Design: angle limits, compiled to links

No new solver machinery. For a joint `a–b–c` with authored bone lengths
`r1, r2`, the interior angle at `b` and the chord `|a–c|` determine each other
by the law of cosines:

```
|ac|(θ) = sqrt(r1² + r2² − 2·r1·r2·cos θ)        # 180° = straight
```

So a **minimum angle is a PUSH link** between `a` and `c` with rest
`|ac|(θ_min)` — it engages only when the joint folds tighter — and a
**maximum angle is a PULL link** with rest `|ac|(θ_max)`. Both constraint types
already exist, already run in the just-fixed momentum-conserving projection,
already serialize as links, and already participate in undo. Rain World builds
its bodies from exactly this kind of brace rod.

This is a cone limit about straightness — no bend-plane or twist control — and
that is stated rather than hidden. It complements `SpineFollow.max_bend_degrees`
rather than replacing it: the controller force is a posture *preference*, the
link is a joint *stop*.

**The one real contract gap: links needed a `render` flag.** The surface is
built from links — one round cone each — so a brace across a knee would grow a
membrane of flesh across the joint. Mechanism links must carry force without
carrying geometry. A chunk whose every link is mechanism still emits its own
sphere.

**Acceptance**
- A min-limit joint driven into a hairpin stops at the chord distance — a hard
  stop within solver softness, not a soft force that rings.
- Limits conserve momentum (free chain with limits: centre of mass stationary).
- **The payoff:** a two-bone leg with a knee stop supports a load with no
  postural height spring — a leg becomes a column.
- A braced knee adds no SDF primitive; the surface is unchanged.
- Limits round-trip through save as ordinary links.

## 4. Sequencing

1. **Contract (done first, single-owner):** `render` flag through `Body`,
   `sdf.pack`, and the v3 reader/writer. Additive attribute, default true —
   an existing document's meaning is unchanged, so no version bump.
2. **Parallel:** Coulomb friction (`core/solver.py`) and angle limits
   (`core/angles.py`, new) — disjoint files, both against the landed contract.
3. **Integration (after both):** the braced walker on Coulomb feet — knee
   stops plus grip, postural spring weakened toward zero. Also the SolverPanel
   friction tooltip, which still describes the bleed.

## 5. Not in scope

Terrain itself (this is its prerequisite, not it). Orientation constraints on
single chunks. Twist along chains. Contact manifolds beyond the ground plane
(the body-vs-itself part of that has since landed — §7).

## 6. What implementation taught the design

Everything in §2–3 landed and every acceptance criterion is a passing test,
but four things in the plan as written were wrong or incomplete, found by
building it:

* **min and max limits do not compose on one joint.** Both compile to the
  single ``a–c`` link a ``Body`` can hold, so a joint takes a floor or a
  ceiling below 180°, not both. The helper raises rather than silently keeping
  one. Composing them needs a second-link mechanism nothing else yet justifies.
* **A single free-standing leg cannot demonstrate the payoff.** A point
  contact under a top-heavy load is an inverted pendulum; it topples in
  ~0.35 s whatever the knee does. A stop rigidifies a leg — it cannot balance
  one. The payoff test is therefore two mirrored legs under one load, the
  smallest structure whose standing is decided by knees alone.
* **Near-straight stops are soft.** Chord sensitivity goes as sin θ, so a 90°
  floor holds to half a degree while a ~155° floor gives ~11° under full load.
  Inherent to the compilation; stated in the module docstring.
* **The walker overshot its command once the floor could grip.** Footholds led
  by velocity *plus* drive — positive feedback the slipping floor had been
  damping (measured open-loop gain 1.6–1.8). A command now replaces the
  measurement; gain brackets unity.

And the integration measurement that sets expectations for rig authors:
**stops are safety rails below the gait's working range, not springs inside
it.** A knee floor at 100° — under the swing's flexion — keeps full walking
speed; at 150° it halves travel for about one percent of standing height. And
stops do not replace the postural controller on a *walking* creature: with
posture off, the collapse simply moves to the hip, the joint without a stop.
Static structure belongs to stops; dynamic balance stays a controller job.

## 7. Later addition: self-contact

§5 ruled contact manifolds beyond the ground plane out of scope. The
body-vs-itself manifold has since landed, opt-in (`SolverConfig.self_contact`,
off by default), built from parts this document already argued for.

**Discovery is per tick, and the exclusion is structural.** Every chunk pair
that overlaps in space while sitting at graph distance three or more becomes a
transient PUSH contact: rest at the radii sum, one Jacobi pass, corrections
shared by inverse mass, scaled by `self_contact_elasticity`. Distance one is
the constraint set itself; distance two is the pair every folding joint drives
into contact, whose governed form is §3's angle-limit brace — itself a link,
so already at distance one. Neighbours *must* stay exempt: a limb is authored
out of overlap, or the surface would not read as one volume.

**The symmetric weight carries over unchanged.** Contact corrections reuse the
projection pass's per-pair weight for the two reasons written there: a pile-up
must stay a convex combination, and both ends must move equal and opposite in
mass-weighted terms or contact becomes a momentum source. Measured: a free
cluster with contact degrees 1-2-2-1 — the smallest uneven shape — holds its
centre of mass to 1e-3 over 300 ticks, and an eight-chunk pile dropped on one
point resolves without a NaN.

**Elasticity is the restoring force, spelled position-level.** The knob closes
that fraction of each overlap per tick: 1.0 opens an isolated pair in one tick
and reads as billiard balls; the 0.5 default spreads it over a few ticks and
reads as flesh. Discovery repeats every tick, so residual overlap is found
again until it is gone. Penalty-force repulsion was rejected for the same
reason the velocity bleed lost §2: a force stiff enough to stop penetration at
a fixed 40 Hz tick is a stability ceiling, while a partial position correction
is unconditionally convex.

**Ordering: shape, separation, world.** The pass runs after link projection
and before the ground/terrain passes, so the floor keeps the last word and a
contact push can never end a tick with a chunk underground. Measured: two
overlapping strangers dropped under gravity stack at exactly one and three
radii above the floor, every tick of the way down.

What it deliberately is not: of the three non-features this paragraph once
named, two have since landed on their own arc — self-friction
(`SolverConfig.self_friction`) and the capsule-capsule test are
docs/SELF_CONTACT.md's work; and the surface is untouched — the SDF still
smooth-mins everything, so nearby limbs merge visually at blend distance `k`
before their chunks ever touch. Keeping limbs visually distinct needs margin
authored into radii or blend groups in the field, and that is a render
decision, not a contact one.
