# Cast shadows: the key learns what stands in its way

**Status:** design, revised under three adversarial reviewers
(GL/perf, look-doctrine, io/tests) before first commit — they
materially changed it: the pricing was re-stated as new spend (the
floor's history proves one march's cost, not this arc's coverage),
zero's reach got owned as doctrine (it silences the floor too, and
the per-receiver look it forecloses is foreclosed on purpose), the
back-compat claim moved off §5's precedent and onto the
additive-lineage break it actually is, the azimuth-flip test learned
to fail on a dead feature, and the glow ruling swapped a false
provenance for the ledger's.
**Arc:** the residual LIGHTING.md §7 names first, taken up. The
lighting arc built the key, the fill, and the occlusion that
discounts the fill; it left the key omniscient — a limb over a body,
a body over the ground, and the key shines through both.
**North star:** the shadow the floor already catches lands on
everything else — the creature's own flesh and the terrain it stands
on — from the same march, gated by the same bounds argument, priced
honestly as new spend at the extents the AO fight measured, and off
exactly at zero.

---

## 1. What stands, and where the key still lies

The frame already contains a working cast shadow. `floor_shadow` in
`metaball.frag` marches from a reference-floor point toward
`u_light_direction` through `sd_field`, carries Quilez's penumbra
estimate (`shade = min(shade, 8.0 * d / t)` — clearance over distance
along), hard-breaks into umbra at `d < 1e-3`, walks at most
`SHADOW_MAX_T = 20` units in steps of `clamp(d, 0.02, 0.5)`, and stands
the tile mask down for the duration because shadow rays leave the
camera ray's cone. In front of it stands `floor_maybe_shadowed`: every
group's bounding sphere, inflated by `SHADOW_REACH = SHADOW_MAX_T / 8`,
slab-tested against the shadow ray — if none crosses, the running min
provably never drops below one and the march is skipped without being
wrong, the argument that beat AO's screen tiles
(`test_the_cull_is_invisible_under_the_floor_shadow_too` holds it to
byte equality).

That machine serves exactly one receiver. Everything the march itself
shades — creature flesh, terrain — takes the key unshadowed:

- **A limb over a body casts nothing.** Self-contact holds poses where
  a creature grips itself (docs/SELF_CONTACT.md); the paw pressed
  against the flank darkens by AO's fill discount only, and the key
  shines through the paw onto the flank it covers.
- **A creature on terrain casts nothing.** The floor is vetoed the
  moment a terrain loads (`_floor_state`), so the one surface that
  catches shadows is exactly the surface a world replaces — a creature
  standing in a room casts *less* shadow than one in the empty editor.
- **The honest cap holds.** LIGHTING.md §2 states it: AO scales
  ambient and glow only, so "a crease in full key stays clean *by
  construction* until the cast-shadow arc." This is that arc; the cap
  comes off.

## 2. Soft, and exactly the floor's softness

The shadow is soft — the penumbra estimate `floor_shadow` already
runs, unchanged, penumbra factor 8 baked. Both alternatives are
rejected by name:

- **Hard shadows** cost the same march (the steps are the price; the
  min-ratio is one multiply on top) and buy an aliased edge — a
  binary hit test at raymarch resolution stair-steps along every
  silhouette, at exactly the place a penumbra reads as flesh over
  flesh. Soft is not the expensive option here; it is the same-price
  option that looks like a body.
- **A softness knob** is rejected because the factor 8 is
  load-bearing twice over: `SHADOW_REACH = SHADOW_MAX_T / 8` is the
  bounds gate's inflation, so a dialable factor turns the gate's
  exactness constant into a uniform expression — computable, but a
  second dial on a term whose first dial nobody has turned, and every
  value of it re-tunes the floor's already-authored look. Softness is
  the vocabulary's opinion, like `GLOW_SKIN`.

The march's start deserves a new proof, because creature receivers
are the first self-intersecting origins `t = 0.05` has ever faced —
the reference floor sits outside `sd_field` and never once marched
through its own body. From a point on the receiver's own skin, a
locally flat lit surface gives `d(t) ≈ t·dot(n, L)`, and the start
constant splits the terminator into named bands:

- **`dot(n, L) < 0.02` is exact black.** The first sample at
  `t = 0.05` reads `d ≈ 0.05·dot(n, L) < 1e-3` and fires the umbra
  hard-break — not a rolloff, a band of zero.
- **`0.02 < dot(n, L) < 1/8` is the squared rolloff.** The ratio
  `8·d/t` is the constant `8·dot(n, L)`, so the key falls off as
  `8·dot²` instead of `dot` — the classic soft look of this
  estimator, landing entirely inside the band where lambert is
  already dying.
- **Above `1/8` a flat origin contributes nothing**, and only real
  occluders move the min.

The band edges breathe with the landing: the receiver point sits
within the march's growing tolerance (`u_surface_epsilon · max(1, t)`)
of the true zero set, which at world distances is the same order as
the `1e-3` umbra threshold — so the bands are surface-exact like
every post-hit term, never byte-pinned. The standard countermeasure —
lifting the ray origin off the skin along the normal — is rejected
because it leaks light through exactly the creases this arc exists to
darken. Concave receivers self-shadow by the same arithmetic, and
that is the feature, not the artefact.

## 3. Casters and receivers: the creature shadows, everything catches

**The shadow ray marches `sd_field` — the creature field, never the
composite.** What ships and what is cut, each with its reason:

- **Self-shadowing ships.** A creature's own limbs occlude its body:
  the march through `sd_field` from a skin point finds the rest of the
  creature at full fidelity, group-bounds cull live per step.
- **Creature-on-creature ships.** `sd_field` folds every group; a
  neighbour's body shadows yours exactly as your own does.
- **Creature-on-terrain ships.** The call site is the shaded hit path,
  which is receiver-agnostic — a terrain hit asks the same question a
  flesh hit does, and the marquee image lands: a body standing in a
  world finally darkens the ground under it. This is the floor's
  shadow generalized to the surface that vetoes the floor.
- **Creature-on-grid stands.** The drawn reference floor keeps the
  call it has today.
- **Terrain-on-anything is cut.** Marching `sd_scene` instead would
  not merely add the fold — it would kill the gate: `sd_terrain` has
  no cull and no bound (the AO cost note already says so), so
  `maybe_shadowed`'s "no bound crossed, therefore full light" proof
  is void the moment terrain can cast, and every pixel marches. The
  ratio is the quantitative half of the argument: AO pays the
  un-culled terrain fold five times per hit pixel and that is the
  lighting arc's accepted spend; a shadow march would pay the same
  fold up to 48 times per pixel — the same fold, payable there and
  ruinous here. Terrain still darkens itself by form — `dot(n, L)`
  shades a canyon's far wall — but a cliff casts nothing onto a
  creature at its base. Named residual, with its reopening condition
  stated precisely: **per-primitive** terrain bounds, by necessity —
  a half-space plane admits no finite global bound, so no single
  terrain sphere can ever exist for the gate to test.
- **The grid plane never casts.** An infinite plane under an
  above-horizon key shadows nothing above it; below-horizon uplights
  already have their ruling (LIGHTING.md §6: the march heads
  underground, finds no creature, answers full light). The plane
  stays outside `sd_scene` and outside the shadow's world, exactly as
  it stays outside the march's.

One asymmetry worth stating so nobody rediscovers it as a bug: the
world receives the creature's shadow but cannot return the favour.
A creature at a cliff base under a low key stays lit while the wall
behind it goes dark. That is the cut's visible face, accepted at this
arc's budget, and the residual above is its price tag.

## 4. One function: `floor_shadow` becomes `key_shadow`

The existing march and the new one are the same question — how much
of the key does the creature field block along `light` from `point` —
so there is one function, not two. `floor_shadow` renames to
`key_shadow`, `floor_maybe_shadowed` to `maybe_shadowed`, the march
body unchanged, and the strength knob (§7) applies inside the
function so every receiver speaks it identically:

```glsl
float key_shadow(vec3 point, vec3 light) {
    // key shadow -- begin. The markers are load-bearing: the
    // exact-skip test excises everything between them and asserts
    // shadow_strength 0 renders the same bytes as the deleted march.
    if (u_shadow_strength > 0.0 && maybe_shadowed(point, light)) {
        bool was_tiled = sdf_tiled;
        sdf_tiled = false;
        float t = 0.05;
        float shade = 1.0;
        for (int i = 0; i < 48; ++i) {
            float d = sd_field(point + light * t);
            if (d < 1e-3) { shade = 0.0; break; }
            shade = min(shade, 8.0 * d / t);
            t += clamp(d, 0.02, 0.5);
            if (t > SHADOW_MAX_T) break;
        }
        sdf_tiled = was_tiled;
        shade = clamp(shade, 0.0, 1.0);
        // The endpoint is exact by branch, not by arithmetic: GLSL
        // pins no rounding for a lerp, and the default must
        // reproduce a pre-shadow room's floor shadow bit for bit.
        return u_shadow_strength == 1.0
            ? shade
            : clamp(1.0 - u_shadow_strength * (1.0 - shade), 0.0, 1.0);
    }
    // key shadow -- end
    return 1.0;
}
```

The `== 1.0` comparison is not merely sound here, it is required: the
lerp at strength 1 is not an IEEE identity and no spelling of it is
guaranteed to return `shade` unrounded, so the branch is the only
honest way to make the default exact. And the gate's byte-equality
proof survives the strength spelling only because `f(1.0) = 1.0`
exactly in both branches: the skipped march would have answered
`shade = 1.0`, the equality branch returns it raw, and the lerp
branch computes `1.0 - s·(1.0 - 1.0) = 1.0 - s·0.0`, exact for every
finite `s` — which `validate()` guarantees.

Two call sites. The floor branch keeps its exact line — `float lit =
mix(0.45, 1.0, key_shadow(point, light));` — and at the default
strength the equality branch hands it the raw penumbra, so a
pre-shadow room's reference floor renders today's bytes. The shaded
hit path gains the second call beside the terms it feeds:

```glsl
vec3 light = normalize(u_light_direction);
float shadow = key_shadow(position, light);
```

The separate-function alternative — a creature-receiver march tuned
apart from the floor's — is rejected as the twin-spelling shape:
two marches answering one question drift, and the first tuning
applied to one and not the other ships a frame where the same limb
casts two different shadows on two receivers in one image.

Unification also means the floor's knobless shadow becomes knobbed:
`shadow_strength` governs the one shadow everywhere, floor included.
At the default 1.0 nothing moves; what zero means is §7's to own.

## 5. What the shadow gates: the key, only

```glsl
vec3 shaded = albedo * (ambient * ao + key * shadow) + glow * ao;
```

The occlusion authorities stay disjoint, one term each. The shader's
own closing comment already keeps the ledger — "AO scales the fill
and the glow, never the key: occlusion is the environment failing to
reach a crease; the key light failing to reach one is a cast shadow"
— and this arc adds the ledger's missing row without touching an
existing one: AO owns ambient and glow, the shadow owns the key. The
consequences are the model working, not defects:

- **Umbra is fill, not black.** A fully shadowed patch shades as
  `albedo * ambient * ao + glow * ao` — the backdrop's hues doing
  exactly the job §2 of LIGHTING.md gave them. The environment is
  the fill, and a shadow is where you finally see only the fill. At
  the defaults the depth is statable in the house convention: a
  full-key up-facing patch keeps ≈ 27% of its lit level in umbra
  (the fill's ≈ 0.28 against 0.75 of key plus the same fill).
- **Glow survives shadow, deliberately — because glow is
  key-blind.** The term reads `n` and `v` and never `L`: it is
  silhouette backlight whose *colour* borrows the key's, but whose
  geometry knows nothing about where the key stands. Gating a
  key-blind term by the key's occlusion would make it track the key
  — and the pinned countershade test
  (`test_a_countershaded_belly_ignores_the_keys_azimuth`, belly
  luminance glow included, invariant under a 180° swing) would break,
  because the swing moves the occlusion geometry even where it moves
  no lambert. The accepted artefact is named instead, the
  `damping="1.02"` convention: a key-coloured rim can glow inside a
  neighbour's umbra, confined to silhouette bands by the pow-3
  falloff.
- **Contact lines double-darken.** A planted foot takes AO's fill
  discount and the shadow's key discount at once. Both lights really
  do fail there; the two dips are two mechanisms agreeing.

## 6. The budget: new spend, priced at true extents

The plateau doctrine says frame time is shading-bound, and this arc
is a shading feature: **everything below is new spend, accepted
explicitly at the doctrine, with `shadow_strength = 0` as the exact
exit.** The floor's history is evidence for one number only — the
per-pixel cost of a single march on a grid scene — and for nothing
about this arc's coverage: `floor_shadow` has never once run in a
world, because the floor is vetoed the moment terrain loads (§1's own
veto), and on grid scenes it pays only on corridor floor pixels,
where `floor_maybe_shadowed` fails to excuse the march.

The new spend, with a terrain loaded — where **every pixel is a hit
pixel** and the shaded path runs full-frame:

- **Full-frame, the gate.** One `maybe_shadowed` pass per pixel: an
  O(G) ray-sphere scan, the same order as the bounds scan each AO tap
  already pays, with no folds behind it.
- **Corridor terrain pixels march.** The corridor — the creatures'
  inflated bounds swept toward the key, projected on screen — is
  where the gate stops excusing. A lit corridor pixel walks
  `SHADOW_MAX_T = 20` in steps capped at 0.5: roughly forty `sd_field`
  evaluations, each an O(G) bounds scan plus the surviving group
  folds, tiles stood down, and **never the terrain fold** — the
  decisive difference in shape from an AO tap, which folds
  `sd_terrain` un-culled five times per pixel.
- **Lit flesh pixels always march, from a base of nothing.** A
  receiver point on a skin sits inside its own group's bounding
  sphere by construction, so the gate can never excuse it — every
  lit creature pixel pays the ~40 evaluations, ≈ 8× the AO ladder's
  five creature-side evaluations, on pixels that until this arc paid
  zero shadow cost. This is the honest worst case and the design
  accepts it by name.
- **Shadow-side flesh is nearly free.** A ray from key-averted skin
  heads into its own body and hard-breaks in about one evaluation —
  the deeper the shadow, the earlier the exit, so the dark half of
  every creature costs almost nothing.

Skipping the gate on flesh — the hit path could know a creature won
and jump straight to the march — is rejected: it saves one sphere
scan ahead of a forty-evaluation march, couples the shadow to
`scene_albedo`'s creature/terrain split, and is the twin-spelling
shape §4 rejects. One question, one path.

No new early-out is invented: a "good enough" break on the running
min is unsound (`d` can still shrink later), and re-tuning the step
clamp is march optimization, which is closed. The knobs the author
holds are the real ones: the gate scales with geometry, umbra
self-limits, and zero removes everything.

**Zero is an exact skip.** `shadow_strength = 0` returns before the
gate exists, on a uniform branch with no derivatives inside — the AO
ladder's precedent, enforced the same way: the `key shadow -- begin/
end` markers are load-bearing, a test excises everything between them
and asserts byte identity (§8). The excised function body is
`return 1.0;`, and `key * 1.0` is exact, so the claim survives the
call sites too.

**The prepass moves nothing categorical.** Creature-receiver shadows
are computed from `position`, which the cone seed shifts within the
march's surface tolerance — so seeded and unseeded frames disagree
about the shadow exactly as much as they already disagree about the
normal, the albedo, and every other post-hit term: surface-exact,
never byte-exact, inside the epsilon the prepass tests already
budget. This is the opposite of the halo rejection: the halo measured
the camera march's own interior, which the seed provably skips; the
shadow marches a fresh ray from the landed point and never reads the
camera march at all.

Two standing facts, restated so they are chosen rather than
discovered. Debug mode 3 counts camera-march steps only, so shadow
work (like AO work before it) is invisible to the step-count view.
And **every debug mode is shadow-free**: modes 2, 3 and 4 return
before the floor branch even runs, the floor branch itself is gated
`u_debug_mode == 0`, and mode 1's hit path returns the normal before
any shading term — the shadow exists in mode 0 alone, the oracles
never meet it, and the CPU twin stays lighting-free with no
`field.py` counterpart owed.

## 7. The knob and the record

One new field, and only one:

```python
shadow_strength: float = 1.0    # Look, render/look.py
```

It slots **between `ao_strength` and `glow_strength`** — the two
occlusion authorities adjacent in the record as on the panel — and
the position is load-bearing three times over: the writer's
field-order promise, the ordered key list of
`test_a_written_look_record_is_complete_and_in_field_order`, and the
Finish row order all enumerate the dataclass. Mid-insertion is safe because every
`Look(...)` construction in the tree is keyword-only in practice —
the composition rewrite already turned the constructor sites into
`Look(field=...)` kwargs, so no positional caller exists to reorder.
The field joins `_STRENGTH_FIELDS` and inherits §4's validation
verbatim — finite, non-negative, no ceiling. Above 1 extrapolates
the penumbra toward black and clamps there: a contrast dial with a
named artefact, the `damping="1.02"` convention.

**The default is 1.0, and the argument is the arc's own continuity
anchor:** 1.0 is the unique value at which the one shadow that
already ships — the floor's — renders bit for bit, through §4's
equality branch. Full removal of a blocked key is also the physical
answer, and the environment fill already keeps umbra off black
(§5's ≈ 27%); fractions are stylization, available but not shipped
as the truth.

**Zero makes the key omniscient everywhere — floor included — and
the reach is doctrine, not accident.** In the default grid scene,
strength 0 removes the floor shadow that predates the lighting arc
and never had an off switch: a grid frame no shipped default has
shown since the floor first learned its shadow. And the in-between
look — floor shadowed, flesh not — is unauthorable at every
strength, deliberately: per-receiver shadow state is exactly §1's
indicted lie, the key shining through some bodies and not others, so
the knob refuses to author it. A planted foot still seats at zero:
AO's drawn-floor fold keeps darkening the contact — the fill's
discount, which was never the shadow's to give.

`<look>` gains the attribute; the writer stays all-or-nothing,
writes it always, at repr precision like every look float.
Back-compat is a break, and the design says so rather than borrowing
a precedent that does not cover it: **`shadow_strength` is the
format's first additive attribute whose default relights a saved
record.** The additive lineage until now — `self_friction`, `<look>`
itself as a section — preserved old behaviour at the default;
LIGHTING.md §5's "exposure-similar, not pixel-identical" covered
section-less documents under a changed model, not saved records
under a grown one. The break stands on two landed feet: `_read_look`'s
rule that an absent attribute inside a present section means the
dataclass default, and the AO precedent — `ao_strength = 0.5`
darkened every section-less pre-arc room and §5 defended it as the
arc's point. A lighting-arc room reopens shadowed, deliberately, and
the attribute exists so its author can state otherwise — knowing
zero's reach. Two consequences ride along, named: a lighting-arc
editor resaving a post-shadow file drops the attribute (its writer
writes its own full record — inherent to additive attributes, chosen
not accidental); and `xml_v2.py`'s format docstring sentence — a
full-written section means "a dataclass default that later changes
cannot relight a saved room" — is falsified by this arc and re-scopes
in §9: full writing protects the *stated attributes'* values, not
against new terms.

The panel row lands in the Lighting dock's **Finish** group, between
Occlusion and Glow. The placement is confirmed by what is already on
screen: Occlusion's landed tooltip teaches the two-authority table —
"darkens the fill and the glow, never the key" — and the shadow row
beside it makes the table the panel's visible shape. Same
`_spin(0.0, 1.0, 0.05)` as its neighbours, `_admit` widening so a
file-authored `2.0` displays as 2.0 rather than being clamp-committed
by the next nudge, and the edit calls `settings_edited()` — it rides
the file, so it is file-grade, the §5 line. Tooltip, owning the
casters, the split, and the reach: "How much key the creature field
blocks — a limb on its own flank, a neighbour's body, a foot's
shadow on the ground. Occlusion darkens the fill; this darkens the
key. Zero skips the shadow march everywhere, floor included."

Rejected spellings: a `<shadow>` section (one scalar is not a
record); twin fields on `MetaballSettings` (the composition exists so
this arc adds one line, not a copy chain); per-receiver toggles
(three booleans authoring §1's lie); a shadow colour (a shadow is
the key's absence, and its colour is already the fill's job — tinted
shadows fall out of the ambient's hemisphere today).

## 8. Tests

Content through `MetaballViewport` with the prepass on, the armed-skin
lesson, in the landed style:

- **The strength-off-minus-on darkening flips sides under a 180°
  azimuth swing** — the covariance mirror of
  `test_a_countershaded_belly_ignores_the_keys_azimuth`, specified so
  it can fail on a dead feature: a tall chunk stands on a wide low
  slab (two groups, hard-min seam), and at each swing the test
  renders strength 1 *and* strength 0 and takes the luminance
  increment between them. Raw luminance is not compared across the
  swing — lambert already moves under a swung key, and a dropped
  `uniforms()` entry would leave `u_shadow_strength` at GL's 0.0
  default with a raw test still green. The increment cannot lie:
  it is zero everywhere on a dead feature, and alive it must be
  substantial on the caster's anti-key side and swap sides with the
  key.
- **`shadow_strength` 0 is byte-identical to an excised march** —
  `test_ao_strength_zero_is_byte_identical_to_an_excised_ladder`'s
  pattern verbatim: excise between the `key shadow` markers through
  the patched `fragment_source`, assert the penumbra token
  (`8.0 * d / t`) leaked nowhere, render with the floor enabled so
  both call sites are exercised, compare bytes.
- **Strength 1 is byte-identical to a strength-blind shader** — the
  §4 endpoint made testable rather than trusted: patch
  `fragment_source` to replace the endpoint ternary with raw
  `shade`, floor uniforms on (the
  `test_the_cull_is_invisible_under_the_floor_shadow_too` patching
  precedent), and assert `np.array_equal` at strength 1.0 — the
  default provably adds no arithmetic to the shadow that already
  shipped.
- **A pixel in full umbra shades as if the key were off** — the §5
  split as bytes: a fat occluder directly over a receiver, and the
  umbra patch under its core renders byte-identical between
  `shadow_strength = 1` and `light_strength = 0`. The docstring
  carries the argument so nobody "fixes" it into a tolerance: neither
  knob feeds `trace()` or the coarse pass, so both frames share a
  byte-identical camera march and seed contents within the run;
  hard-break shade is exactly 0.0, and `key * 0.0` and
  `0.0 * DIFFUSE * lambert` are both exact zeros; ambient, AO and
  glow are untouched in both frames. The patch inset from the
  occluder's silhouette is an explicit constant; if a driver ever
  flakes the margin, the fix is to shrink the patch toward the core,
  never to weaken byte equality.
- **A body on terrain darkens the ground on the anti-key side** —
  worm on a terrain slab, low key: terrain pixels beside the body
  darken at strength 1 against strength 0, and the same patch with
  the body absent does not.
- **A limb shadows the flank it covers** — two stacked chunks,
  overhead key: the lower chunk's top under the upper darkens at
  strength 1 against 0. The AO crease test's shape, aimed at the key
  term instead of the fill.
- **A lone convex body keeps its lit side** — the bright-half mean
  luminance of a sphere agrees between strengths within 2/255: §2's
  bands are the only self-darkening a convex body takes, and they
  must stay confined to the terminator.
- **The standing suite re-runs unchanged**: the countershade
  invariance survives (belly normals carry no key for the shadow to
  remove, and both swings zero it symmetrically); the two
  cull-invisibility tests now cover the new call site for free
  (`maybe_shadowed`'s skip and the per-step group cull are both
  byte-exact by the same proofs); depth and albedo oracles are
  byte-untouched; `test_every_uniform_the_settings_supply_is_declared`
  picks up `u_shadow_strength` the moment `uniforms()` maps it.

io, on `test_gait_io`'s conventions, with the two teeth named:
`tuned_look()` gains a `shadow_strength` away from the default — its
own docstring is the reason ("every field away from its default, so a
dropped one cannot hide"; at 1.0 a dropped writer or reader entry
hides exactly) — and
`test_a_written_look_record_is_complete_and_in_field_order`'s ordered
key list grows the attribute in its §7 slot. Then: round-trip a
non-default `shadow_strength`; a `<look>` without the attribute reads
1.0; a section-less file still reads `None`; SchemaError on negative
and non-finite; `save()` without a look writes today's bytes;
quadruped unchanged. Editor: dirty on the shadow spin, `_admit`
widening displays a loaded 2.0, refresh round-trips it.

## 9. Churn

The implementation touches, exhaustively:

- `mesh_xml/render/gl/shaders/metaball.frag` — the renames
  (`floor_shadow` → `key_shadow`, `floor_maybe_shadowed` →
  `maybe_shadowed`), `u_shadow_strength` + the endpoint branches +
  the marker fence, the hit-path call, the `shaded` line — and the
  line's own comment: "the key failing to reach one is a cast
  shadow, which this arc deliberately does not add" sits on the
  exact line being edited and rewrites to state the landed split.
- `mesh_xml/render/look.py` — the field in its §7 slot,
  `_STRENGTH_FIELDS` entry, docstring sentence.
- `mesh_xml/render/gl/metaball.py` — one `uniforms()` entry:
  `float(look.shadow_strength)`, no arithmetic between parse and
  upload, so the shader's `== 1.0` sees exactly the 1.0 the file or
  the default stated.
- `mesh_xml/io/xml_v2.py` — `_read_look` entry, the writer's
  attribute, and the format docstring's re-scope: "a dataclass
  default that later changes cannot relight a saved room" is
  falsified by this arc's additive attribute — full writing protects
  the stated attributes' values, not against new terms (§7).
- `mesh_xml/editor/panels.py` — `LightingPanel`: spin, Finish row,
  tooltip, handler, refresh pair, `_admit` registration — and the
  elevation tooltip, which names the renamed function by behaviour
  ("the floor shadow just finds nothing underground") and
  understates an uplight: creature receivers do shadow from below.
- `docs/LIGHTING.md` — the §7 cross-reference already rides this
  design's own commit; the implementation's sweep is the falsified
  sentences, enumerated: §1's "this arc keeps" framing of the
  floor-only shadow; §2's "a crease in full key stays clean by
  construction until the cast-shadow arc"; §4's "the four strengths"
  (now five); §5's full `<look>` example (gains the attribute); §6's
  Finish list "occlusion strength, glow strength" and the uplight
  ruling's floor-only truth; §7's "The key light shadows nothing but
  the reference floor, exactly as today."
- `docs/SETTINGS.md` — §5's ledger: "the four strengths" becomes
  five and "eighteen in all" becomes nineteen `settings_edited`
  sites.
- `README.md` — the doc table's new row.
- `tests/test_metaball.py` — §8's additions, and the
  cull-invisibility test's docstring, which names `floor_shadow` by
  its old spelling.
- `tests/test_settings_io.py`, `tests/test_panels.py` — §8's io and
  editor halves, `tuned_look()` included.

No solver, core, scene, or anim code moves. `field.py` owes nothing
(§6's oracle note). The firewall test holds without edits — the new
field is a float on a stdlib dataclass.

## 10. Deliberately not done

- **No shadow maps.** A shadow map is a rasterizer's answer: render
  depth from the light, resample it per pixel, then spend the rest of
  the feature's life tuning bias against acne and peter-panning. This
  renderer rasterizes nothing — a light-view depth pass would be a
  second full raymarch at map resolution, paying the same field
  evaluations to produce a *worse* answer (fixed-resolution, biased,
  resampled) than the march already gives analytically along the
  actual shadow ray. In a raymarcher the field is its own shadow
  query; importing the rasterizer's cache and its artefacts would be
  building the wrong machine beside the right one.
- **No terrain casters.** Cut in §3, priced, residual on
  per-primitive terrain bounds existing first.
- **No softness knob, no shadow colour, no per-receiver toggles.**
  §2 and §7's rejections — the last one twice over, since
  per-receiver shadow state is §1's indicted lie with a checkbox.
- **No second light.** One key is the model; a fill that casts is a
  contradiction in it (the fill is the environment, and the
  environment arrives from everywhere).
- **No halo or bloom, no per-creature look, no tone mapping.**
  LIGHTING.md §7's rejections stand exactly as written; nothing here
  weakens their arguments, and the halo's categorical-difference
  argument is this design's own honesty benchmark (§6).
- **No march-quality reopening.** `SHADOW_MAX_T`, the step clamps,
  and the 48-step cap stay `floor_shadow`'s numbers; the plateau
  doctrine closed march optimization and this arc does not knock.

## 11. Packages

**Package 1 — the shadow.** The shader work in full (§4's function,
both call sites, the uniform, the markers, the `shaded` line and its
comment); `Look.shadow_strength` in its slot + `_STRENGTH_FIELDS`;
the `uniforms()` entry. Tests: the strength-differenced azimuth
flip, the excision byte-identity, the strength-1 strength-blind
byte-identity, the umbra ≡ key-off bytes, creature-on-terrain, the
stacked-limb crease, the convex lit side at 2/255, and the standing
suite green — the two cull tests and both oracles explicitly re-run.

**Package 2 — the record and the panel.** `<look>` attribute in
reader and writer with the docstring re-scope; `tuned_look()` and
the ordered-keys test grow the field; `LightingPanel`'s Finish row
with `_admit` widening and `settings_edited`, plus the elevation
tooltip; the LIGHTING.md and SETTINGS.md sweeps of §9, enumerated;
the README table row; io and editor tests (§8's second half).

Live verification, the standing recipe (`.claude/skills/run`), on
the terrain scene where zero's semantics are clean: the catch scene
on its world under a low key — a paw's shadow crossing the flank,
the body's shadow pooling on the terrain, and strength 0 returning
the lighting arc's look of the same frame. The grid scene's zero is
§7's story, not this still's.
