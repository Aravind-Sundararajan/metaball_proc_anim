# Lighting: the environment becomes the light, and the look becomes a record

**Status:** design, revised under three adversarial reviewers (GL/shader,
io/editor, look/scope) before first commit — they materially changed it:
the hemisphere gained real luminance verticality (chroma alone left it
inert at the defaults), the glow learned to carry the skin (the draft
shipped the exact white rim its own §1 indicts), the chroma lift got a
named black boundary, the AO ladder learned the reference floor, the
elevation clamp died (it silently rewrote file-authored pole vectors),
and every exactness claim was re-stated at what the prepass actually
guarantees.
**Arc:** the fork's other side — contact landed (docs/SELF_CONTACT.md);
this is the look thread it deferred to.
**North star:** the light can move, the environment has depth, creases
read where the fill reaches them — and a file carries its look on
exactly the additive terms SETTINGS.md §6 reserved when it declined to
persist a flat `background_colour` "for tomorrow's gradient, designed
blind". The fields are now in hand.

---

## 1. What the render does today, and where it lies

One directional light, `(0.4, 0.8, 0.45)`, reachable from no panel.
`metaball.frag`'s whole shading model is four hardcoded terms:

```glsl
lambert = max(dot(n, L), 0)
bounce  = max(dot(n, -L), 0) * 0.2
rim     = pow(1 - max(dot(n, -v), 0), 3) * 0.35
shaded  = albedo * (0.25 + 0.75 * lambert + bounce) + rim
```

Each term is defensible alone; the showcase stills caught them lying
together.

- **The fill fights the countershade.** `bounce` is a white fill from
  exactly `-L`. Countershade (docs/TEXTURES.md) pales the belly to
  *cancel shadow with pigment* — that is the entire illusion — and the
  renderer then cancels the same shadow again with light. A pale belly
  plus a white underside fill reads flat and overbright, and because
  the fill tracks the key light's opposite, it is brightest from
  precisely the direction the pattern already accounted for. The two
  systems disagree about whose job the underside is.
- **The rim is additive white.** It brightens dark backs and painted
  skins alike, unscaled by anything the author controls, and washes
  the darkest tones the countershade just authored.
- **The ambient is a number.** `0.25` carries no hue, so shadowed flesh
  desaturates toward grey instead of cooling toward the environment.
- **The background is a void.** One flat colour behind everything; the
  creature floats in it. The floor grid helps where it is drawn, and
  `show_floor` off returns the void.
- **Creases do not read.** A limb pressed against a body shades exactly
  like open skin — no occlusion of any kind. Now that contact lets
  creatures fold against themselves and grip (SELF_CONTACT.md), the
  poses that need crease shading are poses the solver actually holds.

What already works, and this arc keeps: the floor's penumbra shadow
march follows `u_light_direction` (`floor_shadow`), so a movable light
gets a moving shadow for free; and the oracle contract confines
*lighting* to the eyeballed modes — debug modes 2 (packed depth) and 4
(raw albedo) are the CPU-compared paths, return before the miss branch,
and while mode 4 reads the surface and terrain colours (they are data,
and the CPU twin takes the same values), no light, ambient, occlusion,
or glow term touches either. Mode 1 (normals) shares the miss branch
with mode 0, so its backdrop becomes the gradient — harmless, it is
eyeballed and never oracle-compared, and stated here so nobody
rediscovers it as a surprise.

## 2. The model: one key, and the background is the fill

The design principle that replaces the four freelancing terms: **the
key light is authored, and everything else is the environment seen in
the backdrop.** The background stops being a void and starts being the
thing that fills shadows.

```glsl
// the backdrop: a vertical gradient, sampled once per fragment
sky      = mix(u_background_colour, u_background_top, v_uv.y)

// a backdrop colour's contribution to the fill: its hue lifted to unit
// brightness — the colours state the hue, ambient_strength states the
// level — with the lift fading out across a *visible* near-black range,
// so a black sky fills nothing and approaches that boundary
// continuously instead of at a one-picker-quantum cliff
chroma(c) = (c / max(maxc(c), 1e-3)) * smoothstep(0.0, 0.04, maxc(c))

// the fill: hemispheric on world-up, and the ground half sits a named
// step below the sky half — verticality in luminance, not just in two
// nearly identical hues — and none of it tracks the key
ambient  = u_ambient_strength
         * mix(AMBIENT_GROUND * chroma(u_background_colour),
               chroma(u_background_top),
               0.5 + 0.5 * n.y)                     // AMBIENT_GROUND = 0.7

// the key: direction, colour, level — all authored
key      = u_light_colour * u_light_strength * DIFFUSE * max(dot(n, L), 0)

// the glow: the rim term promoted to a knob, carrying the key's colour
// *through the skin* — dark skins rim dimmer, so the backlight cannot
// bleach the tones countershade just authored (the draft that carried
// pure light colour reproduced §1's white rim exactly, at exactly its
// old strength, and celebrated the continuity)
glow     = u_light_colour * u_glow_strength
         * mix(vec3(1.0), albedo, GLOW_SKIN)        // GLOW_SKIN = 0.5
         * pow(1 - max(dot(n, -v), 0), 3)

shaded   = albedo * (ambient * ao + key) + glow * ao
```

`DIFFUSE = 0.75`, `AMBIENT_GROUND = 0.7`, `GLOW_SKIN = 0.5` are baked
constants — the vocabulary's opinion, like `SDF_STRIPE_BAND` — so
`light_strength = 1` lands today's diffuse level exactly. The old
`bounce` term is deleted, not parameterized: its job (keep the
underside off flat black) passes to the ground half of the ambient,
which is dimmer than the sky half by construction, hued, and
**independent of the key direction** — so the belly reads as pigment
against quiet ground light, and countershade's cancelled-shadow
illusion survives because the renderer stops pre-cancelling it. The
ambient keys on world-up exactly as countershade does (`radial.y`,
TEXTURES.md §2): the two systems finally agree about which way the
world is, in luminance and not merely in code.

At the shipped defaults the fill has real verticality where the flat
`0.25 + bounce` was: an up-facing surface takes ≈ 0.28 of cool sky
tint (`0.35 · chroma(top)`), a down-facing one ≈ 0.20 of ground tint
(`0.35 · 0.7 · chroma(horizon)`), and neither moves when the key
does. Two consequences worth stating plainly. A default belly now
sits *near the backdrop's own band* — the glow is what separates an
unlit underside from the sky behind it, and `glow_strength 0` lets
bellies melt into the gradient; that is the knob doing what it says,
not a defect, but the panel tooltip says it. And the chroma lift has
boundaries: a black backdrop fills nothing (the environment IS the
fill, and the smoothstep makes the approach visible rather than a
cliff at one picker quantum), while a fully saturated backdrop fills
in its own monochrome — a complementary-coloured skin multiplies
toward black in shadow, and only the key and glow bring it back.
Deliberate, named, and the sky swatch is sitting right there.

**AO scales the ambient and the glow, never the key.** Occlusion is
the environment failing to reach a crease; the key light failing to
reach one is a cast shadow, which this arc deliberately does not add
(§7). The honest cap follows and is worth one sentence: AO can only
discount the fill's share of a pixel — a crease in full key stays
clean *by construction* until the cast-shadow arc, and the creases
this arc makes read are the shadow-side and contact ones (a
right-angle crease dips ≈ 17% at the defaults; a deep self-contact
slot, far more). Five taps along the normal, Quilez's ladder, with
the reference floor folded in where it is actually drawn:

```glsl
occ = 0.0; fall = 1.0;
for (i = 1; i <= 5; ++i) {
    float h = float(i) * AO_STEP;                   // AO_STEP = 0.06
    float d = sd_scene(p + n * h);
    if (u_floor_enabled != 0)
        d = min(d, p.y + n.y * h - u_floor_height); // the drawn grid occludes
    occ  += (h - d) * fall;
    fall *= 0.7;
}
ao = clamp(1.0 - u_ao_strength * 3.0 * occ, 0.0, 1.0)
```

The floor fold matters because the default editor scene *is* the
grid: without it a foot planted on the reference floor takes zero
contact darkening while the same foot on a terrain darkens — the
design's own "that is what resting looks like" argument, failing in
the out-of-box state. The plane is deliberately outside `sd_scene`
(march and oracle exactness, metaball.frag's own comment), and it
stays outside: the fold above lives in the AO ladder only, on the
post-hit shaded path, gated by the same `u_floor_enabled` the grid
draws by — modes 2 and 4 never reach it.

`AO_STEP = 0.06` reaches 0.3 world units — the scale of the default
blend's creases on bodies "a few units across" (the camera module's
own sizing assumption). Step and falloff are baked; `ao_strength` is
the knob, and **zero skips the loop entirely** — the taps are the only
new field evaluations in the frame, so their off switch must be
exact (a uniform branch, no derivatives in the skipped region —
achievable and asserted by test, not approximate).

The AO taps run against `sd_scene` (a limb resting on terrain darkens
at the contact line) with the tile mask stood down for the duration,
`floor_shadow`'s own argument: taps leave the camera ray's cone, and
an occluder one tile over is real. The per-group bounds cull stays
live — it is position-based, not tile-based — so the taps pay for the
groups near the surface, not for the room.

## 3. The backdrop: screen-keyed, and the floor fades into it

`background_colour` keeps its name and its meaning — the base of the
frame, the horizon — and `background_top` joins it. The gradient runs
in **screen space** (`v_uv.y`), not world space, deliberately:

- Under the orthographic toggle every ray shares one direction; a
  direction-keyed gradient collapses to a single flat colour the
  moment the author presses O. Screen-keyed survives both projections.
- The backdrop is a backdrop, not a sky dome. The floor grid already
  supplies the world's horizon; a second, ray-keyed horizon behind it
  would tilt against the grid every time the camera pitched. The
  screen-keyed lerp cannot contradict the grid's vanishing line
  because a two-stop vertical wash has no horizon *line* to
  contradict it with, and the orbit camera has no roll to skew it.

Misses write `sky`, and the floor's distance fade retargets from flat
`u_background_colour` to the same `sky` sample the miss path would
write at that pixel — otherwise the arc ships a visible seam where
the faded grid meets the gradient it no longer matches. The floor's
lit/base arithmetic is otherwise untouched.

Defaults: `background_top = (0.15, 0.17, 0.21)`, horizon keeps
`(0.11, 0.12, 0.14)`. Lighter at the top, so the horizon remains the
deepest tone in the app and `theme.FIELD`'s provenance — "the
raymarcher's background as chrome sees it, deepest tone in the app" —
stays true as written; the theme comment re-points at the horizon
field by name and `test_theme` keeps holding it to the source. The
top value is chosen a clear step *below* the chrome's RAISED
(`#2c3039`) — the draft's `(0.17, 0.19, 0.23)` landed within one
quantum of the button colour, which would have made the sky's
brightest band read as chrome by coincidence; package 1 verifies the
final value against the docked greys with the standing screenshot
recipe before it freezes.

## 4. The record: `Look`, GL-free, composed not copied

The look fields move into one dataclass:

```python
@dataclass
class Look:                      # mesh_xml/render/look.py — no GL import
    light_direction: tuple[float, float, float] = (0.4, 0.8, 0.45)
    light_colour:    tuple[float, float, float] = (1.0, 1.0, 1.0)
    light_strength:  float = 1.0
    ambient_strength: float = 0.35
    ao_strength:     float = 0.5
    glow_strength:   float = 0.35
    surface_colour:  tuple[float, float, float] = (0.82, 0.55, 0.42)
    terrain_colour:  tuple[float, float, float] = (0.36, 0.44, 0.31)
    background_colour: tuple[float, float, float] = (0.11, 0.12, 0.14)
    background_top:  tuple[float, float, float] = (0.15, 0.17, 0.21)
```

It lives in `mesh_xml/render/look.py` because the render package's
own init states the boundary this rides on: "Only the gl/ subpackage
may import OpenGL." The chain holds today — `mesh_xml.render`'s init
pulls only `draw_points` (numpy and `core.body`) — but io importing
render is a first-of-its-kind edge, and a docstring is not an
enforcement mechanism: package 2 lands a firewall test that imports
`mesh_xml.io.xml_v2` in a fresh interpreter and asserts OpenGL never
entered `sys.modules`, so the GL-less-collection property SETTINGS.md
§3 protected becomes tested instead of trusted. (`core/look.py` was
weighed — every record io constructs today lives in core/ or scene/ —
and declined: `Look` is a render preference, render/ is its semantic
home, and the firewall makes the boundary real.)

`MetaballSettings` **composes** it — `settings.look: Look` via
`default_factory` — rather than growing ten twin fields with a
hand-written copy in every open/save path. Twin spellings of one
record is the shape SETTINGS.md §2 rejected for the world ("the
rarely-exercised spelling rots first"); composition makes the io
record and the render record the same object, so they cannot drift.
`uniforms()` maps `look.*` onto the `u_*` names, and
`test_every_uniform_the_settings_supply_is_declared` keeps holding
the map to the shader.

The churn is shallow but wider than "panels and uniforms", and one
field is not purely a uniform: `surface_colour` is **data** as well —
`Document.group_rows()` seeds `sdf.group_table` with it (group 0
wears it exactly; later groups hue-rotate from it) and
`attribute_rows()` seeds `pack_attributes`' unpainted fallback, which
is how it already reaches OBJ vertex colours through `export_mesh`.
The re-spell sites, enumerated: the uniforms map, the SurfacePanel
swatch/picker sites, those two `Document` packing methods, and the
tests — including `MetaballSettings(surface_colour=..., ...)`
*constructor kwargs* in test_metaball, which composition turns into
`TypeError` and which are rewritten as `MetaballSettings(look=
Look(...))`, a different shape of edit from attribute respelling.
No solver or io code reads any of them; the live-edit path is
already safe because `_repack`'s array comparison re-uploads the
tables whenever the packed bytes change.

Validation on read, the §4 house rule — refuse what breaks
downstream, leave taste to the author: every float finite;
`light_direction` non-zero (the shader normalizes it, and zero has
no direction to state); colour components ≥ 0; the four strengths
≥ 0 (negative light is not a tuning, it is a typo — the
`iterations` ruling). No ceiling on strengths or colours:
`light_colour` above 1 overdrives on purpose, `damping="1.02"`'s
precedent — and like damping's named ringing, overdrive names its
artefact: `shaded` clamps per-channel as today, so an overdriven
warm key saturates red first and hue-rotates toward yellow. The
convention the panel keeps (§6): swatches speak chromaticity in
0..1, the strengths are the level channel; components above 1 are
file-only authoring.

## 5. The section: `<look>`, whole or absent

```xml
<look light_direction="0.4 0.8 0.45" light_colour="1.0 1.0 1.0"
      light_strength="1.0" ambient_strength="0.35" ao_strength="0.5"
      glow_strength="0.35" surface_colour="0.82 0.55 0.42"
      terrain_colour="0.36 0.44 0.31"
      background_colour="0.11 0.12 0.14"
      background_top="0.15 0.17 0.21"/>
```

Written after `<metaball>`, and a separate section rather than more
attributes on it: `<metaball>` states the surface's *geometry* — its
one field, `blend`, changes the silhouette, the pick, and what the
solver's skin means — while `<look>` states shading and the palette's
base. Two records that change for different reasons, named
separately, on the terms `<terrain>` and `<doors>` established:
additive, no version bump. Absent means the shipped defaults — which
reproduce the four previously-existing authored values exactly
(light direction, the three colours), and land today's *exposure*
approximately rather than today's pixels: the bounce term is
deleted, not defaulted, so a pre-arc document reopens
exposure-similar, not pixel-identical. That is the arc's point, and
the section exists so the first author who disagrees can state
otherwise per room.

Writer policy is SETTINGS.md §3's verbatim: all or nothing, the
editor writes all, floats at repr precision, `save(..., look=None)`
default writes no section so every existing script and test writes
today's bytes and `examples/quadruped.xml` does not change. `Rig`
grows `look: Look | None = None`; the legacy ladder answers `None`
for every era that could not say otherwise. Absent semantics are the
`<metaball>` kind, not the `<solver>` kind: absent section and absent
attribute both mean the default, because every default is a plain
value — there is no `ground_y`-style "present section changes what
absence inside it means" trap to document.

Per room, like everything on the room file: `save_room` passes
`state.metaball.look`, the three open doors install `rig.look` (or a
fresh `Look()` when `None` — the file is the truth at every door,
the `adopt()` ruling; `adopt()` grows a `look` parameter beside its
solver and blend ones), and `open()`'s existing `settings_changed`
emit carries it to every panel with zero new plumbing.

## 6. The panel, and the dirty line moving

A new **Lighting** dock. `build_panels` gains the entry — and
`app.py` gains the matching name in its right-dock tab tuple, which
is the *real* second point of addition the build_panels docstring
does not mention: a dock absent from that tuple lands untabbed and
stretches the right column, the exact failure the tuple exists to
prevent. Three groups:

- **Key light**: azimuth (−180..180°, wrapping) and elevation
  (−90..90°) spins, colour swatch, strength. The *record* stays the
  vector — one truth, the uniform's spelling — and the panel derives
  az/el on refresh and writes back a unit vector on edit. The full
  domain is representable, so no display clamp exists to lie: the
  draft's ±85° clamp meant a file-authored noon light displayed as
  85° and the next unrelated nudge silently committed the clamp —
  the exact hazard `_admit`'s docstring names. At the poles azimuth
  is geometrically meaningless, so `_refresh` *skips rewriting the
  azimuth spin* when `|cos(elevation)|` falls below a small
  threshold (under the sync latch — a refresh-time retention rule,
  stated as such; `_admit` is a range-widener and cannot do this
  job), and writing back from (az, ±90°) yields `(0, ±1, 0)` for
  every azimuth, so the round-trip is stable. Below-horizon
  elevations are legal — an uplight is a look — and degrade
  gracefully: the floor shadow march heads underground, finds no
  creature, and answers full light.
- **Environment**: sky-top and horizon swatches (the Background row
  *moves here* from Appearance — it stopped being a preference and
  became the fill light's source, and `set_background_colour` moves
  with its row), terrain colour's first-ever swatch, ambient
  strength.
- **Finish**: occlusion strength, glow strength.

The four strength spins take the `_admit` widening
(`TestLoadedValuesDisplayed`'s convention): a loaded
`glow_strength="5"` displays as 5, not as the spin's cap about to be
silently committed. The colour swatches speak 0..1 (§4's
convention): a file-authored component above 1 shows as its clamped
tone, and re-picking commits what was picked — a pick is an authored
act, not a drive-by clamp.

Appearance keeps the creature-facing look: surface colour, paint,
patterns, and the view toggles. The panel note states the model in
one sentence — "the backdrop lights the creature: shadows fill with
the sky and horizon tones" — and the glow tooltip owns its edge:
glow is what separates an unlit underside from the backdrop, and
zero lets them meet.

**Every look edit is file-grade.** `<look>` saves with the room, so
the look controls call `settings_edited()` — including the surface
and background pickers that emitted raw `settings_changed` until
now; they graduate the day their fields start riding the file, the
exact line SETTINGS.md §5 drew ("split exactly along the persistence
line"). View-only survivors stay raw and clean: debug mode, max
steps, overlay, ortho, show floor. No existing test asserts colour
edits stay clean (the view-only dirty test covers debug/steps/floor
only), so the graduation is unobstructed — but
`test_colours_reach_the_document_and_the_swatches` splits along the
row move, and its background half moves panels and asserts dirty.

## 7. What this arc deliberately does not do

- **No cast shadows on creatures.** The key light shadows nothing but
  the reference floor, exactly as today. A per-hit shadow march is
  the floor's 48-step cost on every creature pixel — the plateau
  note says shading is where frame time now lives, and this arc
  already spends that budget on AO's five taps. Named residual, next
  look arc's first candidate (designed: docs/CAST_SHADOWS.md) — and
  until it lands, AO's ambient-only
  reach (§2) is the honest cap on how much this arc makes creases
  read.
- **No halo or bloom.** A miss-pixel halo keyed on the march's
  closest approach would differ *categorically* between seeded and
  unseeded frames: the cone prepass provably skips exactly the
  near-miss interval the halo would measure. A look feature that
  changes with a perf toggle is a bug by construction, so the glow
  lives on the surface (rim), where seeded and unseeded frames agree
  within the march's surface tolerance — the same drift every
  existing shading term already shows and the prepass tests already
  budget (surface-exact, never byte-exact: the seed moves where the
  march starts, and the landing `t` breathes within the growing
  epsilon).
- **No per-creature look.** `u_light_*` and friends are one uniform
  per view; a per-creature spelling in the file before the renderer
  can honour it is the format promising what the system cannot keep —
  the per-creature-blend refusal, verbatim.
- **No tone mapping or exposure system.** `shaded` clamps as today;
  an exposure knob without a tone curve is a second light_strength,
  and the clip's artefact is named in §4.
- **No march-quality persistence.** §6 of SETTINGS.md already ruled
  the epsilons and step budgets machine preferences; nothing here
  reopens that.

Cost honesty, at its true extent: the new spend is AO's five
`sd_scene` taps per hit pixel — **and a hit pixel is any scene pixel
once a terrain is loaded** (the repo's own terrain test pins the
floor catching ~every ray), so AO runs full-frame in a world, not
just on the creature's silhouette. Each tap, tiles stood down, pays
the O(G) bounds scan and folds the terrain un-culled (`sd_terrain`
has no cull of its own). Per tap that is the same order as a few
march steps; the coverage is the part worth saying out loud, and
`ao_strength = 0` removes all of it exactly. Ambient, gradient, and
glow are arithmetic on values the frame already had.

## 8. Packages

**Package 1 — the model.** `Look` + validation in `render/look.py`;
`MetaballSettings` composes it, `uniforms()` re-maps, the §4 site
list re-spells (both `Document` packing methods included; the
test-side constructor kwargs rewritten); `metaball.frag` gains the
gradient, hemispheric ambient with `AMBIENT_GROUND`, the AO ladder
with the drawn-floor fold and the exact `ao_strength == 0` skip,
skin-carrying glow, and loses `bounce`; floor fade retargets to the
gradient sample. Tests, shaded-mode content through
`MetaballViewport` with the prepass on (the armed-skin lesson):
light swung left/right flips the bright side of a sphere;
**countershade pins key-independence** — a countershaded belly's
luminance is invariant under a 180° azimuth swing of the key (and
demonstrably not under the old constants, whose bounce tracked
`-L`); `ao_strength` 1 darkens a two-chunk crease against 0, and 0
is byte-identical to a shader with the loop deleted; a foot planted
on the *drawn grid* darkens at the contact against `show_floor`
off; top ≠ bottom orders an empty frame's rows and top == bottom
restores a flat frame, under both projections; glow 0 kills the
silhouette lift, and a dark skin rims dimmer than a pale one
(`GLOW_SKIN` observed, not assumed); depth and albedo oracles
byte-untouched; `test_live_metaball_settings_reach_the_shader`
re-anchors — its "differs from flat background" surface-pixel count
saturates under a gradient, so it measures against the per-row
gradient reference (or switches to mode 2's hit mask);
`test_an_empty_body_renders_only_background` stays green untouched
(it reads mode 2's white sentinel, which the gradient never
touches). The `background_top` default is verified against the
chrome greys with the standing screenshot recipe before the value
freezes (§3's RAISED near-collision).

**Package 2 — the io.** `<look>` after `<metaball>`; `Rig.look`;
`save(..., look=None)`; `_read_look` with §4's refusals; format
docstring section; the **io↛GL firewall test** — fresh interpreter
imports `mesh_xml.io.xml_v2`, asserts `OpenGL` never entered
`sys.modules`. Tests on `test_gait_io`'s conventions: full
round-trip of a non-default look; absent on v3 and v2 reads `None`;
SchemaError on zero light vector, negative strength, negative colour
component, non-finite anything, wrong arity; `save()` without the
argument writes today's bytes; quadruped unchanged.

**Package 3 — the editor.** Lighting panel (az/el ↔ vector with the
pole-retention refresh rule, the swatches, the strengths with
`_admit` widening); the `app.py` tab-tuple entry beside the
`build_panels` one; Background row and `set_background_colour`
relocate; look edits call `settings_edited()` including the
graduated pickers; `save_room` / three open doors / `adopt()` carry
the record; falsified prose catches up, enumerated — SETTINGS.md
§5's "exactly ten panel sites" count *and* its view-only list that
names "surface/background colour", §6's look-group paragraph;
`Document.settings_edited()`'s docstring ("colours" leaves its
view-only list); `_RoomState`'s "the rest — colours, march quality —
stays session-only"; `xml_v2.py`'s header and `_read_metaball`'s
landing-zone prose (`<metaball>` stays one attribute — `<look>` took
the landing this arc, and the sentence should say so);
SCENE_BINDING.md's two session-only mentions; `theme.py`'s FIELD
comment naming the horizon field; the SurfacePanel docstring;
TEXTURES.md's countershade note gains the
ambient-agrees-with-world-up sentence. Tests: editor round-trip per
room, two rooms with different looks, dirty on every look control,
clean on view-only, a section-less open lands defaults clean, the
split colour test's background half asserts dirty.

Live verification, the standing recipe (`.claude/skills/run`): the
catch scene with the light swung low-left, AO on — the still that
shows a crease, a filled shadow, and a horizon.
