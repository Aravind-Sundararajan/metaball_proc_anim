# Textures: a skin that varies, on a surface with no UVs

**Status:** landed — all four packages. Colour parity holds against the
oracle through debug mode 4 with every pattern in frame; an unpainted body
packs an empty attribute table and renders — and exports — byte-for-byte
as before; a painted, patterned creature round-trips through v3 and
arrives in Blender wearing its skin as OBJ vertex colours. One vocabulary
note against §3 as designed: the four named patterns landed as two
mechanisms — on a limb, rings around the axis *are* stripes along it, and
a gradient is per-chunk colours doing their ordinary job.
**Provenance:** user-requested (2026-08-07), the first arc after rooms.
**North star:** Rain World again — creatures read as individuals, and after
shape and gait the next carrier of individuality is markings: banding,
countershading, a bright head on a dull body. Today every creature is one
flat colour; this arc makes the skin vary across one creature without ever
inventing a UV.

---

## 1. Where the stack already is

* **The colour store is already per chunk.** ``sdf.COLOUR_META`` rides chunk
  meta and round-trips through v3. Only the *gesture* is per creature:
  ``Document.set_creature_colour`` paints every chunk of the component, and
  ``group_table`` reads the first parseable value and colours the whole
  group with it. The data model for a varied skin exists; the surface just
  never looks past the first chunk.
* **The surface is a fold, and the fold already computes the blend.** The
  polynomial smooth minimum's ``h`` is a convex weight — the module
  docstrings lean on exactly that for the Lipschitz argument — and its
  gradient identity ``h·∇a + (1−h)·∇b`` is the statement that ``h`` *is*
  "how much of each primitive is here." Nothing reads ``h`` out today; the
  fold returns the distance and drops the weight on the floor.
* **Colour is applied flat at the hit.** ``metaball.frag`` asks
  ``sd_nearest_group`` which creature won and looks up one RGB
  (``sd_group_colour``). The march itself never touches colour — which is
  the property to preserve, because every acceleration structure (bound
  cull, tiles, cone prepass) carries an exactness contract about the march.
* **The oracle stops at geometry.** ``field.py`` twins the distance, normal
  and march; no CPU function computes an albedo, so today's colour logic is
  the one shaded thing the parity harness cannot check numerically.
* **A sixth texture unit is free.** Units 0–4 carry primitives, terrain,
  groups, bounds and tiles, all on the same stride-8 ``RGBA32F`` machinery;
  a per-primitive attribute table is one more rider.

## 2. The design, in the surface's vocabulary

**There are no UVs and there will be none.** A metaball surface is the zero
set of a fold; it has no intrinsic parameterization, and any imposed one
tears where blobs merge. So attributes live where the geometry lives — on
the primitives — and reach the surface the way distance does: through the
fold.

**The weighted skin.** Give each packed primitive an albedo, and fold
colour beside distance using the *same* ``h`` the smooth minimum already
computes: seed with the first primitive's colour, and at each step
``colour = mix(colour_i, colour_so_far, h)``. Every surface point then
wears a convex blend of the primitives that shaped it — continuous
everywhere the field is, sharp at ``k = 0`` where ``h`` collapses to a
step, and inheriting the array-order contract the distance fold already
declares. Crucially this runs **once per pixel at the hit**, next to
``sd_nearest_group``, never inside the march: the march stays untouched
and every cull's exactness claim survives by construction.

**Within a primitive, colour interpolates end to end.** A round cone spans
two chunks; its albedo record carries both chunks' colours, lerped by the
*orthogonal projection fraction* ``clamp(dot(p−a, b−a) / |b−a|², 0, 1)`` —
deliberately not the cone maths' own three-case parameter, because the
projection is smooth, branchless, cheap to mirror in numpy, and equal to
the exact parameter wherever they could disagree visibly. A chain painted
chunk by chunk grades smoothly down its whole length; a degenerate sphere
record reads its ``a`` end.

**The link's own segment is the only stable frame — so the pattern
vocabulary is axisymmetric.** Chunks are points and carry no orientation;
world-anchored noise would swim through a walking body (the reason this
arc was scoped away from naive solid noise from the start). But a link's
endpoints define an axial coordinate ``t`` and a radial distance, both
derived per frame from the segment itself, so patterns built from
``(t, radial)`` ride the limb through translation *and* rotation with no
frame to invent. Stripes, rings, gradients, and a countershade (mixing by
the radial offset's world-``y``, which keeps a belly a belly) all live in
that vocabulary. Since the lighting arc, the shaded frame's ambient keys
on the same world-up (docs/LIGHTING.md §2), so the pattern and the fill
agree about which way the world is — the pale belly reads as pigment
against quiet ground light instead of being cancelled twice. What does
not live in the vocabulary: azimuthal detail (no stable perpendicular
reference on a link) and true 3-D noise (needs the frame that does not
exist). That is a real ceiling and the design names it rather than
wallpapering it.

**The oracle grows an albedo, and the shader grows a debug mode to be
checked by.** ``field.albedo(points, primitives, attributes, k, groups)``
twins the weighted fold in float64; debug mode 4 writes the raw unlit
albedo, so the parity test compares colour numerically through the same
readback the depth modes use, with lighting never in the frame. The
pairing invariant gains one entry: the weight-returning smooth minimum
lands in ``core/distance.py`` and ``sdf.glsl`` in the same commit, stated
in both files.

## 3. Work packages, ordered by leverage

1. **The weighted skin (mechanism, lands invisible).** The
   weight-returning smooth minimum beside the plain one in
   ``core/distance.py`` and ``sdf.glsl``; ``sdf.pack_attributes`` building
   the ``(P, 8)`` per-end colour table from chunk meta (falling back to
   the group's colour for unpainted chunks); the attribute texture on
   unit 5; ``sd_group_albedo`` at the hit; ``field.albedo``; debug mode 4;
   the parity test. Acceptance is the null result plus one new power: a
   document whose every chunk agrees on colour renders **bit-identically**
   to today (all ends equal makes every mix an identity), and a two-colour
   chain hand-packed in the test shows a smooth gradient with CPU and GPU
   agreeing within readback quantization.
2. **Paint the selection (the gesture).** ``Document.set_chunk_colour`` —
   the existing creature gesture scoped to the selected chunks instead of
   their whole components; one undo step of ``SetMeta`` commands; a
   second button in the Surface panel beside the creature-wide one.
   Persistence is already free: meta rides v3, ``duplicate_creature``
   copies it, transfer carries it through doors. Acceptance: paint the
   worm's head, save, reopen — still red-headed, and it swims with it.
3. **The pattern vocabulary.** Pattern id, second colour and one parameter
   as creature-painted meta (the tempo mechanism: painted across the
   component, one undo step); stripes, rings, gradient and countershade
   evaluated in the albedo fold from ``(t, radial, radial-up)``; the
   Surface panel grows the combo, the second swatch and the parameter
   spin. Every pattern is added to the parity test the day it is added to
   the shader — per pattern, CPU and GPU stay within the same tolerance
   the flat skin holds. Acceptance: a striped grazer keeps its stripes
   mid-stride (the pattern rides the limb frames, measured as near-zero
   pattern drift over a gait cycle at a marked surface point).
4. **Painted exports.** ``polygonise`` interpolates the albedo fold at its
   vertices; ``write_obj`` emits the widely-read ``v x y z r g b``
   extension (STL stays geometry — the format has no honest colour).
   Acceptance: a two-tone creature arrives in Blender two-toned.

1 is the arc's spine; 2 and 3 both need it and are file-disjoint from each
other; 4 needs 1 and reads whatever 2–3 authored.

## 4. Decisions taken in advance

* **The march never reads attributes.** Albedo folds once at the hit.
  Every existing exactness contract — bound cull byte-exact, tiles
  surface-exact, prepass seeded — survives untouched because the thing
  they protect never changes.
* **One ``h``, two folds.** The colour fold reuses the distance fold's
  weight rather than computing its own notion of nearness, so the skin
  can never disagree with the flesh about where a blend is.
* **Unpainted means unchanged.** Every package's first acceptance is that
  a document not using the new power renders as before — bit-identical
  where the maths allows it, within readback tolerance where it is the
  shaded path.
* **Attributes are meta, additive, no version bump.** Colour stays under
  ``COLOUR_META``; patterns take sibling keys. Every older file reads;
  every older reader ignores.
* **The editor's paint gestures are commands** (undoable ``SetMeta``,
  like colour and tempo today) — this arc adds no file-grade edits.

## 5. Not in scope

UV mapping or image textures (no parameterization exists to map onto).
Azimuthal or noise-based patterns (no stable frame; revisit only if
links ever carry an authored twist reference). Lighting, AO, glow, and
the background gradient (a separate backlog item about light, not skin).
Per-link meta (links stay records; ends inherit from chunks). Texturing
the terrain (the world keeps its one colour this arc).
