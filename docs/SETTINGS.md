# Settings persistence: the file states its physics and its blend

**Status:** designed — no packages landed.
**Arc:** first half of the authoring-completeness branch; SDF-pick chunk
placement is the second half and gets its own design.
**North star:** a saved creature reopens exactly as tuned, and a file can
ship its tuning — nobody verifies an example "walkable at defaults" again.

---

## 1. The gap, measured

v3 grew a section every time a save was caught losing something an author
made: `<controllers>` because a save kept the skeleton and lost the gait,
`<terrain>` because it kept the creature and lost the world, `<doors>`
because it kept the room and lost its exits, chunk `<meta>` because the
first v3 save dropped what the legacy importer had carefully kept. Skins
ride that meta today — paint, `pattern`, `pattern_param`, all per-chunk.

Two records still don't ride, and they are the two that decide how the
body moves and how it wears that skin:

- **`SolverConfig`** (`core/solver.py`) — gravity, damping, iterations,
  max_velocity, ground_y, ground_friction, self_contact,
  self_contact_elasticity. Owned by the `Room` ("the solver settings that
  are its physics" — `scene/room.py`), edited live by the sim panel,
  reset to `SolverConfig(ground_y=0.0)` on every open. A spine tuned to
  read as bone at `iterations=8` reopens as rubber at 4. The quadruped
  example had to be verified walkable at defaults precisely because a
  tuned config had nowhere to be.
- **`MetaballSettings.blend`** (`render/gl/metaball.py`) — the smooth-min
  radius. Its own docstring has said all along that it is "per-body
  authoring data, not a global constant", and the interaction it warns
  about (a linked chunk reads ~`blend/4` fatter than its radius) means
  the silhouette a skin was painted against *is* a function of `blend`.
  Skins persist; the blend they were authored against snaps back to 0.25
  on open. `_RoomState`'s docstring calls the surface look "session-only
  — v3 does not carry it". This arc retires that sentence.

Everything per-chunk found a home in chunk meta. These two are per-file
records with no element to hang on. They get the missing nouns: two
additive sections, on exactly the terms `<terrain>` and `<doors>`
established — no version bump, absent means what every older document
already meant.

## 2. Where the records ride, and where they don't

**The room file.** The room is the unit of simulation and one rig file is
one room; its physics belongs beside its terrain and its doors. The scene
manifest is "deliberately nothing more" than assembly (`io/scene_xml.py`)
and stays settings-free: two scenes listing one room file agree about its
physics because the file states it.

Rejected homes, so they stay rejected:

- **Chunk meta, the way `pattern` rides.** `blend` is one value per body.
  Stamping it on every chunk invents a disagreement channel — two chunks
  claiming different blends is a document that argues with itself.
- **Attributes on the root `<body>`.** Ten attributes flattened onto the
  root is a section without the courtesy of a name, and the next arc
  (lighting) would keep flattening. Sections are the house pattern for "a
  record beside the body".
- **The manifest.** Per-room facts live in room files — the doors ruling.

## 3. The sections

Written after `<doors>`, in this order:

```xml
<solver gravity="0.0 -9.81 0.0" damping="0.98" iterations="4"
        max_velocity="100.0" ground_y="0.0" ground_friction="0.35"
        self_contact="false" self_contact_elasticity="0.5"/>
<metaball blend="0.25"/>
```

**`<solver>` carries every `SolverConfig` field except `terrain`.** The
world already has its own section; a config that also carried it would be
two spellings of one world, and the rarely-exercised spelling rots first
— the flat-layout ruling. `_read_solver` returns a config with
`terrain=None` and the caller folds `Rig.terrain` into whichever config
it installs, which is the fold `open_scene` already performs today.

**`<metaball>` carries `blend` and only `blend` this round.** The
section is named for the record so the lighting arc has its landing zone,
but it takes the number, not the class: `MetaballSettings` lives beside
the OpenGL imports and `io` does not follow it there (that import is the
exact one that makes a GL-less machine unable to even *collect* the
editor tests). `save(..., blend: float | None)`, `Rig.blend: float |
None`. Which subset of `MetaballSettings` persists is therefore spelled
where the split lives anyway — in the editor, which owns the view-only
half (§6).

**Writer policy: all or nothing, and the editor writes all.** A section,
when written, is written in full — every attribute explicit, so the file
states its physics completely and a default that later changes cannot
reshape a saved creature. `save()` defaults both new parameters to
`None` = no section, so every existing script and test writes today's
bytes; `examples/quadruped.xml` does not change, and defaults remain the
truth about it. Floats at full `float64` precision (`repr`, not `_fmt`),
the terrain section's reason: these numbers are inputs to every tick, and
a float32 shortening would change the simulation a little on every save.

**Absent semantics, stated honestly.** For `<metaball>`, absent section
and absent attribute mean the same thing — the dataclass default, which
is what every pre-arc document ran under. For `<solver>` they do **not**
coincide, and the format docstring must say so: an absent *section* means
the config the caller already runs — for a room, floor at zero, the
room's founding opinion — while a present section means the document
states its physics entirely, floor included. `ground_y` absent inside a
present `<solver>` is therefore *no floor*, the dataclass truth, because
the alternative is a magic string for `None` and a section that cannot
say "this world has no ground". The editor always writes the record in
full, so the partial-section path belongs only to hand-authors, and the
docstring warns them in one sentence.

## 4. Reading

`Rig` grows `solver: SolverConfig | None = None` and `blend: float |
None = None` — `None` for every document without the section, "which is
all of them written before it existed". v2 documents read exactly as
they always have. The legacy ladder in `Document._read_rig` returns
`(None, None)` for eras that could not describe either.

Validation, by the house rule — refuse what was validated at authoring
time or would break downstream, leave taste to the author:

- `blend` through `core.distance.as_blend`, the validator terrain's
  blend already answers to: finite, non-negative, same error vocabulary.
- `gravity` must have `dim` components, the chunk-`pos` rule; a 3-vector
  falling into a dim-2 body is a broadcast error three layers later.
- `iterations` ≥ 0; a negative count is not a tuning, it is a typo.
- Every float finite — `as_blend`'s spirit; a NaN damping poisons every
  position in the room by the second tick.
- No range policing beyond that: `damping="1.02"` rings, `blend="0"`
  reads as beads, and both are authoring choices the panels also permit.

## 5. The editor half

**Save.** `Document.save_room` passes `solver=room.solver_config` and
`blend=state.metaball.blend` — per room, active or not, so Save All and
`save_scene_as` carry every room's tuning without new plumbing.

**Open, all three doors.** `open`, `open_as_room`, and `open_scene`
install what `_read_rig` now also returns: `rig.solver` (falling back to
the room's `SolverConfig(ground_y=0.0)` when `None`), then the terrain
fold, then `state.metaball.blend = rig.blend` when present. `open()`
must now also emit `settings_changed` — `_replace` never needed it
because open used to *reset* settings rather than restore them, and that
reasoning (SCENE_BINDING §2.1) just changed for the persisted fields.
The panels already pull on that signal for room switches, so loaded
values reach every spinbox with zero panel changes.

**Dirty, split exactly along the persistence line.** Panels mutate
settings live and emit `settings_changed` themselves. The persisted
fields become file-grade edits on the doors precedent — an explicit
authoring act, no undo stack, but they DO dirty, because they save with
the room's file and a clean flag over an unsaved tuning is a lie Save
All would act on. Mechanically: `Document` grows one method,
`settings_edited()` — dirty the active room, then emit — and exactly
eight panel sites call it instead of emitting raw: `_write_blend` and
the seven solver mutations (iterations, damping, max_velocity,
ground_friction, self_contact, self_contact_elasticity, gravity — both
the vector edit and the zero-G toggle). Every view-only site
(debug_mode, max_steps, show_floor, surface/background colour, overlay,
ortho) keeps emitting raw and keeps the file clean, which is the
document/view split of §6 enforced where edits happen.

**New rooms and New File** keep today's defaults and start clean; the
sections appear at first save.

## 6. Deliberately not persisted

- **March quality and view state**: `max_distance`, `max_steps`, the
  epsilons, `step_relaxation`, `debug_mode`, `show_floor`. Machine and
  session preferences; a file must not reopen slow because it was saved
  on a fast machine, and `debug_mode`'s own panel comment already rules
  it "no business on the document".
- **The look group**: `light_direction`, `surface_colour`,
  `terrain_colour`, `background_colour`. This is the lighting arc's
  design space, and persisting a flat `background_colour` today creates
  a compat obligation for tomorrow's gradient, designed blind. The
  lighting arc extends `<metaball>` or adds its own section on the same
  additive terms, with the actual fields in hand.
- **`max_velocity` rides, despite being "a backstop, not a tuning
  knob"**: the line is *whether a field changes the tick's arithmetic*,
  not whether changing it is wise — two machines opening one file must
  simulate identically, and only `terrain` is excluded, because it
  already rides under its own name.
- **The manifest** stays assembly-only; **no undo** for settings edits
  (doors precedent); **no per-creature blend** within a room — `u_blend`
  is one uniform per view, and inventing per-creature blend in the file
  before the renderer can honour it would be the format promising what
  the system cannot keep.

## 7. Packages

**Package 1 — the format.** `Rig.solver`/`Rig.blend`; `save(...,
solver=None, blend=None)` writing full records after `<doors>`;
`_read_solver`/`_read_metaball` with §4's validation; the two docstring
sections in the `xml_v2` header. Tests (`test_gait_io` conventions):
full round-trip of a non-default config and blend; absent sections read
as `(None, None)` on v3 *and* v2; partial `<solver>` yields dataclass
defaults including `ground_y=None`; `<metaball>` alone; SchemaError on
gravity arity, negative iterations, non-finite floats, negative blend;
`save()` without the new arguments writes no sections
(`examples/quadruped.xml` reads unchanged: `solver is None`).

**Package 2 — the editor binding.** `save_room` passes both; the three
open paths install both (solver fallback, terrain fold, blend); `open()`
emits `settings_changed`; `Document.settings_edited()`; the eight panel
sites; `_RoomState` docstring drops "session-only — v3 does not carry
it" for `blend` specifically. Tests: editor save→open round-trips a
tuned config and blend (through `Document`, not just `io`); a two-room
scene round-trips *different* settings per room; blend and solver edits
dirty the room and title, view-only edits don't; open of a
section-less file lands today's defaults with a clean flag; Save All
carries an inactive room's tuning.
