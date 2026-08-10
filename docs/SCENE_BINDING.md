# The editor scene binding: one document, re-pointed

**Status:** in progress — packages 1 (doors through open and save) and 2
(the scene on the document) landed, the null result held: the full suite
passes unchanged. 3 (rooms dock), 4 (doors section) and 5 (manifest) open.
**Arc:** the last rooms package. Packages 1–5 built the simulation side;
this is the editor's half.
**North star:** the viewport switches rooms the way Rain World's camera
does — a cut.

---

## 1. The gap, measured from both sides

The scene tier is finished and the editor cannot see it. `Scene` holds rooms
and *the* clock (aliased into every room, so one accumulator exists rather
than being agreed upon); `Scene.tick` ticks every room and then carries the
crossings; doors persist in v3 and `xml_v2.read` returns them on `Rig.doors`.
The editor, meanwhile, has zero references to `Scene` or `Door`: a file with
doors opens cleanly and silently drops them on the floor at save time, and
Play ticks exactly one room because `Document.tick` calls `Room.tick`.

Package 2 already did the hard half of the identity rework: a `Document` *is*
one `Room` wearing undo, signals and a file path, with properties
(`document.body`, `document.controllers`, `document.solver_config`,
`document.clock`) forwarding to the room's own state. Every panel and the
viewport read through those properties. That is the hinge this design turns
on.

## 2. The constraints, answered

### 2.1 Identity: one document re-pointed, not several documents

Every panel holds one document reference, taken at construction, and
subscribes to its signals once (`_Panel._follow`). The viewport does the
same, six signals deep, with GPU caches hanging off them. Several documents
behind one window means every one of those subscriptions learns to
disconnect and reconnect, and a missed disconnect is a stale write into a
background room — precisely the silently-disagreeing-UI failure the
document/command architecture was built to make impossible.

The other branch costs nothing, because it already exists. To a panel, a
room switch is indistinguishable from File > Open: the body, controllers,
solver config, terrain and selection all change at once, and
`Document._replace` already owns the emission suite for exactly that event.
Every panel demonstrably survives it — opening a file over a loaded rig is
the most-travelled path in the editor. So: **the `Document` grows a
`Scene`, its `room` attribute becomes "the active room", and switching
rooms is `_replace` generalized** — stash the outgoing room's editor state,
re-point, restore the incoming room's, emit the full suite
(`body_replaced`, `topology_changed`, `selection_changed`,
`controllers_changed`, `history_changed`, plus `settings_changed`,
`terrain_changed` and `dirty_changed`, which `_replace` itself never needed
because open resets rather than restores). No panel changes for the core
switch; the viewport's terrain re-upload falls out of its existing identity
check the first frame after the swap.

A fresh `Document` builds a scene of one room. That is the null result the
whole design is accountable to: a one-room scene must be bit-for-bit
today's editor, every existing test unchanged.

### 2.2 Undo: per-room histories, and two kinds of edit

Commands address chunks by id and controllers by stack index, resolved
against whatever `document.body` points at when they apply. Ids are
per-body: a command replayed against the wrong room would edit the wrong
creature without an error. A single history of room-tagged commands could
guard that by switching rooms before undoing — Ctrl+Z yanking the viewport
to another room — or by applying invisibly, which is the stale-UI sin with
an undo stack. **Per-room histories** have neither failure: undo undoes
what you are looking at, and a command structurally cannot reach a room it
did not come from. The undo/redo menu items re-read on switch, which the
suite's `history_changed` already causes.

That answers room *content*. Scene *structure* — add room, remove room,
rename, and door records — takes the stance terrain already holds:
`set_terrain` is deliberately not a `Command` and not a dirty-flag edit
("terrain authoring stays in Python this round"), and `Room.doors` is
authored "like terrain is authored" by its own docstring. **Rooms and doors
are file-grade edits this round**: explicit, immediate, not on any undo
stack — the same weight as File > New. This dissolves the paradox a
per-room design otherwise trips on (undoing "Remove Room" from a history
that died with the room) without inventing a third history for it.

One guard the histories need: **undo and redo disable while the active
room is abstracted.** Abstraction removes every non-anchor chunk;
a dormant command's targets are parked data, and `RemoveChunk.apply`
against a parked id raises. Ids survive the abstract/realize round trip
(`realize` re-adds chunks with their own ids), so the history wakes
intact — it just cannot run while the body is elsewhere.

### 2.3 Per-room editor state: the side table

The room stays Qt-free — that firewall has a test standing on it — so
everything the *editor* keeps per room lives in a table on the document,
keyed by the `Room` object:

```python
@dataclass
class _RoomState:
    draw: DrawPoints          # the easing springs, with their per-point tuning
    rest_pose: dict[int, tuple[float, ...]]
    selection: set[int]
    undo: list[Command]
    redo: list[Command]
    path: Path | None         # a room is a file; the scene is several
    dirty: bool
    metaball: MetaballSettings  # session-only today (v3 does not carry it)
```

`document.draw`, `document.path`, `document.dirty`, `document.selection`,
`document.metaball` and the history accessors become views of the active
room's entry, the same move the package-2 properties made for the room's own
state. Switching stashes and restores the whole record; the incoming room's
draw points `snap` to its body (the cut is a cut — easing across it would
animate a camera change). Rest poses are captured per room: at pause for
every realized room, and whenever a room realizes, so Rest Pose restores
the active room and only it. What Rest Pose deliberately cannot do is
un-cross a creature that changed rooms during play — a door is a
serialization boundary, and the pose dictionary's existing
`has_chunk` guard already makes the miss silent and honest.

The surface look rides the table because it is per-place authoring — a
swamp and a den want different blends — but its absence from v3 is a
pre-existing gap this package does not close: the look still resets with
the session, now per room instead of per window. Panel-local view state
(rig overlay, orthographic, the curve scratchpad) stays window state on
purpose; none of it addresses a room.

### 2.4 Play and the transport: one clock shown because one clock exists

Play drives the scene. `Document.tick` becomes `scene.tick()` — every room,
then the crossings — plus the one thing only a window adds: easing the
*active* room's draw points, once per tick, which is why the editor keeps
its own accumulator loop instead of calling `Scene.advance`. Step steps the
scene once, so a paused author single-stepping a crossing watches it happen.

The transport question answers itself by construction. `document.clock`
forwards to `room.clock`, and `Scene.add_room` aliases the scene's clock
into every room — so the clock the Playback panel polls *is* the scene's
clock whichever room is active, identity rather than convention. Ticks,
dropped and alpha are scene-level truths already; the readout needs no
change and could not show a per-room clock if it wanted to, because there
is no such object.

## 3. The decision the constraints did not name: the cut is not activate()

`Scene.activate` is the game's camera: realize the entered room, abstract
every other. The editor must not couple the dock's room switch to it, for a
reason sitting in `Scene._cross`: **an abstracted far side is a closed
door.** A creature walks to its exit and waits, correctly, until the
destination realizes. An editor that abstracted every background room would
therefore show the doors feature as every creature in the scene queueing in
doorways forever — the package-5 integration story failing as a matter of
policy, in the tool meant to author it.

So the editor decouples what the game couples: **switching rooms moves the
viewport only; every room stays realized and ticks at full cost.** Editor
scenes are a handful of rooms, not Rain World's hundreds — fidelity is
worth more here than the 5.5×. Abstraction stays reachable as an explicit
per-room toggle in the dock, because the cheap tier is part of what an
author needs to see and test: toggle the meadow abstract, watch its grazers
become anchors, watch a traveller wait at the den's east door until the
meadow is realized again. The toggle is the closed-door demo, on purpose,
where the automatic policy would have been the closed-door bug.

## 4. Work packages, ordered by leverage

1. **Doors through open and save (lands value alone).** `Document.open`
   carries `rig.doors` onto the room; `Document.save` passes `room.doors`.
   Closes the silent drop: a file with doors survives an editor session
   today's build would strip. Acceptance: a saved room with doors reopens
   field-for-field, and every pre-door file still opens with none.
2. **The scene on the document (the identity rework).** `Document` owns a
   `Scene` of one room; the `_RoomState` table; `set_active_room` with the
   full emission suite; `tick` through `scene.tick`; per-room dirty and
   path; histories per room with the abstraction guard. No UI. Acceptance
   is the null result — the entire editor suite passes unchanged — plus:
   panels show room B's values after a scripted switch, an edit in A is
   undoable only with A active, and A's per-point draw tuning survives a
   round trip through B.
3. **The rooms dock.** The room list (name, creature count, dirty mark,
   abstract toggle); click activates — stash, restore, `frame_all`, a cut.
   New Room, Open Into Scene… (a file becomes a new room named from its
   body, suffixed on collision), Remove Room (refused for the last one),
   rename (file-grade; warns when doors elsewhere name the old room,
   because door addressing does not chase renames). Save saves the active
   room to its path; Save All walks the dirty ones; the close prompt lists
   every dirty room. A drag in flight releases without a command when the
   room switches under it. `Scene` grows `remove_room` — scene-tier, so it
   lands with a headless test.
4. **The doors section.** In the rooms dock: the active room's doors as
   rows — name, at, to_room, to_door, radius — with add (placed at the
   selection's centroid, else the origin), edit and remove. File-grade like
   the rooms themselves; saved by package 1's plumbing. A door whose
   destination does not exist in the scene shows as closed rather than
   erroring — the scene's own stance, surfaced.
5. **The scene manifest.** A scene evaporates on close without one: room
   files persist, assembly does not. A new root tag —
   `<scene name="…"><room name="den" src="den.xml"/>…</scene>` — in a new
   `io` module; `src` relative to the manifest; the manifest's `name` wins
   (it is the addressing the doors compiled against). File > Open Scene…
   replaces the scene wholesale after the discard prompt; Save Scene writes
   the manifest and every dirty room. Rig files are untouched by this
   package, and a rig reader shown a `<scene>` refuses it by tag as it
   refuses any unknown root. Acceptance: a two-room scene with a door pair
   round-trips whole from a cold start; a missing room file fails naming
   the room, not the line number.

1 is independent and file-disjoint from 2; 3 and 4 need 2; 5 needs 3.

**The money test**, after 4: open two rooms, author a door pair in the
dock, press Play, and watch the grazer arrive in the meadow while the den
is off-screen — then save all, reopen (by hand before package 5, by
manifest after), and watch it again.

## 5. Decisions taken in advance

* **A room's name is its body's name.** The format already carries
  `body.name`; the room adopts it at open (file stem when absent) and the
  dock's rename writes both in one gesture. One value, one owner (the
  document), two write sites — not two fields drifting.
* **Switching is not editing.** Activation, abstraction toggles and dock
  selection touch no dirty flag and no history. Only commands and
  file-grade authoring dirty a room.
* **The viewport draws one room, still.** No cross-room rendering, no
  transition animation; `frame_all` on arrival because a cut that lands
  pointed at nothing reads as a broken app.
* **Editing while the simulation runs stays legal across the scene** —
  the document's founding rule, unchanged: commands act on the active
  room's body while every room's solver keeps acting too.

## 6. Not in scope

A move-creature-to-room gesture (doors are the sanctioned crossing;
authoring-time population is Open Into Scene and Duplicate). Cross-room
undo or a scene-level history. Coupling the editor to `Scene.activate` (a
"simulate like the game" mode can exist someday; it is a policy toggle,
not this rework). Seamless cameras, split views, or a second viewport.
Persisting `MetaballSettings`. Abstract-space migration — the transfer
module names it as its own future work, and the closed door remains the
honest behaviour until it lands.
