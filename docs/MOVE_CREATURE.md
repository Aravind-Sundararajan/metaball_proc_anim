# The creature move: a gesture that states a destination

**Status:** design — revised under two adversarial reviewers
(editor-binding lens, model/runtime lens) before first commit; they
materially changed it: the wholesale refusal became an editor pre-flight
(the transfer's own gate cannot see an unregistered controller), both
histories now clear on a successful move rather than standing armed, the
rest pose learned both directions (a partial fold-in for arrivals, a
source prune against id reuse), and the arrival frame became the
gesture's own act over the per-room cameras.
**Arc:** the first authoring gesture after the scene binding;
`docs/SCENE_BINDING.md` §6 named it out of scope and this picks it up.
**North star:** select the grazer, say "meadow", and be looking at the
grazer standing in the meadow — the door's mechanism at authoring speed,
with no door required.

---

## 1. The gap, measured from both sides

The core is finished and the editor cannot reach it.
`scene/transfer.py` moves a creature between rooms whole — chunks with
their masses, pins and meta, links with their rests and types, the
controllers rebuilt through the decode path — and `test_scene.py` proves
it five ways (the creature crosses whole and returns whole, an offset
arrival moves flesh and goals together, the other creature stays behind,
a transferred hunter hunts on arrival, refusals leave both rooms
untouched). The scene calls it every tick a door crossing lands. The
editor calls it never.

So the only way a creature changes rooms today is Play plus a door pair
plus a brain that wants out — the simulation's crossing, which is
correct and useless for authoring. The workaround measures the gap: to
stage the hunt split across two rooms (package 3's own acceptance
story), an author opens `hunt.xml` **twice** as two rooms and deletes
complementary creatures from each — five gestures and two lies to get
what one verb should say. Population gestures exist (Open Into Scene,
Duplicate) but both grow contents in place; nothing *moves*.

This design is the editor face of the landed mechanism, and deliberately
nothing more: no new core motion, no new persistence, no new crossing
semantics. One verb, wired to the module that already owns the journey.

## 2. The gesture surface: selection-grown, fired from the Creature dock

The PICK arc's lesson holds one noun up: gestures in this editor grow
from the selection. Every per-creature verb — colour, tempo, rename,
duplicate — resolves the selection to whole connected components through
`_selected_creatures` and acts on those, so **any chunk of a creature
arms the move for the whole creature**, and a selection touching several
creatures moves each of them. A whole-creature-only selection
requirement would be a stricter rule with no precedent and nothing to
protect: the component is derived either way, and the transfer moves
components, not chunks.

The surface is **a "Move to…" button in the Creature dock, beside
Duplicate, popping a menu of the scene's other rooms** — built fresh on
each click from `scene.rooms`, so it is never stale and needs no
subscription. A parked room appears disabled with the reason in the
door-status vocabulary ("'meadow' is parked"), a room of another
dimension likewise (the transfer's refusal, answered before it is
asked), and a scene of one room shows one disabled entry saying so. The
dock is where the creature's own verbs already live, the roster above it
is how an author aims the selection, and the popup is a Qt shape the
docks already own input-gathering with (the colour pickers gather a
colour for a document verb; this gathers a room). What the dock does
*not* own is the camera: the arrival frame leaves as a signal (§4, §6),
the RoomPanel rule scoped to what it was always about — window-owned
resources.

Rejected by name:

- **A menubar "Move to Room ▸" submenu.** The window's menus are built
  once at startup; a live room list means `aboutToShow` machinery, and
  the Edit menu is command country — undoable edits with shortcuts. This
  verb is neither (see §5), and a room list cannot hold stable
  shortcuts. The creature's vocabulary lives in the creature's dock.
- **A context menu on the selection.** No context menu exists anywhere
  in this editor — introducing the species for one verb buys a whole
  interaction idiom for a single item, and right-click in the viewport
  cannot distinguish click from pan-drag without inventing a threshold
  (PICK's own finding, unchanged).
- **A rooms-dock drop target** (drag the creature onto a room row). The
  viewport's drag vocabulary is fully claimed by chunk dragging, a drop
  target is invisible until taught, and the binding already documents
  drag-in-flight as the fragile path ("a drag in flight releases
  without a command when the room switches under it"). The heaviest Qt
  machinery for the least discoverable surface.
- **Door-linked semantics** ("send through door east"). Doors are the
  *simulation's* crossing: a brain spends a goal, the scene carries the
  arrival to the paired far door. An authoring move must not require a
  door to exist — the commonest use is populating a fresh room that has
  none — and overloading the door with an editor teleport muddies the
  one object whose honesty ("a closed door is a state, not a fault")
  the whole tier leans on. Play remains the way to watch a door work.

## 3. What moves: the transfer's unit, plus the editor's bookkeeping

The unit is `move_creature`'s own: **the connected component of each
selected seed**, which is the codebase's entire definition of a creature
("the graph's own answer to how many creatures live here"). Per
component: the chunks with name, position, radius, mass, pin state and
meta — and meta *is* the skin and the identity: colour, `pattern`,
`pattern_param`, tempo, the creature's authored name all ride there —
the intra-component links with rest, type, elasticity and render flags,
and every controller referencing the component, rebuilt through
`serialize.remap_ids` with fresh target ids. Runtime state (a plan
mid-hunt, a foothold mid-swing) resets on the way through, because a
door is a serialization boundary and this gesture is that boundary held
by hand.

**Refusal is wholesale — and the editor is what makes it so.** The
transfer's docstring promises that a creature which cannot be carried
whole is not carried at all, but its gate cannot keep that promise
alone: travel is decided by `tempo.chunk_refs`, which answers
`frozenset()` for a class the serializer does not know — the format
error is swallowed there on purpose ("membership, not authority") — so
an unregistered controller is never *travelling* and therefore never
refused. Left to the mechanism, the creature would move and the
controller would stay behind on the source stack, its references
dangling until id reuse quietly re-binds them to the wrong flesh. The
verb therefore **pre-flights the whole gesture before any motion**:
every controller on the source stack is probed for registration
(`serialize.fields_of` — an unreadable class refuses the gesture
outright, because unreadable references are unprovable ones), and every
travelling controller of every moving component is rebuilt in a dry run
through the identity remap (`serialize.remap_ids(c, {})`, the
`Room.realize` idiom), so anything the real rebuild would trip on
refuses now, while nothing has moved. One refusal refuses everything —
a three-creature gesture that fails on the third moves none of them,
because a gesture that half-happens is the mutated-but-clean lie this
section exists to forbid. The status names the creature where the
references are readable and the controller class where they are not:
the editor composes the name from the component's meta; the serializer
only ever knew the class. Only a clean pre-flight moves anything, and
then everything moves.

What the editor adds around the motion, none of it new machinery:

- **Both rooms dirty, as each component's transfer lands** (§6) — a
  room that lost a creature and a room that gained one both differ
  from disk, and a clean flag over either is a lie Save All would act
  on.
- **The target's rest pose gains the arrivals** — the mapped ids only,
  folded in at their arrival positions beside the incumbents' entries,
  never a whole-room recapture (that is `realize`'s act, for its own
  room). So Reset in the meadow treats the newcomer as having always
  stood there instead of skipping it until the next pause.
- **The source's rest pose prunes the departed ids in the same
  breath.** Chunk ids allocate at max-plus-one, so after a full
  departure the next Add reuses the low ids — and a stale pose entry
  would pass `has_chunk` and teleport the newcomer to where the
  departed creature once stood: binding wrong, not missing silently.
  A door crossing leaves that window open only until the next pause
  recaptures; a paused editor's window never closes on its own, so the
  verb closes it. One dict comprehension, the symmetry the selection
  already gets.
- **The arrivals replace the target's selection** (the union of the
  mappings' values) — a gesture's product takes the selection, and
  takes it the way Add and Duplicate take it: replacement, not
  accumulation.
- Source-side switch hygiene is already landed: door crossings during
  play remove chunks from stashed rooms today, and the switch's
  prune-and-snap covers selection and springs. The gesture leans on
  that and adds nothing there.

An authored `exit` goal survives the move unspent, and the distinction
is protocol versus mechanism, stated once: `_cross` clears the goal
because the *crossing protocol* spends what the brain used to earn its
crossing; the editor's move is mechanism only — the brain asked for
nothing, so the editor spends nothing, and blanking the field would be
silently editing an authored controller. The consequence is loud by
design — exit outranks every think ("leaving is louder than living"),
so a creature authored to want out via "east" will walk to the meadow's
"east" the moment one exists — and the status line says so on arrival:
"Moved 'grazer' to 'meadow'; 'grazer' still wants out via 'east'."

## 4. Arrival: preserved coordinates, and the cut lands on the proof

**The creature arrives at its own coordinates** — `offset=None`, the
transfer's stated honest default "while doors do not exist to say where
'here' is". A room is its own frame — that is exactly why a door
crossing carries an offset — and with no door to do the translating,
the same numbers in the new frame are the only arrival that states no
position the author did not author. Rejected by name:

- **Door-position arrival** — which door? It requires one to exist and
  re-invents `_cross` without the brain's half of the contract.
- **Frame-center drop** — a position invented from where a camera
  happened to point: the legacy editor's origin-fallback sin with a
  view matrix, and camera state is per-room *editor* state besides —
  an authored position must not depend on it.

**Terrain and floor mismatch is the solver's standing answer, not the
gesture's — and the answer is a fling, not a settle.** A penetrating
arrival is lifted by its full depth in a single tick with `prev_pos`
untouched, which the integrator reads as a velocity of depth over the
timestep: a deeply buried arrival snaps to the surface and *launches*,
decaying at the damping. And a **pinned** chunk never ejects at all —
the contact set masks on inverse mass — so a pinned anchor moved into a
cliff hangs buried until the author drags it out. The gesture softens
neither fact and does not auto-project the arrival onto the terrain:
`Terrain.project` writing arrival positions would put unauthored
numbers in authored fields (the PICK arc's no-compensation stance), and
a deep fold can project to the wrong side. What makes the sharp edges
honest is that the author is looking at them, because of the next
decision:

**The move cuts to the target room and then frames, as the gesture's
own act.** The verb ends with `set_active_room(target)`; the dock face
then asks the window for one explicit frame. The frame is the gesture's
act, not the cut's: since the per-room-cameras work, the viewport
carries the camera across every cut itself — each room's stashed pose
restored, framing only a first entry — so a bare cut would land on the
target's *last* camera, pointed wherever the author last left it and
not at the arrival. The proof of arrival is the gesture's product on
screen — the placed-chunk-takes-the-selection rule at camera scale —
and the explicit frame composes with the per-room machinery instead of
fighting it: the stash is taken on leaving, so the frame simply becomes
the pose that is stashed later. Staying in the source was considered
and rejected: the move's only visible evidence there is a creature
vanishing, which reads as deletion — "a cut that lands pointed at
nothing reads as a broken app", and a move whose proof is an absence is
worse. The cost is real (herding several creatures means switching back
per move) and accepted: repeated population is what Duplicate and doors
serve, and the dock's room list is one click.

## 5. Edit grade: file-grade, and both histories clear

**The move dirties both rooms and lands on no undo stack** — the same
weight as rooms, doors and terrain. This is not caution but structure:
per-room histories were chosen precisely so "a command structurally
cannot reach a room it did not come from", and a move is *two* rooms'
edit by definition. Parking it in the source's history means undo while
the target is active either applies invisibly (the stale-UI sin the
architecture exists to prevent) or yanks the viewport across a cut (the
other named sin); parking it in both means one gesture, two histories,
and a half-undone creature the first time either side diverges. The
two-room structure alone carries the ruling.

**A successful move clears both rooms' histories.** Not tidiness — a
defusal. The stacks on either side are armed against bodies that no
longer exist: a source undo can revert a Move Chunk into a departed id
and raise; a Remove Chunk revert can *resurrect* part of a departed
creature onto a freed id; a target redo can collide with an arrival
("id already in use"); and under max-plus-one reuse a stale command can
silently edit the wrong creature — the exact sin per-room histories
exist to make impossible (`SCENE_BINDING.md` §2.2). The precedent is
`_replace`: file-grade body surgery clears the history it invalidates,
and "to a panel this is File > Open" is the binding's own story for the
cut this gesture ends in. Refusals clear nothing: the pre-flight runs
before any motion, so a refused move leaves both histories standing.

**Abstraction refuses; it is never forced.** The transfer refuses a
parked end wholesale ("would tear a parked creature from the half of it
that is data"), and the editor surfaces that refusal rather than
working around it: a parked target is a disabled menu entry, and with
the *active* room parked the gesture disarms in the same breath as
every other edit ("Realize 'den' to edit it" — `do()`'s own sentence).
Force-realizing on the author's behalf is rejected by the binding's own
ruling: moving a room between tiers is "an explicit authoring act", and
a move that silently changes a room's simulation cost is a menu click
with a second meaning. Undo's sleep while abstracted is untouched —
nothing here writes history at all.

## 6. Core versus editor: the split is already cut

**Nothing lands in `scene/`.** `transfer.move_creature(source, target,
chunk_id, offset=None) -> dict[int, int]` is the whole mechanism,
landed and tested; `Scene` grows no `move()` because the scene's move
is `_cross` — simulation policy, brain-driven, goal-spending — and the
editor is not a second policy, it is a second *caller*, exactly as
`_cross` itself calls the module function directly. `Room` grows
nothing. Abstract-space migration stays the transfer's named future
work; this gesture neither needs it nor pretends at it.

The editor grows one verb, the `add_room`/`remove_room` precedent of
wearing the lower tier's name one level up:

```python
def move_creature(self, target: Room) -> None:
    """Move every creature the selection touches into ``target``.

    File-grade: dirties both rooms, clears both histories, refuses
    whole with the pre-flight's reasons. Ends with the cut to
    ``target``, the arrivals selected and in its rest pose.
    """
```

In prose: the first line is the landed room-verb guard — `if target not
in self._states: raise ValueError(...)` — because a foreign `Room` must
die before it can half-apply. Then the refusals, each a status line and
no motion: an empty selection, a parked end (either), a dimension
mismatch (the transfer's refusal, restated before any motion), and the
§3 pre-flight over the stack and the travelling controllers. Then the motion: per component,
`transfer.move_creature(self.room, target, min(component))` — `min` for
determinism; the transfer re-derives the component from any member —
**dirtying both rooms as each transfer lands**, not once after the
loop, so even a failure the pre-flight failed to imagine leaves
mutated rooms already marked. Then the bookkeeping of §3 — rest poses
both directions, the selection replaced, both histories cleared — then
`set_active_room(target)`, `scene_changed`, and the status.

The dock face calls the verb and emits an `activated`-style signal (the
RoomPanel pattern); the window connects it to the explicit frame of §4
— one connect in `app.py`. That is the whole window-side churn, and it
is the reconciliation with the per-room cameras in one sentence: the
viewport owns carrying cameras across cuts, the gesture owns declaring
that *this* cut's pose is "look at the arrival", and it declares it
through the window because the camera is the window's resource, not the
dock's. Persistence cost: zero — the move writes no XML; two dirtied
room files carry it through the save paths that exist.

## 7. Settings friction: the room's records win, on purpose

Solver settings and the look are per-room records, so **a moved
creature changes physics and lighting the moment it arrives** — a gait
tuned to read as bone at `iterations=8` arrives in a room that runs 4,
a countershaded skin walks out of the den's key into the swamp's. This
design keeps that and says so, because it is the rooms arc's point, not
its leak: per-room records are per-place authoring ("a swamp and a den
want different blends"), and a creature is a visitor. Carrying solver
or look fragments with the creature would dissolve what a room *is*
and re-open the one-value-two-owners drift the settings arc closed.
The same truth already holds at every door crossing during play; the
gesture invents nothing, and an author who wants the den's physics in
the meadow has the per-room panels to say so — per room, where that
statement lives.

One adjacent truth, distinct from the look: an *unpainted* creature's
fallback hue is its blend-group index, and groups number in first-seen
component order — so an unpainted creature can arrive wearing a
different auto-colour than it left with. Paint is meta and travels;
the fallback is a room-local ordinal and never did.

## 8. Edges, answered

- **Moving the room's only creature** — allowed, an empty room is a
  legal room (New Room creates one; an empty room even abstracts,
  honestly, to nothing). The roster's empty-state note already says
  what an emptied room shows.
- **Moving into the active room** — unrepresentable rather than
  refused: the menu lists only *other* rooms, and the transfer's
  source-is-target refusal stands behind the UI as the belt behind the
  suspenders.
- **Rooms of different dimension** — the transfer refuses; the menu
  disables such a room up front with the reason, the parked-room
  treatment for the same kind of impossibility.
- **Name collisions** — none to handle, by two landed rulings: chunk
  identity is "a stable integer id, never a name" (and ids are the
  transfer's to remap), and creature names are meta, not addressing —
  the roster already tolerates duplicates and `None`. Two "grazer"s in
  one meadow are two roster rows, exactly as Duplicate makes today.
- **A controller straddling two creatures moved in one gesture** —
  it travels with the first component asked about, and its references
  into the second stay old source ids forever; those numbers can
  coincide with ids living in the target and *mis-bind*, not miss.
  A straddling controller "was already a design smell" — the
  transfer's own words — and this gesture inherits the caveat naming
  its worst case rather than promising `has_chunk` will always save
  it.
- **Selection and stash after** — arrivals selected in the target;
  the source's stashed selection holds dead ids until its next
  activation prunes them, the landed hygiene door crossings already
  exercise every play session. The rest poses do not wait (§3).
- **During play** — legal, the founding rule: "editing while the
  simulation runs stays legal across the scene". The gesture runs
  between ticks in the event loop; the transfer is atomic from the
  scene's point of view.

## 9. Tests and churn

Core: **no new tests** — the mechanism's five proofs in
`tests/test_scene.py` are the mechanism's proofs; this design adds no
core motion to test.

Editor, app-less beside the binding's own tests
(`tests/test_scene_binding.py` builds bare `Document`s — no
QApplication, no GL): moved creature gone from the source and whole in
the target (chunks, links, controller counts); one selected chunk moves
its whole component and the other creature stays; both rooms dirty and
**both histories cleared** on success; the active room is the target
with the arrivals as the *replaced* selection, at preserved
coordinates; the target's rest pose gains exactly the mapped arrivals
with every incumbent entry untouched, and the source's rest pose no
longer holds the departed ids — proven the unkind way: move everything
out, Add a chunk (which reuses id 0), Reset, and assert the newcomer
did not teleport. Refusals: with the active room parked, with the
target parked, on a dimension mismatch, and with an unregistered
controller on the stack — the last being the test the transfer alone
cannot pass, which is the pre-flight's whole reason — and after every
refusal both rooms are bit-identical, still clean, and both histories
still standing. The only creature moves and leaves a legal empty room.

Dock face, offscreen behind the suite's existing Qt-platform guard
(`tests/test_panels.py`): the button disarms with an empty selection;
the popup lists exactly the other rooms, parked and wrong-dimension
ones disabled; a chosen room moves for real (asserted on the document,
not the mock) and the activated-style signal fires once.

**The money test:** open the den and the meadow, select one chunk of
the grazer, Move to… meadow — the viewport is looking at the grazer in
the meadow, both dock counts agree, `step_once` ticks it where it
stands. Save All, reopen the scene: the move is in the files.

Churn inventory: `mesh_xml/editor/document.py` (one import, one method
— pre-flight included, ~60 lines), `mesh_xml/editor/panels.py`
(CreaturePanel: one button, one popup, one signal, one handler),
`mesh_xml/editor/app.py` (one connect, signal to frame),
`tests/test_scene_binding.py`, `tests/test_panels.py`, this document,
and one cross-reference line in `docs/SCENE_BINDING.md` §6. `core/`,
`scene/`, `io/`, `render/`: untouched.

## 10. Deliberately not done

- **Copy-to-room.** Duplicate then Move composes it in two honest
  steps; a fused gesture is a third grammar for zero new capability.
- **Cross-file and cross-scene moves.** The scene is the addressing
  boundary; Open Into Scene is the import gesture that brings a file
  *inside* it first.
- **An arrival placement picker** (click where it lands). Placing in a
  room you can see means rendering two rooms; the viewport draws one,
  by the binding's oldest ruling. Move, then drag — the tools exist.
- **Per-creature arrival layout for multi-creature moves.** Every
  touched creature arrives at its own coordinates; arranging a herd is
  authoring, not arrival policy.
- **Auto-authoring doors, or touching `exit` goals.** The move is not
  a door and does not pretend; doors stay the simulation's crossing
  and the dock's to author.
- **Abstract-space migration.** Still the transfer's named future
  work; until it lands, parked ends refuse and the closed door stays
  the honest behaviour.
