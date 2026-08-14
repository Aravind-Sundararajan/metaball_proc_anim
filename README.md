# mesh_xml

A procedural animation and metaball framework for soft, jointed creatures, modelled on
Rain World's approach: a body is a handful of masses joined by distance constraints, the
surface is a smooth field implied by them rather than a mesh anyone authored, and a
creature is something that thinks, walks, and exists whether or not anyone is looking.

![The editor: a quadruped mid-stride under an authored key light, the Lighting dock open](docs/editor.png)

## Quick start

```bash
poetry install
poetry run mesh-xml
```

Opens the editor on `examples/chubby_worm.xml`, or on any path given as the first
argument — a rig, or a `<scene>` manifest of several rooms. **Space** plays and pauses,
**F** frames the creature, **A** grows a chunk out of the selection, **R** toggles the
rig overlay, **O** goes orthographic, **1/3/7** snap the view, click selects, drag moves.

For a creature that walks: **File → Open Example → quadruped**, press Space. To author
one yourself, build a spine at standing height, grow three-chunk legs from it (hip,
knee, foot — ending near the floor), and press **Rig Legs** in the toolbar; the recipe
it compiles is the measured self-propelled walker, and the creature walks the way its
spine points, on footfalls against friction, with no pins and no kinematic carry.

![The quadruped walking: footfalls against friction, no pins, no kinematic carry](docs/walk.gif)

*The same clip, smaller and unquantized: [walk.webm](docs/walk.webm).*

## What's here

* **A soft-body core** — chunks and typed links under a fixed-timestep Verlet solver
  with momentum-conserving projection, position-level Coulomb friction, angle limits
  compiled to brace links, opt-in self-contact, and SDF terrain
  ([docs/CONTACT_MODEL.md](docs/CONTACT_MODEL.md), [docs/TERRAIN.md](docs/TERRAIN.md)).
* **A metaball surface** — round cones along the links, folded by smooth minimum within
  a creature and hard minimum between them, sphere-traced in a GL 3.3 raymarcher that a
  float64 CPU oracle checks numerically. Bound culling, tiled broad phase and a cone
  prepass keep herds cheap; every optimisation carries a stated exactness contract.
* **Procedural gaits** — spines that follow, two-bone IK, foothold state machines,
  posture and balance, an autorig that fits them to a shape, and `Rig Legs`, which
  compiles a measured walker onto any leg-shaped appendage.
* **Brains** — a behavior tier above the controllers, obeying the same law one level
  up (brains write goals, never forces): a state-machine grazer, a planning hunter, a
  utility scavenger. `examples/scavenge.xml` is a three-species food chain in one file.
* **Rooms and scenes** — `Room` as the unit of simulation, scenes of many rooms on one
  clock, doors that carry creatures between them, abstraction that drops an unwatched
  room to anchors-and-brains, and a scene manifest so the assembly survives a cold
  start ([docs/ROOMS.md](docs/ROOMS.md), [docs/SCENE_BINDING.md](docs/SCENE_BINDING.md)).
* **Skins** — per-chunk colour folded beside the distance field by the same blend
  weight, stripe and countershade patterns riding each link's own frame, painted from
  the editor and saved with the rig ([docs/TEXTURES.md](docs/TEXTURES.md)).
* **An authored look** — one key light on azimuth/elevation knobs, a background
  gradient that doubles as the ambient fill (shadows cool toward the sky and horizon
  tones instead of tracking the key), SDF occlusion in the creases, a rim the skin
  carries, and the whole record saved per room as `<look>`
  ([docs/LIGHTING.md](docs/LIGHTING.md)).
* **Exports** — watertight OBJ/STL by surface nets over the same field, with vertex
  colours carrying the skin; a painted herd arrives in Blender as separate, painted
  animals.
* **The editor** — document/command architecture with full undo, dockable panels, a
  rooms dock, paint and pattern gestures, a toolbar of the core verbs, and a dark
  theme sourced from the viewport's own palette ([docs/CHROME.md](docs/CHROME.md)).

![The key light swinging a full turn around a frozen stride: the shadow follows the light while the gradient holds the fill](docs/keylight.gif)

*The same clip, smaller and unquantized: [keylight.webm](docs/keylight.webm).*

## Architecture

```
mesh_xml/
  core/     body.py solver.py curve.py distance.py terrain.py   pure math — no Qt, no GL
  io/       xml_v2.py xml_legacy.py scene_xml.py mesh_export.py serialization, versioned
  anim/     clock.py controller.py autorig.py behavior.py       fixed-timestep driving
    controllers/  spine, limb IK, footholds, balance, drive     force, never position
  scene/    room.py scene.py door.py transfer.py                simulation without a window
  render/   draw_points.py sdf.py field.py tiles.py meshing.py  surface + visual smoothing
    gl/     context.py camera.py metaball.py shaders/           core-profile 3.3
  editor/   document.py viewport.py panels.py theme.py app.py   Qt, commands, undo
```

The dependency rule is one-directional: `core` imports nothing from the others, `scene`
imports no Qt (a test proves it), and only `render/gl/` may import OpenGL.

### The body is a graph, not a hierarchy

A `Body` holds chunks — position, radius, inverse mass — joined by undirected distance
constraints. There is no parent transform at simulation time, exactly as Rain World's
`PhysicalObject` has none. Cycles are legal, because the topologies worth building (a hip
girdle, a ribcage, a palm) are cyclic. Hierarchy is derived on demand when something needs
a tree, and the derivation is total: non-tree edges are reported, never dropped.

Chunk identity is a stable integer id. Names are an editable field and may repeat.

### Links do the work

A `Link` carries a rest length and a type — `NORMAL` holds distance in both directions,
`PULL` resists only stretching, `PUSH` only compression. That distinction is what separates
a spine from a rope, and it is Rain World's `BodyChunkConnection` in all but name. Angle
limits compile to exactly these links, and a `render` flag lets a mechanism brace carry
force without carrying flesh.

### Draw positions are not chunk positions

`render/draw_points.py` keeps a second set of points that chase the chunks through damped
springs, arriving late and overshooting. That lag is the secondary motion. Rain World's
bodies read as gelatinous because of this layer, not because the physics is soft.

### Controllers contribute force, never positions — and brains write goals

Constraint projection runs immediately after integration and will undo anything a
controller places. So controllers state where they want to go — via `ControlContext.seek` —
and the solver decides what the body can do about it. One level up, a brain writes goals
(destinations, choices) and steers only its own pinned anchor; `ctx.peers` is the room's
population, and a creature in another room is unsensed by construction.

### The surface is built from links

Every SDF primitive is a round cone — a capsule with a different radius at each end — so
radius varies continuously along a chain. A chunk contributes a primitive only when nothing
links it, since a cone already contains its end spheres. The skin rides the same fold: the
smooth minimum's own blend weight mixes per-chunk colour, so the paint can never disagree
with the flesh about where a blend is.

![A procedurally animated worm rendered as metaballs](docs/metaballs.png)

## The design docs

Each arc lands with a design document that records its decisions, measurements, and the
things implementation taught the design. They are the honest history of the project:

| Doc | What it holds |
| --- | --- |
| [REFACTOR_PLAN.md](docs/REFACTOR_PLAN.md) | The original diagnosis, the phase plan, and the running ledger of everything landed since |
| [CONTACT_MODEL.md](docs/CONTACT_MODEL.md) | Coulomb friction and angle limits — the two blockers that gated terrain — plus self-contact |
| [TERRAIN.md](docs/TERRAIN.md) | The world as a second field: collision, footholds on slopes, the slope-μ law |
| [ROOMS.md](docs/ROOMS.md) | Rooms, scenes, transfer, abstraction, doors — the world model |
| [SCENE_BINDING.md](docs/SCENE_BINDING.md) | The editor learning the plural: one document re-pointed across rooms |
| [TEXTURES.md](docs/TEXTURES.md) | The weighted skin, paint gestures, patterns, painted exports |
| [CHROME.md](docs/CHROME.md) | The editor's dress: a theme sourced from the viewport's palette, the toolbar |
| [SETTINGS.md](docs/SETTINGS.md) | The file states its physics: solver and blend persistence, and the dirty line between file-grade and view-only |
| [PICK.md](docs/PICK.md) | Ctrl+click placement on the rendered surface: the ray, the trace, and the one Add rule |
| [SELF_CONTACT.md](docs/SELF_CONTACT.md) | Capsule self-contact and self-friction: mid-spans are real, end zones belong to the spheres |
| [LIGHTING.md](docs/LIGHTING.md) | The environment becomes the fill: the key, the gradient, occlusion, glow, and the `<look>` record |

## Development

```bash
poetry run pytest -q
```

```bash
poetry run ruff check .
```

`examples/` holds authored rigs, every one a v3 document that carries its own gait —
open it, press Play, it moves. `tests/data/` holds fixtures, including one deliberately
kept in the pre-versioning coordinate format as a regression case for the legacy reader.
The retired original application is recoverable at the baseline commit:
`git show ec05cdf:main.py`.
