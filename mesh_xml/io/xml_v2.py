"""The on-disk body format: flat, world-space, explicitly versioned, v3 current.

The module keeps the name of the era it introduced; the version that matters is
the one stamped on the root, not the one in the filename. :func:`save` writes v3.
:func:`load` reads v3 and v2, and refuses everything else by name.

Why the version stamp
---------------------
The v1 ``<armature>`` format changed the meaning of ``position`` from world space
to parent-relative without leaving a single byte in the file to say so (refactor
plan §2.3). Both eras are still on disk here, and nothing but a heuristic can now
tell them apart — see :mod:`mesh_xml.io.xml_legacy`. The version on the root
exists so that failure can never repeat: a reader that does not recognise the
version refuses the document instead of quietly misinterpreting it. Any future
semantic change to an attribute bumps the version rather than reusing a name.

Which is what v3 is. v2 described a body and said nothing about what drove it, so
saving a creature kept its chunks and lost its entire gait — silently, because
there was nowhere in the document for a gait to have been. Every creature with
one therefore had to stay a Python script. v3 adds ``<controllers>`` and nothing
else; a v2 document is still read exactly as it always was and simply has no
gait, which is the truth about it rather than a default being invented for it.
Reading both eras is not politeness, it is the whole reason the stamp is there.

Why ``<controllers>`` is a section beside ``<chunks>``, not part of a chunk
--------------------------------------------------------------------------
A controller spans chunks — a spine chain, a three-chunk limb — so there is no
one chunk that owns it, and hanging it off any of them would make the body's
structure and the gait's structure disagree about which is which. It also has to
be able to name a chunk that is not there: the editor deletes chunks while the
simulation runs, and :meth:`~mesh_xml.anim.controller.ControlContext.add_force`
ignores a dead id on purpose rather than taking the simulation down over a
keypress. So a dangling controller id is **not** refused the way a dangling link
id is. A link to a missing chunk is a document that cannot be built; a controller
holding one is a document whose runtime already has a defined answer, and a
loader stricter than the runtime would make a rig unopenable for a reason
nothing else in the system considers an error.

Order matters here and nowhere else in the document. Chunks and links are records
that can be shuffled freely and the reader treats them that way. Controllers are
a pipeline: force accumulates commutatively, but a controller that carries pinned
chunks along a path has to run before the limbs that solve against where those
chunks now are. So ``<controllers>`` is read and written in document order, and
that order is part of the gait rather than an artefact of how it was written.

Why a flat ``<chunks>`` + ``<links>`` layout, not a nested tree
--------------------------------------------------------------
:meth:`~mesh_xml.core.body.Body.spanning_forest` can render any body as a nested
tree plus leftover edges, and nesting does read nicely for a spine. It was still
rejected:

* A tree cannot express a cyclic body by itself, so the document would need a
  second, differently shaped section for ``extra_edges``. That gives two spellings
  of one concept, and the cyclic spelling is the one that gets exercised least and
  therefore rots first. Losslessness must not depend on the rare path.
* Nesting makes a chunk's coordinates mean different things depending on where the
  element sits. That is the exact mechanism behind defect §2.3. Flat records give
  ``pos`` one meaning — world space — that cannot be changed by moving an element.
* The spanning forest is derived data. Persisting it means persisting something
  that can contradict the links it was derived from, in exchange for nothing:
  ``spanning_forest()`` regenerates it in one call, at read time, always current.
* Flat records are order-independent and diff per-chunk rather than per-subtree.

Identity
--------
``id`` is identity. ``name`` is a plain attribute with no uniqueness requirement,
may repeat and may be empty (defect §2.6 was making a name a dict key). Links
reference chunks by ``id``, never by name or by document position, so reordering
or renaming cannot change the topology.

Deliberately not persisted
--------------------------
``prev_pos``. A saved body is a rest pose, not a paused simulation; the loader
starts every chunk at rest, which is what :meth:`Body.add_chunk` already does.
Persisting velocity would make a file's meaning depend on the tick it was written
on.

The gait is written on exactly the same terms, and
:mod:`mesh_xml.anim.serialize` says how it finds the line: the clock on
:class:`~mesh_xml.anim.controller.ControllerStack`, the foothold a leg happens to
be standing on and the mode it happens to be in are all products of ticking, and
a creature reloads at the start of its first step rather than half way through
whichever one was in flight when the file was written. ``seed`` *is* written,
because the per-rig randomness is a property of the individual and replaying a
gait requires it.

``<terrain>``: additive in v3, absent means none
------------------------------------------------
The world the creature stands in is a ``<terrain>``
section beside ``<controllers>``: a ``blend`` attribute and ``<plane>`` /
``<cone>`` children in authoring order — which is fold order, and therefore
part of the world, exactly as controller order is part of the gait. A document
without the section has no terrain, which is the truth about every document
written before terrain existed, so this is an addition on the ``render``
attribute's terms and not a version bump: absent preserves the old meaning.
An empty ``<terrain/>`` is not the same statement as no section — it is a
world with nothing in it, and it reloads as one.

Terrain floats are written at full ``float64`` precision, like controller
parameters and unlike chunk positions: :class:`~mesh_xml.core.terrain.Terrain`
stores ``float64`` (it is the solver's collision oracle, not just the
renderer's), so :func:`_fmt`'s float32 shortening would quietly move the world
by an ulp on every save.

``<doors>``: additive in v3, absent means none
----------------------------------------------
Where the room leads is a ``<doors>`` section
beside ``<terrain>``, on exactly its terms: one ``<door>`` per exit — a name,
a world point, the destination room and door by name, an arrival radius —
written only when there is at least one door, and absent meaning none, which
is the truth about every document written before doors existed. No version
bump. Door records are order-independent like chunks and links: unlike
controllers and terrain primitives, nothing about a doorway depends on the
doorway before it. Floats are full ``float64`` precision (``repr``, not
:func:`_fmt`) for the terrain section's reason — the point and radius are the
scene's arrival test, and a float32 shortening would nudge the doorway on
every save.

``<solver>``: additive in v3, absent means the config the caller already runs
-----------------------------------------------------------------------------
The physics a body was tuned under is a ``<solver>``
section beside ``<doors>``: every :class:`~mesh_xml.core.solver.SolverConfig`
field except ``terrain``, because the world already rides above under its own
name, and a config that also carried it would be two spellings of one world —
of which the rarely-exercised spelling rots first, the flat-layout ruling.
:func:`_read_solver` therefore returns ``terrain=None`` and the caller folds
:attr:`Rig.terrain` into whichever config it installs.

Absence is asymmetric here, and honestly so: an absent *section* is not "no
physics", it is the config the caller already runs — for a room, a floor at
zero, the room's founding opinion — so every document written before the
section existed keeps its meaning and there is no version bump. A *present*
section states the physics entirely, floor included, which is why ``ground_y``
absent inside one means *no floor*, the dataclass truth: the alternative is a
magic string standing for ``None`` and a section that cannot say "this world
has no ground". The editor always writes the record in full, so the
partial-section reading exists for hand-authors alone. Floats are full
``float64`` precision (``repr``, not :func:`_fmt`) for the terrain section's
reason — these numbers are inputs to every tick, and a float32 shortening
would change the simulation a little on every save.

``<metaball>``: additive in v3, absent means the default
--------------------------------------------------------
The smooth-min radius a skin was painted against is a ``<metaball>`` section
with one ``blend`` attribute. It takes the number rather than
``MetaballSettings`` because that class lives beside the OpenGL imports and
``io`` does not follow it there. The section once held itself open as the
landing zone for the rest of that record; the rest landed as ``<look>``
below instead — shading changes for different reasons than geometry — so
``blend`` stays this section's only attribute. Absent section and absent
attribute mean the same thing — the dataclass default, which is what every
pre-arc document ran under, so no version bump and no asymmetry to state.
The value answers to :func:`~mesh_xml.core.distance.as_blend`, the validator
terrain's blend already answers to: finite, non-negative, the same error
vocabulary.

``<look>``: additive in v3, absent means the shipped defaults
-------------------------------------------------------------
The shading a room reads under is a ``<look>`` section
beside ``<metaball>``: every :class:`~mesh_xml.render.look.Look` field — the
key light's direction, colour and strength, the ambient, occlusion, shadow
and glow levels, and the palette's base colours — written in full when
written at all, the ``<solver>`` ruling. Full writing protects the *stated*
attributes' values, not against new terms: ``shadow_strength`` arrived after
the section did, defaulting to 1.0, so a pre-shadow ``<look>`` reopens
shadowed — the format's first additive attribute whose default relights a
saved record, deliberate on the precedent ``ao_strength`` set: an
additive attribute whose default changes an old file's picture is allowed
when the new default is the one the author would have asked for. A
separate section rather than more attributes on
``<metaball>`` because the two records change for different reasons:
``blend`` states the surface's *geometry* — the silhouette, the pick, what
the solver's skin means — while ``<look>`` states its shading and the
palette's base. Absent semantics are the ``<metaball>`` kind, not the
``<solver>`` kind: absent section and absent attribute both mean the
dataclass default, because every look default is a plain value — there is no
``ground_y``-style "present section changes what absence inside it means"
asymmetry to state. Validation is the record's own
:meth:`~mesh_xml.render.look.Look.validate`, wrapped into this module's
vocabulary: every float finite, the light direction non-zero, colours and
strengths non-negative, no ceilings anywhere. ``Look`` is this module's
first import from ``render``, and it holds because the record imports
nothing beyond the standard library — the render package's own "only gl/
imports OpenGL" rule, which the firewall test in ``test_settings_io`` runs
a fresh interpreter to enforce rather than trusting a docstring. Floats are
full ``float64`` precision (``repr``, not :func:`_fmt`) on the solver
section's terms: these numbers are the authored look, and a float32
shortening would drift it a little on every save.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mesh_xml.anim.controller import Controller, ControllerStack
from mesh_xml.anim.serialize import decode_controller, encode_controller
from mesh_xml.core.body import Body, LinkType
from mesh_xml.core.distance import as_blend
from mesh_xml.core.solver import SolverConfig
from mesh_xml.core.terrain import Terrain
from mesh_xml.render.look import Look
from mesh_xml.scene.door import Door

__all__ = [
    "FORMAT_VERSION",
    "READABLE_VERSIONS",
    "Rig",
    "SchemaError",
    "read",
    "save",
    "load",
]

FORMAT_VERSION = "3"

READABLE_VERSIONS = ("2", "3")
"""Every era :func:`load` understands, newest last.

v2 is here in full rather than through a converter: the only difference is a
section it does not have, so a v2 document needs nothing done to it beyond being
allowed through.
"""

_MAX_ID = int(np.iinfo(np.int32).max)


class SchemaError(ValueError):
    """A document is not a well-formed body."""


@dataclass(frozen=True)
class Rig:
    """A document's body, the gait that drives it, and the world it stands in.

    :func:`load` returns only the body, the way it always has, so a caller that
    has no use for controllers is not made to know they exist. This is the
    richer record, mirroring :class:`~mesh_xml.io.xml_legacy.LegacyImport`.
    ``terrain`` is ``None`` for every document without a ``<terrain>`` section,
    which is all of them written before it existed; ``doors`` is empty on the
    same terms for every document without a ``<doors>`` section. ``solver``
    and ``blend`` are ``None`` on the same absent-section terms — which for
    ``solver`` means the config the caller already runs, not an empty world;
    the module docstring states the asymmetry. ``look`` is ``None`` on those
    terms too, and means the shipped defaults — the truth about every
    document written before a look could be stated.
    """

    body: Body
    controllers: ControllerStack
    terrain: Terrain | None = None
    doors: tuple[Door, ...] = ()
    solver: SolverConfig | None = None
    blend: float | None = None
    look: Look | None = None


def save(
    body: Body,
    path: str | Path,
    controllers: ControllerStack | None = None,
    terrain: Terrain | None = None,
    doors: Sequence[Door] = (),
    solver: SolverConfig | None = None,
    blend: float | None = None,
    look: Look | None = None,
) -> None:
    """Write ``body``, any gait driving it, any world it stands in, its doors,
    and — when stated — its physics, its blend, and its look.

    Every field :func:`read` needs is written explicitly; nothing is inferred from
    element nesting, and nothing but the controller and terrain-primitive order
    from document order.

    Raises :class:`~mesh_xml.anim.serialize.ControllerFormatError` naming the
    controller if any of them cannot be written, rather than saving the body and
    the part of the gait that happened to encode.
    """
    root = ET.Element(
        "body",
        {"version": FORMAT_VERSION, "name": body.name, "dim": str(body.dim)},
    )

    chunks_el = ET.SubElement(root, "chunks")
    for chunk in body.chunks():
        attrs = {
            "id": str(chunk.id),
            "name": chunk.name,
            "pos": " ".join(_fmt(v) for v in chunk.pos),
            "radius": _fmt(chunk.radius),
        }
        if chunk.pinned:
            attrs["pinned"] = "true"
        else:
            attrs["mass"] = _fmt(chunk.mass)
        chunk_el = ET.SubElement(chunks_el, "chunk", attrs)
        # Meta rides as string children. Additive — no children means the
        # empty dict every earlier document already meant, so no version
        # bump. Dropping it was defect §2.4 arriving one save late: the
        # legacy importer carefully kept a capsule's type and height as
        # opaque meta, and the first v3 save silently discarded them.
        for key in sorted(chunk.meta):
            meta_el = ET.SubElement(chunk_el, "meta", {"name": str(key)})
            meta_el.text = str(chunk.meta[key])

    links_el = ET.SubElement(root, "links")
    for link in body.links():
        attrs = {
            "a": str(link.a),
            "b": str(link.b),
            "rest": _fmt(link.rest),
            "type": link.type.name,
            "elasticity": _fmt(link.elasticity),
        }
        # Written only when false: an absent attribute means what every document
        # written before the flag existed already meant, so no version bump.
        if not link.render:
            attrs["render"] = "false"
        ET.SubElement(links_el, "link", attrs)

    seed = 0 if controllers is None else controllers.seed
    stack_el = ET.SubElement(root, "controllers", {"seed": str(seed)})
    if controllers is not None:
        for controller in controllers.controllers:
            stack_el.append(encode_controller(controller))

    # Full float64 precision (repr, not _fmt): Terrain stores float64 and the
    # solver collides against it, so a float32 shortening would move the world
    # a little on every save. See the module docstring.
    if terrain is not None:
        terrain_el = ET.SubElement(root, "terrain", {"blend": repr(terrain.blend)})
        for plane in terrain.planes:
            ET.SubElement(
                terrain_el,
                "plane",
                {
                    "normal": " ".join(repr(float(v)) for v in plane[:3]),
                    "offset": repr(float(plane[3])),
                },
            )
        for cone in terrain.cones:
            ET.SubElement(
                terrain_el,
                "cone",
                {
                    "a": " ".join(repr(float(v)) for v in cone[0:3]),
                    "ra": repr(float(cone[3])),
                    "b": " ".join(repr(float(v)) for v in cone[4:7]),
                    "rb": repr(float(cone[7])),
                },
            )

    # Written only when there is something to say, so a doorless document has
    # no section and keeps meaning what every older document meant. Floats at
    # full precision on the terrain section's terms: the point and radius are
    # the scene's arrival test, not a rendering hint.
    if doors:
        doors_el = ET.SubElement(root, "doors")
        for door in doors:
            ET.SubElement(
                doors_el,
                "door",
                {
                    "name": door.name,
                    "at": " ".join(repr(float(v)) for v in door.at),
                    "to_room": door.to_room,
                    "to_door": door.to_door,
                    "radius": repr(float(door.radius)),
                },
            )

    # Written in full when written at all, so a dataclass default that later
    # changes cannot reshape a saved creature. terrain is not an attribute —
    # the world rides above under its own name — and an absent ground_y is the
    # spelling for "no floor". Floats at full float64 precision: these numbers
    # are inputs to every tick. See the module docstring.
    if solver is not None:
        # Trimmed or zero-padded to the root's dim — the projection the
        # solver itself applies at tick time (``_gravity_vector``), because
        # the file must state a gravity its own reader can accept: a dim-2
        # room carrying core's 3-D default would otherwise save a document
        # that reopens as a refusal.
        shared = tuple(float(v) for v in solver.gravity[: body.dim])
        gravity = shared + (0.0,) * (body.dim - len(shared))
        attrs = {
            "gravity": " ".join(repr(v) for v in gravity),
            "damping": repr(float(solver.damping)),
            "iterations": str(solver.iterations),
            "max_velocity": repr(float(solver.max_velocity)),
        }
        if solver.ground_y is not None:
            attrs["ground_y"] = repr(float(solver.ground_y))
        attrs["ground_friction"] = repr(float(solver.ground_friction))
        attrs["self_contact"] = "true" if solver.self_contact else "false"
        attrs["self_contact_elasticity"] = repr(float(solver.self_contact_elasticity))
        attrs["self_friction"] = repr(float(solver.self_friction))
        ET.SubElement(root, "solver", attrs)

    # The number rather than MetaballSettings: that class lives beside the
    # OpenGL imports, and io does not follow it there.
    if blend is not None:
        ET.SubElement(root, "metaball", {"blend": repr(float(blend))})

    # Written in full when written at all — the solver ruling — and in the
    # record's own field order, so the file reads as the dataclass does.
    # The attributes are Look's fields verbatim: the record this module
    # round-trips and the record the frame renders are one class, which is
    # what its imports-nothing-beyond-stdlib rule buys. Floats at full
    # float64 precision (repr): an authored look, not a rendering hint.
    if look is not None:
        ET.SubElement(
            root,
            "look",
            {
                "light_direction": " ".join(
                    repr(float(v)) for v in look.light_direction
                ),
                "light_colour": " ".join(repr(float(v)) for v in look.light_colour),
                "light_strength": repr(float(look.light_strength)),
                "ambient_strength": repr(float(look.ambient_strength)),
                "ao_strength": repr(float(look.ao_strength)),
                "shadow_strength": repr(float(look.shadow_strength)),
                "glow_strength": repr(float(look.glow_strength)),
                "surface_colour": " ".join(
                    repr(float(v)) for v in look.surface_colour
                ),
                "terrain_colour": " ".join(
                    repr(float(v)) for v in look.terrain_colour
                ),
                "background_colour": " ".join(
                    repr(float(v)) for v in look.background_colour
                ),
                "background_top": " ".join(
                    repr(float(v)) for v in look.background_top
                ),
            },
        )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(Path(path), encoding="utf-8", xml_declaration=True)


def load(path: str | Path) -> Body:
    """Read a document into a :class:`Body`, discarding any gait it carries.

    Raises :class:`SchemaError` rather than guessing whenever the document is not
    a version this module knows, including when it is a v1 ``<armature>`` file.
    """
    return read(path).body


def read(path: str | Path) -> Rig:
    """Read a document into a :class:`Body` and the :class:`ControllerStack` on it.

    A v2 document has no gait and yields an empty stack — the honest reading,
    since v2 could not describe one.
    """
    root = ET.parse(Path(path)).getroot()

    if root.tag == "armature":
        raise SchemaError(
            f"{path} is a pre-versioned <armature> document; "
            "read it with mesh_xml.io.xml_legacy.load_legacy"
        )
    if root.tag == "scene":
        raise SchemaError(
            f"{path} is a <scene> manifest, not a rig; "
            "read it with mesh_xml.io.scene_xml.read_scene"
        )
    if root.tag != "body":
        raise SchemaError(f"expected root element <body>, got <{root.tag}>")

    version = root.get("version")
    if version not in READABLE_VERSIONS:
        raise SchemaError(
            f"unsupported body format version {version!r}, "
            f"expected one of {', '.join(repr(v) for v in READABLE_VERSIONS)}"
        )

    dim = _int_attr(root, "dim", default=3)
    if dim < 2:
        raise SchemaError(f"dim must be at least 2, got {dim}")

    body = Body(dim=dim, name=root.get("name", "body"))

    file_ids: list[int] = []
    for el in root.findall("chunks/chunk"):
        chunk_id = _int_attr(el, "id")
        if not 0 <= chunk_id <= _MAX_ID:
            raise SchemaError(
                f"chunk id must be between 0 and {_MAX_ID}, got {chunk_id}; "
                "Body stores ids in an int32 array and would truncate it"
            )
        if chunk_id in file_ids:
            raise SchemaError(f"duplicate chunk id {chunk_id}")

        pos = _float_list(el, "pos")
        if len(pos) != dim:
            raise SchemaError(
                f"chunk {chunk_id}: pos has {len(pos)} components, expected {dim}"
            )

        pinned = _bool_attr(el, "pinned", default=False)
        meta = {
            child.get("name", ""): (child.text or "")
            for child in el.findall("meta")
            if child.get("name")
        }
        try:
            body.add_chunk(
                name=el.get("name", ""),
                pos=pos,
                radius=_float_attr(el, "radius", default=0.5),
                mass=_float_attr(el, "mass", default=1.0),
                pinned=pinned,
                id=chunk_id,
                meta=meta or None,
            )
        except ValueError as exc:
            raise SchemaError(f"chunk {chunk_id}: {exc}") from None
        file_ids.append(chunk_id)

    for el in root.findall("links/link"):
        a = _int_attr(el, "a")
        b = _int_attr(el, "b")
        for end in (a, b):
            if not body.has_chunk(end):
                raise SchemaError(f"link {a}-{b} references unknown chunk id {end}")
        try:
            added = body.add_link(
                a,
                b,
                rest=_float_attr(el, "rest", default=None),
                type=_link_type(el.get("type", "NORMAL")),
                elasticity=_float_attr(el, "elasticity", default=1.0),
                render=_bool_attr(el, "render", default=True),
            )
        except ValueError as exc:
            raise SchemaError(f"link {a}-{b}: {exc}") from None
        if not added:
            raise SchemaError(f"duplicate link between chunks {a} and {b}")

    return Rig(
        body=body,
        controllers=_read_controllers(root, body),
        terrain=_read_terrain(root),
        doors=_read_doors(root),
        solver=_read_solver(root, dim),
        blend=_read_metaball(root),
        look=_read_look(root),
    )


def _read_controllers(root: ET.Element, body: Body) -> ControllerStack:
    """Rebuild the stack in document order.

    Controller ids are not checked against the body; see the module docstring for
    why a stale one is a runtime condition rather than a malformed document.
    """
    element = root.find("controllers")
    if element is None:
        return ControllerStack(body)

    stack = ControllerStack(body, seed=_int_attr(element, "seed", default=0))
    for child in element:
        if child.tag != "controller":
            raise SchemaError(f"unexpected <{child.tag}> inside <controllers>")
        decoded = decode_controller(child)
        # A FootholdPicker is registered and nests inside a limb, but has no
        # update(): on the stack it would raise once per tick rather than here.
        if not isinstance(decoded, Controller):
            raise SchemaError(
                f"{child.get('type')!r} is not something that can drive a body; "
                "it has no update() and belongs inside a controller, not beside one"
            )
        stack.add(decoded)
    return stack


def _read_terrain(root: ET.Element) -> Terrain | None:
    """Rebuild the world, in document order — which is fold order.

    Absent means none, exactly like a missing ``render`` attribute means
    rendered: the default is what every older document already meant, so no
    version bump. Children rebuild through :meth:`Terrain.add_plane` and
    :meth:`Terrain.add_cone` so a hand-edited zero normal or negative radius is
    refused here, by the same validation authoring gets, rather than becoming a
    NaN in the solver three layers later.
    """
    element = root.find("terrain")
    if element is None:
        return None

    try:
        terrain = Terrain(blend=_float_attr(element, "blend", default=0.3))
    except ValueError as exc:
        raise SchemaError(f"terrain: {exc}") from None

    for child in element:
        if child.tag not in ("plane", "cone"):
            raise SchemaError(f"unexpected <{child.tag}> inside <terrain>")
        try:
            if child.tag == "plane":
                terrain.add_plane(
                    _float_list(child, "normal"), _required_float(child, "offset")
                )
            else:
                terrain.add_cone(
                    _float_list(child, "a"),
                    _required_float(child, "ra"),
                    _float_list(child, "b"),
                    _required_float(child, "rb"),
                )
        except SchemaError:
            raise
        except ValueError as exc:
            raise SchemaError(f"terrain <{child.tag}>: {exc}") from None
    return terrain


def _read_doors(root: ET.Element) -> tuple[Door, ...]:
    """Rebuild the room's exits. Absent means none, like ``<terrain>``.

    Records rebuild through :class:`~mesh_xml.scene.door.Door` itself, so a
    hand-edited empty name or negative radius is refused here, by the same
    validation authoring gets, rather than becoming a doorway the scene can
    never route. Order is preserved because everything else here preserves
    it, but nothing may depend on it: doors address each other by name.
    """
    element = root.find("doors")
    if element is None:
        return ()

    doors: list[Door] = []
    for child in element:
        if child.tag != "door":
            raise SchemaError(f"unexpected <{child.tag}> inside <doors>")
        try:
            doors.append(
                Door(
                    name=_required_str(child, "name"),
                    at=tuple(_float_list(child, "at")),
                    to_room=_required_str(child, "to_room"),
                    to_door=_required_str(child, "to_door"),
                    radius=_float_attr(child, "radius", default=0.9),
                )
            )
        except SchemaError:
            raise
        except ValueError as exc:
            raise SchemaError(f"door {child.get('name', '')!r}: {exc}") from None
    return tuple(doors)


def _read_solver(root: ET.Element, dim: int) -> SolverConfig | None:
    """Rebuild the physics the document states. Absent means the caller's config.

    Not "none", the way absent terrain means none: a document without the
    section runs under whatever config the caller already has, so ``None`` is
    returned rather than defaults being invented for it. A present section
    starts from the dataclass defaults and overlays each attribute it carries —
    ``ground_y`` absent inside one is therefore no floor, the dataclass truth.
    ``terrain`` stays ``None``: the world rides under its own name, and the
    caller folds :attr:`Rig.terrain` into whichever config it installs.
    """
    element = root.find("solver")
    if element is None:
        return None

    defaults = SolverConfig()

    gravity = defaults.gravity
    if element.get("gravity") is not None:
        gravity = tuple(_float_list(element, "gravity"))
        if len(gravity) != dim:
            raise SchemaError(
                f"solver: gravity has {len(gravity)} components, expected {dim}; "
                "a wrong-arity gravity is a broadcast error three layers later"
            )

    config = SolverConfig(
        gravity=gravity,
        damping=_float_attr(element, "damping", default=defaults.damping),
        iterations=_int_attr(element, "iterations", default=defaults.iterations),
        max_velocity=_float_attr(
            element, "max_velocity", default=defaults.max_velocity
        ),
        ground_y=_float_attr(element, "ground_y", default=None),
        ground_friction=_float_attr(
            element, "ground_friction", default=defaults.ground_friction
        ),
        self_contact=_bool_attr(element, "self_contact", default=defaults.self_contact),
        self_contact_elasticity=_float_attr(
            element,
            "self_contact_elasticity",
            default=defaults.self_contact_elasticity,
        ),
        self_friction=_float_attr(
            element, "self_friction", default=defaults.self_friction
        ),
    )

    if config.iterations < 0:
        raise SchemaError(
            f"solver: iterations must be at least 0, got {config.iterations}"
        )

    for name, values in (
        ("gravity", config.gravity),
        ("damping", (config.damping,)),
        ("max_velocity", (config.max_velocity,)),
        ("ground_y", () if config.ground_y is None else (config.ground_y,)),
        ("ground_friction", (config.ground_friction,)),
        ("self_contact_elasticity", (config.self_contact_elasticity,)),
        ("self_friction", (config.self_friction,)),
    ):
        for value in values:
            if not np.isfinite(value):
                raise SchemaError(
                    f"solver: {name} must be finite, got {value}; a NaN here "
                    "poisons every position in the room by the second tick"
                )

    return config


def _read_metaball(root: ET.Element) -> float | None:
    """The blend the skin was painted against. Absent means the default.

    An absent attribute means the same as an absent section — the dataclass
    default. That reading was written for a landing zone this section no
    longer is — the rest of the record landed as ``<look>`` instead, so
    ``blend`` stays the only attribute — and it keeps, because an empty
    section still has to mean something and it already means this. The value
    answers to :func:`~mesh_xml.core.distance.as_blend`, terrain's own
    validator.
    """
    element = root.find("metaball")
    if element is None:
        return None

    value = _float_attr(element, "blend", default=None)
    if value is None:
        return None
    try:
        return as_blend(value)
    except ValueError as exc:
        raise SchemaError(f"metaball: {exc}") from None


def _read_look(root: ET.Element) -> Look | None:
    """The shading the room was authored under. Absent means the defaults.

    Absent semantics are ``<metaball>``'s, not ``<solver>``'s: an attribute
    absent inside a present section means the dataclass default, exactly as
    the absent section does, because every look default is a plain value.
    The record refuses its own poisons — :meth:`Look.validate` gates
    non-finite floats, a zero light direction, negative colours and
    strengths, wrong arity — and its ``ValueError`` folds into this module's
    vocabulary, the same wrap ``as_blend`` gets one reader up.
    """
    element = root.find("look")
    if element is None:
        return None

    defaults = Look()
    look = Look(
        light_direction=_vector_attr(
            element, "light_direction", default=defaults.light_direction
        ),
        light_colour=_vector_attr(
            element, "light_colour", default=defaults.light_colour
        ),
        light_strength=_float_attr(
            element, "light_strength", default=defaults.light_strength
        ),
        ambient_strength=_float_attr(
            element, "ambient_strength", default=defaults.ambient_strength
        ),
        ao_strength=_float_attr(element, "ao_strength", default=defaults.ao_strength),
        shadow_strength=_float_attr(
            element, "shadow_strength", default=defaults.shadow_strength
        ),
        glow_strength=_float_attr(
            element, "glow_strength", default=defaults.glow_strength
        ),
        surface_colour=_vector_attr(
            element, "surface_colour", default=defaults.surface_colour
        ),
        terrain_colour=_vector_attr(
            element, "terrain_colour", default=defaults.terrain_colour
        ),
        background_colour=_vector_attr(
            element, "background_colour", default=defaults.background_colour
        ),
        background_top=_vector_attr(
            element, "background_top", default=defaults.background_top
        ),
    )
    try:
        return look.validate()
    except ValueError as exc:
        raise SchemaError(f"look: {exc}") from None


def _fmt(value: float) -> str:
    """Shortest decimal that reloads to the same ``float32`` the body stores."""
    return str(np.float32(value))


def _int_attr(el: ET.Element, name: str, default: int | None = None) -> int:
    raw = el.get(name)
    if raw is None:
        if default is None:
            raise SchemaError(f"<{el.tag}> is missing required attribute {name!r}")
        return default
    try:
        return int(raw)
    except ValueError:
        raise SchemaError(
            f"<{el.tag}> attribute {name}={raw!r} is not an integer"
        ) from None


def _float_attr(el: ET.Element, name: str, default: float | None) -> float | None:
    raw = el.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise SchemaError(
            f"<{el.tag}> attribute {name}={raw!r} is not a number"
        ) from None


def _required_float(el: ET.Element, name: str) -> float:
    """Like :func:`_float_attr`, for attributes with no sensible default.

    A separate helper because ``_float_attr``'s ``default=None`` is a *value* —
    an absent link ``rest`` means "measure it" — so it cannot also mean
    "required" the way :func:`_int_attr`'s does.
    """
    value = _float_attr(el, name, default=None)
    if value is None:
        raise SchemaError(f"<{el.tag}> is missing required attribute {name!r}")
    return value


def _required_str(el: ET.Element, name: str) -> str:
    """A string attribute that must be present. Empty is allowed through:
    whether ``""`` is a legal value is the record's own ruling — a chunk name
    may be empty, a door name may not — and the record is where that lives.
    """
    raw = el.get(name)
    if raw is None:
        raise SchemaError(f"<{el.tag}> is missing required attribute {name!r}")
    return raw


def _float_list(el: ET.Element, name: str) -> list[float]:
    raw = el.get(name)
    if raw is None:
        raise SchemaError(f"<{el.tag}> is missing required attribute {name!r}")
    try:
        return [float(part) for part in raw.split()]
    except ValueError:
        raise SchemaError(
            f"<{el.tag}> attribute {name}={raw!r} is not a vector"
        ) from None


def _vector_attr(
    el: ET.Element, name: str, default: tuple[float, ...]
) -> tuple[float, ...]:
    """Like :func:`_float_list`, with ``<metaball>``-kind absence.

    An absent attribute answers ``default`` — the dataclass value — rather
    than being required, because every absence inside ``<look>`` means the
    shipped look. A tuple rather than a list, because the record it feeds
    compares by field and a list that means the same triple would not be
    equal to it.
    """
    raw = el.get(name)
    if raw is None:
        return default
    return tuple(_float_list(el, name))


def _bool_attr(el: ET.Element, name: str, default: bool) -> bool:
    raw = el.get(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in ("true", "1"):
        return True
    if lowered in ("false", "0"):
        return False
    raise SchemaError(f"<{el.tag}> attribute {name}={raw!r} is not a boolean")


def _link_type(raw: str) -> LinkType:
    """Accept the member name, and the integer only as a courtesy to hand edits.

    Names are written rather than integers because the numeric values of
    :class:`LinkType` are an implementation detail; a file that says ``PUSH``
    stays correct if the enum is ever renumbered.
    """
    try:
        return LinkType[raw.strip().upper()]
    except KeyError:
        pass
    try:
        return LinkType(int(raw))
    except ValueError:
        raise SchemaError(f"unknown link type {raw!r}") from None
