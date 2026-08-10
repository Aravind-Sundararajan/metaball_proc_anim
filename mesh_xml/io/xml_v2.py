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
The world the creature stands in (``docs/TERRAIN.md``) is a ``<terrain>``
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
Where the room leads (``docs/ROOMS.md`` package 5) is a ``<doors>`` section
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
from mesh_xml.core.terrain import Terrain
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
    same terms for every document without a ``<doors>`` section.
    """

    body: Body
    controllers: ControllerStack
    terrain: Terrain | None = None
    doors: tuple[Door, ...] = ()


def save(
    body: Body,
    path: str | Path,
    controllers: ControllerStack | None = None,
    terrain: Terrain | None = None,
    doors: Sequence[Door] = (),
) -> None:
    """Write ``body``, any gait driving it, any world it stands in, its doors.

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
