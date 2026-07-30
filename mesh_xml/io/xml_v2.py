"""Version 2 of the on-disk body format: flat, world-space and explicitly versioned.

Why the version stamp
---------------------
The v1 ``<armature>`` format changed the meaning of ``position`` from world space
to parent-relative without leaving a single byte in the file to say so (refactor
plan §2.3). Both eras are still on disk here, and nothing but a heuristic can now
tell them apart — see :mod:`mesh_xml.io.xml_legacy`. ``version="2"`` on the root
exists so that failure can never repeat: a reader that does not recognise the
version refuses the document instead of quietly misinterpreting it. Any future
semantic change to an attribute bumps the version rather than reusing a name.

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
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from mesh_xml.core.body import Body, LinkType

__all__ = ["FORMAT_VERSION", "SchemaError", "save", "load"]

FORMAT_VERSION = "2"

_MAX_ID = int(np.iinfo(np.int32).max)


class SchemaError(ValueError):
    """A document is not a well-formed v2 body."""


def save(body: Body, path: str | Path) -> None:
    """Write ``body`` to ``path`` as a v2 document.

    Every field :func:`load` needs is written explicitly; nothing is inferred from
    element nesting or document order.
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
        ET.SubElement(chunks_el, "chunk", attrs)

    links_el = ET.SubElement(root, "links")
    for link in body.links():
        ET.SubElement(
            links_el,
            "link",
            {
                "a": str(link.a),
                "b": str(link.b),
                "rest": _fmt(link.rest),
                "type": link.type.name,
                "elasticity": _fmt(link.elasticity),
            },
        )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(Path(path), encoding="utf-8", xml_declaration=True)


def load(path: str | Path) -> Body:
    """Read a v2 document into a :class:`Body`.

    Raises :class:`SchemaError` rather than guessing whenever the document is not
    recognisably v2, including when it is a v1 ``<armature>`` file.
    """
    root = ET.parse(Path(path)).getroot()

    if root.tag == "armature":
        raise SchemaError(
            f"{path} is a pre-versioned <armature> document; "
            "read it with mesh_xml.io.xml_legacy.load_legacy"
        )
    if root.tag != "body":
        raise SchemaError(f"expected root element <body>, got <{root.tag}>")

    version = root.get("version")
    if version != FORMAT_VERSION:
        raise SchemaError(
            f"unsupported body format version {version!r}, expected {FORMAT_VERSION!r}"
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
        try:
            body.add_chunk(
                name=el.get("name", ""),
                pos=pos,
                radius=_float_attr(el, "radius", default=0.5),
                mass=_float_attr(el, "mass", default=1.0),
                pinned=pinned,
                id=chunk_id,
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
            )
        except ValueError as exc:
            raise SchemaError(f"link {a}-{b}: {exc}") from None
        if not added:
            raise SchemaError(f"duplicate link between chunks {a} and {b}")

    return body


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
