"""Persisting a gait: stable names for controller classes, and their fields.

Why this exists
---------------
The body format describes a graph of masses and says nothing about what drives
it, so every creature with a gait had to be a Python script and
:meth:`~mesh_xml.editor.document.Document.adopt` existed to get one into the
editor. What is missing is not a file section — that is easy — it is a way to
turn a controller back into itself without the reader having to know, in
advance, every controller that will ever exist.

Why a registered name and not the class path
--------------------------------------------
``mesh_xml.anim.controllers.limb.LimbIK`` in a document would make every saved
creature depend on the package layout: move a module and files somebody else
wrote stop loading, with the breakage surfacing in their project rather than in
the commit that caused it. A short registered name is a promise this package
makes and can keep across any refactor. It also means a document can never name
an import path that a loader then resolves and calls, which a class-path reader
must do by construction.

Why the constructor is the schema
---------------------------------
:func:`register` reads the field list off the class's own constructor rather
than taking a hand-written one. A second, hand-maintained list is a second place
to remember, and the failure when it is forgotten is silent: a field added to
:class:`~mesh_xml.anim.controllers.limb.FootholdPicker` would simply stop being
written, and the creature would reload subtly differently with nothing raised.
Deriving the list means a parameter is persisted the day it is added.

The same rule says what *not* to write, which is the point. ``mode`` and
``foothold`` on a picker, and ``_phase`` on :class:`Breathe`, are declared
``init=False`` and so are not constructor parameters. That is not a coincidence
to exploit — it is the same statement in both directions: what the constructor
accepts is the authored definition, and everything else is derived from it or
accumulated while running. :mod:`mesh_xml.io.xml_v2` refuses to persist
``prev_pos`` for exactly this reason. A saved creature is a rest pose plus a
gait definition, not a paused simulation, so it reloads at the start of its
first step rather than half way through whichever one was in flight when the
file was written.

Why values say what they are
----------------------------
:attr:`LimbIK.target` is a :class:`~mesh_xml.anim.controllers.limb.TipTarget`,
or a world point, or ``None``; :attr:`LookAt.target` is only ever a world point.
One field name, three shapes, and no way to tell them apart from the name. So
each value is written as an element whose *tag* is its kind, and nesting a
controller inside a controller becomes the same mechanism as nesting a float
inside a tuple rather than a special case in the writer.

Why floats are written at full precision
----------------------------------------
Unlike a chunk position, which :mod:`mesh_xml.io.xml_v2` shortens to the
``float32`` the body actually stores. A controller parameter is a Python float
used in ``float64`` arithmetic: ``step_duration`` divides a ``float64`` clock,
and rounding it to ``float32`` moves a footfall by a tick. The creature would
still walk, which is what makes it dangerous — the gait it walks would not be
the gait that was saved.

Chunk ids need no translation table
-----------------------------------
Controllers hold chunk ids and this module writes them unchanged, because
:meth:`~mesh_xml.core.body.Body.add_chunk` restores a body's ids exactly. The
alternative — renumbering on load — would have to know which integers are ids,
and nothing here can: a world point is three floats and a chunk id is one, and
both are just numbers by the time they reach a field. Ids are the only identity
a body has, so they are the one thing that must survive untouched.

Refusing rather than half-writing
---------------------------------
An unregistered class, or a value with no encoding, raises
:class:`ControllerFormatError` naming the controller. A gait that saves nine of
its ten controllers is worse than one that refuses: it loads, it looks like a
creature, and it walks wrong.
"""

from __future__ import annotations

import inspect
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mesh_xml.anim.behavior import Grazer, Hunter
from mesh_xml.anim.controller import Controller
from mesh_xml.anim.controllers.balance import Level, Posture
from mesh_xml.anim.controllers.drive import WaveAnchor
from mesh_xml.anim.controllers.limb import FootholdPicker, LimbIK, TipTarget
from mesh_xml.anim.controllers.spine import Breathe, LookAt, SpineFollow

__all__ = [
    "CONTROLLER_TAG",
    "ControllerFormatError",
    "Persisted",
    "decode_controller",
    "encode_controller",
    "fields_of",
    "register",
    "registered",
]

CONTROLLER_TAG = "controller"

Persisted = Controller | TipTarget
"""What the registry indexes: a controller, or a target a controller holds.

:class:`~mesh_xml.anim.controllers.limb.FootholdPicker` is not a
:class:`~mesh_xml.anim.controller.Controller` — it has no ``update`` and never
sits on the stack — but it is half of what makes a leg walk, and it is written
by the same mechanism. A second registry for targets would double this module
in exchange for a distinction the file format does not need to make.
"""


class ControllerFormatError(ValueError):
    """A controller cannot be written, or a document does not describe one."""


@dataclass(frozen=True)
class _Entry:
    name: str
    cls: type
    fields: tuple[str, ...]


_BY_NAME: dict[str, _Entry] = {}
_BY_CLASS: dict[type, _Entry] = {}


def register(
    name: str, cls: type, *, fields: Sequence[str] | None = None
) -> type:
    """Bind a controller class to the name its documents will carry.

    This is the whole of adding a new controller to the format; no reader or
    writer changes with it. ``fields`` defaults to the class's constructor
    parameters — pass it only to persist fewer of them, never to rename one,
    since the names written are the keyword arguments the loader calls back
    with.

    Re-registering the identical binding is a no-op, so a module may register at
    import time without caring how often it is imported. Rebinding a live name
    to a different class raises instead, because the alternative is a document
    that reloads as something other than what wrote it.
    """
    entry = _Entry(
        name=name,
        cls=cls,
        fields=tuple(fields) if fields is not None else _constructor_fields(cls),
    )

    claimed_name = _BY_NAME.get(name)
    if claimed_name is not None and claimed_name != entry:
        raise ControllerFormatError(
            f"controller name {name!r} is already registered to "
            f"{claimed_name.cls.__name__}"
        )
    claimed_class = _BY_CLASS.get(cls)
    if claimed_class is not None and claimed_class.name != name:
        raise ControllerFormatError(
            f"{cls.__name__} is already registered as {claimed_class.name!r}"
        )

    _BY_NAME[name] = entry
    _BY_CLASS[cls] = entry
    return cls


def registered() -> dict[str, type]:
    """Every registered name and the class it names, as a copy."""
    return {name: entry.cls for name, entry in _BY_NAME.items()}


def fields_of(cls: type) -> tuple[str, ...]:
    """The field names the registry persists for ``cls``.

    The editor's controller panel builds its widgets from this, which is what
    keeps "what the panel can edit" and "what the file can carry" the same
    list by construction — a knob the format cannot save would be a lie that
    survives exactly one session.
    """
    entry = _BY_CLASS.get(cls)
    if entry is None:
        raise ControllerFormatError(
            f"{cls.__name__} is not a registered controller"
        )
    return entry.fields


def encode_controller(controller: Persisted, name: str | None = None) -> ET.Element:
    """Write one controller as a ``<controller>`` element.

    ``name`` labels the element as the value of an enclosing controller's field,
    which is how a picker ends up inside the limb that holds it. A controller
    sitting directly on the stack has no such label.
    """
    entry = _BY_CLASS.get(type(controller))
    if entry is None:
        raise ControllerFormatError(
            f"{type(controller).__name__} is not a registered controller, so "
            "its settings cannot be written; call "
            "mesh_xml.anim.serialize.register() for it first"
        )

    attrs = {} if name is None else {"name": name}
    attrs["type"] = entry.name
    element = ET.Element(CONTROLLER_TAG, attrs)

    for field in entry.fields:
        where = f"{entry.cls.__name__}.{field}"
        try:
            value = getattr(controller, field)
        except AttributeError:
            raise ControllerFormatError(
                f"{where} is a constructor parameter with no attribute of that "
                "name, so it cannot be written; pass fields= to register() to "
                "leave it out"
            ) from None
        element.append(_encode_value(value, field, where))

    return element


def decode_controller(element: ET.Element) -> Persisted:
    """Rebuild one controller from a ``<controller>`` element."""
    if element.tag != CONTROLLER_TAG:
        raise ControllerFormatError(
            f"expected <{CONTROLLER_TAG}>, got <{element.tag}>"
        )

    name = element.get("type")
    if name is None:
        raise ControllerFormatError(
            f"<{CONTROLLER_TAG}> is missing required attribute 'type'"
        )
    entry = _BY_NAME.get(name)
    if entry is None:
        known = ", ".join(sorted(_BY_NAME)) or "none"
        raise ControllerFormatError(
            f"unknown controller type {name!r}; registered types are {known}"
        )

    arguments: dict[str, object] = {}
    for child in element:
        field = child.get("name")
        if field is None:
            raise ControllerFormatError(
                f"{entry.cls.__name__}: <{child.tag}> is missing required "
                "attribute 'name'"
            )
        # Breadcrumbs name the class rather than the document's type string, so
        # a field that fails reads the same whichever direction it failed in.
        arguments[field] = _decode_value(child, f"{entry.cls.__name__}.{field}")

    try:
        return entry.cls(**arguments)
    except (TypeError, ValueError) as exc:
        raise ControllerFormatError(
            f"cannot rebuild {entry.cls.__name__} from this document: {exc}"
        ) from None


def _constructor_fields(cls: type) -> tuple[str, ...]:
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError) as exc:
        raise ControllerFormatError(
            f"cannot read {cls.__name__}'s constructor ({exc}); pass fields= "
            "to register() to say what to persist"
        ) from None
    return tuple(
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
    )


def _element(tag: str, name: str | None, **attrs: str) -> ET.Element:
    """One value element. ``name`` leads, so a document reads field-first."""
    ordered = {} if name is None else {"name": name}
    ordered.update(attrs)
    return ET.Element(tag, ordered)


def _encode_value(value: object, name: str | None, where: str) -> ET.Element:
    """Encode one field value, recursing through sequences and nested controllers.

    ``bool`` is tested before ``int`` because it is a subclass of it, and an
    ``enabled`` flag written as ``1`` would reload as an integer.
    """
    if value is None:
        return _element("none", name)
    if isinstance(value, (bool, np.bool_)):
        return _element("bool", name, value="true" if value else "false")
    if isinstance(value, (int, np.integer)):
        return _element("int", name, value=str(int(value)))
    if isinstance(value, (float, np.floating)):
        return _element("float", name, value=repr(float(value)))
    if isinstance(value, str):
        return _element("str", name, value=value)
    if isinstance(value, np.ndarray):
        return _encode_array(value, name, where)
    if isinstance(value, (list, tuple)):
        element = _element("list" if isinstance(value, list) else "tuple", name)
        for index, item in enumerate(value):
            element.append(_encode_value(item, None, f"{where}[{index}]"))
        return element
    if type(value) in _BY_CLASS:
        return encode_controller(value, name)

    raise ControllerFormatError(
        f"{where}: cannot write a {type(value).__name__}. A persisted field has "
        "to be a number, a string, a sequence of those, or another registered "
        "controller."
    )


def _encode_array(value: np.ndarray, name: str | None, where: str) -> ET.Element:
    """Numpy arrays carry their dtype, since a world point may be either width.

    Only one-dimensional numeric arrays occur here — a raw
    :attr:`LimbIK.target` is a world point — and refusing the rest keeps the
    reader from having to reconstruct a shape it was never given.
    """
    if value.ndim != 1 or value.dtype.kind not in "fiu":
        raise ControllerFormatError(
            f"{where}: only a one-dimensional numeric array can be written, "
            f"got shape {value.shape} of {value.dtype}"
        )
    return _element(
        "vector",
        name,
        dtype=value.dtype.name,
        value=" ".join(repr(item.item()) for item in value),
    )


def _decode_value(element: ET.Element, where: str) -> object:
    if element.tag == CONTROLLER_TAG:
        return decode_controller(element)
    if element.tag == "none":
        return None
    if element.tag == "bool":
        return _bool(element, where)
    if element.tag == "int":
        return _number(element, where, int)
    if element.tag == "float":
        return _number(element, where, float)
    if element.tag == "str":
        return _raw(element, where)
    if element.tag == "vector":
        return _vector(element, where)
    if element.tag in ("list", "tuple"):
        items = [
            _decode_value(child, f"{where}[{index}]")
            for index, child in enumerate(element)
        ]
        return items if element.tag == "list" else tuple(items)

    raise ControllerFormatError(f"{where}: unknown value element <{element.tag}>")


def _raw(element: ET.Element, where: str) -> str:
    raw = element.get("value")
    if raw is None:
        raise ControllerFormatError(
            f"{where}: <{element.tag}> is missing required attribute 'value'"
        )
    return raw


def _bool(element: ET.Element, where: str) -> bool:
    raw = _raw(element, where).strip().lower()
    if raw in ("true", "1"):
        return True
    if raw in ("false", "0"):
        return False
    raise ControllerFormatError(f"{where}: {raw!r} is not a boolean")


def _number(element: ET.Element, where: str, kind: type) -> object:
    raw = _raw(element, where)
    try:
        return kind(raw)
    except ValueError:
        raise ControllerFormatError(
            f"{where}: {raw!r} is not a {kind.__name__}"
        ) from None


def _vector(element: ET.Element, where: str) -> np.ndarray:
    dtype = element.get("dtype", "float32")
    try:
        parts = [float(part) for part in _raw(element, where).split()]
        return np.array(parts, dtype=np.dtype(dtype))
    except (TypeError, ValueError) as exc:
        raise ControllerFormatError(f"{where}: not a {dtype} vector ({exc})") from None


register("spine_follow", SpineFollow)
register("look_at", LookAt)
register("breathe", Breathe)
register("limb_ik", LimbIK)
register("foothold_picker", FootholdPicker)
register("wave_anchor", WaveAnchor)
register("grazer", Grazer)
register("hunter", Hunter)
register("posture", Posture)
register("level", Level)
