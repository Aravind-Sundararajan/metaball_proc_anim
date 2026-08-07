"""Placement: the world points a creature's gait carries, and moving them.

A duplicated or transferred creature brings its controllers along, and some
of their fields are *positions in the world*: a wave driver's ``origin``, a
brain's ``home``, a spine's ``lead``. Copy those verbatim and the copy is
haunted — its driver drags it back to where the original lives, which reads
as two creatures fighting over one soul. Directions must not move with it:
a ``pole`` or a ``stance`` translated like a point stops being a direction
at the first non-zero offset.

Nothing about a tuple of floats says which of the two it is — the registry's
untyped-field caveat, third appearance (ids, tempo, now points) — so each
controller class declares its placed fields in a ``PLACED`` attribute beside
the fields themselves, the same co-location ruling ``TEMPO`` follows.
:func:`translate` walks that declaration intersected with the registry's
persisted list and recurses into nested registered values; heights riding a
field the world owns (a picker's ``ground_y``) are deliberately absent,
which is also why offsets here are horizontal in practice — a creature
placed elsewhere stands on the same floor.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from mesh_xml.anim import serialize

__all__ = ["translate"]


def translate(controller: object, offset) -> dict[str, object]:
    """New values for every placed field of ``controller``, moved by ``offset``.

    Returns field name to new value, empty when nothing is placed — the same
    shape :func:`mesh_xml.anim.tempo.retime` returns, for the same reason:
    the caller lands the changes through commands or ``dataclasses.replace``
    and the original is never touched. ``offset`` is sliced to each value's
    own width, so a 3-D offset moves a 2-D home without ceremony.
    """
    cls = type(controller)
    try:
        fields = serialize.fields_of(cls)
    except serialize.ControllerFormatError:
        return {}
    placed = getattr(cls, "PLACED", ())
    shift = np.asarray(offset, dtype=np.float64)

    changes: dict[str, object] = {}
    for name in fields:
        value = getattr(controller, name)
        if value is None:
            continue
        if name in placed and not _registered(type(value)):
            changes[name] = _moved(value, shift, f"{cls.__name__}.{name}")
        elif _registered(type(value)):
            nested = translate(value, offset)
            if nested:
                changes[name] = dataclasses.replace(value, **nested)
    return changes


def _registered(cls: type) -> bool:
    try:
        serialize.fields_of(cls)
    except serialize.ControllerFormatError:
        return False
    return True


def _moved(value: object, shift: np.ndarray, where: str) -> object:
    """One placed value, moved. Container type survives the trip.

    A placed field holding a nested controller (a limb whose ``target`` is a
    picker) never reaches here — the picker owns no world point, so the limb
    recurses instead — which is what lets one field name hold either shape.
    """
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.dtype.kind not in "f":
            raise TypeError(f"{where}: not a world point: {value!r}")
        return value + shift[: value.size].astype(value.dtype)
    if isinstance(value, (list, tuple)):
        moved = [float(item) + float(shift[i]) for i, item in enumerate(value)]
        return type(value)(moved)
    raise TypeError(f"{where}: not a world point: {value!r}")
