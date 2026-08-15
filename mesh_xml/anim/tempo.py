"""Creature tempo: one knob that retimes a whole gait coherently.

A creature's pace is smeared across its stack. The worm's is one number
(:attr:`WaveAnchor.rate`), but a rigged walker's is at least five: trot
``period``, ``step_duration``, ``lead_time`` and ``drive`` on every foothold
picker, and the brain's travel ``speed`` above them. Editing those spins one
by one *can* speed a creature up, and almost always speeds it up wrong — a
doubled drive over an unchanged period is a longer stride, not a faster one,
and a halved ``lead_time`` on its own aims the feet short.

Tempo is the coherent version: a single factor ``s`` under which every field
with time in its units rescales by its dimension. Rates and speeds (1/s) gain
an exponent of ``+1``; periods and durations (s) gain ``-1``; everything
spatial stays put. The pleasant consequence is that shape survives — a walker
at tempo 2 takes the *same* strides twice as often, because the spatial
triggers (``step_threshold``, amplitudes, step heights) never move, and the
prediction windows shrink exactly as fast as the velocities they multiply.

What tempo deliberately is not: cognition. ``think_period`` and ``feast_time``
keep their values, because how often a creature reconsiders and how long it
digests are behaviour, not gait — a fast creature is not thereby a hasty one.
And it is not a force knob: strengths and dampings stay authored, so a gait
pushed far enough to out-run its own muscles fails honestly in the solver
rather than being silently propped up.

Each controller class declares its own tempo fields in a ``TEMPO`` class
attribute — field name to exponent — beside the fields themselves, the same
co-location argument the registry makes for constructors: a table here would
be a second place to remember. :func:`retime` intersects that declaration
with the registry's persisted field list, so a knob the file cannot carry can
never be silently rescaled, and recurses into nested registered values, which
is how a :class:`LimbIK`'s foothold picker retimes. The nested controller
comes back as a *replacement* built by :func:`dataclasses.replace`, not a
mutation — the caller is expected to land it through an undoable command, and
``replace`` re-derives the runtime fields (``mode``, ``foothold``), which is
the serializer's own rule: a retimed gait starts its first step, it does not
resume half of an old one at the wrong speed.

The authored value itself lives in chunk meta under :data:`TEMPO_META`,
painted across the creature's component exactly as its colour is — a creature
has no object of its own, so per-creature facts ride its chunks and die with
them. Meta already round-trips through v3, so the knob reopens where it was
left without a format change.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from mesh_xml.anim import serialize

__all__ = ["TEMPO_META", "chunk_refs", "retime"]

#: Chunk-meta key carrying the creature's authored tempo, ``repr(float)``.
TEMPO_META = "tempo"


def retime(controller: object, factor: float) -> dict[str, object]:
    """New values for every tempo field of ``controller``, ``factor`` faster.

    Returns field name to new value, empty when the controller has nothing
    with time in its units — the shape :class:`SetControllerField` commands
    are built from. The controller itself is never touched; nested registered
    values (a picker inside a limb) come back as retimed replacements.
    """
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"tempo factor must be a positive number, got {factor!r}")
    cls = type(controller)
    try:
        fields = serialize.fields_of(cls)
    except serialize.ControllerFormatError:
        return {}
    tempo = getattr(cls, "TEMPO", {})

    changes: dict[str, object] = {}
    for name in fields:
        value = getattr(controller, name)
        if value is None:
            continue
        if name in tempo:
            changes[name] = _scale(value, factor ** float(tempo[name]), f"{cls.__name__}.{name}")
        elif serialize.is_registered(type(value)):
            nested = retime(value, factor)
            if nested:
                changes[name] = dataclasses.replace(value, **nested)
    return changes


def chunk_refs(controller: object) -> frozenset[int]:
    """Every chunk id the controller's persisted fields mention.

    This is membership, not authority: the ids say which creature a
    controller belongs to, so a per-creature edit can find its stack. An
    integer field is presumed to be a chunk id — true of every registered
    controller today, and unverifiable in general for the reason the registry
    gives: a world point is three floats and a chunk id is one, and both are
    just numbers by the time they reach a field. Callers cross-check against
    the body, which turns a future non-id integer into at worst a stray
    membership, never a crash.
    """
    refs: set[int] = set()
    try:
        fields = serialize.fields_of(type(controller))
    except serialize.ControllerFormatError:
        return frozenset()
    for name in fields:
        _collect(getattr(controller, name), refs)
    return frozenset(refs)


def _scale(value: object, mult: float, where: str) -> object:
    """Scale one tempo field, preserving its container type.

    A ``drive`` is a velocity vector, so a sequence scales element-wise; the
    scalar fields are rates and durations. Anything else declared in a
    ``TEMPO`` table is a declaration bug, and naming the field beats bending
    the arithmetic around it.
    """
    if isinstance(value, (bool, str)):
        raise TypeError(f"{where}: a {type(value).__name__} has no tempo dimension")
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value) * mult
    if isinstance(value, np.ndarray):
        return value * np.asarray(mult, dtype=value.dtype)
    if isinstance(value, (list, tuple)):
        return type(value)(float(item) * mult for item in value)
    raise TypeError(f"{where}: a {type(value).__name__} has no tempo dimension")


def _collect(value: object, refs: set[int]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, np.integer)):
        refs.add(int(value))
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect(item, refs)
        return
    if not isinstance(
        value, (float, str, bytes, np.ndarray)
    ) and serialize.is_registered(type(value)):
        for name in serialize.fields_of(type(value)):
            _collect(getattr(value, name), refs)
