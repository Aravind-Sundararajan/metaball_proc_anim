"""Angle limits compiled to distance links: a joint stop with no new machinery.

The solver knows nothing but distance constraints, and nothing here amends
that. For a joint ``a-b-c`` whose bone lengths ``r1 = rest(a, b)`` and
``r2 = rest(b, c)`` are already held by links, the one free variable left at
the joint is the chord ``|a-c|``, and the law of cosines makes the chord and
the interior angle at ``b`` interchangeable::

    |ac|(theta) = sqrt(r1**2 + r2**2 - 2*r1*r2*cos(theta))    # 180 = straight

So a **minimum** angle compiles to a ``PUSH`` link across the joint with rest
``|ac|(theta_min)`` — slack until the joint folds tighter, a wall once it
does — and a **maximum** compiles to a ``PULL`` at ``|ac|(theta_max)``, slack
until the joint hyperextends past it. The emitted links are ordinary links:
they run in the same momentum-conserving projection as everything else, they
serialize and round-trip as links, and they carry ``render=False`` because
they are mechanism, not flesh — the surface is built from links, and a brace
across a knee must not grow a membrane over the joint it exists to limit.

Bone lengths are read from the *authored* rests, never from current positions,
for the reason :meth:`LimbIK.bone_lengths` reads them there: a momentarily
squashed pose must not compile its own squash into the limit. :func:`brace` is
the one place current positions enter at all, and only to *measure* the angle
being frozen; the chord it emits is still built from the authored lengths, so
the brace and the bones agree on a satisfiable triangle.

Three limits of the compilation, stated rather than hidden:

* **It is a cone limit about straightness.** The chord measures how far from
  straight the joint sits and nothing else — no bend plane, no twist. It
  complements :attr:`SpineFollow.max_bend_degrees` rather than replacing it:
  the controller force is a posture *preference* the body leans against, the
  link is a joint *stop* it cannot pass.
* **A joint carries a floor or a ceiling, not both.** :class:`Body` holds one
  link per chunk pair and both limits compile onto the same ``a-c`` pair. The
  one two-sided request that is expressible is ``max_degrees=180``, which is
  no ceiling at all — the bones' own triangle inequality already caps the
  chord at ``r1 + r2`` — so it compiles to nothing and may accompany a floor.
* **Hardness fades near straight.** ``d|ac|/d(theta)`` is proportional to
  ``sin(theta)``, so a stop at 90 degrees is hard to within a few degrees of
  solver softness while a limit at 175 degrees maps its last five degrees onto
  a thousandth of a unit of chord, below what projection holds against a real
  load. Limits well away from 180 are walls; limits near 180 are firm
  suggestions.

These are structure, not behaviour: no controller runs them, nothing ticks,
and a body that loads from disk has its stops the moment it exists.
"""

from __future__ import annotations

import math

import numpy as np

from mesh_xml.core.body import Body, LinkType

__all__ = ["angle_limit", "brace", "joint_degrees"]

_MIN_ARM = 1e-6
_STRAIGHT = 180.0


def angle_limit(
    body: Body,
    a: int,
    b: int,
    c: int,
    *,
    min_degrees: float | None = None,
    max_degrees: float | None = None,
    elasticity: float = 1.0,
) -> None:
    """Stop the joint ``a-b-c`` folding past ``min_degrees`` or straightening
    past ``max_degrees``.

    Angles are interior degrees at ``b``, 180 straight, and must lie in
    ``(0, 180]``; at least one bound is required, and a floor must sit below a
    ceiling. The bones ``a-b`` and ``b-c`` must already exist as links, because
    the chord is compiled from their authored rests — a limit is a statement
    about the skeleton, and there is no skeleton to state it about until those
    links do.

    ``min_degrees=180`` is legal and extreme: its chord is ``r1 + r2``, so the
    emitted PUSH engages at any bend at all and the joint is held straight.
    ``max_degrees=180`` is the opposite extreme — a ceiling the bones' own
    triangle inequality already enforces — and compiles to *nothing* rather
    than to a harmless link: emitting it would spend the pair's single link
    slot on a constraint with no content and block a real floor from ever
    occupying it. Given alone it is therefore a no-op, which is the truthful
    reading of "may straighten fully".

    Because both bounds compile onto the one ``a-c`` link a :class:`Body` can
    hold, a joint takes a floor or a ceiling but not both below 180; asking
    for both raises rather than silently keeping one.

    Recompiling is an edit, not an error: when ``a-c`` already carries a
    mechanism link (``render=False`` — a previous limit, or a brace) it is
    updated in place via :meth:`Body.set_link`, which is what an editor
    scrubbing a limit slider needs. A *rendered* ``a-c`` link is different —
    that chord is authored flesh, a triangle rig whose third side is already
    held by its own link — and a limit over it would give one distance two
    owners, so it raises naming the triple.

    ``elasticity`` is the emitted link's per-iteration correction share:
    ``1.0`` is a wall, lower is a progressively softer stop.
    """
    r1, r2 = _bone_rests(body, a, b, c)

    if min_degrees is None and max_degrees is None:
        raise ValueError(
            f"joint {a}-{b}-{c}: neither min_degrees nor max_degrees given"
        )
    for name, value in (("min_degrees", min_degrees), ("max_degrees", max_degrees)):
        if value is not None and not 0.0 < value <= _STRAIGHT:
            raise ValueError(f"{name} must be in (0, 180], got {value}")
    if min_degrees is not None and max_degrees is not None:
        if min_degrees >= max_degrees:
            raise ValueError(
                f"min_degrees must be below max_degrees, "
                f"got {min_degrees} >= {max_degrees}"
            )

    if max_degrees == _STRAIGHT:
        max_degrees = None
    if min_degrees is not None and max_degrees is not None:
        raise ValueError(
            f"joint {a}-{b}-{c}: a floor and a ceiling both compile to the single "
            f"{a}-{c} link and a body holds one link per pair; limit the side "
            "that matters, or pass max_degrees=180 if the ceiling only meant "
            "'may straighten fully'"
        )

    if min_degrees is not None:
        _emit(body, (a, b, c), LinkType.PUSH, _chord(r1, r2, min_degrees), elasticity)
    elif max_degrees is not None:
        _emit(body, (a, b, c), LinkType.PULL, _chord(r1, r2, max_degrees), elasticity)


def brace(body: Body, a: int, b: int, c: int, *, elasticity: float = 1.0) -> None:
    """Rigidify the joint ``a-b-c`` at the angle it currently holds.

    A ``NORMAL`` mechanism link at the chord of the *measured* angle: where a
    limit is a wall on one side, a brace is a wall on both, and the joint
    keeps the pose it was posed in. The angle is the only thing read from
    positions; the chord is rebuilt from the authored bone rests, so a joint
    braced while its bones are momentarily squashed is held at its measured
    angle rather than at its squashed size, and the three links agree on a
    triangle they can all satisfy at once.

    Collision rules are :func:`angle_limit`'s: an existing mechanism link on
    ``a-c`` is overwritten, an existing rendered one raises.
    """
    r1, r2 = _bone_rests(body, a, b, c)
    rest = _chord(r1, r2, joint_degrees(body, a, b, c))
    _emit(body, (a, b, c), LinkType.NORMAL, rest, elasticity)


def joint_degrees(body: Body, a: int, b: int, c: int) -> float:
    """The interior angle at ``b`` in degrees, measured from current positions.

    180 is straight, 0 is folded shut. A measurement rather than a constraint:
    it needs the chunks but not the links, so tests and tools can read a joint
    the author has not finished rigging. Raises ``ValueError`` when either arm
    has no measurable length, because a direction is exactly what a coincident
    pair does not have.
    """
    _require_triple(body, a, b, c)
    pivot = body.pos[body.slot(b)]
    u = body.pos[body.slot(a)] - pivot
    v = body.pos[body.slot(c)] - pivot
    length_u = float(np.linalg.norm(u))
    length_v = float(np.linalg.norm(v))
    if length_u < _MIN_ARM or length_v < _MIN_ARM:
        raise ValueError(
            f"joint {a}-{b}-{c}: an arm of the joint has no length, "
            "so its angle is undefined"
        )
    cosine = float(np.dot(u, v)) / (length_u * length_v)
    return math.degrees(math.acos(min(max(cosine, -1.0), 1.0)))


def _require_triple(body: Body, a: int, b: int, c: int) -> None:
    if len({a, b, c}) != 3:
        raise ValueError(f"joint {a}-{b}-{c}: the three chunks must be distinct")
    for chunk_id in (a, b, c):
        body.slot(chunk_id)


def _bone_rests(body: Body, a: int, b: int, c: int) -> tuple[float, float]:
    _require_triple(body, a, b, c)
    rests = []
    for i, j in ((a, b), (b, c)):
        if not body.has_link(i, j):
            raise ValueError(
                f"joint {a}-{b}-{c}: no {i}-{j} link; limits compile against "
                "authored bone lengths, so the skeleton links come first"
            )
        rest = body.link(i, j).rest
        if rest <= 0.0:
            raise ValueError(
                f"joint {a}-{b}-{c}: bone {i}-{j} has zero rest length, "
                f"which leaves the angle at {b} undefined"
            )
        rests.append(rest)
    return rests[0], rests[1]


def _chord(r1: float, r2: float, degrees: float) -> float:
    span = r1 * r1 + r2 * r2 - 2.0 * r1 * r2 * math.cos(math.radians(degrees))
    return math.sqrt(max(span, 0.0))


def _emit(
    body: Body,
    joint: tuple[int, int, int],
    type: LinkType,
    rest: float,
    elasticity: float,
) -> None:
    a, b, c = joint
    if body.has_link(a, c):
        existing = body.link(a, c)
        if existing.render:
            raise ValueError(
                f"joint {a}-{b}-{c}: {a} and {c} already share a rendered "
                f"{existing.type.name} link, so their distance is authored "
                "flesh; a limit over it would give one chord two owners"
            )
        body.set_link(a, c, rest=rest, type=type, elasticity=elasticity)
    else:
        body.add_link(a, c, rest=rest, type=type, elasticity=elasticity, render=False)
