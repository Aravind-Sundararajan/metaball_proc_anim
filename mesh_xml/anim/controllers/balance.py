"""Postural control: hold a trunk at a height, and hold it flat.

A creature that pushes on the ground needs something holding it up, and in this
model that something cannot be the limbs. Two bones joined by distance links
have a free knee angle, so a leg folds under load like a drawing compass and the
body sinks whatever the IK is aiming at — the elbow target is solved from the
root's *current* position, so it follows the trunk down rather than resisting.
Nothing in :mod:`mesh_xml.core` expresses "this leg holds its own length": there
are no angular constraints, no muscles along a bone, and a distance link is
isotropic. Standing therefore has to be asserted, and this is where it is
asserted.

:class:`Posture` seeks a height. :class:`Level` removes tilt from an ordered
chain. They are separate because they answer different questions — how high, and
which way up — and a rig usually wants both at different gains.

**How much of standing this is.** Not much, and the gap is worth naming.

*It does not know where the feet are.* There is no support polygon and no
centre-of-mass test, so it will hold a trunk at its target height with perfect
composure while the creature topples off the side of its own stance. Staying
over the feet is the gait's job here, which means a gait that puts a foot in a
bad place gets no help recovering.

*It cannot step to catch itself.* Real balance is a stepping problem — pushed
hard enough, an animal puts a foot out. This is a spring, and a spring pushed
past the point where the legs can reach just sags.

*There is no orientation to control.* A :class:`~mesh_xml.core.body.Body` is
points, not frames, so "level" can only mean the height difference between two
named chunks. Roll and pitch are whatever the caller's choice of chain implies,
and yaw is not expressible at all.

*The height term is external unless it is told what to press against.* On its
own it is a hand under the belly: it lifts the trunk with force that came from
nowhere, which is stable and cheap and a lie. Name the planted feet in
``support`` and the same lift is charged back to them as an equal and opposite
push into the floor, which is what a leg extending actually does — the ground
takes the load, the pair sums to zero, and the controller stops being a source
of momentum. Prefer it. The one-sided form is kept because a limb with nothing
under it has nothing to press on, and sagging quietly beats diverging.

:class:`Level` never needed the escape hatch: it is an equal and opposite pair
by construction, for the reason ``SpineFollow``'s bend limit is — a trunk
pressed hard against a levelling term must not walk sideways across the world
as a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mesh_xml.anim.controller import ControlContext

__all__ = ["Level", "Posture"]

_UP = 1
"""Index of the vertical axis, matching ``solver._collide_ground``."""


def _live(ctx: ControlContext, chunk_ids: list[int]) -> list[int]:
    return [cid for cid in chunk_ids if ctx.body.has_chunk(cid)]


def _vertical(ctx: ControlContext, amount: float) -> np.ndarray:
    out = np.zeros(ctx.body.dim, dtype=np.float32)
    out[_UP] = np.float32(amount)
    return out


def _clamp(value: float, limit: float | None) -> float:
    if limit is None:
        return value
    return float(np.clip(value, -limit, limit))


@dataclass
class Posture:
    """Hold chunks at a world height, optionally by pressing on the feet.

    ``height`` is a world ``y``, not a clearance: it is where the trunk should
    sit, and a rig that stands 0.9 above the floor passes 0.9 whatever the floor
    is. ``damping`` opposes vertical velocity, and it is not optional — an
    undamped height spring on a body that also bounces off the ground is two
    oscillators feeding each other.

    ``max_force`` clamps each chunk's share. A creature dropped from a height
    otherwise arrives with an error of several units and receives a lift
    proportional to it, which throws the rig rather than catching it.

    ``support`` names the chunks the lift is charged to, and is what makes this
    a leg rather than a hook; see the module docstring. The total is split
    evenly rather than by contact, because this controller cannot see the
    ground: a foot in mid-swing is pushed down as hard as a planted one, which
    costs a little swing clearance and buys not having to guess at contact.

    Gains live under an explicit stability ceiling that the solver does not
    enforce. Verlet integrates these forces explicitly, so a chunk of mass ``m``
    needs ``strength < 4m/dt**2`` and ``damping < 2m/dt``; at the 40 Hz tick and
    a mass of one that is roughly 6400 and 80. Exceeding the damping bound is
    the surprising one — it diverges rather than over-damping, and it does so
    silently on whichever chunk happens to be lightest.
    """

    chunks: list[int]
    height: float
    strength: float = 150.0
    damping: float = 12.0
    max_force: float | None = 60.0
    support: list[int] = field(default_factory=list)
    enabled: bool = True

    def update(self, ctx: ControlContext) -> None:
        held = _live(ctx, self.chunks)
        if not held:
            return

        total = 0.0
        for chunk_id in held:
            lift = self.strength * (self.height - float(ctx.position(chunk_id)[_UP]))
            lift -= self.damping * float(ctx.velocity(chunk_id)[_UP])
            lift = _clamp(lift, self.max_force)
            ctx.add_force(chunk_id, _vertical(ctx, lift))
            total += lift

        pressing = _live(ctx, self.support)
        if not pressing:
            return
        share = _vertical(ctx, -total / len(pressing))
        for chunk_id in pressing:
            ctx.add_force(chunk_id, share)


@dataclass
class Level:
    """Flatten an ordered chain by pushing neighbouring pairs level.

    Each adjacent pair is nudged toward equal height with an equal and opposite
    vertical pair, so the chain flattens without the trunk gaining momentum.
    Applying it pairwise rather than end to end means a long spine straightens
    along its whole length instead of hinging in the middle.

    ``strength`` and ``damping`` are subject to the same explicit-integration
    ceiling as :class:`Posture`, and the pair form doubles the effective gain
    on a chunk shared by two pairs.

    This levels against world vertical, not against a heading. A creature on a
    slope is held horizontal rather than parallel to the ground, which is the
    right answer for a trunk and the wrong one for a foot.
    """

    chain: list[int]
    strength: float = 60.0
    damping: float = 6.0
    max_force: float | None = 60.0
    enabled: bool = True

    def update(self, ctx: ControlContext) -> None:
        chain = _live(ctx, self.chain)
        if len(chain) < 2:
            return

        for lower, upper in zip(chain, chain[1:]):
            drop = float(ctx.position(upper)[_UP] - ctx.position(lower)[_UP])
            rate = float(ctx.velocity(upper)[_UP] - ctx.velocity(lower)[_UP])
            push = _clamp(
                0.5 * (self.strength * drop + self.damping * rate), self.max_force
            )
            ctx.add_force(lower, _vertical(ctx, push))
            ctx.add_force(upper, _vertical(ctx, -push))
