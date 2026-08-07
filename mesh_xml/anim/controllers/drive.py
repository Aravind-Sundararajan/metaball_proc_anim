"""Drivers: motion sources that lead a rig from a pinned chunk.

The controller contract everywhere else in this package is force-only —
:mod:`mesh_xml.core.solver` documents why a controller that writes positions
fights the projection running right after it. A driver is the one legitimate
exception, and it is legitimate for a stated reason: it moves a *pinned*
chunk, and a pinned chunk is one the solver owns no part of. Inverse mass
zero means no constraint and no force will ever move it, so there is nothing
to fight and nothing to reconcile — the pin is a hand, and the driver is the
arm holding it.

This is also the missing half of a saved idle. ``SpineFollow`` and
``Breathe`` shape how a body carries motion, but a hanging rig with nothing
moving its anchor simply hangs. Baking a driver beside them is what makes a
shipped file wriggle when opened rather than pose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mesh_xml.anim.controller import ControlContext

__all__ = ["WaveAnchor"]

_LISSAJOUS = (2.0, 1.3, 1.0)
"""Per-axis frequency ratios. Irrational-ish on purpose: a closed figure reads
as a loop within seconds, while these phases take long enough to realign that
the motion reads as wandering."""


@dataclass
class WaveAnchor:
    """Carry a pinned anchor on a slow lissajous so the body has somewhere to go.

    Promoted from ``examples/animate_rig.py`` once gaits became files: the
    example rebuilt it per run, so nothing saved could wriggle on its own.

    ``origin`` is authored rather than sampled on the first tick, so
    :meth:`ControllerStack.reset` genuinely rewinds the motion — a sampled
    origin would re-anchor wherever the rig happened to be. The y amplitude is
    a fraction of the horizontal one because the figure reads as swimming;
    equal amplitudes read as a fly bouncing in a bottle.
    """

    anchor: int
    origin: tuple[float, ...]
    amplitude: float = 0.8
    rate: float = 0.6
    enabled: bool = True

    def swing(self, time: float) -> np.ndarray:
        """The wave's offset from ``origin`` at ``time``, sized to ``origin``.

        Public because a grab release needs it: re-authoring ``origin`` so the
        anchor stays exactly where the hand left it means subtracting the swing
        the next tick will add, and duplicating the figure there would let the
        two drift apart.
        """
        t = time * self.rate
        return np.array(
            [
                np.sin(t * _LISSAJOUS[0]) * self.amplitude,
                np.sin(t * _LISSAJOUS[1]) * self.amplitude * 0.35,
                np.cos(t * _LISSAJOUS[2]) * self.amplitude,
            ]
        )[: len(self.origin)]

    def update(self, ctx: ControlContext) -> None:
        # A held anchor belongs to the hand holding it. Writing over the drag
        # is not a fight the driver can even see, let alone win gracefully.
        if not ctx.has(self.anchor) or self.anchor in ctx.held:
            return
        origin = np.asarray(self.origin, dtype=np.float64)
        ctx.body.set_pos(self.anchor, tuple(origin + self.swing(ctx.time)))
