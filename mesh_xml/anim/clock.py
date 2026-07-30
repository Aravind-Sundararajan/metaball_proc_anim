"""Fixed-timestep clock.

Simulation runs at a fixed rate and rendering does not. Rain World ticks at
40 Hz and draws whenever the frame lands, passing a ``timeStacker`` into
``DrawSprites`` to blend the last tick toward the current one. Everything about
how its creatures move is tuned against that fixed tick; a variable timestep
would make the same spring constants behave differently on different hardware.

:func:`mesh_xml.core.solver.fixed_step` already consumes whole ticks internally,
which is enough when the solver is the only thing that ticks. It stops being
enough as soon as anything else has to run *between* integration and the next
frame — the draw-point springs in :mod:`mesh_xml.render.draw_points` now, the
controllers in Phase 4 next. So this clock hands ticks back to the caller rather
than consuming them, and the caller decides what happens inside one:

.. code-block:: python

   for _ in clock.advance(frame_seconds):
       solver.step(body, clock.dt, config)
       draw_points.update(body, clock.dt)
   positions = draw_points.render_pos(clock.alpha)
"""

from __future__ import annotations

from mesh_xml.core.solver import TICK_DT, TICK_HZ

__all__ = ["Clock", "TICK_DT", "TICK_HZ"]

_MAX_CATCHUP_TICKS = 8

# Relative slack when floor-dividing the accumulator, in units of one tick.
_EPSILON_TICKS = 1e-9


class Clock:
    """Accumulates wall-clock time and doles it out in whole fixed ticks.

    ``max_ticks`` caps how much catch-up a single frame may attempt. Without a
    cap, a frame that stalls hands the next one a backlog it cannot clear inside
    its own budget, which makes the following frame stall worse — the spiral of
    death. Overshoot past the cap is dropped and counted in :attr:`dropped`,
    so the simulation runs slow rather than progressively falling further behind.
    """

    def __init__(self, dt: float = TICK_DT, max_ticks: int = _MAX_CATCHUP_TICKS) -> None:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if max_ticks < 1:
            raise ValueError(f"max_ticks must be at least 1, got {max_ticks}")
        self.dt = dt
        self.max_ticks = max_ticks
        self._accumulator = 0.0
        self.ticks = 0
        self.dropped = 0

    def advance(self, elapsed: float) -> range:
        """Bank ``elapsed`` seconds and return a range over the ticks now due.

        Returning a ``range`` rather than an int is deliberate: the call site
        reads as ``for _ in clock.advance(dt)``, so there is no way to bank the
        time and then forget to run the ticks it bought.
        """
        if elapsed > 0.0:
            self._accumulator += elapsed

        # The epsilon is not paranoia. Banking 0.03 then 0.02 against a 0.025
        # tick leaves the accumulator one ulp *below* 0.025, so a plain floor
        # yields zero ticks and the frame silently starves. Over a long run that
        # reads as the simulation drifting slow for no visible reason.
        due = int(self._accumulator / self.dt + _EPSILON_TICKS)
        if due > self.max_ticks:
            self.dropped += due - self.max_ticks
            self._accumulator -= (due - self.max_ticks) * self.dt
            due = self.max_ticks

        self._accumulator = max(0.0, self._accumulator - due * self.dt)
        self.ticks += due
        return range(due)

    @property
    def alpha(self) -> float:
        """How far the next tick already is, in ``[0, 1)``. Blend factor for drawing."""
        return self._accumulator / self.dt

    @property
    def accumulator(self) -> float:
        return self._accumulator

    def reset(self) -> None:
        """Drop banked time. Call after a pause, or the clock tries to catch up on it."""
        self._accumulator = 0.0

    def __repr__(self) -> str:
        return (
            f"<Clock {1.0 / self.dt:.1f}Hz ticks={self.ticks} "
            f"dropped={self.dropped} alpha={self.alpha:.3f}>"
        )
