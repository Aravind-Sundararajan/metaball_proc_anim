"""A pure 1-D animation curve: ``x -> y`` through cubic Bezier segments.

This replaces the sampling half of ``SimpleCurveEditor``. It is deliberately
headless: no Qt, no scene coordinates, no pixel units, no radius mapping. ``x``
and ``y`` are plain floats in whatever domain the caller chose — arc length for
taper, chunk index for stiffness-along-chain, normalized phase for gait timing,
impact speed for squash response. The old widget baked a ``2.0 - (y/180)*1.9``
pixel-to-radius conversion into eight separate places; nothing of the sort lives
here, and callers scale the result themselves.

Three things in here are subtle enough to be worth stating outright, because all
three were wrong in the widget this replaces.

**Which control points a segment uses.** Segment ``i`` is the cubic Bezier over
``(key[i].pos, key[i].out_handle, key[i+1].in_handle, key[i+1].pos)``. The old
sampler paired ``key[i].out_handle`` with ``key[i+1].out_handle`` while the
renderer correctly used the *in*-handle, so the numbers a rig was built from did
not match the curve the author had drawn. :meth:`Curve1D.segment_control_points`
is the single source of truth for that tuple; a renderer and the sampler must
both read it rather than each assembling their own.

**``x`` is not the Bezier parameter.** A cubic Bezier's *x*-component is itself a
cubic in ``t``, so ``t = (x - x0) / (x1 - x0)`` is only the right answer in the
special case where both handles sit at exact x-thirds. Feeding that fraction in
as the parameter — which the old code did — silently discards the ease the
author authored. :func:`_solve_t` instead solves ``x(t) = x`` the way a CSS
``cubic-bezier()`` easing is evaluated: Newton-Raphson seeded with the x-fraction
(a bad answer but an excellent starting guess), falling back to bisection for
anything Newton cannot land. See that function for the exact tolerances.

**Handles are rigid with their key.** Moving a key translates its handles by the
same delta, preserving their offsets bit-for-bit. The old widget's version of
this reduced to ``setPos(pos + (handle - pos))``, i.e. ``setPos(handle)``, a
no-op that left handles behind whenever a key was dragged.

The x-ordering invariant — keys strictly increasing in ``x`` — is re-established
after every mutation, so no caller can leave the curve in a state where sampling
is ambiguous. See :data:`X_EPSILON` for what happens to coincident keys.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

__all__ = ["Key", "Curve1D", "X_EPSILON"]

X_EPSILON = 1e-9
"""Minimum separation in ``x`` between adjacent keys.

Two keys at the same ``x`` would make ``y`` a relation rather than a function.
Of the three available answers — reject, clamp, nudge — this module **nudges**:
after any mutation the keys are sorted by ``x`` and then swept left to right,
and any key within :data:`X_EPSILON` of its predecessor is pushed just past it
(carrying its handles rigidly). Nothing is ever silently dropped and no mutation
can fail, which matters because the editor drags keys through each other
routinely. Where ``x`` is large enough that adding :data:`X_EPSILON` rounds away,
:func:`math.nextafter` supplies the separation instead, so strict monotonicity
holds at every scale.
"""

_NEWTON_ITERATIONS = 8
_BISECTION_ITERATIONS = 64
_T_TOLERANCE = 1e-12
_MIN_SLOPE = 1e-12

_KEEP: Any = object()


def _bezier(t, p0, p1, p2, p3):
    """Cubic Bezier in Bernstein form. Elementwise over numpy arrays.

    Written so that ``t == 0`` returns exactly ``p0`` and ``t == 1`` exactly
    ``p3``, with no rounding: sampling at a key's own ``x`` must return that
    key's ``y`` bit-for-bit.
    """
    mt = 1.0 - t
    return (
        mt * mt * mt * p0
        + 3.0 * mt * mt * t * p1
        + 3.0 * mt * t * t * p2
        + t * t * t * p3
    )


def _bezier_slope(t, p0, p1, p2, p3):
    """``d/dt`` of :func:`_bezier`, for the Newton step."""
    mt = 1.0 - t
    return (
        3.0 * mt * mt * (p1 - p0) + 6.0 * mt * t * (p2 - p1) + 3.0 * t * t * (p3 - p2)
    )


def _solve_t(u: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    """Solve ``x(t) = u`` for ``t`` on a segment normalized to the unit interval.

    The caller maps the segment's four control x-values so the endpoints are
    ``0`` and ``1``; ``u1`` and ``u2`` are the interior control x-values under
    that same map and ``u`` is the target. Normalizing is what makes a fixed
    absolute tolerance meaningful: ``x`` is an abstract quantity here and might
    span ``[0, 1]`` or ``[0, 1e6]``, so a tolerance stated in the caller's units
    would be either unreachable or useless depending on the domain.

    **Convergence is measured in ``t``, never in ``x``.** ``_T_TOLERANCE =
    1e-12`` is an absolute bound on the parameter within the unit interval.
    Stopping on the x-residual instead is tempting and wrong: wherever an author
    crowds a handle against a key the segment is near-vertical, ``dx/dt`` is
    tiny, and an x-residual of ``1e-10`` corresponds to a ``t`` — and therefore a
    ``y`` — that is out by several orders of magnitude more. Solving ``t`` to
    ``1e-12`` bounds the error in the value actually returned at
    ``1e-12 * |dy/dt|``, which is what the caller cares about.

    Newton-Raphson runs at most ``_NEWTON_ITERATIONS = 8`` passes, seeded with
    ``t = u`` — the naive x-fraction, which is the wrong *answer* but a starting
    guess within a few percent for any sane handle placement, so eight passes of
    quadratic convergence is generous. An element converges when its step falls
    to ``_T_TOLERANCE``, and is handed to bisection if the derivative goes flat
    (``|x'(t)| <= _MIN_SLOPE``, where the step would explode) or the step lands
    outside ``[0, 1]``.

    Bisection then runs at most ``_BISECTION_ITERATIONS = 64`` passes over
    ``[0, 1]``, stopping when the bracket narrows below ``_T_TOLERANCE`` — about
    40 halvings. It cannot fail to bracket: the target is clamped into
    ``[0, 1]`` and ``x(0) = 0``, ``x(1) = 1``, so a sign change is always
    enclosed. The cap is a hard safety bound, not a working limit.

    The interval endpoints are answered directly rather than iterated. ``t = 0``
    and ``t = 1`` are exact roots for ``u = 0`` and ``u = 1``, and short-cutting
    them is what guarantees that sampling at a key's own ``x`` returns that key's
    ``y`` bit-for-bit even when a handle has zero x-extent and the derivative
    there is zero.

    Handles whose x-component falls outside the segment make ``x(t)``
    non-monotone, and ``y`` is then not a function of ``x`` at all. This routine
    stays total and deterministic in that case — bisection converges on one of
    the roots — but the curve is the author's problem, not the solver's.

    The whole routine is masked rather than short-circuited, so a single-element
    call performs exactly the same arithmetic as a thousand-element one. That is
    what lets :meth:`Curve1D.sample` be defined as a one-element
    :meth:`Curve1D.sample_many` and still agree bit-for-bit.
    """
    target = np.clip(u, 0.0, 1.0)
    t = target.copy()
    pending = (target > 0.0) & (target < 1.0)

    for _ in range(_NEWTON_ITERATIONS):
        if not pending.any():
            return t
        err = _bezier(t, 0.0, u1, u2, 1.0) - target
        slope = _bezier_slope(t, 0.0, u1, u2, 1.0)
        usable = np.abs(slope) > _MIN_SLOPE
        step = err / np.where(usable, slope, 1.0)
        candidate = t - step
        take = pending & usable & (candidate >= 0.0) & (candidate <= 1.0)
        t = np.where(take, candidate, t)
        pending &= ~(take & (np.abs(step) <= _T_TOLERANCE))

    lo = np.zeros(target.shape)
    hi = np.ones(target.shape)
    for _ in range(_BISECTION_ITERATIONS):
        if not pending.any():
            break
        mid = 0.5 * (lo + hi)
        undershoot = _bezier(mid, 0.0, u1, u2, 1.0) < target
        t = np.where(pending, mid, t)
        lo = np.where(pending & undershoot, mid, lo)
        hi = np.where(pending & ~undershoot, mid, hi)
        pending &= (hi - lo) > _T_TOLERANCE

    return t


@dataclass(frozen=True)
class Key:
    """One keyframe: a point the curve passes through, plus two Bezier handles.

    Handles are **absolute positions in curve space**, not offsets from the key.
    That is the same convention a renderer needs for ``cubicTo`` and the same one
    the editor drags, so no code path has to remember which representation it is
    holding. :meth:`translated` exists precisely so that "absolute" does not turn
    handle-follow into a manual bookkeeping chore at every call site.

    ``None`` means *auto*: the handle sits one third of the way toward the
    neighbouring key along the straight chord, and is re-derived whenever that
    neighbour moves. Two auto handles make the control polygon exactly collinear
    at thirds, which is exact linear interpolation — that is what
    :meth:`Curve1D.linear` relies on. An auto handle on the outermost side of a
    terminal key collapses onto the key itself, where it has no effect.

    Frozen, so a :class:`Curve1D` can hand out its keys without a caller being
    able to mutate one behind the curve's back and invalidate its sort order or
    its cached arrays.
    """

    x: float
    y: float
    in_handle: tuple[float, float] | None = None
    out_handle: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "y", float(self.y))
        for name in ("in_handle", "out_handle"):
            handle = getattr(self, name)
            if handle is not None:
                object.__setattr__(self, name, (float(handle[0]), float(handle[1])))

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def in_offset(self) -> tuple[float, float] | None:
        """In-handle relative to the key, or ``None`` if the handle is auto."""
        if self.in_handle is None:
            return None
        return (self.in_handle[0] - self.x, self.in_handle[1] - self.y)

    @property
    def out_offset(self) -> tuple[float, float] | None:
        """Out-handle relative to the key, or ``None`` if the handle is auto."""
        if self.out_handle is None:
            return None
        return (self.out_handle[0] - self.x, self.out_handle[1] - self.y)

    def translated(self, dx: float, dy: float) -> Key:
        """Return a copy displaced by ``(dx, dy)``, handles carried rigidly.

        Both handles move by the same delta as the key, so their offsets survive
        unchanged. Auto handles stay auto and re-derive themselves from their new
        neighbours.
        """
        return Key(
            x=self.x + dx,
            y=self.y + dy,
            in_handle=(
                None
                if self.in_handle is None
                else (self.in_handle[0] + dx, self.in_handle[1] + dy)
            ),
            out_handle=(
                None
                if self.out_handle is None
                else (self.out_handle[0] + dx, self.out_handle[1] + dy)
            ),
        )


class Curve1D:
    """A piecewise-cubic-Bezier function of one variable.

    Keys are kept sorted and strictly increasing in ``x`` at all times; every
    mutator re-establishes that before returning. Sampling outside
    :meth:`domain` clamps to the first or last key's ``y`` rather than
    extrapolating, because every intended use — taper past the tail, gait phase
    before the cycle starts — wants a held value there, not a cubic running off
    to infinity.

    ``y_range`` is an optional output clamp for callers whose codomain is
    constrained: a radius curve must stay positive no matter where an author
    drags a handle. It applies at read time only, so authored key values survive
    a clamp being tightened and then loosened again. The default is ``None``,
    meaning the curve reports exactly what the geometry says.
    """

    def __init__(
        self,
        keys: Iterable[Key] | None = None,
        y_range: tuple[float, float] | None = None,
    ) -> None:
        self._keys: list[Key] = list(keys) if keys is not None else []
        self._cache: tuple[np.ndarray, ...] | None = None
        self._y_range: tuple[float, float] | None = None
        self.y_range = y_range
        self._commit()

    # ------------------------------------------------------------------
    # convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def linear(
        cls,
        x0: float = 0.0,
        y0: float = 0.0,
        x1: float = 1.0,
        y1: float = 1.0,
        y_range: tuple[float, float] | None = None,
    ) -> Curve1D:
        """A straight line between two points, exact to floating point.

        Both keys take auto handles, which land on the chord at thirds and
        therefore degenerate the cubic to the line through the endpoints.
        """
        return cls([Key(x0, y0), Key(x1, y1)], y_range=y_range)

    @classmethod
    def constant(cls, y: float, y_range: tuple[float, float] | None = None) -> Curve1D:
        """A single key, so every ``x`` clamps to the same ``y``."""
        return cls([Key(0.0, y)], y_range=y_range)

    # ------------------------------------------------------------------
    # inspection
    # ------------------------------------------------------------------

    @property
    def keys(self) -> list[Key]:
        """The keys, sorted by ``x``. A fresh list of immutable records."""
        return list(self._keys)

    @property
    def y_range(self) -> tuple[float, float] | None:
        return self._y_range

    @y_range.setter
    def y_range(self, value: tuple[float, float] | None) -> None:
        if value is not None:
            lo, hi = float(value[0]), float(value[1])
            if lo > hi:
                raise ValueError(f"y_range lower bound exceeds upper: {value}")
            value = (lo, hi)
        self._y_range = value

    def domain(self) -> tuple[float, float]:
        """``(first_x, last_x)``, or ``(0.0, 0.0)`` when there are no keys."""
        if not self._keys:
            return (0.0, 0.0)
        return (self._keys[0].x, self._keys[-1].x)

    def segment_control_points(
        self, index: int
    ) -> tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]:
        """The four control points of segment ``index``, auto handles resolved.

        Ordered ``(P0, P1, P2, P3)`` = ``(key[i].pos, key[i].out_handle,
        key[i+1].in_handle, key[i+1].pos)`` — ready to hand straight to a
        ``cubicTo``. Both the renderer and :meth:`sample` must source the tuple
        from here; the defect this class exists to fix was two call sites each
        assembling it independently and disagreeing about ``P2``.
        """
        count = len(self._keys)
        if count < 2:
            raise IndexError(f"curve has {count} key(s); no segments exist")
        if not 0 <= index < count - 1:
            raise IndexError(f"segment {index} out of range (0..{count - 2})")

        kx, ky, out, inn = self._arrays()
        return (
            (float(kx[index]), float(ky[index])),
            (float(out[index, 0]), float(out[index, 1])),
            (float(inn[index + 1, 0]), float(inn[index + 1, 1])),
            (float(kx[index + 1]), float(ky[index + 1])),
        )

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------

    def sample(self, x: float) -> float:
        """Evaluate the curve at ``x``.

        Delegates to :meth:`sample_many` rather than duplicating the solve, so
        the scalar and vector paths cannot drift apart — they are the same
        arithmetic on an array of length one, and therefore agree bit-for-bit
        rather than merely to within a tolerance. A second hand-written scalar
        solve is precisely the shape of the bug this module was written to fix.

        The price is numpy dispatch overhead, which dominates entirely at one
        element: a scalar call costs on the order of a hundred microseconds
        against a quarter of a microsecond per element through
        :meth:`sample_many`. Anything in a per-tick loop should batch.
        """
        return float(self.sample_many((x,))[0])

    def sample_many(self, xs) -> np.ndarray:
        """Evaluate the curve at every ``x`` in ``xs``, shape preserved.

        Vectorized end to end: one binary search to locate segments, one gather
        of control points, one masked parameter solve across all inputs.
        """
        xs = np.asarray(xs, dtype=np.float64)
        count = len(self._keys)

        if count == 0:
            return self._clamp_y(np.zeros(xs.shape, dtype=np.float64))

        kx, ky, out, inn = self._arrays()
        if count == 1:
            return self._clamp_y(np.full(xs.shape, ky[0], dtype=np.float64))

        flat = np.atleast_1d(xs).ravel()
        seg = np.clip(np.searchsorted(kx, flat, side="right") - 1, 0, count - 2)

        p0x, p3x = kx[seg], kx[seg + 1]
        p1x, p2x = out[seg, 0], inn[seg + 1, 0]
        p0y, p3y = ky[seg], ky[seg + 1]
        p1y, p2y = out[seg, 1], inn[seg + 1, 1]

        width = p3x - p0x
        u = np.clip((flat - p0x) / width, 0.0, 1.0)
        t = _solve_t(u, (p1x - p0x) / width, (p2x - p0x) / width)
        y = _bezier(t, p0y, p1y, p2y, p3y)

        y = np.where(flat <= kx[0], ky[0], y)
        y = np.where(flat >= kx[-1], ky[-1], y)

        return self._clamp_y(y.reshape(xs.shape))

    # ------------------------------------------------------------------
    # mutation
    # ------------------------------------------------------------------

    def add_key(
        self,
        x: float | None = None,
        y: float | None = None,
        in_handle: tuple[float, float] | None = None,
        out_handle: tuple[float, float] | None = None,
    ) -> int:
        """Insert a key and return its index.

        ``x`` defaults to the midpoint of the widest existing segment, which is
        the sane reading of an editor's bare "add key" button. The old widget
        hardcoded a single scene coordinate there, so repeated presses stacked
        every new point on top of the last.

        With ``x`` inside the domain and neither ``y`` nor a handle specified,
        the containing segment is split by de Casteljau at the solved parameter.
        That is exact: the sampled curve is unchanged everywhere, the neighbours'
        facing handles are rewritten to preserve their halves, and the author
        gets a new control point without the shape moving under them. The key
        lands at exactly the requested ``x``, which perturbs the split by at most
        the parameter-solve tolerance.

        Supplying ``y`` or a handle means the caller is asking for a specific
        shape, so the key is inserted plainly and the curve does change. Outside
        the domain the curve is extended, and an omitted ``y`` picks up the
        clamped end value.
        """
        if x is None:
            x = self._default_insert_x()
        x = float(x)

        if y is None and in_handle is None and out_handle is None:
            index = self._splittable_segment(x)
            if index is not None:
                return self._split_segment(index, x)

        if y is None:
            y = self.sample(x)

        key = Key(x, y, in_handle=in_handle, out_handle=out_handle)
        self._keys.append(key)
        return self._commit(key)

    def remove_key(self, index: int) -> None:
        """Delete a key.

        Neighbouring handles are left exactly as authored, so removing a key
        with explicit handles changes the shape. Re-fitting them would be a
        guess, and a curve editor's undo stack is the right place for that
        decision.
        """
        del self._keys[self._resolve_index(index)]
        self._commit()

    def move_key(self, index: int, x: float, y: float) -> int:
        """Move a key to ``(x, y)`` and return its index after re-sorting.

        Handles travel with the key, preserving their offsets exactly. Dragging
        a key past a neighbour reorders the list, which is why the new index is
        returned — an editor holding the old one would otherwise start moving
        somebody else's key.
        """
        position = self._resolve_index(index)
        key = self._keys[position]
        moved = key.translated(float(x) - key.x, float(y) - key.y)
        self._keys[position] = moved
        return self._commit(moved)

    def set_handles(
        self,
        index: int,
        in_handle: tuple[float, float] | None = _KEEP,
        out_handle: tuple[float, float] | None = _KEEP,
    ) -> None:
        """Replace a key's handles. Passing ``None`` reverts one to auto.

        Omitting an argument leaves that handle alone; ``None`` is a meaningful
        value here, so the two cases cannot share a default.
        """
        position = self._resolve_index(index)
        changes: dict[str, Any] = {}
        if in_handle is not _KEEP:
            changes["in_handle"] = in_handle
        if out_handle is not _KEEP:
            changes["out_handle"] = out_handle
        if changes:
            self._keys[position] = replace(self._keys[position], **changes)
            self._cache = None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _commit(self, anchor: Key | None = None) -> int:
        """Restore the sort and separation invariants; locate ``anchor``.

        The sort settles first so the anchor's index can be read off by
        identity; the separation sweep only ever substitutes a translated copy
        at the same list position, so that index survives it.
        """
        self._keys.sort(key=lambda k: k.x)
        index = -1
        if anchor is not None:
            index = next(i for i, k in enumerate(self._keys) if k is anchor)
        self._separate()
        self._cache = None
        return index

    def _separate(self) -> None:
        for i in range(1, len(self._keys)):
            previous = self._keys[i - 1].x
            if self._keys[i].x - previous >= X_EPSILON:
                continue
            bumped = previous + X_EPSILON
            if bumped <= previous:
                bumped = math.nextafter(previous, math.inf)
            self._keys[i] = self._keys[i].translated(bumped - self._keys[i].x, 0.0)

    def _resolve_index(self, index: int) -> int:
        count = len(self._keys)
        position = index + count if index < 0 else index
        if not 0 <= position < count:
            raise IndexError(f"key index {index} out of range (n={count})")
        return position

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._cache is None:
            count = len(self._keys)
            kx = np.array([k.x for k in self._keys], dtype=np.float64)
            ky = np.array([k.y for k in self._keys], dtype=np.float64)
            out = np.empty((count, 2), dtype=np.float64)
            inn = np.empty((count, 2), dtype=np.float64)
            for i, key in enumerate(self._keys):
                out[i] = (
                    key.out_handle if key.out_handle is not None else self._auto_out(i)
                )
                inn[i] = (
                    key.in_handle if key.in_handle is not None else self._auto_in(i)
                )
            self._cache = (kx, ky, out, inn)
        return self._cache  # type: ignore[return-value]

    def _auto_out(self, index: int) -> tuple[float, float]:
        key = self._keys[index]
        if index + 1 >= len(self._keys):
            return key.pos
        nxt = self._keys[index + 1]
        return (key.x + (nxt.x - key.x) / 3.0, key.y + (nxt.y - key.y) / 3.0)

    def _auto_in(self, index: int) -> tuple[float, float]:
        key = self._keys[index]
        if index == 0:
            return key.pos
        prv = self._keys[index - 1]
        return (key.x - (key.x - prv.x) / 3.0, key.y - (key.y - prv.y) / 3.0)

    def _clamp_y(self, y: np.ndarray) -> np.ndarray:
        if self._y_range is None:
            return y
        return np.clip(y, self._y_range[0], self._y_range[1])

    def _default_insert_x(self) -> float:
        if not self._keys:
            return 0.0
        if len(self._keys) == 1:
            return self._keys[0].x + 1.0
        widest = max(
            range(len(self._keys) - 1),
            key=lambda i: self._keys[i + 1].x - self._keys[i].x,
        )
        return 0.5 * (self._keys[widest].x + self._keys[widest + 1].x)

    def _splittable_segment(self, x: float) -> int | None:
        """Index of the segment strictly containing ``x``, with room to insert."""
        if len(self._keys) < 2:
            return None
        for i in range(len(self._keys) - 1):
            lo, hi = self._keys[i].x, self._keys[i + 1].x
            if lo + X_EPSILON <= x <= hi - X_EPSILON:
                return i
        return None

    def _split_segment(self, index: int, x: float) -> int:
        p0, p1, p2, p3 = self.segment_control_points(index)
        width = p3[0] - p0[0]
        u = np.clip(np.array([(x - p0[0]) / width]), 0.0, 1.0)
        t = float(
            _solve_t(
                u,
                np.array([(p1[0] - p0[0]) / width]),
                np.array([(p2[0] - p0[0]) / width]),
            )[0]
        )

        a = _lerp2(p0, p1, t)
        b = _lerp2(p1, p2, t)
        c = _lerp2(p2, p3, t)
        d = _lerp2(a, b, t)
        e = _lerp2(b, c, t)
        midpoint = _lerp2(d, e, t)

        self._keys[index] = replace(self._keys[index], out_handle=a)
        self._keys[index + 1] = replace(self._keys[index + 1], in_handle=c)
        inserted = Key(x, midpoint[1], in_handle=d, out_handle=e)
        self._keys.insert(index + 1, inserted)
        return self._commit(inserted)

    def __len__(self) -> int:
        return len(self._keys)

    def __repr__(self) -> str:
        lo, hi = self.domain()
        return f"<Curve1D keys={len(self._keys)} domain=({lo:g}, {hi:g}) y_range={self._y_range}>"


def _lerp2(
    a: tuple[float, float], b: tuple[float, float], t: float
) -> tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
