"""Two-bone IK, and the state machine that decides where a foot should be.

A limb that merely follows a target is a rubber band. What makes it read as a
leg is *commitment*: the foot picks a point in the world, holds it while the
body travels over it, and only then lets go and swings somewhere new. Rain World
spends a whole enum on exactly this — ``Limb.mode`` is ``HuntAbsolutePosition``,
``Reaching`` or ``Dangle`` — and it is the modes, not the IK, that separate
walking from flailing. IK is the easy half.

So this module is two pieces with a deliberate seam between them.

:func:`two_bone_ik` is pure geometry: a root, a target and two bone lengths in,
elbow and tip positions out. :class:`LimbIK` runs it once per tick and feeds the
result to :meth:`ControlContext.seek`. It never writes a position. The elbow and
tip are *targets* expressed as force, so the constraint projection that runs
immediately afterward gets to overrule the IK rather than fight it — a limb
reaching past its own span simply comes up short, which is the correct answer
and costs nothing to arrange. Solving for a pose that already satisfies both
bone lengths is what keeps the seek and the projection pulling the same way
instead of buzzing against each other.

:class:`FootholdPicker` decides what the tip should aim at. The two meet at
:class:`LimbFrame`: a four-field snapshot of where the shoulder is, where the
foot currently is, how fast the shoulder is travelling, and how far the limb can
stretch. That is the entire seam. The picker never learns a chunk id, never sees
the body and never touches the solver; the IK never learns that footholds exist.
Anything else satisfying :class:`TipTarget` — a hand reaching for a carried
object, a scripted path, a test stub — drops into the same slot. The frame is
also why exactly one place has to guard against a chunk the editor deleted: no
frame, no target request, no stale lookup anywhere downstream.

**The state machine.** A picker is always in one of three modes.

*Planted* holds a fixed world point and emits it unchanged every tick. The foot
does not move while the body does, which is the whole reason this class exists.
It leaves when the foothold is out of position — either further than
``max_reach`` from the limb root, or more than ``step_threshold`` from where the
foot *ought* to be standing now, which when walking forward means the plant has
fallen that far behind.

*Reaching* is the swing. On entry a new foothold is chosen ahead of the body:
the neutral stance offset from the root, displaced along the root's velocity by
``lead_time``, and dropped onto the ground plane. The emitted target sweeps from
the old foothold to the new one over ``step_duration`` seconds, lifted by
``step_height`` on a half-sine so the foot arcs over the ground rather than
dragging through it. When the sweep completes the new point becomes the plant.

*Dangling* is the honest failure. If the candidate foothold is itself out of
reach — no ground under the limb, or the body too high above it — the picker
emits ``None``, :class:`LimbIK` contributes no force at all, and gravity gets
the limb. It re-tests every tick and swings back up from wherever the foot is
hanging as soon as something is reachable.

**Coordination.** ``period`` and ``phase`` gate when a step may *start*. Each
picker owns the slice ``[phase, phase + duty)`` of the gait cycle and may only
begin a swing early enough in its own slice for the whole swing to finish
inside it. Two legs at ``phase=0.0`` and ``phase=0.5`` therefore own disjoint
halves of the cycle and can never be swinging at the same time, which is what
keeps a walker supported. ``period=0`` disables the gate entirely and every
picker steps the moment it wants to, which is the right default for a single
limb and the wrong one for four.

Gait timing reads :attr:`ControlContext.time` rather than counting ticks, so
:meth:`ControllerStack.reset` rewinds the gait for free. The plant and the mode
are genuinely stateful and cannot be derived from a clock; :meth:`reset` exists
for those.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol, runtime_checkable

import numpy as np

from mesh_xml.anim.controller import ControlContext
from mesh_xml.core.body import Body

__all__ = [
    "FootholdPicker",
    "LimbFrame",
    "LimbIK",
    "LimbMode",
    "TipTarget",
    "two_bone_ik",
]

_EPS = 1e-6

_UP = 1
"""Index of the vertical axis, matching ``solver._collide_ground``."""

_REACH_MARGIN = 0.95
"""Fraction of a limb's full span it will commit to, when not told otherwise.

Planting at the exact span leaves a limb permanently straight and shaking, since
the smallest body motion puts the plant out of reach.
"""


@dataclass(frozen=True, eq=False)
class LimbFrame:
    """One tick's worth of a limb's situation, with nothing rig-specific in it.

    The seam between :class:`LimbIK` and whatever supplies its target. Built by
    the IK from chunk ids it owns, consumed by a :class:`TipTarget` that has
    never heard of chunks.

    Frozen because it is a snapshot; ``eq=False`` because a generated ``__eq__``
    over numpy fields returns an array and raises the moment anything treats it
    as a truth value.
    """

    root: np.ndarray
    tip: np.ndarray
    velocity: np.ndarray
    reach: float


@runtime_checkable
class TipTarget(Protocol):
    """Something that decides where a limb's tip should be aiming this tick.

    Returning ``None`` means "no opinion": the limb contributes no force and
    falls under gravity, which is what a dangling leg should do.
    """

    def tip_target(
        self, ctx: ControlContext, limb: LimbFrame
    ) -> np.ndarray | None: ...


def two_bone_ik(
    root,
    target,
    length_a: float,
    length_b: float,
    pole=None,
    flip: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a two-bone chain analytically. Returns ``(mid, tip)`` world points.

    The elbow lies on a circle around the root-to-target axis, so a pose is only
    determined once something picks a point on that circle. ``pole`` is that
    something: the elbow is placed in the half-plane the pole points into, and
    ``flip`` mirrors it to the other side. One vector plus one sign is Rain
    World's whole answer to limb orientation, and it is why a left and a right
    leg built from the same template bend away from each other rather than both
    the same way.

    Three degeneracies, each resolved rather than guarded against:

    * **Target beyond full extension.** The distance is clamped to
      ``length_a + length_b``, which drives the elbow height to exactly zero:
      the limb points straight at the target and stops at its own length. The
      same clamp at the other end, ``abs(length_a - length_b)``, folds it.
    * **Target on the root.** There is no axis to build a pose around, so the
      first basis vector is used — the same arbitrary-but-stable choice
      ``solver._project`` makes for coincident chunks, for the same reason: a
      wrong direction recovers on the next tick, no direction never does.
    * **Pole parallel to the limb axis.** Its perpendicular component vanishes
      and cannot say which way to bend, so a deterministic perpendicular of the
      axis is substituted. A zero-length or wrong-dimensional pole lands here
      too, which is how a 2-D body gets a sensible bend from the 3-D default.
    """
    root = np.asarray(root, dtype=np.float32)
    dim = int(root.shape[0])
    target = _fit(target, dim)
    length_a = max(float(length_a), 0.0)
    length_b = max(float(length_b), 0.0)

    axis, distance = _normalized(target - root)
    if distance < _EPS:
        axis = np.zeros(dim, dtype=np.float32)
        axis[0] = 1.0

    span = length_a + length_b
    fold = abs(length_a - length_b)
    reachable = min(max(distance, fold), span)

    if reachable > _EPS:
        along = (length_a**2 - length_b**2 + reachable**2) / (2.0 * reachable)
    else:
        along = 0.0
    height = np.sqrt(max(length_a**2 - along**2, 0.0))

    pole_vector = _fit(pole, dim) if pole is not None else np.zeros(dim, np.float32)
    bend, residual = _normalized(pole_vector - axis * np.dot(pole_vector, axis))
    if residual < _EPS:
        bend = _perpendicular(axis)
    if flip < 0.0:
        bend = -bend

    mid = root + axis * np.float32(along) + bend * np.float32(height)
    tip = root + axis * np.float32(reachable)
    return mid.astype(np.float32), tip.astype(np.float32)


@dataclass(eq=False)
class LimbIK:
    """Drives a ``(root, mid, tip)`` chunk chain toward a tip target with force.

    ``target`` is either a :class:`TipTarget` — usually a
    :class:`FootholdPicker` — or a fixed world point, or ``None`` for a limb
    that is temporarily off.

    Only ``mid`` and ``tip`` are pushed. The root is the limb's attachment and
    belongs to whatever moves the body; a controller that also shoved the root
    would be quietly steering the creature from inside its knee.

    ``mid_weight`` scales the elbow's spring relative to the tip's. The elbow is
    a hint about *how* to bend and the tip is the actual goal, so giving them
    equal authority makes a limb that fights itself whenever the constraint
    solver disagrees with the pole.

    ``lengths`` overrides the bone lengths, which are otherwise taken from the
    rest lengths of the ``root-mid`` and ``mid-tip`` links — the authored bone
    lengths, not the current squashed separations, which is what keeps the
    solved pose from chasing its own compression.
    """

    root: int
    mid: int
    tip: int
    target: TipTarget | Sequence[float] | np.ndarray | None = None
    pole: Sequence[float] = (0.0, 0.0, 1.0)
    flip: float = 1.0
    lengths: tuple[float, float] | None = None
    strength: float = 200.0
    damping: float = 14.0
    mid_weight: float = 0.35
    max_force: float | None = 500.0
    enabled: bool = True

    def update(self, ctx: ControlContext) -> None:
        frame = self.frame(ctx)
        if frame is None:
            return

        goal = self.resolve(ctx, frame)
        if goal is None:
            return

        length_a, length_b = self.bone_lengths(ctx.body)
        mid_goal, tip_goal = two_bone_ik(
            frame.root, goal, length_a, length_b, self.pole, self.flip
        )
        ctx.seek(
            self.mid,
            mid_goal,
            self.strength * self.mid_weight,
            self.damping * self.mid_weight,
            self.max_force,
        )
        ctx.seek(self.tip, tip_goal, self.strength, self.damping, self.max_force)

    def frame(self, ctx: ControlContext) -> LimbFrame | None:
        """Snapshot this limb, or ``None`` if the editor deleted part of it.

        The single point where a stale chunk id is caught. Everything downstream
        works from the frame and so cannot go looking for a chunk that is gone.
        """
        body = ctx.body
        if not all(body.has_chunk(cid) for cid in (self.root, self.mid, self.tip)):
            return None
        return LimbFrame(
            root=ctx.position(self.root),
            tip=ctx.position(self.tip),
            velocity=ctx.velocity(self.root),
            reach=sum(self.bone_lengths(body)),
        )

    def resolve(self, ctx: ControlContext, frame: LimbFrame) -> np.ndarray | None:
        if self.target is None:
            return None
        if isinstance(self.target, TipTarget):
            return self.target.tip_target(ctx, frame)
        return _fit(self.target, ctx.body.dim)

    def bone_lengths(self, body: Body) -> tuple[float, float]:
        if self.lengths is not None:
            return float(self.lengths[0]), float(self.lengths[1])
        return (
            _rest_length(body, self.root, self.mid),
            _rest_length(body, self.mid, self.tip),
        )

    def reset(self) -> None:
        """Rewind the target source, if it has state to rewind."""
        reset = getattr(self.target, "reset", None)
        if callable(reset):
            reset()

    def __repr__(self) -> str:
        return (
            f"<LimbIK {self.root}-{self.mid}-{self.tip} "
            f"flip={self.flip:+.0f} target={type(self.target).__name__}>"
        )


class LimbMode(IntEnum):
    """Rain World's ``Limb.mode``, under names that say what they do."""

    PLANTED = 0
    REACHING = 1
    DANGLING = 2


@dataclass(eq=False)
class FootholdPicker:
    """Commits a limb to a world point and decides when to let go of it.

    Supplies :class:`LimbIK` with a tip target; see the module docstring for the
    transitions and for why the two are joined by :class:`LimbFrame` rather than
    by inheritance or by a shared body reference.

    ``ground_y`` is where a planted foot sits. :class:`ControlContext` carries no
    world geometry — the solver's floor lives in
    :class:`~mesh_xml.core.solver.SolverConfig` and controllers never see it — so
    the floor is handed in here. Pass ``config.ground_y`` plus the tip's radius,
    since that is the height the collision response will actually hold the chunk
    at, and aiming below it means seeking into a floor forever. ``None`` means no
    floor at all: the foothold is planted at whatever height the stance offset
    puts it, which is what a climbing or swimming limb wants.

    ``max_reach`` defaults to :data:`_REACH_MARGIN` of the limb's own span, so a
    rig author does not have to keep two numbers in sync. ``scatter`` jitters
    each new foothold through :attr:`ControlContext.rng`, which is what stops
    two legs on identical templates from landing in perfect lockstep.
    """

    stance: Sequence[float] = (0.0, -1.0, 0.0)
    ground_y: float | None = 0.0
    max_reach: float | None = None
    step_threshold: float = 0.5
    step_height: float = 0.3
    step_duration: float = 0.25
    lead_time: float = 0.3
    scatter: float = 0.0
    period: float = 0.0
    phase: float = 0.0
    duty: float = 0.5

    mode: LimbMode = field(init=False, default=LimbMode.DANGLING)
    foothold: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.step_duration <= 0.0:
            raise ValueError(
                f"step_duration must be positive, got {self.step_duration}"
            )
        if self.step_threshold <= 0.0:
            raise ValueError(
                f"step_threshold must be positive, got {self.step_threshold}"
            )
        if self.step_height < 0.0:
            raise ValueError(
                f"step_height must be non-negative, got {self.step_height}"
            )
        if self.scatter < 0.0:
            raise ValueError(f"scatter must be non-negative, got {self.scatter}")
        if not 0.0 < self.duty <= 1.0:
            raise ValueError(f"duty must be in (0, 1], got {self.duty}")
        # Otherwise the swing can never fit inside the limb's slice of the gait
        # and the leg simply never steps again, silently.
        if self.period > 0.0 and self.step_duration > self.duty * self.period:
            raise ValueError(
                f"step_duration {self.step_duration} does not fit in a "
                f"{self.duty} duty of period {self.period}"
            )
        self._swing_from = np.zeros(0, dtype=np.float32)
        self._swing_to = np.zeros(0, dtype=np.float32)
        self._step_start = 0.0

    def tip_target(self, ctx: ControlContext, limb: LimbFrame) -> np.ndarray | None:
        if self.mode is LimbMode.REACHING:
            return self._swing(ctx.time)

        reach = self._reach(limb)
        if self.mode is LimbMode.PLANTED and not self._out_of_position(limb, reach):
            return self.foothold

        if not self._window_open(ctx.time):
            return self.foothold

        candidate = self._candidate(ctx, limb)
        if float(np.linalg.norm(candidate - limb.root)) > reach:
            self.mode = LimbMode.DANGLING
            self.foothold = None
            return None

        origin = limb.tip if self.foothold is None else self.foothold
        self._swing_from = np.array(origin, dtype=np.float32)
        self._swing_to = candidate
        self._step_start = ctx.time
        self.foothold = None
        self.mode = LimbMode.REACHING
        return self._swing(ctx.time)

    def reset(self) -> None:
        """Forget the plant and the mode. The gait clock rewinds with ``ctx.time``."""
        self.mode = LimbMode.DANGLING
        self.foothold = None
        self._swing_from = np.zeros(0, dtype=np.float32)
        self._swing_to = np.zeros(0, dtype=np.float32)
        self._step_start = 0.0

    def _swing(self, time: float) -> np.ndarray:
        progress = (time - self._step_start) / self.step_duration
        if progress >= 1.0:
            self.foothold = self._swing_to
            self.mode = LimbMode.PLANTED
            return self.foothold

        progress = max(progress, 0.0)
        point = self._swing_from + (self._swing_to - self._swing_from) * np.float32(
            progress
        )
        if point.size > _UP:
            point[_UP] += np.float32(self.step_height * np.sin(np.pi * progress))
        return point

    def _out_of_position(self, limb: LimbFrame, reach: float) -> bool:
        if self.foothold is None:
            return True
        if float(np.linalg.norm(self.foothold - limb.root)) > reach:
            return True
        drift = float(np.linalg.norm(self.foothold - self._neutral(limb)))
        return drift > self.step_threshold

    def _neutral(self, limb: LimbFrame) -> np.ndarray:
        """Where the foot ought to be standing right now, ignoring travel."""
        return self._on_ground(limb.root + _fit(self.stance, limb.root.shape[0]))

    def _candidate(self, ctx: ControlContext, limb: LimbFrame) -> np.ndarray:
        point = self._neutral(limb) + limb.velocity * np.float32(self.lead_time)
        if self.scatter > 0.0:
            point = point + ctx.rng.normal(0.0, self.scatter, size=point.shape).astype(
                np.float32
            )
        return self._on_ground(point)

    def _on_ground(self, point: np.ndarray) -> np.ndarray:
        out = np.array(point, dtype=np.float32)
        if self.ground_y is not None and out.size > _UP:
            out[_UP] = np.float32(self.ground_y)
        return out

    def _reach(self, limb: LimbFrame) -> float:
        if self.max_reach is not None:
            return self.max_reach
        return limb.reach * _REACH_MARGIN

    def _window_open(self, time: float) -> bool:
        if self.period <= 0.0:
            return True
        position = ((time / self.period) - self.phase) % 1.0
        remaining = (self.duty - position) * self.period
        return position < self.duty and remaining >= self.step_duration

    def __repr__(self) -> str:
        if self.foothold is None:
            plant = "none"
        else:
            plant = np.array2string(self.foothold, precision=2)
        return f"<FootholdPicker {self.mode.name.lower()} at {plant}>"


def _rest_length(body: Body, id_a: int, id_b: int) -> float:
    if body.has_link(id_a, id_b):
        return float(body.link(id_a, id_b).rest)
    return float(np.linalg.norm(body.pos[body.slot(id_a)] - body.pos[body.slot(id_b)]))


def _fit(vector, dim: int) -> np.ndarray:
    """Trim or zero-pad to ``dim``, as ``solver._gravity_vector`` does.

    ``core`` is dimension-agnostic, so a 3-D pole or stance authored once has to
    survive being handed to a 2-D rig without the caller reconfiguring it.
    """
    out = np.zeros(dim, dtype=np.float32)
    values = np.asarray(vector, dtype=np.float32).ravel()
    shared = min(values.size, dim)
    out[:shared] = values[:shared]
    return out


def _normalized(vector: np.ndarray) -> tuple[np.ndarray, float]:
    length = float(np.linalg.norm(vector))
    if length < _EPS:
        return np.zeros_like(vector), 0.0
    return (vector / np.float32(length)).astype(np.float32), length


def _perpendicular(axis: np.ndarray) -> np.ndarray:
    """A unit vector orthogonal to ``axis``, chosen the same way every time.

    The least-aligned basis vector is never parallel to a unit axis, so the
    Gram-Schmidt residual is always well clear of zero and no second fallback is
    needed.
    """
    basis = np.zeros_like(axis)
    basis[int(np.argmin(np.abs(axis)))] = 1.0
    unit, _ = _normalized(basis - axis * np.dot(basis, axis))
    return unit
