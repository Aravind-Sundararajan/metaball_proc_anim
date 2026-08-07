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
``lead_time``, and dropped onto the ground — flattened to the ``ground_y``
plane, or projected onto the terrain's surface when the picker has a world to
stand on. The emitted target sweeps from
the old foothold to the new one over ``step_duration`` seconds, lifted by
``step_height`` on a half-sine so the foot arcs over the ground rather than
dragging through it. When the sweep completes the new point becomes the plant.

*Dangling* is the honest failure. If the candidate foothold is itself out of
reach — no ground under the limb, or the body too high above it — the picker
emits ``None``, :class:`LimbIK` contributes no force at all, and gravity gets
the limb. On a terrain this is a statement about the world rather than about
the rig: the candidate arrives already projected onto the nearest surface, so
a candidate still out of reach means the world offers nothing in range — a gap
under the limb, a ledge beyond it — and the leg hangs into it honestly. It
re-tests every tick and swings back up from wherever the foot is hanging as
soon as something is reachable.

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

**Propulsion.** Everything above describes a limb that reaches. Two opt-in
fields turn it into a limb that *pushes*, which is a different thing and worth
stating plainly, because a creature whose body is carried along a path and whose
feet merely keep up is being dragged, not walking.

:attr:`LimbIK.reaction` pays the seek back. The IK pushes mid and tip; with a
reaction the root is pushed by the exact negative of that sum, so the limb's
contribution to the tick nets to zero and the limb becomes a closed actuator
between the root and the foot rather than a thruster hanging off it.

:attr:`FootholdPicker.drive` is what gives the actuator something to do. A
planted foothold retreats through the world at ``-drive`` while it is held, so
the limb is continuously pushing its own foot backward. That is a stance phase:
the foot presses back, the ground's friction refuses to let it slide, and the
reaction on the root carries the body forward over it.

The momentum bookkeeping is the whole argument, and it is short. Gravity is
vertical, constraint projection is internal, and a limb with ``reaction=1``
contributes nothing net — so the only external horizontal force on the creature
is ground friction, which can do nothing but *remove* tangential speed from a
chunk it is touching. A body therefore gains forward momentum only by having its
feet carry backward momentum for the floor to absorb. Feet that push backward
while planted is not one way to walk; in this model it is the only way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol, runtime_checkable

import numpy as np

from mesh_xml.anim.controller import ControlContext
from mesh_xml.core.body import Body
from mesh_xml.core.terrain import Terrain

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

    ``tip_radius`` crosses the seam because a foothold is aimed, not touched:
    contact holds a chunk one radius off the surface, so a picker projecting a
    candidate onto a terrain must aim that far off it, and the frame is the
    only channel body facts are allowed to reach the picker through. It
    defaults to zero — a point foot — so a frame built by hand keeps working
    and keeps meaning what it always meant.

    Frozen because it is a snapshot; ``eq=False`` because a generated ``__eq__``
    over numpy fields returns an array and raises the moment anything treats it
    as a truth value.
    """

    root: np.ndarray
    tip: np.ndarray
    velocity: np.ndarray
    reach: float
    tip_radius: float = 0.0


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

    ``mid_weight`` scales the elbow's spring relative to the tip's. The elbow is
    a hint about *how* to bend and the tip is the actual goal, so giving them
    equal authority makes a limb that fights itself whenever the constraint
    solver disagrees with the pole.

    ``lengths`` overrides the bone lengths, which are otherwise taken from the
    rest lengths of the ``root-mid`` and ``mid-tip`` links — the authored bone
    lengths, not the current squashed separations, which is what keeps the
    solved pose from chasing its own compression.

    ``reaction`` is the fraction of the seek that the root is charged for. At
    ``0`` — the default — only ``mid`` and ``tip`` are pushed and the root is
    never touched; at ``1`` the root takes the exact negative of what the other
    two were given.

    That default was the original behaviour and it was a reasonable one. A limb
    whose root is untouched can be aimed anywhere without disturbing the body it
    hangs from, which is exactly what you want when the root is pinned or
    carried: the pin absorbs the reaction, and the limb solves against a hip
    that is already where the animation says it should be. Charging the root
    unconditionally would instead let the IK steer the creature from inside its
    own knee, because a seek is one-sided (see :meth:`ControlContext.seek`) and
    a one-sided force on a free body is a thruster. Point four such limbs at
    footholds and the creature flies.

    The catch, and the reason this field exists, is that the same one-sidedness
    runs the other way too: an unreacted limb also holds *itself* up, so a
    creature standing on four of them levitates on its own IK and its feet never
    load the floor. At ``reaction=1`` the limb's three forces sum to exactly
    zero, the levitation goes, the body's weight reaches the ground through the
    links, and a planted foot pushing backward moves the body forward. That is
    propulsion, and it is unavailable at any other setting.

    The reaction is **not** withheld while the foot swings. It is tempting —
    the swing pushes the body backward, which looks like a loss — but the swing
    force has to go somewhere, and the only alternatives are to invent momentum
    or to leave the leg levitating. What is true is that the reaction is zero
    whenever the limb contributes nothing, which is precisely when the target
    source has no opinion: a :class:`FootholdPicker` with nothing in reach
    returns ``None``, and a dangling limb charges the root nothing.
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
    reaction: float = 0.0
    enabled: bool = True

    #: World points that move with the creature (:mod:`mesh_xml.anim.place`):
    #: a raw point ``target`` travels; a picker there places itself and the
    #: ``pole`` is a direction, which is exactly why it is not listed.
    PLACED = ("target",)

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
        pushed = ctx.seek(
            self.mid,
            mid_goal,
            self.strength * self.mid_weight,
            self.damping * self.mid_weight,
            self.max_force,
        )
        pushed = pushed + ctx.seek(
            self.tip, tip_goal, self.strength, self.damping, self.max_force
        )
        if self.reaction:
            # The clamped forces, not the requested ones, or the pair stops
            # summing to zero exactly when max_force starts to matter.
            ctx.add_force(self.root, -pushed * np.float32(self.reaction))

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
            tip_radius=float(body.radius[body.slot(self.tip)]),
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

    ``terrain`` is the world as a surface rather than as a height. When one is
    in effect, candidates are *projected* onto it — the nearest point sitting
    one tip radius off the surface, reached from either side — instead of being
    flattened to ``ground_y``, and a foothold on a slope or a mound falls out
    of the same machinery that planted one on the plane. The radius rides in on
    :attr:`LimbFrame.tip_radius`, and the offset exists for the same reason
    :meth:`ControlContext.ground_level` adds one: contact holds a chunk a
    radius off the surface, so aiming at the surface itself is aiming
    underground. A set ``terrain`` supersedes ``ground_y``, mirroring the
    precedence :class:`~mesh_xml.core.solver.SolverConfig` documents — the
    picker must aim at the same floor collision will actually hold the foot on.
    Left ``None``, the picker follows :attr:`ControlContext.terrain`, the world
    the solver collides against, which is what lets one creature be dropped
    into any world and stand on it; the field exists to *win* over the context
    when a limb must work a surface the solver does not own. A picker carrying
    its own terrain refuses to serialize, deliberately: a world belongs to the
    document, not to one foot of one creature, and a saved rig reaches its
    world through the context.

    ``max_reach`` defaults to :data:`_REACH_MARGIN` of the limb's own span, so a
    rig author does not have to keep two numbers in sync. ``scatter`` jitters
    each new foothold through :attr:`ControlContext.rng`, which is what stops
    two legs on identical templates from landing in perfect lockstep.

    ``drive`` is the locomotion intent, in world units per second, and ``None``
    — the default — leaves the plant exactly as fixed as it has always been. Set
    it and a held foothold retreats through the world at ``-drive``, so the limb
    spends its stance pushing its own foot backward; new footholds are placed
    that much further ahead to keep the stance centred under the root rather
    than trailing behind it. Paired with :attr:`LimbIK.reaction` this is what
    propels a creature, and without that reaction it is only a foot that
    scuffs. Nothing here reads the body's actual speed, so ``drive`` is a
    command and not a measurement: a creature that cannot achieve it simply
    slips.
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
    drive: Sequence[float] | None = None
    terrain: Terrain | None = None

    mode: LimbMode = field(init=False, default=LimbMode.DANGLING)
    foothold: np.ndarray | None = field(init=False, default=None)

    #: Tempo dimensions (:mod:`mesh_xml.anim.tempo`). Durations divide and the
    #: drive velocity multiplies, so a faster walk is the same strides more
    #: often: the spatial trigger (``step_threshold``) never moves, and the
    #: shrunken ``lead_time`` times the grown velocities predicts the same
    #: distance ahead. ``phase`` and ``duty`` are fractions of the period and
    #: are dimensionless.
    TEMPO = {"step_duration": -1, "lead_time": -1, "period": -1, "drive": 1}

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

    @property
    def planted(self) -> bool:
        """Whether a foot is currently bearing on a committed point."""
        return self.mode is LimbMode.PLANTED and self.foothold is not None

    def tip_target(self, ctx: ControlContext, limb: LimbFrame) -> np.ndarray | None:
        if self.mode is LimbMode.REACHING:
            return self._swing(ctx.time)

        surface = self._surface(ctx)
        self._retreat(ctx, limb, surface)

        reach = self._reach(limb)
        if self.mode is LimbMode.PLANTED and not self._out_of_position(
            limb, reach, surface
        ):
            return self.foothold

        if not self._window_open(ctx.time):
            return self.foothold

        # The candidate arrives already dropped onto the ground, so this
        # distance is measured to a point that really is standable. Failing it
        # therefore means the world offers nothing in range — over a terrain,
        # a gap or a ledge — not merely that an unprojected guess happened to
        # float too far away.
        candidate = self._candidate(ctx, limb, surface)
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

    def _surface(self, ctx: ControlContext) -> Terrain | None:
        """The ground in effect this tick: the picker's own, else the world's.

        The constructor field wins so a rig author can aim a limb at a surface
        the solver does not collide with; left ``None``, the picker follows
        :attr:`ControlContext.terrain` and the creature stands on whatever
        world it was dropped into, with nothing re-wired per limb.
        """
        return self.terrain if self.terrain is not None else ctx.terrain

    def _retreat(
        self, ctx: ControlContext, limb: LimbFrame, surface: Terrain | None
    ) -> None:
        """Slide a held plant backward at ``-drive``. This is the stance phase.

        Reading ``ctx.dt`` rather than the gait clock means the plant travels the
        commanded distance per *second* whatever the tick rate, and means a
        picker whose limb was skipped this tick does not retreat without having
        pushed on anything.

        The slid plant goes back through :meth:`_on_ground` because the retreat
        is straight and a slope is not: each unprojected step would leave the
        held foothold a little further off an incline, and the foot would spend
        its stance being pulled into the hill or off it. Re-projecting keeps
        the plant one tip radius off whatever it stands on — which on the plane
        is the flatten it always was.
        """
        plant = self.foothold
        if self.drive is None or plant is None:
            return
        if self.mode is not LimbMode.PLANTED:
            return
        travel = _fit(self.drive, plant.shape[0]) * np.float32(ctx.dt)
        self.foothold = self._on_ground(plant - travel, surface, limb.tip_radius)

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

    def _out_of_position(
        self, limb: LimbFrame, reach: float, surface: Terrain | None
    ) -> bool:
        if self.foothold is None:
            return True
        if float(np.linalg.norm(self.foothold - limb.root)) > reach:
            return True
        drift = float(np.linalg.norm(self.foothold - self._neutral(limb, surface)))
        return drift > self.step_threshold

    def _neutral(self, limb: LimbFrame, surface: Terrain | None) -> np.ndarray:
        """Where the foot ought to be standing right now, ignoring travel."""
        return self._on_ground(
            limb.root + _fit(self.stance, limb.root.shape[0]),
            surface,
            limb.tip_radius,
        )

    def _candidate(
        self, ctx: ControlContext, limb: LimbFrame, surface: Terrain | None
    ) -> np.ndarray:
        # The command replaces the measurement; it must not add to it. Leading
        # by velocity + drive feeds achieved speed back into stride length,
        # which is positive feedback: on a slipping floor the leak damped it,
        # and on a gripping floor the walker overshot its command by sixty
        # percent (measured open-loop gain 1.6 to 1.8). A driven limb strides
        # for the speed it was asked for; an undriven one still leads the
        # motion it can only observe.
        if self.drive is not None:
            travel = _fit(self.drive, limb.root.shape[0])
        else:
            travel = limb.velocity
        point = self._neutral(limb, surface) + travel * np.float32(self.lead_time)
        if self.scatter > 0.0:
            point = point + ctx.rng.normal(0.0, self.scatter, size=point.shape).astype(
                np.float32
            )
        return self._on_ground(point, surface, limb.tip_radius)

    def _on_ground(
        self, point: np.ndarray, surface: Terrain | None, clearance: float
    ) -> np.ndarray:
        """Drop a desired foothold onto whatever there is to stand on.

        A terrain projects the point to sit ``clearance`` — the tip's radius —
        off the surface, working from either side, and supersedes ``ground_y``
        entirely: the solver gives terrain the same precedence, and aiming at
        any other floor than the one collision enforces is seeking into it, or
        hovering over it, forever. An *empty* terrain projects to nowhere and
        the point keeps its stance height, exactly as
        :meth:`~mesh_xml.core.terrain.Terrain.project` documents — no world
        means no floor, which is ``ground_y=None`` spelled in the new
        vocabulary.
        """
        out = np.array(point, dtype=np.float32)
        if surface is not None and out.size > _UP:
            return surface.project(out, clearance).astype(np.float32)
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
