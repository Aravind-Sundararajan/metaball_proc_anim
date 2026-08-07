"""The behavior tier: brains that decide, controllers that act, a solver that reconciles.

This is the layer above the controller contract, and it obeys the same law one
level up. Controllers write forces, never positions; brains write *goals* —
destinations, targets, choices — never forces and never anything they do not
own. A brain that reached into the solver, or into another creature's
controllers, would make tick order load-bearing, which is the failure the
force accumulator exists to prevent. Rain World layers itself the same way:
creature AI thinks at its own cadence, sets targets, and the body physics
follows without ever being written to directly.

Two brain architectures ship, deliberately different in kind:

* :class:`Grazer` is a **finite state machine** (:class:`Fsm`): a handful of
  named states with explicit transitions. FSMs are the right tool when the
  behaviour is a small set of modes with clear switching rules — which is most
  ambient creatures.
* :class:`Hunter` runs **GOAP** — goal-oriented action planning
  (:func:`plan`): actions with symbolic preconditions and effects, an A*
  search from the perceived world state to a goal, and replanning whenever the
  world stops matching the plan's assumptions. GOAP earns its complexity when
  behaviour should *fall out* of goals rather than be enumerated by hand.

Both are ordinary registered controllers: they sit on the stack, save into v3
documents, grow a panel UI, and tick at 40 Hz while *thinking* on their own
slower ``think_period``. They act through exactly one output — their
creature's pinned anchor, the same kinematic authority :class:`WaveAnchor`
holds and the editor's hand overrides — so a grabbed creature obeys the hand,
not its brain, with no special case anywhere (``ctx.held`` already says so).

Perception runs through ``ctx.peers``, the stack's own controller list: a
brain finds other creatures by scanning peers for a ``species`` tag and an
``anchor`` chunk. Read peers, write yourself; that convention is the whole
protocol.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import count

import numpy as np

from mesh_xml.anim.controller import ControlContext

__all__ = ["Fsm", "GoapAction", "plan", "Grazer", "Hunter"]

#: A destination this close counts as reached — half a typical chunk radius,
#: so arrival does not oscillate across the threshold.
_ARRIVED = 0.25

#: Search cap for the planner. The domains this module ships are a handful of
#: booleans; a search that runs long has been handed an impossible goal, and
#: the honest answer there is "no plan", not a spin.
_MAX_EXPANSIONS = 4096


class Fsm:
    """A finite state machine small enough to trust.

    ``states`` maps a name to a callable; :meth:`step` runs the current one
    and follows the transition it returns (``None`` means stay). States are
    callables rather than classes because every brain here keeps its state
    *data* on itself — the machine only needs to know which behaviour is in
    charge, and a string it validates is exactly that and nothing more.
    """

    def __init__(self, states: Mapping[str, Callable[..., str | None]], initial: str) -> None:
        if not states:
            raise ValueError("an Fsm needs at least one state")
        if initial not in states:
            raise ValueError(f"unknown initial state {initial!r}, expected one of {sorted(states)}")
        self._states = dict(states)
        self.state = initial

    def step(self, *args: object) -> str:
        """Run the current state and follow transitions until one settles.

        The new state acts in the same step that chose it — a machine that
        merely recorded the transition would leave the creature one whole
        think behind its own decision, which at a 4 Hz think rate is a
        visible stutter between "saw the threat" and "started running". A
        chain longer than the state count is two states handing off forever
        within one step, which is a bug in the states, and raising names it.
        """
        for _ in range(len(self._states) + 1):
            outcome = self._states[self.state](*args)
            if outcome is None or outcome == self.state:
                return self.state
            if outcome not in self._states:
                raise ValueError(
                    f"state {self.state!r} transitioned to unknown state {outcome!r}"
                )
            self.state = outcome
        raise RuntimeError(
            f"states {sorted(self._states)} did not settle within one step"
        )


@dataclass(frozen=True)
class GoapAction:
    """One planning operator: do ``name`` when ``pre`` holds and ``eff`` becomes true.

    Symbolic on purpose. The planner never runs an action — it reasons about
    these summaries, and whatever executes the plan maps the chosen name onto
    real steering. Keeping the two apart is what lets the same executor
    survive a re-plan mid-stride.
    """

    name: str
    pre: Mapping[str, bool]
    eff: Mapping[str, bool]
    cost: float = 1.0


def plan(
    state: Mapping[str, bool],
    goal: Mapping[str, bool],
    actions: Sequence[GoapAction],
) -> list[str] | None:
    """A* from ``state`` to ``goal`` over ``actions``. Names in order, or ``None``.

    Facts absent from ``state`` are false. The heuristic is the count of
    unmet goal facts, which is admissible while every action cost is at least
    one — an action can settle at most the facts it writes, and these domains
    write few. Returns ``[]`` when the goal already holds, ``None`` when the
    search space is exhausted or the expansion cap trips: an impossible goal
    is an answer, not an error.
    """
    keys = set(state) | set(goal)
    for action in actions:
        keys |= set(action.pre) | set(action.eff)

    def freeze(facts: Mapping[str, bool]) -> frozenset[tuple[str, bool]]:
        return frozenset((key, bool(facts.get(key, False))) for key in sorted(keys))

    def unmet(facts: frozenset[tuple[str, bool]]) -> int:
        held = dict(facts)
        return sum(1 for key, want in goal.items() if held.get(key, False) != want)

    start = freeze(state)
    tie = count()
    frontier: list[tuple[float, int, frozenset, tuple[str, ...], float]] = [
        (float(unmet(start)), next(tie), start, (), 0.0)
    ]
    best_cost: dict[frozenset, float] = {start: 0.0}

    for _ in range(_MAX_EXPANSIONS):
        if not frontier:
            return None
        _, _, facts, path, spent = heapq.heappop(frontier)
        held = dict(facts)
        if unmet(facts) == 0:
            return list(path)
        for action in actions:
            if any(held.get(key, False) != want for key, want in action.pre.items()):
                continue
            after = dict(held)
            after.update(action.eff)
            nxt = freeze(after)
            cost = spent + action.cost
            if cost >= best_cost.get(nxt, float("inf")):
                continue
            best_cost[nxt] = cost
            heapq.heappush(
                frontier,
                (cost + unmet(nxt), next(tie), nxt, (*path, action.name), cost),
            )
    return None


@dataclass
class _Brain:
    """The shared body of a brain: cadence, steering, and perception.

    Movement runs every tick — a kinematic anchor that only moved on think
    ticks would stutter at 4 Hz — but decisions run on ``think_period``,
    because a brain that reconsiders forty times a second is paying attention
    it cannot possibly need. The split is Rain World's: AI slow, body fast.

    The one thing written is the brain's own pinned anchor, stepped toward
    ``_destination`` at ``speed`` with a bounded swim sway overlaid while
    travelling. A held anchor is left entirely alone: the hand outranks the
    brain through the same ``ctx.held`` that already outranks
    :class:`~mesh_xml.anim.controllers.drive.WaveAnchor`.
    """

    anchor: int
    home: tuple[float, ...]
    species: str = ""
    speed: float = 1.4
    sway: float = 0.5
    swing_rate: float = 3.0
    wander_range: float = 2.5
    think_period: float = 0.25
    enabled: bool = True

    _destination: np.ndarray | None = field(default=None, init=False, repr=False)
    _next_think: float = field(default=0.0, init=False, repr=False)

    def reset(self) -> None:
        self._destination = None
        self._next_think = 0.0

    def update(self, ctx: ControlContext) -> None:
        if ctx.time >= self._next_think:
            self._next_think = ctx.time + self.think_period
            self._think(ctx)
        self._move(ctx)

    def _think(self, ctx: ControlContext) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # steering
    # ------------------------------------------------------------------

    def _move(self, ctx: ControlContext) -> None:
        if not ctx.has(self.anchor) or self.anchor in ctx.held:
            return
        position = np.asarray(ctx.body.chunk(self.anchor).pos, dtype=np.float64)
        if self._destination is None:
            return
        to = self._destination[: position.size] - position
        distance = float(np.linalg.norm(to))
        if distance <= _ARRIVED:
            return

        step = to * min(1.0, self.speed * ctx.dt / distance)
        target = position + step
        if position.size >= 3:
            # The sway is added as a derivative times dt, so the integrated
            # wiggle stays bounded at ``sway`` world units instead of the
            # per-tick sum amplifying it by 1/dt.
            forward = to / distance
            perp = np.zeros_like(position)
            perp[0], perp[2] = forward[2], -forward[0]
            wiggle = (
                self.sway
                * self.swing_rate
                * np.cos(ctx.time * self.swing_rate)
                * ctx.dt
            )
            target = target + perp * wiggle
        ctx.body.set_pos(self.anchor, tuple(target))

    def _position(self, ctx: ControlContext) -> np.ndarray:
        return np.asarray(ctx.body.chunk(self.anchor).pos, dtype=np.float64)

    def _wander_point(self, ctx: ControlContext) -> np.ndarray:
        angle = float(ctx.rng.uniform(0.0, 2.0 * np.pi))
        radius = float(ctx.rng.uniform(0.4, 1.0)) * self.wander_range
        offset = np.array([np.cos(angle) * radius, 0.0, np.sin(angle) * radius])
        home = np.asarray(self.home, dtype=np.float64)
        return home + offset[: home.size]

    def _arrived(self, ctx: ControlContext) -> bool:
        if self._destination is None:
            return True
        gap = self._destination[: self._position(ctx).size] - self._position(ctx)
        return float(np.linalg.norm(gap)) <= _ARRIVED

    def _nearest_peer(
        self, ctx: ControlContext, species: str
    ) -> tuple[np.ndarray, float] | None:
        """Position and distance of the closest peer of ``species``, if any."""
        if not species:
            return None
        mine = self._position(ctx)
        best: tuple[np.ndarray, float] | None = None
        for peer in ctx.peers:
            if peer is self or getattr(peer, "species", None) != species:
                continue
            anchor = getattr(peer, "anchor", None)
            if anchor is None or not ctx.has(anchor):
                continue
            theirs = np.asarray(ctx.body.chunk(anchor).pos, dtype=np.float64)
            distance = float(np.linalg.norm(theirs - mine))
            if best is None or distance < best[1]:
                best = (theirs, distance)
        return best


@dataclass
class Grazer(_Brain):
    """An FSM prey animal: wanders its home range, flees what it fears.

    Two states with two transitions is the honest size of this behaviour, and
    an :class:`Fsm` keeps it that size: WANDER picks ambling points around
    ``home`` and switches to FLEE the moment anything of ``fears`` species
    comes inside ``sense_range``; FLEE runs directly away, re-aiming every
    think, and returns to WANDER once the threat is out of range. ``state``
    is a persisted field, so a saved document reopens mid-flight rather than
    amnesiac.
    """

    species: str = "grazer"
    fears: str = "hunter"
    sense_range: float = 4.0
    flee_distance: float = 3.5
    state: str = "wander"

    _fsm: Fsm | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        super().reset()
        self.state = "wander"
        self._fsm = None

    def _machine(self) -> Fsm:
        if self._fsm is None or self._fsm.state != self.state:
            self._fsm = Fsm(
                {"wander": self._wander, "flee": self._flee}, self.state
            )
        return self._fsm

    def _think(self, ctx: ControlContext) -> None:
        self.state = self._machine().step(ctx)

    def _wander(self, ctx: ControlContext) -> str | None:
        threat = self._nearest_peer(ctx, self.fears)
        if threat is not None and threat[1] <= self.sense_range:
            return "flee"
        if self._arrived(ctx):
            self._destination = self._wander_point(ctx)
        return None

    def _flee(self, ctx: ControlContext) -> str | None:
        threat = self._nearest_peer(ctx, self.fears)
        if threat is None or threat[1] > self.sense_range:
            self._destination = self._wander_point(ctx)
            return "wander"
        mine = self._position(ctx)
        away = mine - threat[0][: mine.size]
        length = float(np.linalg.norm(away))
        direction = away / length if length > 1e-6 else np.eye(mine.size)[0]
        self._destination = mine + direction * self.flee_distance
        return None


#: The hunt, as planning operators. PROWL discovers prey, CLOSE_IN converts
#: visibility into reach, STRIKE converts reach into a meal. The point of
#: spelling it this way is what happens when the world disagrees mid-plan:
#: prey slipping out of range falsifies CLOSE_IN's precondition and the
#: planner is simply asked again from the state that is actually true.
HUNT = (
    GoapAction("prowl", pre={"prey_visible": False}, eff={"prey_visible": True}, cost=2.0),
    GoapAction("close_in", pre={"prey_visible": True}, eff={"in_reach": True}, cost=1.0),
    GoapAction("strike", pre={"in_reach": True}, eff={"fed": True}, cost=1.0),
)

FED = {"fed": True}


@dataclass
class Hunter(_Brain):
    """A GOAP predator: plans its way from the perceived world to a full belly.

    Each think it senses ``prey_visible`` / ``in_reach`` / ``fed``, keeps its
    current plan only while the head action's preconditions still hold, pops
    actions whose effects have come true, and executes the head by steering —
    PROWL sweeps wander points, CLOSE_IN and STRIKE aim at the prey. A strike
    inside ``catch_range`` sets ``fed`` for ``feast_time`` seconds, during
    which it ambles home; hunger returning re-plans from scratch.

    ``_plan`` is runtime state, not a field: a plan is a cache of reasoning
    about a world that may have changed by the time a document reopens, and
    re-deriving it costs one A* over three booleans.
    """

    species: str = "hunter"
    hunts: str = "grazer"
    sense_range: float = 6.5
    catch_range: float = 1.05
    feast_time: float = 4.0
    speed: float = 1.9

    _plan: list[str] = field(default_factory=list, init=False, repr=False)
    _fed_until: float = field(default=-1.0, init=False, repr=False)

    def reset(self) -> None:
        super().reset()
        self._plan = []
        self._fed_until = -1.0

    @property
    def intent(self) -> str:
        """The action currently steering the body — what a debug label shows."""
        return "digest" if not self._plan else self._plan[0]

    def _think(self, ctx: ControlContext) -> None:
        prey = self._nearest_peer(ctx, self.hunts)
        state = {
            "prey_visible": prey is not None and prey[1] <= self.sense_range,
            "in_reach": prey is not None and prey[1] <= self.catch_range,
            "fed": ctx.time < self._fed_until,
        }

        if state["fed"]:
            self._plan = []
            if self._arrived(ctx):
                self._destination = self._wander_point(ctx)
            return

        def satisfied(facts: Mapping[str, bool]) -> bool:
            return all(state.get(key, False) == want for key, want in facts.items())

        while self._plan and satisfied(self._actions[self._plan[0]].eff):
            self._plan.pop(0)
        if not self._plan or not satisfied(self._actions[self._plan[0]].pre):
            self._plan = plan(state, FED, HUNT) or []
        if not self._plan:
            return

        head = self._plan[0]
        if head == "prowl":
            if self._arrived(ctx):
                self._destination = self._wander_point(ctx)
        elif prey is not None:
            self._destination = prey[0].copy()
            if head == "strike" and prey[1] <= self.catch_range:
                self._fed_until = ctx.time + self.feast_time
                self._plan = []

    @property
    def _actions(self) -> dict[str, GoapAction]:
        return {action.name: action for action in HUNT}
