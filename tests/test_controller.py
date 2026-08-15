"""Tests for the controller contract.

The load-bearing test here is :meth:`TestForcesNotPositions.test_a_controller_cannot_move_a_body`.
The rule it guards is easy to break and hard to notice broken: a controller that
writes positions still animates, right up until a link tightens or the iteration
count changes, and then it buzzes instead of failing.
"""

import numpy as np
import pytest

from mesh_xml.anim import ControlContext, Controller, ControllerStack
from mesh_xml.core import Body
from mesh_xml.core.solver import SolverConfig


def chain(n: int = 4) -> Body:
    body = Body()
    ids = [body.add_chunk(f"c{i}", [float(i), 0.0, 0.0], radius=0.2) for i in range(n)]
    for a, b in zip(ids, ids[1:]):
        body.add_link(a, b)
    return body


class Pusher:
    """Minimal well-behaved controller: pushes one chunk along +x."""

    def __init__(self, chunk_id: int, magnitude: float = 1.0) -> None:
        self.enabled = True
        self.chunk_id = chunk_id
        self.magnitude = magnitude
        self.calls = 0

    def update(self, ctx: ControlContext) -> None:
        self.calls += 1
        ctx.add_force(self.chunk_id, (self.magnitude, 0.0, 0.0))


class TestForcesNotPositions:
    def test_a_controller_cannot_move_a_body(self):
        """Running controllers alone must change force and nothing else."""
        body = chain()
        stack = ControllerStack(body)
        stack.add(Pusher(body.chunk_ids()[0], 10.0))

        before = body.pos.copy()
        prev_before = body.prev_pos.copy()
        forces = stack.accumulate(body, 0.025)

        assert np.array_equal(body.pos, before)
        assert np.array_equal(body.prev_pos, prev_before)
        assert forces[0][0] == pytest.approx(10.0)

    def test_force_reaches_the_body_only_through_a_tick(self):
        body = chain()
        stack = ControllerStack(body)
        stack.add(Pusher(body.chunk_ids()[0], 50.0))
        config = SolverConfig(gravity=(0.0, 0.0, 0.0))

        stack.tick(body, 0.025, config)
        assert body.pos[0][0] > 0.0

    def test_the_accumulator_is_zeroed_each_tick(self):
        """Otherwise force integrates over time and a constant push accelerates."""
        body = chain()
        stack = ControllerStack(body)
        stack.add(Pusher(body.chunk_ids()[0], 3.0))

        first = stack.accumulate(body, 0.025)[0][0]
        second = stack.accumulate(body, 0.025)[0][0]
        assert first == pytest.approx(second)


class TestMass:
    def test_force_is_divided_by_mass(self):
        """Force, not acceleration: a heavier chunk must accelerate less."""
        displacement = {}
        for mass in (1.0, 8.0):
            body = Body()
            chunk = body.add_chunk("c", [0.0, 0.0, 0.0], mass=mass)
            stack = ControllerStack(body)
            stack.add(Pusher(chunk, 20.0))
            stack.tick(body, 0.025, SolverConfig(gravity=(0.0, 0.0, 0.0)))
            displacement[mass] = float(body.pos[0][0])

        assert displacement[1.0] > displacement[8.0] * 4

    def test_a_pinned_chunk_absorbs_force(self):
        """Zero inverse mass means a controller need not check for pinning."""
        body = Body()
        chunk = body.add_chunk("anchor", [0.0, 0.0, 0.0], pinned=True)
        stack = ControllerStack(body)
        stack.add(Pusher(chunk, 1000.0))
        stack.tick(body, 0.025, SolverConfig())

        assert body.pos[0] == pytest.approx([0.0, 0.0, 0.0])


class TestSeek:
    def test_seek_pulls_toward_the_target(self):
        body = chain()
        stack = ControllerStack(body)
        target = np.array([0.0, 5.0, 0.0], dtype=np.float32)
        ctx = ControlContext(body, stack.forces, 0.025, 0.0, np.random.default_rng(0))

        force = ctx.seek(body.chunk_ids()[0], target, strength=2.0)
        assert force[1] > 0.0

    def test_seek_is_proportional_to_distance(self):
        body = chain()
        stack = ControllerStack(body)
        ctx = ControlContext(body, stack.forces, 0.025, 0.0, np.random.default_rng(0))
        first = body.chunk_ids()[0]

        near = ctx.seek(first, (0.0, 1.0, 0.0), strength=1.0).copy()
        stack.forces[:] = 0.0
        far = ctx.seek(first, (0.0, 4.0, 0.0), strength=1.0).copy()

        assert far[1] == pytest.approx(near[1] * 4, rel=1e-4)

    def test_max_force_clamps(self):
        """A distant target otherwise injects enough energy to throw the rig."""
        body = chain()
        stack = ControllerStack(body)
        ctx = ControlContext(body, stack.forces, 0.025, 0.0, np.random.default_rng(0))

        force = ctx.seek(body.chunk_ids()[0], (0.0, 1000.0, 0.0), strength=50.0, max_force=7.0)
        assert np.linalg.norm(force) == pytest.approx(7.0, rel=1e-5)

    def test_damping_opposes_velocity(self):
        body = chain()
        stack = ControllerStack(body)
        first = body.chunk_ids()[0]
        body.pos[0] = [0.0, 1.0, 0.0]

        ctx = ControlContext(body, stack.forces, 0.025, 0.0, np.random.default_rng(0))
        undamped = ctx.seek(first, (0.0, 1.0, 0.0), strength=1.0).copy()
        stack.forces[:] = 0.0
        damped = ctx.seek(first, (0.0, 1.0, 0.0), strength=1.0, damping=0.5).copy()

        assert damped[1] < undamped[1]

    def test_velocity_is_in_units_per_second(self):
        body = chain()
        stack = ControllerStack(body)
        body.pos[0] = [0.25, 0.0, 0.0]
        ctx = ControlContext(body, stack.forces, 0.025, 0.0, np.random.default_rng(0))

        assert ctx.velocity(body.chunk_ids()[0])[0] == pytest.approx(10.0)


class TestStaleIds:
    """Deleting a chunk is a routine edit, not a programming error."""

    def test_add_force_ignores_a_dead_chunk(self):
        body = chain()
        stack = ControllerStack(body)
        doomed = body.chunk_ids()[1]
        stack.add(Pusher(doomed))
        body.remove_chunk(doomed)

        stack.accumulate(body, 0.025)

    def test_seek_ignores_a_dead_chunk(self):
        body = chain()
        doomed = body.chunk_ids()[1]
        body.remove_chunk(doomed)
        ctx = ControlContext(body, np.zeros((body.n_chunks, 3), np.float32), 0.025, 0.0, np.random.default_rng(0))

        assert ctx.seek(doomed, (0.0, 1.0, 0.0), strength=1.0) == pytest.approx([0, 0, 0])

    def test_has_reports_liveness(self):
        body = chain()
        ctx = ControlContext(body, np.zeros((body.n_chunks, 3), np.float32), 0.025, 0.0, np.random.default_rng(0))
        alive = body.chunk_ids()[0]

        assert ctx.has(alive)
        assert not ctx.has(9999)


class TestGroundLevel:
    def test_the_stack_passes_the_config_through(self):
        seen = {}

        class Spy:
            enabled = True

            def update(self, ctx):
                seen["config"] = ctx.config

        body = chain()
        stack = ControllerStack(body)
        stack.add(Spy())
        config = SolverConfig(ground_y=0.0)
        stack.tick(body, 0.025, config)

        assert seen["config"] is config


class TestStack:
    def test_reset_cascades_to_controllers(self):
        """A foothold and a gait mode are not functions of a clock."""

        class Stateful:
            enabled = True

            def __init__(self):
                self.was_reset = False

            def update(self, ctx):
                pass

            def reset(self):
                self.was_reset = True

        body = chain()
        stack = ControllerStack(body)
        stateful = stack.add(Stateful())
        stack.reset()

        assert stateful.was_reset

    def test_reset_tolerates_controllers_without_one(self):
        body = chain()
        stack = ControllerStack(body)
        stack.add(Pusher(body.chunk_ids()[0]))
        stack.reset()


    def test_disabled_controllers_are_skipped(self):
        body = chain()
        stack = ControllerStack(body)
        pusher = Pusher(body.chunk_ids()[0])
        pusher.enabled = False
        stack.add(pusher)

        stack.accumulate(body, 0.025)
        assert pusher.calls == 0

    def test_controllers_compose_additively(self):
        body = chain()
        stack = ControllerStack(body)
        first = body.chunk_ids()[0]
        stack.add(Pusher(first, 2.0))
        stack.add(Pusher(first, 3.0))

        assert stack.accumulate(body, 0.025)[0][0] == pytest.approx(5.0)

    def test_the_accumulator_resizes_with_the_body(self):
        body = chain()
        stack = ControllerStack(body)
        stack.accumulate(body, 0.025)

        body.add_chunk("extra", [9.0, 0.0, 0.0])
        forces = stack.accumulate(body, 0.025)
        assert forces.shape == (body.n_chunks, body.dim)

    def test_time_advances_with_ticks(self):
        body = chain()
        stack = ControllerStack(body)
        for _ in range(4):
            stack.tick(body, 0.025, SolverConfig())
        assert stack.time == pytest.approx(0.1)

    def test_reset_rewinds_time_and_randomness(self):
        body = chain()
        stack = ControllerStack(body, seed=7)
        first = stack.rng.random()
        stack.tick(body, 0.025, SolverConfig())

        stack.reset()
        assert stack.time == 0.0
        assert stack.rng.random() == pytest.approx(first)

    def test_seeds_differ_between_rigs(self):
        a = ControllerStack(chain(), seed=1)
        b = ControllerStack(chain(), seed=2)
        assert a.rng.random() != b.rng.random()

    def test_an_empty_stack_still_ticks(self):
        body = chain()
        stack = ControllerStack(body)
        stack.tick(body, 0.025, SolverConfig())
        assert np.isfinite(body.pos).all()

    def test_remove_and_clear(self):
        body = chain()
        stack = ControllerStack(body)
        pusher = stack.add(Pusher(body.chunk_ids()[0]))
        stack.remove(pusher)
        assert stack.controllers == []

        stack.add(Pusher(body.chunk_ids()[0]))
        stack.clear()
        assert stack.controllers == []

    def test_pusher_satisfies_the_protocol(self):
        assert isinstance(Pusher(0), Controller)


class TestHeld:
    """A grab is authority, not force: held chunks reach every controller."""

    def test_held_chunks_flow_into_the_context(self):
        body = Body()
        a = body.add_chunk("a", (0.0, 0.0, 0.0))
        stack = ControllerStack(body)
        seen: list[frozenset[int]] = []

        class Probe:
            enabled = True

            def update(self, ctx: ControlContext) -> None:
                seen.append(ctx.held)

        stack.add(Probe())
        stack.accumulate(body, 0.025, SolverConfig())
        stack.held.add(a)
        stack.accumulate(body, 0.025, SolverConfig())

        assert seen == [frozenset(), frozenset({a})]
