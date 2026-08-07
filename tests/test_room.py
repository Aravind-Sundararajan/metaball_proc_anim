"""Tests for the headless room: the simulation with no editor in sight.

Package 2's acceptance is mostly the null result — the whole editor suite
passing unchanged over a Document that is now one room wearing signals — so
what belongs here is the other half: a room works with no window anywhere,
and the scene tier can never quietly grow a Qt import.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from mesh_xml.anim.controllers.drive import WaveAnchor
from mesh_xml.anim.controllers.spine import SpineFollow
from mesh_xml.scene import Room


def test_the_scene_tier_never_imports_qt():
    """Run a bare interpreter, import the scene tier, and look for PySide6.

    In-process this proves nothing — any earlier test may have loaded Qt —
    so a fresh interpreter is the only honest witness. This is the guard
    that keeps the extraction extracted: a future convenience import of
    anything editor-shaped fails here, not in the first game that tries to
    run rooms on a server.
    """
    probe = (
        "import sys\n"
        "import mesh_xml.scene\n"
        "import mesh_xml.scene.room\n"
        "loaded = [m for m in sys.modules if m.startswith('PySide6')]\n"
        "assert not loaded, f'the scene tier dragged in {loaded}'\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def worm(room: Room) -> int:
    """A five-chunk driven worm in the room; returns its anchor id."""
    ids = [
        room.body.add_chunk(f"w{i}", (i * 0.6, 0.5, 0.0), radius=0.3)
        for i in range(5)
    ]
    for a, b in zip(ids, ids[1:]):
        room.body.add_link(a, b)
    room.body.set_pinned(ids[0], True)
    room.controllers.add(WaveAnchor(anchor=ids[0], origin=(0.0, 0.5, 0.0)))
    room.controllers.add(SpineFollow(chain=ids))
    return ids[0]


class TestRoom:
    def test_a_room_ticks_a_creature_on_its_own(self):
        room = Room(name="den")
        anchor = worm(room)
        start = np.asarray(room.body.chunk(anchor).pos)

        for _ in range(300):
            room.tick()

        assert np.isfinite(room.body.pos).all()
        moved = float(np.linalg.norm(np.asarray(room.body.chunk(anchor).pos) - start))
        assert moved > 0.05, "the driver never led the worm anywhere"

    def test_the_game_loop_is_the_accumulator(self):
        """Drive a room by wall time alone, the way a game would."""
        room = Room()
        worm(room)

        ticked = 0
        before = room.clock.dt
        for _ in range(120):
            alpha = room.advance(1.0 / 60.0)
            assert 0.0 <= alpha <= 1.0
        ticked = room.controllers.time / room.clock.dt

        assert room.clock.dt == before, "advance must not retune the clock"
        # Two seconds of wall time at a fixed timestep is two seconds of
        # simulation, give or take the partial tick the accumulator banks.
        assert abs(ticked * room.clock.dt - 2.0) < 2.0 * room.clock.dt

    def test_a_room_starts_with_a_floor(self):
        assert Room().solver_config.ground_y == 0.0