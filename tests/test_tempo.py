"""Tests for creature tempo: the retiming rules, and the document knob.

The claim worth the hardest test is coherence — that one factor moves every
clock in a creature and nothing else in it, so shape survives speed. The
money test at the bottom is the whole promise in one number: the shipped worm
at tempo 2 sweeps exactly twice the path with its anchor, because a kinematic
driver retimed by 2 *is* the same figure at double speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from mesh_xml.anim import tempo
from mesh_xml.anim.behavior import Grazer, Hunter
from mesh_xml.anim.controllers.balance import Posture
from mesh_xml.anim.controllers.drive import WaveAnchor
from mesh_xml.anim.controllers.limb import FootholdPicker, LimbIK, LimbMode
from mesh_xml.anim.controllers.spine import Breathe, SpineFollow
from mesh_xml.core import Body
from mesh_xml.editor.document import Document, Macro

TICK = 1.0 / 40.0


class TestRetime:
    def test_a_rate_multiplies_and_shape_stays(self):
        wave = WaveAnchor(anchor=0, origin=(1.0, 2.0, 3.0), amplitude=0.8, rate=0.6)
        changes = tempo.retime(wave, 2.0)
        assert changes == {"rate": pytest.approx(1.2)}
        assert wave.rate == pytest.approx(0.6), "retime must not touch the original"

    def test_durations_divide_and_the_drive_multiplies(self):
        picker = FootholdPicker(
            step_duration=0.25,
            lead_time=0.3,
            period=0.8,
            phase=0.5,
            duty=0.6,
            step_threshold=0.5,
            drive=(1.5, 0.0, 0.0),
        )
        changes = tempo.retime(picker, 2.0)
        assert changes["step_duration"] == pytest.approx(0.125)
        assert changes["lead_time"] == pytest.approx(0.15)
        assert changes["period"] == pytest.approx(0.4)
        assert changes["drive"] == pytest.approx((3.0, 0.0, 0.0))
        assert isinstance(changes["drive"], tuple)
        for untouched in ("phase", "duty", "step_threshold"):
            assert untouched not in changes, f"{untouched} is dimensionless or spatial"

    def test_a_nested_picker_comes_back_as_a_fresh_replacement(self):
        picker = FootholdPicker(period=0.8, drive=(1.0, 0.0, 0.0))
        picker.mode = LimbMode.PLANTED
        picker.foothold = np.array([1.0, 0.0, 0.0])
        limb = LimbIK(root=1, mid=2, tip=3, target=picker, strength=200.0)

        changes = tempo.retime(limb, 2.0)
        assert set(changes) == {"target"}, "the limb's own gains are muscle, not clock"
        retimed = changes["target"]
        assert retimed is not picker
        assert retimed.period == pytest.approx(0.4)
        assert retimed.mode is LimbMode.DANGLING, "runtime state re-derives"
        assert retimed.foothold is None
        assert picker.period == pytest.approx(0.8)

    def test_brains_scale_locomotion_but_not_cognition(self):
        for brain in (
            Grazer(anchor=0, home=(0.0, 0.0, 0.0)),
            Hunter(anchor=0, home=(0.0, 0.0, 0.0)),
        ):
            changes = tempo.retime(brain, 3.0)
            assert changes["speed"] == pytest.approx(brain.speed * 3.0)
            assert changes["swing_rate"] == pytest.approx(brain.swing_rate * 3.0)
            assert "think_period" not in changes
            assert "feast_time" not in changes

    def test_a_spine_turns_its_heading_faster_and_breath_keeps_time(self):
        assert tempo.retime(SpineFollow(chain=[1, 2]), 2.0) == {
            "heading_speed": pytest.approx(4.0)
        }
        assert tempo.retime(Breathe(chunks=[1, 2]), 2.0) == {
            "frequency": pytest.approx(0.5)
        }

    def test_doubling_then_halving_returns_home(self):
        wave = WaveAnchor(anchor=0, origin=(0.0, 0.0, 0.0), rate=0.6)
        up = tempo.retime(wave, 2.0)["rate"]
        wave.rate = up
        down = tempo.retime(wave, 0.5)["rate"]
        assert down == pytest.approx(0.6)

    @pytest.mark.parametrize("factor", [0.0, -1.0, float("inf"), float("nan")])
    def test_a_degenerate_factor_is_refused(self, factor):
        with pytest.raises(ValueError):
            tempo.retime(WaveAnchor(anchor=0, origin=(0.0, 0.0, 0.0)), factor)

    def test_an_unregistered_controller_has_no_tempo(self):
        @dataclass
        class Local:
            rate: float = 1.0
            TEMPO = {"rate": 1}

        assert tempo.retime(Local(), 2.0) == {}


class TestChunkRefs:
    def test_ids_are_found_and_floats_are_not(self):
        grazer = Grazer(anchor=3, home=(5.0, 0.34, 0.6))
        assert tempo.chunk_refs(grazer) == frozenset({3})

    def test_lists_of_ids_and_nested_controllers_are_walked(self):
        limb = LimbIK(root=1, mid=2, tip=3, target=FootholdPicker())
        assert tempo.chunk_refs(limb) == frozenset({1, 2, 3})
        posture = Posture(chunks=[4, 5], height=1.0, support=[6])
        assert tempo.chunk_refs(posture) == frozenset({4, 5, 6})

    def test_booleans_are_not_ids(self):
        wave = WaveAnchor(anchor=0, origin=(0.0, 0.0, 0.0), enabled=True)
        assert tempo.chunk_refs(wave) == frozenset({0})


@pytest.fixture
def herd() -> Document:
    """Two creatures, one document: a wave-driven pair and a breathing pair."""
    document = Document()
    body = Body()
    a0 = body.add_chunk("a0", (0.0, 0.5, 0.0), radius=0.3, pinned=True)
    a1 = body.add_chunk("a1", (0.6, 0.5, 0.0), radius=0.3)
    b0 = body.add_chunk("b0", (5.0, 0.5, 0.0), radius=0.3, pinned=True)
    b1 = body.add_chunk("b1", (5.6, 0.5, 0.0), radius=0.3)
    body.add_link(a0, a1)
    body.add_link(b0, b1)
    document._replace(body, None)
    document.controllers.add(WaveAnchor(anchor=a0, origin=(0.0, 0.5, 0.0), rate=0.6))
    document.controllers.add(WaveAnchor(anchor=b0, origin=(5.0, 0.5, 0.0), rate=0.6))
    document.controllers.add(Breathe(chunks=[b0, b1], frequency=0.25))
    return document


class TestDocumentTempo:
    def test_retimes_only_the_selected_creature_as_one_undo_step(self, herd):
        herd.set_selection({0})
        herd.set_creature_tempo(2.0)

        mine, theirs, breath = herd.controllers.controllers
        assert mine.rate == pytest.approx(1.2)
        assert theirs.rate == pytest.approx(0.6), "the other creature keeps its clock"
        assert breath.frequency == pytest.approx(0.25)
        assert herd.body.meta_of(0)[tempo.TEMPO_META] == repr(2.0)
        assert herd.body.meta_of(1)[tempo.TEMPO_META] == repr(2.0)
        assert tempo.TEMPO_META not in herd.body.meta_of(2)
        assert len(herd._undo) == 1 and isinstance(herd._undo[0], Macro)

        herd.undo()
        assert mine.rate == pytest.approx(0.6)
        assert tempo.TEMPO_META not in herd.body.meta_of(0)
        herd.redo()
        assert mine.rate == pytest.approx(1.2)

    def test_the_knob_is_absolute_so_a_second_edit_uses_the_ratio(self, herd):
        herd.set_selection({0})
        herd.set_creature_tempo(2.0)
        herd.set_creature_tempo(3.0)
        assert herd.controllers.controllers[0].rate == pytest.approx(0.6 * 3.0)
        assert herd.creature_tempo() == pytest.approx(3.0)

    def test_a_herd_selection_lands_every_creature_at_the_value(self, herd):
        herd.set_selection({0})
        herd.set_creature_tempo(2.0)
        herd.set_selection({0, 2})
        herd.set_creature_tempo(4.0)

        mine, theirs, breath = herd.controllers.controllers
        assert mine.rate == pytest.approx(0.6 * 4.0)
        assert theirs.rate == pytest.approx(0.6 * 4.0), "reached 4 from 1, not from 2"
        assert breath.frequency == pytest.approx(0.25 * 4.0)

    def test_setting_the_current_value_records_nothing(self, herd):
        herd.set_selection({0})
        herd.set_creature_tempo(2.0)
        herd.set_creature_tempo(2.0)
        assert len(herd._undo) == 1

    def test_no_selection_is_a_message_not_an_edit(self, herd):
        notes: list[str] = []
        herd.status.connect(notes.append)
        herd.set_creature_tempo(2.0)
        assert not herd._undo
        assert any("Select" in note for note in notes)

    def test_garbage_meta_reads_as_one(self, herd):
        herd.body.meta_of(0)[tempo.TEMPO_META] = "banana"
        herd.set_selection({0})
        assert herd.creature_tempo() == pytest.approx(1.0)
        herd.set_creature_tempo(2.0)
        assert herd.controllers.controllers[0].rate == pytest.approx(1.2)

    def test_tempo_survives_the_file(self, herd, tmp_path):
        herd.set_selection({0})
        herd.set_creature_tempo(2.0)
        path = tmp_path / "herd.xml"
        herd.save(path)

        reopened = Document()
        reopened.open(path)
        reopened.set_selection({0})
        assert reopened.creature_tempo() == pytest.approx(2.0)
        wave = reopened.controllers.controllers[0]
        assert wave.rate == pytest.approx(1.2)


class TestTheWormAtDouble:
    def test_the_shipped_worm_swims_the_same_figure_in_half_the_time(self):
        """The coherence claim, end to end on a shipped creature.

        Tempo is a clock, not a new gait: at ×2 the anchor traverses the
        *same* figure in half the ticks. Same-window path ratios are not the
        invariant — over a window that is not a whole number of figure
        periods they legitimately drift from 2 — but the swept path over a
        matched phase span is, to sampling error. The anchor is pinned and
        its driver kinematic, so nothing but the clock is in this number.
        """

        def path_length(document: Document, ticks: int) -> float:
            anchor = next(
                c.anchor
                for c in document.controllers.controllers
                if isinstance(c, WaveAnchor)
            )
            total = 0.0
            last = np.asarray(document.body.chunk(anchor).pos)
            for _ in range(ticks):
                document.step_once()
                here = np.asarray(document.body.chunk(anchor).pos)
                total += float(np.linalg.norm(here - last))
                last = here
            return total

        plain = Document()
        plain.open("examples/chubby_worm.xml")

        doubled = Document()
        doubled.open("examples/chubby_worm.xml")
        doubled.set_selection({doubled.body.chunk_ids()[0]})
        doubled.set_creature_tempo(2.0)

        whole = path_length(plain, 480)
        halved = path_length(doubled, 240)
        assert halved == pytest.approx(whole, rel=0.01)
