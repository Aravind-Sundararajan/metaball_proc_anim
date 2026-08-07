"""Tests for the creature roster: components, remapping, placement, herding.

The claim that carries the package is wholeness — a duplicated creature is
the same creature somewhere else: same flesh (links with their types and
rests), same authored facts (colour, tempo, name), same gait (controllers
with ids remapped and world points carried), and no shared object anywhere,
so neither copy can haunt the other. The money test at the bottom herds the
shipped worm and watches both swim.
"""

from __future__ import annotations

import numpy as np
import pytest

from mesh_xml.anim import place, serialize
from mesh_xml.anim.behavior import Grazer, Hunter
from mesh_xml.anim.controllers.drive import WaveAnchor
from mesh_xml.anim.controllers.limb import FootholdPicker, LimbIK, LimbMode
from mesh_xml.anim.controllers.spine import SpineFollow
from mesh_xml.core import Body, LinkType
from mesh_xml.editor.document import CREATURE_META, Document
from mesh_xml.render.sdf import _component_labels


class TestComponents:
    def test_components_split_and_order_by_first_seen_slot(self):
        body = Body()
        a0 = body.add_chunk("a0", (0.0, 0.0, 0.0), radius=0.3)
        b0 = body.add_chunk("b0", (5.0, 0.0, 0.0), radius=0.3)
        a1 = body.add_chunk("a1", (0.6, 0.0, 0.0), radius=0.3)
        body.add_link(a0, a1)

        assert body.components() == [[a0, a1], [b0]]

    def test_a_mechanism_brace_still_joins_identity(self):
        body = Body()
        a = body.add_chunk("a", (0.0, 0.0, 0.0), radius=0.3)
        b = body.add_chunk("b", (1.0, 0.0, 0.0), radius=0.3)
        body.add_link(a, b, type=LinkType.PUSH, render=False)

        assert body.components() == [[a, b]]

    def test_the_roster_order_is_the_renderers_group_order(self):
        body = Body()
        ids = [
            body.add_chunk(f"c{i}", (float(i) * 3.0, 0.0, 0.0), radius=0.3)
            for i in range(6)
        ]
        body.add_link(ids[0], ids[1])
        body.add_link(ids[2], ids[3])
        body.add_link(ids[4], ids[5])

        labels = _component_labels(body)
        for index, component in enumerate(body.components()):
            for cid in component:
                assert labels[body.slot(cid)] == index


class TestRemapIds:
    def test_ids_remap_and_unmapped_ids_pass_through(self):
        brain = Hunter(anchor=5, home=(0.0, 1.0, 0.0), speed=2.0)
        clone = serialize.remap_ids(brain, {5: 50})
        assert clone is not brain
        assert clone.anchor == 50
        assert clone.speed == pytest.approx(2.0)
        assert serialize.remap_ids(brain, {1: 11}).anchor == 5

    def test_a_nested_picker_is_rebuilt_fresh(self):
        picker = FootholdPicker(period=0.8)
        picker.mode = LimbMode.PLANTED
        picker.foothold = np.array([1.0, 0.0, 0.0])
        limb = LimbIK(root=1, mid=2, tip=3, target=picker)

        clone = serialize.remap_ids(limb, {1: 11, 2: 12, 3: 13})

        assert (clone.root, clone.mid, clone.tip) == (11, 12, 13)
        assert clone.target is not picker
        assert clone.target.period == pytest.approx(0.8)
        assert clone.target.mode is LimbMode.DANGLING, "runtime state re-derives"
        assert limb.root == 1, "the original is never touched"

    def test_an_array_target_is_copied_not_shared(self):
        point = np.array([4.0, 0.0, 0.0])
        limb = LimbIK(root=1, mid=2, tip=3, target=point)
        clone = serialize.remap_ids(limb, {})
        assert clone.target is not point
        assert clone.target == pytest.approx(point)

    def test_an_unregistered_controller_is_refused(self):
        class Local:
            pass

        with pytest.raises(serialize.ControllerFormatError):
            serialize.remap_ids(Local(), {})


class TestTranslate:
    def test_world_points_move_and_directions_do_not(self):
        wave = WaveAnchor(anchor=0, origin=(1.0, 2.0, 3.0))
        assert place.translate(wave, (10.0, 0.0, 0.0)) == {
            "origin": pytest.approx((11.0, 2.0, 3.0))
        }

        limb = LimbIK(root=1, mid=2, tip=3, pole=(0.0, 0.0, 1.0))
        assert place.translate(limb, (10.0, 0.0, 0.0)) == {}, (
            "a pole is a direction and a missing target is nothing"
        )

    def test_a_brains_home_moves_with_it(self):
        brain = Grazer(anchor=0, home=(5.0, 0.3, 0.6))
        moved = place.translate(brain, (2.0, 0.0, -1.0))
        assert moved == {"home": pytest.approx((7.0, 0.3, -0.4))}

    def test_a_picker_target_places_itself_and_a_point_target_travels(self):
        with_picker = LimbIK(root=1, mid=2, tip=3, target=FootholdPicker())
        assert place.translate(with_picker, (10.0, 0.0, 0.0)) == {}

        with_point = LimbIK(root=1, mid=2, tip=3, target=np.array([4.0, 0.0, 0.0]))
        moved = place.translate(with_point, (10.0, 0.0, 0.0))
        assert moved["target"] == pytest.approx((14.0, 0.0, 0.0))

    def test_an_absent_lead_moves_nothing(self):
        spine = SpineFollow(chain=[1, 2, 3])
        assert place.translate(spine, (10.0, 0.0, 0.0)) == {}
        led = SpineFollow(chain=[1, 2, 3], lead=(0.0, 1.0, 0.0))
        assert place.translate(led, (10.0, 0.0, 0.0)) == {
            "lead": pytest.approx((10.0, 1.0, 0.0))
        }


@pytest.fixture
def pond() -> Document:
    """Two creatures: a named, pinned, braced pair and a loner."""
    document = Document()
    body = Body()
    a0 = body.add_chunk(
        "a0", (0.0, 0.5, 0.0), radius=0.3, pinned=True, meta={CREATURE_META: "newt"}
    )
    a1 = body.add_chunk("a1", (0.6, 0.5, 0.0), radius=0.3, meta={CREATURE_META: "newt"})
    body.add_link(a0, a1, rest=0.55, type=LinkType.PULL, elasticity=0.5, render=False)
    body.add_chunk("b0", (3.0, 0.5, 0.0), radius=0.4)
    document._replace(body, None)
    document.controllers.add(WaveAnchor(anchor=a0, origin=(0.0, 0.5, 0.0), rate=0.6))
    return document


class TestRoster:
    def test_the_roster_reads_names_from_meta(self, pond):
        first, second = pond.creatures()
        assert first.chunks == (0, 1) and first.name == "newt"
        assert second.chunks == (2,) and second.name is None
        assert second.label(1) == "Creature 1"

    def test_rename_paints_the_component_and_clears_honestly(self, pond):
        pond.set_selection({2})
        pond.rename_creature("boulder")
        assert pond.creatures()[1].name == "boulder"
        assert len(pond._undo) == 1

        pond.rename_creature("")
        assert pond.creatures()[1].name is None
        pond.undo()
        assert pond.creatures()[1].name == "boulder"

    def test_duplicate_copies_flesh_facts_and_gait(self, pond):
        pond.set_selection({0})
        pond.duplicate_creature()

        creatures = pond.creatures()
        assert len(creatures) == 3
        copy = creatures[2]
        assert copy.name == "newt", "the authored name rides along"
        assert pond.selection == set(copy.chunks), "the copy takes the selection"

        links = [
            link
            for link in pond.body.links()
            if link.a in copy.chunks and link.b in copy.chunks
        ]
        assert len(links) == 1
        assert links[0].rest == pytest.approx(0.55)
        assert links[0].type is LinkType.PULL
        assert links[0].elasticity == pytest.approx(0.5)
        assert links[0].render is False, "a brace stays a brace"

        anchor = min(copy.chunks)
        assert pond.body.chunk(anchor).pinned, "the pin is part of the creature"

        waves = [
            c for c in pond.controllers.controllers if isinstance(c, WaveAnchor)
        ]
        assert len(waves) == 2
        original, clone = waves
        assert clone.anchor == anchor
        assert clone.origin[0] > original.origin[0], "the copy swims around its own home"
        assert original.anchor == 0 and original.origin[0] == pytest.approx(0.0)

    def test_the_copy_lands_clear_of_everything(self, pond):
        body_right = max(
            body_chunk.pos[0] + body_chunk.radius for body_chunk in pond.body.chunks()
        )
        pond.set_selection({0})
        pond.duplicate_creature()
        copy = pond.creatures()[2]
        copy_left = min(
            pond.body.chunk(cid).pos[0] - pond.body.chunk(cid).radius
            for cid in copy.chunks
        )
        assert copy_left > body_right

    def test_one_undo_takes_the_whole_copy_back(self, pond):
        pond.set_selection({0})
        pond.duplicate_creature()
        assert len(pond._undo) == 1

        pond.undo()
        assert len(pond.creatures()) == 2
        assert len(pond.controllers.controllers) == 1
        pond.redo()
        assert len(pond.creatures()) == 3
        assert len(pond.controllers.controllers) == 2


class TestTheHerd:
    def test_the_shipped_worm_duplicates_into_a_herd_that_swims(self, tmp_path):
        document = Document()
        document.open("examples/chubby_worm.xml")
        document.set_selection({document.body.chunk_ids()[0]})
        document.duplicate_creature()

        assert len(document.creatures()) == 2
        waves = [
            c for c in document.controllers.controllers if isinstance(c, WaveAnchor)
        ]
        assert len(waves) == 2 and waves[0].anchor != waves[1].anchor

        path = tmp_path / "herd.xml"
        document.save(path)
        reopened = Document()
        reopened.open(path)
        assert len(reopened.creatures()) == 2

        anchors = [
            c.anchor
            for c in reopened.controllers.controllers
            if isinstance(c, WaveAnchor)
        ]
        starts = {a: np.asarray(reopened.body.chunk(a).pos) for a in anchors}
        for _ in range(300):
            reopened.step_once()
        for anchor, start in starts.items():
            travelled = float(
                np.linalg.norm(np.asarray(reopened.body.chunk(anchor).pos) - start)
            )
            assert travelled > 0.05, f"anchor {anchor} never swam"
