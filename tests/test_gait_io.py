"""Does a saved creature still walk?

The headline case is :func:`test_the_quadruped_still_walks_after_a_roundtrip`.
Every other test here checks a field or an error message; that one checks the
thing the format is for. It saves ``examples/quadruped.py`` before a single tick
has run, reloads it, walks both for ten seconds and demands the same footfalls
and the same distance travelled. A gait can survive a field-by-field comparison
and still not walk — a step duration rounded to ``float32`` moves a footfall by
a tick, a lost controller ordering solves the limbs against last tick's hips —
and none of that shows up until something takes a walk.

It also registers ``KinematicPath``, which lives in the example and not in the
library. That is the point of the registry stated as a test: a controller the
format has never heard of becomes savable in one call, with nothing in
:mod:`mesh_xml.io.xml_v2` changed or even aware.

Comparisons are exact rather than approximate throughout. Controller parameters
are written at full ``float64`` precision precisely so they come back as the
same numbers, and asking only for closeness would pass on a format that quietly
rounds them.
"""

from __future__ import annotations

import inspect
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from examples import quadruped
from mesh_xml.anim import serialize
from mesh_xml.anim.controller import ControlContext, ControllerStack
from mesh_xml.anim.controllers.limb import (
    FootholdPicker,
    LimbFrame,
    LimbIK,
    LimbMode,
)
from mesh_xml.anim.controllers.spine import Breathe, LookAt, SpineFollow
from mesh_xml.core.body import Body, LinkType
from mesh_xml.core.solver import TICK_DT
from mesh_xml.io import xml_legacy, xml_v2

DATA = Path(__file__).parent / "data"
ABSOLUTE_SPLINE = DATA / "legacy_v1_absolute_spline.xml"

#: Ten seconds at 40 Hz, which is ten gait cycles of the quadruped's 1.0 period.
TICKS = 400

V2_DOCUMENT = """<?xml version='1.0' encoding='utf-8'?>
<body version="2" name="worm" dim="3">
    <chunks>
        <chunk id="0" name="head" pos="0.0 1.0 0.0" radius="0.25" mass="2.0" />
        <chunk id="4" name="tail" pos="1.0 1.0 0.0" radius="0.2" pinned="true" />
    </chunks>
    <links>
        <link a="0" b="4" rest="1.0" type="PULL" elasticity="0.5" />
    </links>
</body>
"""


@dataclass
class Wander:
    """A controller this test file invented, to prove registration is one call."""

    chunks: list[int]
    speed: float = 1.0
    enabled: bool = True

    def update(self, ctx: ControlContext) -> None:
        for chunk_id in self.chunks:
            ctx.add_force(chunk_id, (self.speed, 0.0, 0.0))


@dataclass
class Opaque:
    """Registered, but holding a value the format has no encoding for."""

    chunks: set[int]
    enabled: bool = True

    def update(self, ctx: ControlContext) -> None: ...


@dataclass
class Stranger:
    """Never registered. A rig using one has to refuse to save."""

    enabled: bool = True

    def update(self, ctx: ControlContext) -> None: ...


serialize.register("kinematic_path", quadruped.KinematicPath)
serialize.register("test_wander", Wander)
serialize.register("test_opaque", Opaque)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def fields_of(controller: object) -> dict[str, object]:
    """Every constructor parameter — exactly what the format promises to keep."""
    return {
        name: getattr(controller, name)
        for name in inspect.signature(type(controller)).parameters
    }


def assert_same_fields(loaded: object, original: object) -> None:
    """Compare two controllers on the fields the format persists.

    Types are compared alongside values because a tuple that came back a list,
    or an ``enabled`` flag that came back ``1``, is a controller that still
    behaves and a format that has quietly lost something.
    """
    assert type(loaded) is type(original)
    for name, value in fields_of(original).items():
        got = getattr(loaded, name)
        if type(value) in set(serialize.registered().values()):
            assert_same_fields(got, value)
        elif isinstance(value, np.ndarray):
            assert isinstance(got, np.ndarray)
            assert got.dtype == value.dtype
            assert np.array_equal(got, value)
        else:
            assert got == value
            assert type(got) is type(value)


def roundtrip_controller(controller: object) -> object:
    return serialize.decode_controller(serialize.encode_controller(controller))


def roundtrip_rig(
    body: Body, stack: ControllerStack, tmp_path: Path, name: str = "rig.xml"
) -> xml_v2.Rig:
    path = tmp_path / name
    xml_v2.save(body, path, stack)
    return xml_v2.read(path)


def splice_controller(path: Path, **attrs: str) -> None:
    """Put a raw ``<controller>`` into a saved document, as a hand edit would."""
    tree = ET.parse(path)
    section = tree.getroot().find("controllers")
    assert section is not None
    ET.SubElement(section, "controller", attrs)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def context(body: Body, time: float = 0.0) -> ControlContext:
    return ControlContext(
        body=body,
        forces=np.zeros((body.n_chunks, body.dim), dtype=np.float32),
        dt=TICK_DT,
        time=time,
        rng=np.random.default_rng(0),
    )


def planted_picker() -> FootholdPicker:
    """A picker driven through a full swing, so it holds a real foothold."""
    body = Body(dim=3, name="stub")
    body.add_chunk("a", (0.0, 1.0, 0.0))
    picker = FootholdPicker(stance=(0.0, -1.0, 0.0), ground_y=0.0, step_duration=0.1)
    frame = LimbFrame(
        root=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        tip=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        reach=2.0,
    )
    picker.tip_target(context(body, time=0.0), frame)
    picker.tip_target(context(body, time=1.0), frame)
    return picker


@dataclass(frozen=True)
class Gait:
    """What a walk produced, reduced to what two walks can be compared on."""

    footholds: list[int]
    travelled: float
    pose: np.ndarray


def walk(body: Body, stack: ControllerStack, ticks: int = TICKS) -> Gait:
    """Run the creature and record where its feet went, as ``headless()`` does."""
    settings = quadruped.config()
    head = next(cid for cid in body.chunk_ids() if body.name_of(cid) == "head")
    start = float(body.pos[body.slot(head)][0])
    plants: dict[int, set] = {}

    for _ in range(ticks):
        stack.tick(body, TICK_DT, settings)
        for index, controller in enumerate(stack.controllers):
            picker = getattr(controller, "target", None)
            foothold = getattr(picker, "foothold", None)
            if foothold is not None:
                plants.setdefault(index, set()).add(tuple(np.round(foothold, 2)))

    return Gait(
        footholds=sorted(len(v) for v in plants.values()),
        travelled=float(body.pos[body.slot(head)][0]) - start,
        pose=body.pos.copy(),
    )


# ----------------------------------------------------------------------
# the headline: a saved creature still walks
# ----------------------------------------------------------------------


def test_the_quadruped_still_walks_after_a_roundtrip(tmp_path: Path) -> None:
    body, stack = quadruped.build()
    rig = roundtrip_rig(body, stack, tmp_path, "quadruped.xml")

    original = walk(body, stack)
    reloaded = walk(rig.body, rig.controllers)

    assert original.footholds == [10, 10, 10, 10], "the fixture must actually walk"
    assert reloaded.footholds == original.footholds
    assert reloaded.travelled == pytest.approx(original.travelled)
    assert reloaded.travelled > 5.0, "a creature that lost its carrier goes nowhere"
    assert np.array_equal(reloaded.pose, original.pose)


def test_the_quadruped_keeps_its_topology_and_its_stack(tmp_path: Path) -> None:
    body, stack = quadruped.build()
    rig = roundtrip_rig(body, stack, tmp_path, "quadruped.xml")

    assert rig.body.chunk_ids() == body.chunk_ids()
    assert rig.body.n_links == body.n_links
    assert rig.controllers.seed == stack.seed
    assert [type(c).__name__ for c in rig.controllers.controllers] == [
        type(c).__name__ for c in stack.controllers
    ]
    for loaded, original in zip(rig.controllers.controllers, stack.controllers):
        assert_same_fields(loaded, original)


def test_the_carrier_runs_first_after_a_roundtrip(tmp_path: Path) -> None:
    """Order is part of the gait: the limbs solve against hips already moved."""
    body, stack = quadruped.build()
    rig = roundtrip_rig(body, stack, tmp_path)

    assert isinstance(rig.controllers.controllers[0], quadruped.KinematicPath)
    assert isinstance(stack.controllers[0], quadruped.KinematicPath)


# ----------------------------------------------------------------------
# every shipped controller round-trips its fields
# ----------------------------------------------------------------------


def test_spine_follow_roundtrips_every_field() -> None:
    original = SpineFollow(
        chain=[3, 9, 27],
        strength=17.5,
        damping=2.25,
        max_bend_degrees=31.0,
        bend_strength=650.0,
        lead=(1.5, 2.5, -3.5),
        lead_strength=44.0,
        spacing=0.375,
        heading_speed=1.75,
        max_force=180.0,
        enabled=False,
    )

    loaded = roundtrip_controller(original)

    assert_same_fields(loaded, original)
    assert loaded.chain == [3, 9, 27]
    assert loaded.lead == (1.5, 2.5, -3.5)
    assert loaded.enabled is False


def test_look_at_roundtrips_every_field() -> None:
    original = LookAt(
        chain=[11, 12],
        target=(4.0, 1.25, -2.0),
        strength=19.0,
        damping=1.5,
        spacing=0.4,
        max_force=None,
    )

    loaded = roundtrip_controller(original)

    assert_same_fields(loaded, original)
    assert loaded.target == (4.0, 1.25, -2.0)
    assert loaded.max_force is None


def test_breathe_roundtrips_every_field() -> None:
    original = Breathe(
        chunks=[5, 6, 7], amplitude=2.5, frequency=0.5, axis=(0.0, 1.0, 0.0)
    )

    loaded = roundtrip_controller(original)

    assert_same_fields(loaded, original)
    assert loaded.chunks == [5, 6, 7]
    assert loaded.axis == (0.0, 1.0, 0.0)


def test_foothold_picker_roundtrips_every_field() -> None:
    original = FootholdPicker(
        stance=(0.0, -1.02, 0.125),
        ground_y=0.13,
        max_reach=1.44,
        step_threshold=0.26,
        step_height=0.2,
        step_duration=0.26,
        lead_time=0.22,
        scatter=0.05,
        period=1.0,
        phase=0.5,
        duty=0.44,
    )

    loaded = roundtrip_controller(original)

    assert_same_fields(loaded, original)
    assert loaded.step_duration == 0.26
    assert loaded.stance == (0.0, -1.02, 0.125)


def test_limb_ik_roundtrips_every_field() -> None:
    original = LimbIK(
        root=8,
        mid=9,
        tip=10,
        target=None,
        pole=(-1.0, 0.0, 0.0),
        flip=-1.0,
        lengths=(0.62, 0.58),
        strength=200.0,
        damping=14.0,
        mid_weight=0.35,
        max_force=500.0,
    )

    loaded = roundtrip_controller(original)

    assert_same_fields(loaded, original)
    assert (loaded.root, loaded.mid, loaded.tip) == (8, 9, 10)
    assert loaded.lengths == (0.62, 0.58)
    assert loaded.target is None


def test_a_float_survives_at_full_precision() -> None:
    """A ``float32`` round trip would move a footfall by a tick."""
    original = FootholdPicker(step_duration=0.1 + 0.2, lead_time=1.0 / 3.0)

    loaded = roundtrip_controller(original)

    assert loaded.step_duration == 0.1 + 0.2
    assert loaded.lead_time == 1.0 / 3.0


# ----------------------------------------------------------------------
# the interesting case: a target that is itself a controller
# ----------------------------------------------------------------------


def test_a_limb_keeps_the_picker_it_aims_with() -> None:
    picker = FootholdPicker(
        stance=(0.0, -1.02, 0.0), ground_y=0.13, period=1.0, phase=0.5, duty=0.44
    )
    original = LimbIK(8, 9, 10, target=picker, pole=(-1.0, 0.0, 0.0))

    loaded = roundtrip_controller(original)

    assert isinstance(loaded.target, FootholdPicker)
    assert loaded.target is not picker
    assert_same_fields(loaded.target, picker)
    assert loaded.target.phase == 0.5


def test_a_reloaded_limb_can_still_rewind_its_picker() -> None:
    """``LimbIK.reset`` reaches through to the target, so the nesting must be live."""
    loaded = roundtrip_controller(LimbIK(8, 9, 10, target=planted_picker()))

    loaded.target.mode = LimbMode.PLANTED
    loaded.target.foothold = np.zeros(3, dtype=np.float32)
    loaded.reset()

    assert loaded.target.mode is LimbMode.DANGLING
    assert loaded.target.foothold is None


def test_a_limb_aimed_at_a_bare_world_point_keeps_the_point() -> None:
    original = LimbIK(1, 2, 3, target=(2.5, 0.75, -1.5))

    loaded = roundtrip_controller(original)

    assert loaded.target == (2.5, 0.75, -1.5)
    assert not isinstance(loaded.target, FootholdPicker)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_limb_aimed_at_an_array_keeps_its_width(dtype: str) -> None:
    point = np.array([2.5, 0.75, -1.5], dtype=dtype)

    loaded = roundtrip_controller(LimbIK(1, 2, 3, target=point))

    assert isinstance(loaded.target, np.ndarray)
    assert loaded.target.dtype == np.dtype(dtype)
    assert np.array_equal(loaded.target, point)


def test_a_picker_beside_a_limb_rather_than_inside_one_is_refused(
    tmp_path: Path,
) -> None:
    """It has no ``update``; on the stack it would raise once per tick instead."""
    body = Body(dim=3, name="stub")
    body.add_chunk("a", (0.0, 0.0, 0.0))
    path = tmp_path / "misplaced.xml"
    xml_v2.save(body, path)
    splice_controller(path, type="foothold_picker")

    with pytest.raises(xml_v2.SchemaError, match="foothold_picker"):
        xml_v2.read(path)


# ----------------------------------------------------------------------
# refusing rather than lying
# ----------------------------------------------------------------------


def test_an_unregistered_controller_is_refused_by_name(tmp_path: Path) -> None:
    body = Body(dim=3, name="stub")
    chunk = body.add_chunk("a", (0.0, 0.0, 0.0))
    stack = ControllerStack(body)
    stack.add(SpineFollow([chunk]))
    stack.add(Stranger())
    path = tmp_path / "stranger.xml"

    with pytest.raises(serialize.ControllerFormatError, match="Stranger"):
        xml_v2.save(body, path, stack)

    assert not path.exists(), "half a gait on disk is worse than none"


def test_a_field_with_no_encoding_is_refused_by_name() -> None:
    with pytest.raises(serialize.ControllerFormatError, match=r"Opaque\.chunks"):
        serialize.encode_controller(Opaque(chunks={1, 2}))


def test_an_unregistered_target_names_the_limb_holding_it() -> None:
    with pytest.raises(serialize.ControllerFormatError, match=r"LimbIK\.target"):
        serialize.encode_controller(LimbIK(1, 2, 3, target=Stranger()))


def test_a_nested_controller_that_cannot_encode_names_itself() -> None:
    with pytest.raises(serialize.ControllerFormatError, match=r"Opaque\.chunks"):
        serialize.encode_controller(LimbIK(1, 2, 3, target=Opaque(chunks={1})))


def test_a_controller_type_no_reader_knows_is_refused_by_name(tmp_path: Path) -> None:
    body = Body(dim=3, name="stub")
    body.add_chunk("a", (0.0, 0.0, 0.0))
    path = tmp_path / "unknown.xml"
    xml_v2.save(body, path)
    splice_controller(path, type="wingbeat")

    with pytest.raises(serialize.ControllerFormatError, match="wingbeat"):
        xml_v2.read(path)


def test_a_field_the_class_no_longer_takes_is_refused() -> None:
    element = serialize.encode_controller(Breathe(chunks=[1]))
    ET.SubElement(element, "float", {"name": "wobble", "value": "1.0"})

    with pytest.raises(serialize.ControllerFormatError, match="Breathe"):
        serialize.decode_controller(element)


def test_a_field_value_that_is_not_a_number_is_refused() -> None:
    element = serialize.encode_controller(Breathe(chunks=[1]))
    element.find("float[@name='amplitude']").set("value", "loud")

    with pytest.raises(serialize.ControllerFormatError, match=r"Breathe\.amplitude"):
        serialize.decode_controller(element)


def test_a_name_cannot_be_rebound_to_a_different_class() -> None:
    with pytest.raises(serialize.ControllerFormatError, match="spine_follow"):
        serialize.register("spine_follow", Wander)


def test_registering_the_same_binding_twice_is_a_no_op() -> None:
    serialize.register("test_wander", Wander)

    assert serialize.registered()["test_wander"] is Wander


# ----------------------------------------------------------------------
# runtime state is not a gait
# ----------------------------------------------------------------------


def test_a_planted_foothold_is_not_persisted() -> None:
    picker = planted_picker()
    assert picker.mode is LimbMode.PLANTED
    assert picker.foothold is not None

    loaded = roundtrip_controller(picker)

    assert loaded.mode is LimbMode.DANGLING
    assert loaded.foothold is None
    assert_same_fields(loaded, picker)


def test_a_breathing_phase_is_not_persisted() -> None:
    body = Body(dim=3, name="stub")
    chunks = [body.add_chunk(f"c{i}", (float(i), 0.0, 0.0)) for i in range(3)]
    original = Breathe(chunks)
    original.update(context(body))
    assert original._phase is not None

    loaded = roundtrip_controller(original)

    assert loaded._phase is None


def test_the_stacks_clock_is_not_persisted_but_its_seed_is(tmp_path: Path) -> None:
    body, stack = quadruped.build()
    settings = quadruped.config()
    for _ in range(20):
        stack.tick(body, TICK_DT, settings)
    assert stack.time > 0.0

    rig = roundtrip_rig(body, stack, tmp_path)

    assert rig.controllers.time == 0.0
    assert rig.controllers.seed == 11


# ----------------------------------------------------------------------
# the version stamp, and the eras behind it
# ----------------------------------------------------------------------


def test_a_saved_document_is_stamped_v3_and_carries_a_controllers_section(
    tmp_path: Path,
) -> None:
    import xml.etree.ElementTree as ET

    body = Body(dim=3, name="stub")
    chunk = body.add_chunk("a", (0.0, 0.0, 0.0))
    stack = ControllerStack(body, seed=7)
    stack.add(Wander([chunk], speed=2.0))
    path = tmp_path / "stamped.xml"
    xml_v2.save(body, path, stack)

    root = ET.parse(path).getroot()
    section = root.find("controllers")

    assert root.get("version") == "3"
    assert xml_v2.FORMAT_VERSION == "3"
    assert section is not None
    assert section.get("seed") == "7"
    assert [el.get("type") for el in section] == ["test_wander"]


def test_a_body_saved_without_a_gait_still_says_so(tmp_path: Path) -> None:
    body = Body(dim=3, name="still")
    body.add_chunk("a", (0.0, 0.0, 0.0))
    path = tmp_path / "still.xml"
    xml_v2.save(body, path)

    rig = xml_v2.read(path)

    assert rig.controllers.controllers == []
    assert rig.controllers.seed == 0


def test_a_v2_document_still_loads_and_honestly_has_no_gait(tmp_path: Path) -> None:
    path = tmp_path / "v2.xml"
    path.write_text(V2_DOCUMENT, encoding="utf-8")

    rig = xml_v2.read(path)

    assert rig.body.name == "worm"
    assert sorted(rig.body.chunk_ids()) == [0, 4]
    assert rig.body.chunk(0).mass == pytest.approx(2.0)
    assert rig.body.chunk(4).pinned
    assert rig.body.link(0, 4).type is LinkType.PULL
    assert rig.controllers.controllers == []
    assert xml_v2.load(path).n_chunks == 2


def test_a_v2_document_becomes_v3_when_it_is_saved_again(tmp_path: Path) -> None:
    source = tmp_path / "v2.xml"
    source.write_text(V2_DOCUMENT, encoding="utf-8")
    rig = xml_v2.read(source)
    stack = ControllerStack(rig.body, seed=2)
    stack.add(SpineFollow([0, 4]))

    upgraded = roundtrip_rig(rig.body, stack, tmp_path, "v3.xml")

    assert sorted(upgraded.body.chunk_ids()) == [0, 4]
    assert upgraded.controllers.controllers[0].chain == [0, 4]


def test_a_v1_document_still_loads_through_the_legacy_reader() -> None:
    body = xml_legacy.load_legacy(ABSOLUTE_SPLINE)

    assert body.n_chunks == 8
    assert body.chunk(body.chunk_ids()[-1]).pos[0] == pytest.approx(4.444, abs=1e-3)

    with pytest.raises(xml_v2.SchemaError, match="xml_legacy"):
        xml_v2.load(ABSOLUTE_SPLINE)


def test_a_version_this_reader_does_not_know_is_still_refused(tmp_path: Path) -> None:
    path = tmp_path / "v4.xml"
    path.write_text(
        '<body version="4" dim="3"><chunks/><links/></body>', encoding="utf-8"
    )

    with pytest.raises(xml_v2.SchemaError, match="version"):
        xml_v2.load(path)


# ----------------------------------------------------------------------
# ids are the only identity a body has
# ----------------------------------------------------------------------


def test_controller_chunk_ids_survive_gaps_in_the_id_sequence(tmp_path: Path) -> None:
    body = Body(dim=3, name="gappy")
    ids = [body.add_chunk(f"c{i}", (float(i), 0.0, 0.0)) for i in range(5)]
    body.remove_chunk(ids[1])
    body.remove_chunk(ids[3])
    body.add_link(ids[0], ids[2])
    body.add_link(ids[2], ids[4])

    stack = ControllerStack(body, seed=3)
    stack.add(SpineFollow([ids[0], ids[2], ids[4]]))
    stack.add(LimbIK(ids[0], ids[2], ids[4], target=FootholdPicker()))
    rig = roundtrip_rig(body, stack, tmp_path)

    assert sorted(rig.body.chunk_ids()) == [ids[0], ids[2], ids[4]]
    assert rig.controllers.controllers[0].chain == [ids[0], ids[2], ids[4]]

    limb = rig.controllers.controllers[1]
    assert (limb.root, limb.mid, limb.tip) == (ids[0], ids[2], ids[4])
    assert all(rig.body.has_chunk(cid) for cid in (limb.root, limb.mid, limb.tip))


def test_a_controller_holding_a_deleted_chunk_still_loads(tmp_path: Path) -> None:
    """The runtime skips a stale id by design; a loader must not be stricter."""
    body = Body(dim=3, name="edited")
    kept = body.add_chunk("kept", (0.0, 0.0, 0.0))
    doomed = body.add_chunk("doomed", (1.0, 0.0, 0.0))
    stack = ControllerStack(body)
    stack.add(SpineFollow([kept, doomed]))
    body.remove_chunk(doomed)

    rig = roundtrip_rig(body, stack, tmp_path)

    assert rig.controllers.controllers[0].chain == [kept, doomed]
    assert not rig.body.has_chunk(doomed)
    rig.controllers.accumulate(rig.body, TICK_DT)


# ----------------------------------------------------------------------
# adding a controller to the format
# ----------------------------------------------------------------------


def test_a_new_controller_type_costs_one_call_and_no_writer_change(
    tmp_path: Path,
) -> None:
    """``Wander`` is registered at the top of this file and nowhere else."""
    body = Body(dim=3, name="wanderer")
    chunk = body.add_chunk("a", (0.0, 0.0, 0.0))
    stack = ControllerStack(body, seed=5)
    stack.add(Wander([chunk], speed=3.5, enabled=False))

    rig = roundtrip_rig(body, stack, tmp_path)
    loaded = rig.controllers.controllers[0]

    assert isinstance(loaded, Wander)
    assert loaded.speed == 3.5
    assert loaded.enabled is False
    assert "test_wander" in serialize.registered()
