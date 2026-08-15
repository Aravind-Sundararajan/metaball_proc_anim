"""Does a saved creature reopen under the physics it was tuned under?

The format half of the settings and lighting records: ``<solver>``,
``<metaball>`` and ``<look>`` are additive sections on ``<terrain>``'s
terms, so the tests mirror ``test_gait_io``'s shape — a full round trip
compared exactly, absence meaning what every older document already meant,
and refusal rather than guessing for the hand edits the solver would
otherwise choke on three layers later.

The absent-semantics asymmetry gets its own tests because it is the part a
reader will doubt: an absent ``<solver>`` *section* reads as ``None`` — the
config the caller already runs — while ``ground_y`` absent *inside* a
present section is a floorless world, the dataclass truth. ``<look>`` has
no such asymmetry — absent section and absent attribute both mean the
shipped defaults — and one boundary all its own: the record lives in
``render/``, the first render import io has ever taken, so the io↛GL
firewall test at the bottom holds that package to its "only gl/ imports
OpenGL" rule in a fresh interpreter rather than trusting it. Comparisons
are exact rather than approximate: settings are written at full
``float64`` precision precisely so they come back as the same numbers, and
asking only for closeness would pass on a format that quietly rounds them.
"""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from mesh_xml.core.body import Body
from mesh_xml.core.solver import SolverConfig
from mesh_xml.io import xml_v2
from mesh_xml.render.look import Look

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


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def stub_body(dim: int = 3) -> Body:
    body = Body(dim=dim, name="stub")
    body.add_chunk("a", tuple(0.0 for _ in range(dim)))
    return body


def tuned_config() -> SolverConfig:
    """Every field away from its default, so a dropped one cannot hide."""
    return SolverConfig(
        gravity=(0.0, -3.7, 0.0),
        damping=0.9,
        iterations=8,
        max_velocity=50.0,
        ground_y=1.5,
        ground_friction=0.6,
        self_contact=True,
        self_contact_elasticity=0.25,
        self_friction=0.4,
    )


def tuned_look() -> Look:
    """Every field away from its default, so a dropped one cannot hide.

    ``light_colour`` deliberately carries a component above 1: colours are
    uncapped — overdrive is an authored act, ``damping="1.02"``'s ruling —
    and the round trip is where that gets held to.
    """
    return Look(
        light_direction=(-0.6, 0.2, 0.1),
        light_colour=(1.2, 0.9, 0.7),
        light_strength=1.4,
        ambient_strength=0.6,
        ao_strength=0.9,
        shadow_strength=0.35,
        glow_strength=0.05,
        surface_colour=(0.3, 0.6, 0.5),
        terrain_colour=(0.5, 0.4, 0.2),
        background_colour=(0.02, 0.03, 0.05),
        background_top=(0.4, 0.45, 0.55),
    )


def hand_authored(tmp_path: Path, section: str, dim: int = 3) -> Path:
    """A minimal v3 document with ``section`` written the way a hand edit would."""
    pos = " ".join("0.0" for _ in range(dim))
    path = tmp_path / "authored.xml"
    path.write_text(
        "<?xml version='1.0' encoding='utf-8'?>\n"
        f'<body version="3" name="stub" dim="{dim}">\n'
        "    <chunks>\n"
        f'        <chunk id="0" name="a" pos="{pos}" radius="0.25" mass="1.0" />\n'
        "    </chunks>\n"
        "    <links/>\n"
        f"    {section}\n"
        "</body>\n",
        encoding="utf-8",
    )
    return path


# ----------------------------------------------------------------------
# the headline: a tuned file reopens as tuned
# ----------------------------------------------------------------------


def test_a_tuned_config_and_blend_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "tuned.xml"
    xml_v2.save(stub_body(), path, solver=tuned_config(), blend=0.4)

    rig = xml_v2.read(path)

    assert rig.solver is not None
    assert rig.solver.gravity == (0.0, -3.7, 0.0)
    assert rig.solver.damping == 0.9
    assert rig.solver.iterations == 8
    assert rig.solver.max_velocity == 50.0
    assert rig.solver.ground_y == 1.5
    assert rig.solver.ground_friction == 0.6
    assert rig.solver.self_contact is True
    assert rig.solver.self_contact_elasticity == 0.25
    assert rig.solver.self_friction == 0.4
    assert rig.solver.terrain is None, "the world rides as <terrain>, never here"
    assert rig.blend == 0.4


def test_a_settings_float_survives_at_full_precision(tmp_path: Path) -> None:
    """A ``float32`` round trip would change the simulation a little every save."""
    path = tmp_path / "precise.xml"
    xml_v2.save(
        stub_body(), path, solver=SolverConfig(damping=0.1 + 0.2), blend=1.0 / 3.0
    )

    rig = xml_v2.read(path)

    assert rig.solver.damping == 0.1 + 0.2
    assert rig.blend == 1.0 / 3.0


def test_a_written_solver_record_is_complete_except_terrain(tmp_path: Path) -> None:
    """All or nothing: a later default change must not reshape a saved creature."""
    path = tmp_path / "complete.xml"
    xml_v2.save(stub_body(), path, solver=tuned_config())

    section = ET.parse(path).getroot().find("solver")

    assert section is not None
    assert set(section.keys()) == {
        "gravity",
        "damping",
        "iterations",
        "max_velocity",
        "ground_y",
        "ground_friction",
        "self_contact",
        "self_contact_elasticity",
        "self_friction",
    }


def test_a_dim2_body_saves_a_gravity_its_own_reader_accepts(tmp_path: Path) -> None:
    """core's 3-D default gravity is trimmed to the root's dim on write — the
    solver's own ``_gravity_vector`` projection applied at save time — so the
    editor can always reopen what it just saved."""
    path = tmp_path / "flat.xml"
    xml_v2.save(stub_body(dim=2), path, solver=SolverConfig())

    config = xml_v2.read(path).solver

    assert config is not None
    assert config.gravity == (0.0, -9.81)


def test_a_short_gravity_zero_pads_to_the_body(tmp_path: Path) -> None:
    path = tmp_path / "padded.xml"
    xml_v2.save(stub_body(), path, solver=SolverConfig(gravity=(0.0, -5.0)))

    assert xml_v2.read(path).solver.gravity == (0.0, -5.0, 0.0)


def test_a_tuned_look_round_trips(tmp_path: Path) -> None:
    """What returns is what went in — the same class, field for field, the
    vectors tuples and not lists, the overdriven colour uncapped."""
    path = tmp_path / "lit.xml"
    xml_v2.save(stub_body(), path, look=tuned_look())

    rig = xml_v2.read(path)

    assert rig.look is not None
    assert rig.look == tuned_look()


def test_a_look_float_survives_at_full_precision(tmp_path: Path) -> None:
    """The float32 shortening chunk positions take would drift a stated look
    a little on every save; look floats ride the solver sections' repr rule."""
    path = tmp_path / "precise_look.xml"
    xml_v2.save(
        stub_body(),
        path,
        look=Look(ambient_strength=0.1 + 0.2, background_top=(1.0 / 3.0, 0.2, 0.7)),
    )

    look = xml_v2.read(path).look

    assert look is not None
    assert look.ambient_strength == 0.1 + 0.2
    assert look.background_top == (1.0 / 3.0, 0.2, 0.7)


def test_a_written_look_record_is_complete_and_in_field_order(
    tmp_path: Path,
) -> None:
    """All or nothing, the solver ruling — and in the record's own field
    order, so the file reads as the dataclass does."""
    path = tmp_path / "complete_look.xml"
    xml_v2.save(stub_body(), path, look=Look())

    section = ET.parse(path).getroot().find("look")

    assert section is not None
    assert list(section.keys()) == [
        "light_direction",
        "light_colour",
        "light_strength",
        "ambient_strength",
        "ao_strength",
        "shadow_strength",
        "glow_strength",
        "surface_colour",
        "terrain_colour",
        "background_colour",
        "background_top",
    ]


def test_the_look_section_lands_after_metaball(tmp_path: Path) -> None:
    """Sections are written in one fixed order — <look> directly after
    <metaball>, and directly after <solver> when no blend is stated."""
    both = tmp_path / "both.xml"
    xml_v2.save(stub_body(), both, solver=SolverConfig(), blend=0.4, look=Look())
    tags = [el.tag for el in ET.parse(both).getroot()]
    assert tags == ["chunks", "links", "controllers", "solver", "metaball", "look"]

    unblended = tmp_path / "unblended.xml"
    xml_v2.save(stub_body(), unblended, solver=SolverConfig(), look=Look())
    tags = [el.tag for el in ET.parse(unblended).getroot()]
    assert tags == ["chunks", "links", "controllers", "solver", "look"]


# ----------------------------------------------------------------------
# absence, on each of its three levels
# ----------------------------------------------------------------------


def test_no_floor_is_spelled_by_an_absent_attribute(tmp_path: Path) -> None:
    path = tmp_path / "floorless.xml"
    xml_v2.save(stub_body(), path, solver=SolverConfig(ground_y=None))

    assert "ground_y" not in path.read_text(encoding="utf-8")
    config = xml_v2.read(path).solver
    assert config is not None
    assert config.ground_y is None
    assert config == SolverConfig()


def test_a_save_without_settings_writes_no_sections(tmp_path: Path) -> None:
    """Every pre-arc caller keeps writing today's bytes."""
    path = tmp_path / "plain.xml"
    xml_v2.save(stub_body(), path)

    text = path.read_text(encoding="utf-8")
    assert "<solver" not in text
    assert "<metaball" not in text
    assert "<look" not in text

    rig = xml_v2.read(path)
    assert rig.solver is None
    assert rig.blend is None
    assert rig.look is None


def test_a_v2_document_honestly_has_no_settings(tmp_path: Path) -> None:
    path = tmp_path / "v2.xml"
    path.write_text(V2_DOCUMENT, encoding="utf-8")

    rig = xml_v2.read(path)

    assert rig.body.name == "worm"
    assert rig.solver is None
    assert rig.blend is None
    assert rig.look is None


def test_the_shipped_quadruped_reads_with_no_settings() -> None:
    """Defaults remain the truth about the example; its bytes did not change."""
    rig = xml_v2.read("examples/quadruped.xml")

    assert rig.solver is None
    assert rig.blend is None
    assert rig.look is None


# ----------------------------------------------------------------------
# the hand-authored partial section
# ----------------------------------------------------------------------


def test_a_partial_solver_reads_as_the_dataclass_defaults(tmp_path: Path) -> None:
    """A present section states the physics entirely — unstated means default,
    and for ``ground_y`` the default is no floor at all."""
    path = hand_authored(tmp_path, '<solver iterations="8"/>')

    config = xml_v2.read(path).solver
    defaults = SolverConfig()

    assert config is not None
    assert config.iterations == 8
    assert config.gravity == defaults.gravity
    assert config.damping == defaults.damping
    assert config.max_velocity == defaults.max_velocity
    assert config.ground_y is None
    assert config.ground_friction == defaults.ground_friction
    assert config.self_contact is defaults.self_contact
    assert config.self_contact_elasticity == defaults.self_contact_elasticity
    assert config.terrain is None


def test_an_empty_metaball_section_says_nothing(tmp_path: Path) -> None:
    """The section may grow other fields; bare, it means the default blend."""
    path = hand_authored(tmp_path, "<metaball/>")

    assert xml_v2.read(path).blend is None


def test_a_partial_look_reads_as_the_dataclass_defaults(tmp_path: Path) -> None:
    """Absent attribute and absent section mean the same default — the
    ``<metaball>`` kind of absence, with no ``ground_y``-style asymmetry."""
    path = hand_authored(
        tmp_path, '<look glow_strength="0.9" background_top="0.4 0.5 0.6"/>'
    )

    look = xml_v2.read(path).look

    assert look == Look(glow_strength=0.9, background_top=(0.4, 0.5, 0.6))


def test_a_look_without_a_shadow_strength_reopens_shadowed(tmp_path: Path) -> None:
    """The format's first additive attribute whose default relights a saved
    record: a lighting-arc ``<look>`` predates ``shadow_strength``, absence
    inside the section means the dataclass default, and the default is the
    full shadow — deliberate on the precedent ``ao_strength`` set,
    and the attribute exists so an author can state otherwise."""
    path = hand_authored(
        tmp_path, '<look light_strength="1.4" ao_strength="0.9"/>'
    )

    look = xml_v2.read(path).look

    assert look is not None
    assert look.shadow_strength == 1.0


def test_an_empty_look_section_means_the_shipped_defaults(tmp_path: Path) -> None:
    """Bare, the section states the whole default record — a ``Look``, not
    ``None``, because io can construct one; ``<metaball/>``'s ``None`` exists
    only because io cannot follow ``MetaballSettings`` to the GL imports."""
    path = hand_authored(tmp_path, "<look/>")

    assert xml_v2.read(path).look == Look()


# ----------------------------------------------------------------------
# refusing rather than lying
# ----------------------------------------------------------------------


def test_a_gravity_with_the_wrong_arity_is_refused(tmp_path: Path) -> None:
    """A 3-vector falling into a dim-2 body is a broadcast error three layers
    later; here it is a named refusal."""
    path = hand_authored(tmp_path, '<solver gravity="0.0 -9.81 0.0"/>', dim=2)

    with pytest.raises(xml_v2.SchemaError, match="gravity"):
        xml_v2.read(path)


def test_a_gravity_that_is_not_a_vector_is_refused(tmp_path: Path) -> None:
    path = hand_authored(tmp_path, '<solver gravity="a b c"/>')

    with pytest.raises(xml_v2.SchemaError, match="gravity"):
        xml_v2.read(path)


def test_negative_iterations_are_refused(tmp_path: Path) -> None:
    """A negative count is not a tuning, it is a typo."""
    path = hand_authored(tmp_path, '<solver iterations="-1"/>')

    with pytest.raises(xml_v2.SchemaError, match="iterations"):
        xml_v2.read(path)


def test_a_nan_damping_is_refused(tmp_path: Path) -> None:
    """A NaN here poisons every position in the room by the second tick."""
    path = hand_authored(tmp_path, '<solver damping="nan"/>')

    with pytest.raises(xml_v2.SchemaError, match="finite"):
        xml_v2.read(path)


def test_a_nan_self_friction_is_refused(tmp_path: Path) -> None:
    """A mu answers to the same finiteness gate as every other solver float."""
    path = hand_authored(tmp_path, '<solver self_friction="nan"/>')

    with pytest.raises(xml_v2.SchemaError, match="finite"):
        xml_v2.read(path)


def test_a_negative_blend_is_refused(tmp_path: Path) -> None:
    """The same validator terrain's blend answers to, the same vocabulary."""
    path = hand_authored(tmp_path, '<metaball blend="-0.1"/>')

    with pytest.raises(xml_v2.SchemaError, match="blend"):
        xml_v2.read(path)


def test_a_zero_light_direction_is_refused(tmp_path: Path) -> None:
    """The shader normalizes it, and zero has no direction to state."""
    path = hand_authored(tmp_path, '<look light_direction="0.0 0.0 0.0"/>')

    with pytest.raises(xml_v2.SchemaError, match="light_direction"):
        xml_v2.read(path)


def test_a_negative_look_strength_is_refused(tmp_path: Path) -> None:
    """Negative light is not a tuning, it is a typo — the iterations ruling."""
    path = hand_authored(tmp_path, '<look glow_strength="-0.1"/>')

    with pytest.raises(xml_v2.SchemaError, match="glow_strength"):
        xml_v2.read(path)


def test_a_negative_shadow_strength_is_refused(tmp_path: Path) -> None:
    """The new strength answers to the standing gate: negative shadow is
    not a tuning either."""
    path = hand_authored(tmp_path, '<look shadow_strength="-0.5"/>')

    with pytest.raises(xml_v2.SchemaError, match="shadow_strength"):
        xml_v2.read(path)


def test_a_nan_shadow_strength_is_refused(tmp_path: Path) -> None:
    """Non-finite poisons the shader's == 1.0 endpoint as surely as every
    pixel; the finiteness gate covers the grown record."""
    path = hand_authored(tmp_path, '<look shadow_strength="nan"/>')

    with pytest.raises(xml_v2.SchemaError, match="finite"):
        xml_v2.read(path)


def test_a_negative_colour_component_is_refused(tmp_path: Path) -> None:
    """Above 1 overdrives on purpose; below 0 is not a colour at all."""
    path = hand_authored(tmp_path, '<look light_colour="1.0 -0.2 1.0"/>')

    with pytest.raises(xml_v2.SchemaError, match="light_colour"):
        xml_v2.read(path)


def test_a_nan_look_value_is_refused(tmp_path: Path) -> None:
    """A NaN here poisons every pixel rather than every position."""
    path = hand_authored(tmp_path, '<look ambient_strength="nan"/>')

    with pytest.raises(xml_v2.SchemaError, match="finite"):
        xml_v2.read(path)


def test_an_infinite_colour_component_is_refused(tmp_path: Path) -> None:
    """Vector components answer to the same finiteness gate as the scalars."""
    path = hand_authored(tmp_path, '<look background_top="0.1 inf 0.2"/>')

    with pytest.raises(xml_v2.SchemaError, match="finite"):
        xml_v2.read(path)


def test_a_look_vector_with_the_wrong_arity_is_refused(tmp_path: Path) -> None:
    """Two components or four, a triple is the only spelling the record has."""
    for stated in ("0.4 0.8", "0.4 0.8 0.45 1.0"):
        path = hand_authored(tmp_path, f'<look light_direction="{stated}"/>')

        with pytest.raises(xml_v2.SchemaError, match="components"):
            xml_v2.read(path)


def test_a_look_colour_that_is_not_a_vector_is_refused(tmp_path: Path) -> None:
    path = hand_authored(tmp_path, '<look surface_colour="a b c"/>')

    with pytest.raises(xml_v2.SchemaError, match="surface_colour"):
        xml_v2.read(path)


# ----------------------------------------------------------------------
# the io↛GL firewall
# ----------------------------------------------------------------------


def test_the_format_module_never_imports_opengl() -> None:
    """Run a bare interpreter, import the format module, and look for OpenGL.

    In-process this proves nothing — any earlier test may have loaded GL —
    so a fresh interpreter is the only honest witness (``test_room``'s Qt
    firewall, retold). ``Look`` lives in ``render/``, the first render
    import io has ever taken, and the render package's "only gl/ imports
    OpenGL" rule is what keeps a GL-less machine able to read and write
    documents; a docstring is not an enforcement mechanism.
    """
    probe = (
        "import sys\n"
        "import mesh_xml.io.xml_v2\n"
        "loaded = [m for m in sys.modules if m.startswith('OpenGL')]\n"
        "assert not loaded, f'the format module dragged in {loaded}'\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
