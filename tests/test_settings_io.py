"""Does a saved creature reopen under the physics it was tuned under?

The format half of docs/SETTINGS.md: ``<solver>`` and ``<metaball>`` are
additive sections on ``<terrain>``'s terms, so the tests mirror
``test_gait_io``'s shape — a full round trip compared exactly, absence
meaning what every older document already meant, and refusal rather than
guessing for the hand edits the solver would otherwise choke on three
layers later.

The absent-semantics asymmetry gets its own tests because it is the part a
reader will doubt: an absent ``<solver>`` *section* reads as ``None`` — the
config the caller already runs — while ``ground_y`` absent *inside* a
present section is a floorless world, the dataclass truth. Comparisons are
exact rather than approximate: settings are written at full ``float64``
precision precisely so they come back as the same numbers, and asking only
for closeness would pass on a format that quietly rounds them.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from mesh_xml.core.body import Body
from mesh_xml.core.solver import SolverConfig
from mesh_xml.io import xml_v2

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

    rig = xml_v2.read(path)
    assert rig.solver is None
    assert rig.blend is None


def test_a_v2_document_honestly_has_no_settings(tmp_path: Path) -> None:
    path = tmp_path / "v2.xml"
    path.write_text(V2_DOCUMENT, encoding="utf-8")

    rig = xml_v2.read(path)

    assert rig.body.name == "worm"
    assert rig.solver is None
    assert rig.blend is None


def test_the_shipped_quadruped_reads_with_no_settings() -> None:
    """Defaults remain the truth about the example; its bytes did not change."""
    rig = xml_v2.read("examples/quadruped.xml")

    assert rig.solver is None
    assert rig.blend is None


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


def test_a_negative_blend_is_refused(tmp_path: Path) -> None:
    """The same validator terrain's blend answers to, the same vocabulary."""
    path = hand_authored(tmp_path, '<metaball blend="-0.1"/>')

    with pytest.raises(xml_v2.SchemaError, match="blend"):
        xml_v2.read(path)
