"""Serialization tests: v2 round trips and the two legacy coordinate eras.

The headline case is :func:`test_absolute_era_fixture_is_not_read_cumulatively`,
which pins defect §2.3 — an absolute-era spline read with today's relative rules
lands its last sphere at x=20.0 instead of x≈4.444.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mesh_xml.core.body import Body, LinkType
from mesh_xml.io import xml_legacy, xml_v2

DATA = Path(__file__).parent / "data"
EXAMPLES = Path(__file__).parent.parent / "examples"

ABSOLUTE_SPLINE = DATA / "legacy_v1_absolute_spline.xml"

RELATIVE_ERA_FIXTURES = [
    (EXAMPLES / "chubby_worm.xml", 10),
    (EXAMPLES / "character_armature.xml", 7),
    (EXAMPLES / "branching_armature.xml", 2),
    (DATA / "test_export.xml", 4),
    (DATA / "test_export_multiple_apendages.xml", 6),
]


def link_map(body: Body) -> dict[tuple[int, int], tuple[float, LinkType, float]]:
    return {
        (min(link.a, link.b), max(link.a, link.b)): (
            link.rest,
            link.type,
            link.elasticity,
        )
        for link in body.links()
    }


def roundtrip(body: Body, tmp_path: Path, name: str = "body.xml") -> Body:
    path = tmp_path / name
    xml_v2.save(body, path)
    return xml_v2.load(path)


# ----------------------------------------------------------------------
# v2 round trips
# ----------------------------------------------------------------------


def test_v2_roundtrip_preserves_every_field(tmp_path: Path) -> None:
    body = Body(dim=3, name="worm")
    head = body.add_chunk("head", (1.0, 2.0, 3.0), radius=0.25, mass=2.0)
    mid = body.add_chunk("mid", (1.5, 2.0, 3.0), radius=0.5, mass=4.0)
    tail = body.add_chunk("tail", (2.0, 2.0, 3.0), radius=0.125, pinned=True)
    body.add_link(head, mid, rest=0.5, type=LinkType.PULL, elasticity=0.25)
    body.add_link(mid, tail, rest=0.75, type=LinkType.PUSH, elasticity=0.5)

    loaded = roundtrip(body, tmp_path)

    assert loaded.name == "worm"
    assert loaded.dim == 3
    assert loaded.n_chunks == body.n_chunks
    assert loaded.n_links == body.n_links
    assert loaded.chunk_ids() == body.chunk_ids()

    for original in body.chunks():
        restored = loaded.chunk(original.id)
        assert restored.name == original.name
        assert restored.pos == pytest.approx(original.pos, rel=1e-6)
        assert restored.radius == pytest.approx(original.radius, rel=1e-6)
        assert restored.pinned == original.pinned
        if not original.pinned:
            assert restored.mass == pytest.approx(original.mass, rel=1e-6)

    assert loaded.chunk(tail).pinned
    assert loaded.chunk(tail).inv_mass == 0.0

    for key, (rest, kind, elasticity) in link_map(body).items():
        got_rest, got_kind, got_elasticity = link_map(loaded)[key]
        assert got_rest == pytest.approx(rest, rel=1e-6)
        assert got_kind is kind
        assert got_elasticity == pytest.approx(elasticity, rel=1e-6)


def test_v2_roundtrip_keeps_every_link_of_a_cyclic_body(tmp_path: Path) -> None:
    body = Body(dim=3, name="square")
    corners = [
        body.add_chunk(f"c{i}", pos)
        for i, pos in enumerate(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        )
    ]
    for i in range(4):
        body.add_link(corners[i], corners[(i + 1) % 4])
    body.add_link(corners[0], corners[2])

    assert body.n_links == 5
    _, _, extra = body.spanning_forest()
    assert extra, "fixture must not be a tree, or it proves nothing"

    loaded = roundtrip(body, tmp_path)

    assert loaded.n_links == 5
    assert link_map(loaded).keys() == link_map(body).keys()
    for key, (rest, kind, elasticity) in link_map(body).items():
        got_rest, got_kind, got_elasticity = link_map(loaded)[key]
        assert got_rest == pytest.approx(rest, rel=1e-6)
        assert got_kind is kind
        assert got_elasticity == pytest.approx(elasticity, rel=1e-6)


def test_v2_roundtrip_preserves_duplicate_names(tmp_path: Path) -> None:
    body = Body(dim=3, name="spider")
    left = body.add_chunk("leg", (-1.0, 0.0, 0.0))
    right = body.add_chunk("leg", (1.0, 0.0, 0.0))
    body.add_link(left, right)

    loaded = roundtrip(body, tmp_path)

    assert loaded.n_chunks == 2
    assert loaded.name_of(left) == "leg"
    assert loaded.name_of(right) == "leg"
    assert left != right
    assert loaded.chunk(left).pos[0] == pytest.approx(-1.0)
    assert loaded.chunk(right).pos[0] == pytest.approx(1.0)


def test_v2_roundtrip_preserves_dim(tmp_path: Path) -> None:
    body = Body(dim=2, name="flat")
    a = body.add_chunk("a", (0.0, 0.0))
    b = body.add_chunk("b", (3.0, 4.0))
    body.add_link(a, b)

    loaded = roundtrip(body, tmp_path)

    assert loaded.dim == 2
    assert loaded.chunk(b).pos == pytest.approx((3.0, 4.0))
    assert loaded.chunk(a).pos == pytest.approx((0.0, 0.0))


def test_v2_roundtrip_preserves_ids_with_gaps(tmp_path: Path) -> None:
    body = Body(dim=3, name="gappy")
    first = body.add_chunk("a", (0.0, 0.0, 0.0))
    middle = body.add_chunk("b", (1.0, 0.0, 0.0))
    last = body.add_chunk("c", (2.0, 0.0, 0.0))
    body.remove_chunk(middle)
    body.add_link(first, last)

    loaded = roundtrip(body, tmp_path)

    assert sorted(loaded.chunk_ids()) == sorted([first, last])
    assert not loaded.has_chunk(middle)
    assert loaded.has_link(first, last)
    assert loaded.add_chunk("d", (3.0, 0.0, 0.0)) == last + 1


def test_v2_refuses_an_id_that_would_overflow_the_id_array(tmp_path: Path) -> None:
    path = tmp_path / "huge.xml"
    path.write_text(
        '<body version="2" dim="3"><chunks>'
        '<chunk id="99999999999" name="a" pos="0 0 0" radius="0.5" mass="1"/>'
        "</chunks><links/></body>",
        encoding="utf-8",
    )

    with pytest.raises(xml_v2.SchemaError, match="int32"):
        xml_v2.load(path)


def test_v2_roundtrip_of_an_empty_body(tmp_path: Path) -> None:
    loaded = roundtrip(Body(dim=3, name="empty"), tmp_path)

    assert loaded.n_chunks == 0
    assert loaded.n_links == 0
    assert loaded.name == "empty"


def test_v2_stamps_the_version_on_the_root(tmp_path: Path) -> None:
    import xml.etree.ElementTree as ET

    body = Body(dim=3, name="stamped")
    body.add_chunk("a", (0.0, 0.0, 0.0))
    path = tmp_path / "stamped.xml"
    xml_v2.save(body, path)

    root = ET.parse(path).getroot()
    assert root.tag == "body"
    assert root.get("version") == "2"
    assert root.get("dim") == "3"


def test_v2_refuses_a_legacy_document() -> None:
    with pytest.raises(xml_v2.SchemaError, match="xml_legacy"):
        xml_v2.load(ABSOLUTE_SPLINE)


def test_v2_refuses_an_unknown_version(tmp_path: Path) -> None:
    path = tmp_path / "v3.xml"
    path.write_text(
        '<body version="3" dim="3"><chunks/><links/></body>', encoding="utf-8"
    )

    with pytest.raises(xml_v2.SchemaError, match="version"):
        xml_v2.load(path)


def test_v2_refuses_a_link_to_a_missing_chunk(tmp_path: Path) -> None:
    path = tmp_path / "dangling.xml"
    path.write_text(
        '<body version="2" dim="3">'
        '<chunks><chunk id="0" name="a" pos="0 0 0" radius="0.5" mass="1"/></chunks>'
        '<links><link a="0" b="9" rest="1" type="NORMAL" elasticity="1"/></links>'
        "</body>",
        encoding="utf-8",
    )

    with pytest.raises(xml_v2.SchemaError, match="unknown chunk id 9"):
        xml_v2.load(path)


# ----------------------------------------------------------------------
# legacy: the coordinate eras
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected_chunks",
    RELATIVE_ERA_FIXTURES,
    ids=lambda value: value.stem if isinstance(value, Path) else str(value),
)
def test_relative_era_fixtures_load(path: Path, expected_chunks: int) -> None:
    result = xml_legacy.read_legacy(path)

    assert result.coords == "relative"
    assert result.body.n_chunks == expected_chunks
    assert result.body.n_links == expected_chunks - 1
    assert set(result.meta) == set(result.body.chunk_ids())


def test_absolute_era_fixture_loads() -> None:
    result = xml_legacy.read_legacy(ABSOLUTE_SPLINE)

    assert result.coords == "absolute"
    assert result.body.n_chunks == 8
    assert result.body.n_links == 7


def test_absolute_era_fixture_is_not_read_cumulatively() -> None:
    """Defect §2.3: reading absolute-era offsets as relative accumulates garbage."""
    body = xml_legacy.load_legacy(ABSOLUTE_SPLINE)
    last = body.chunk(body.chunk_ids()[-1])

    assert last.name == "Spline_Sphere_8"
    assert last.pos[0] == pytest.approx(4.444, abs=1e-3)
    assert last.pos[0] != pytest.approx(20.0, abs=1e-3)


def test_absolute_era_fixture_is_evenly_spaced_when_read_correctly() -> None:
    body = xml_legacy.load_legacy(ABSOLUTE_SPLINE)
    xs = [body.chunk(cid).pos[0] for cid in body.chunk_ids()]
    gaps = [b - a for a, b in zip(xs, xs[1:])]

    assert gaps == pytest.approx([5.0 / 9.0] * 7, abs=1e-3)


def test_forcing_relative_reproduces_the_old_bug() -> None:
    """The override must be able to reproduce today's behaviour verbatim."""
    body = xml_legacy.load_legacy(ABSOLUTE_SPLINE, coords="relative")
    last = body.chunk(body.chunk_ids()[-1])

    assert last.pos[0] == pytest.approx(20.0, abs=1e-3)


def test_forcing_absolute_skips_detection() -> None:
    result = xml_legacy.read_legacy(EXAMPLES / "chubby_worm.xml", coords="absolute")

    assert result.coords == "absolute"
    assert result.body.chunk(result.body.chunk_ids()[-1]).pos[0] == pytest.approx(0.556)


def test_detect_coords_agrees_with_the_loader() -> None:
    assert xml_legacy.detect_coords(ABSOLUTE_SPLINE) == "absolute"
    for path, _ in RELATIVE_ERA_FIXTURES:
        assert xml_legacy.detect_coords(path) == "relative"


def test_unknown_coords_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="coords must be"):
        xml_legacy.load_legacy(ABSOLUTE_SPLINE, coords="world")


# ----------------------------------------------------------------------
# legacy: losslessness (defect 2.4)
# ----------------------------------------------------------------------


def test_capsule_type_and_height_survive_the_import() -> None:
    result = xml_legacy.read_legacy(EXAMPLES / "character_armature.xml")
    by_name = {result.body.name_of(cid): cid for cid in result.body.chunk_ids()}

    segment = result.meta[by_name["Segment1"]]
    assert segment.mesh_type == "capsule"
    assert segment.height == pytest.approx(0.6)
    assert segment.material == "body"
    assert result.body.chunk(by_name["Segment1"]).radius == pytest.approx(0.15)

    head = result.meta[by_name["Head"]]
    assert head.mesh_type == "sphere"
    assert head.height == pytest.approx(0.4)
    assert head.material == "head"

    assert sum(1 for m in result.meta.values() if m.mesh_type == "capsule") == 5


def test_rotation_and_scale_survive_the_import() -> None:
    result = xml_legacy.read_legacy(EXAMPLES / "character_armature.xml")
    by_name = {result.body.name_of(cid): cid for cid in result.body.chunk_ids()}

    assert result.meta[by_name["Head"]].scale == pytest.approx((1.2, 1.2, 1.2))
    assert result.meta[by_name["Segment1"]].scale == pytest.approx((1.0, 1.0, 1.0))
    assert result.meta[by_name["Segment1"]].rotation == pytest.approx((0.0, 0.0, 0.0))


def test_a_bone_without_a_mesh_is_reported_not_silently_defaulted() -> None:
    result = xml_legacy.read_legacy(EXAMPLES / "character_armature.xml")
    by_name = {result.body.name_of(cid): cid for cid in result.body.chunk_ids()}
    root = by_name["Root"]

    assert result.missing_mesh == (root,)
    assert result.meta[root].has_mesh is False
    assert result.meta[root].radius_defaulted is True
    assert result.meta[root].mesh_type is None
    assert result.body.chunk(root).radius == pytest.approx(0.5)

    assert all(
        not result.meta[cid].radius_defaulted for cid in by_name.values() if cid != root
    )


def test_the_missing_mesh_default_is_the_callers_choice() -> None:
    result = xml_legacy.read_legacy(
        EXAMPLES / "character_armature.xml", default_radius=0.9
    )
    root = next(
        cid for cid in result.body.chunk_ids() if result.body.name_of(cid) == "Root"
    )

    assert result.body.chunk(root).radius == pytest.approx(0.9)


def test_refusing_the_default_raises_instead_of_inventing_a_radius() -> None:
    with pytest.raises(xml_legacy.LegacyFormatError, match="'Root' has no <mesh>"):
        xml_legacy.read_legacy(EXAMPLES / "character_armature.xml", default_radius=None)


def test_legacy_reader_refuses_a_v2_document(tmp_path: Path) -> None:
    body = Body(dim=3, name="modern")
    body.add_chunk("a", (0.0, 0.0, 0.0))
    path = tmp_path / "modern.xml"
    xml_v2.save(body, path)

    with pytest.raises(xml_legacy.LegacyFormatError, match="xml_v2"):
        xml_legacy.load_legacy(path)


# ----------------------------------------------------------------------
# legacy: the reversed-duplicate-edge corruption (defect 2.1)
# ----------------------------------------------------------------------


REVERSED_DUPLICATE = """<?xml version="1.0" ?>
<armature>
    <bone name="A" position="0.000,0.000,0.000">
        <mesh type="sphere" radius="0.500" material="body"/>
        <bone name="B" position="1.000,0.000,0.000">
            <mesh type="sphere" radius="0.500" material="body"/>
        </bone>
    </bone>
    <bone name="B" position="1.000,0.000,0.000">
        <mesh type="sphere" radius="0.500" material="body"/>
        <bone name="A" position="-1.000,0.000,0.000">
            <mesh type="sphere" radius="0.500" material="body"/>
        </bone>
    </bone>
</armature>
"""


def test_reversed_duplicate_edges_collapse_to_one_link(tmp_path: Path) -> None:
    """Defect §2.1 wiped the whole rig; here it costs one deduplicated edge."""
    path = tmp_path / "reversed.xml"
    path.write_text(REVERSED_DUPLICATE, encoding="utf-8")

    result = xml_legacy.read_legacy(path)

    assert result.body.n_chunks == 2
    assert result.body.n_links == 1
    assert result.merged_names == ("B", "A")

    by_name = {result.body.name_of(cid): cid for cid in result.body.chunk_ids()}
    assert result.body.has_link(by_name["A"], by_name["B"])
    assert result.body.chunk(by_name["A"]).pos[0] == pytest.approx(0.0)


def test_a_bone_parented_to_its_own_name_does_not_self_link(tmp_path: Path) -> None:
    path = tmp_path / "self.xml"
    path.write_text(
        '<?xml version="1.0" ?><armature>'
        '<bone name="A" position="0,0,0"><mesh type="sphere" radius="0.5"/>'
        '<bone name="A" position="1,0,0"><mesh type="sphere" radius="0.5"/></bone>'
        "</bone></armature>",
        encoding="utf-8",
    )

    result = xml_legacy.read_legacy(path)

    assert result.body.n_chunks == 1
    assert result.body.n_links == 0


# ----------------------------------------------------------------------
# legacy -> v2 migration
# ----------------------------------------------------------------------


def test_a_legacy_import_survives_a_v2_roundtrip(tmp_path: Path) -> None:
    original = xml_legacy.load_legacy(ABSOLUTE_SPLINE)
    loaded = roundtrip(original, tmp_path, "migrated.xml")

    assert loaded.n_chunks == original.n_chunks
    assert loaded.chunk_ids() == original.chunk_ids()
    assert link_map(loaded).keys() == link_map(original).keys()
    for cid in original.chunk_ids():
        assert loaded.chunk(cid).pos == pytest.approx(original.chunk(cid).pos, rel=1e-6)
        assert loaded.name_of(cid) == original.name_of(cid)
    assert loaded.chunk(loaded.chunk_ids()[-1]).pos[0] == pytest.approx(4.444, abs=1e-3)
