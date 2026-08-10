"""Tests for SDF primitive packing.

The property worth guarding hardest is that geometry is not emitted twice.
Overlap is invisible in a hard union and would never show up as a crash — it
shows up as a surface quietly fatter than it was authored, by a fraction of the
blend radius, everywhere at once.
"""

import numpy as np
import pytest

from mesh_xml.core import Body
from mesh_xml.render import sdf


def chain(n: int, radius: float = 0.5) -> Body:
    body = Body()
    ids = [body.add_chunk(f"c{i}", [float(i), 0.0, 0.0], radius=radius) for i in range(n)]
    for a, b in zip(ids, ids[1:]):
        body.add_link(a, b)
    return body


class TestCounts:
    def test_a_chain_emits_one_primitive_per_link(self):
        body = chain(10)
        assert sdf.primitive_count(body) == 9
        assert len(sdf.pack(body)) == 9

    def test_an_isolated_chunk_emits_its_own_primitive(self):
        body = Body()
        body.add_chunk("lone", [0.0, 0.0, 0.0])
        assert sdf.primitive_count(body) == 1

    def test_a_linked_chunk_does_not_also_emit_a_sphere(self):
        """Duplicated geometry inflates the smooth-min surface by k/4."""
        body = Body()
        a = body.add_chunk("a", [0.0, 0.0, 0.0])
        b = body.add_chunk("b", [1.0, 0.0, 0.0])
        body.add_link(a, b)

        assert sdf.primitive_count(body) == 1

    def test_mixed_linked_and_isolated(self):
        body = chain(4)
        body.add_chunk("floater", [9.0, 9.0, 9.0])
        body.add_chunk("floater2", [-9.0, 0.0, 0.0])

        assert sdf.primitive_count(body) == 3 + 2

    def test_an_empty_body_packs_to_an_empty_array(self):
        packed = sdf.pack(Body())
        assert packed.shape == (0, sdf.PRIMITIVE_STRIDE)

    def test_a_cycle_emits_a_primitive_per_edge(self):
        body = Body()
        ids = [body.add_chunk(f"c{i}", [float(i), 0.0, 0.0]) for i in range(3)]
        body.add_link(ids[0], ids[1])
        body.add_link(ids[1], ids[2])
        body.add_link(ids[2], ids[0])

        assert sdf.primitive_count(body) == 3


class TestLayout:
    def test_a_link_becomes_a_tapered_cone(self):
        body = Body()
        a = body.add_chunk("a", [0.0, 0.0, 0.0], radius=0.3)
        b = body.add_chunk("b", [2.0, 0.0, 0.0], radius=0.7)
        body.add_link(a, b)

        row = sdf.pack(body)[0]
        assert row[0:3] == pytest.approx([0.0, 0.0, 0.0])
        assert row[3] == pytest.approx(0.3)
        assert row[4:7] == pytest.approx([2.0, 0.0, 0.0])
        assert row[7] == pytest.approx(0.7)

    def test_an_isolated_chunk_is_a_degenerate_cone(self):
        """Both ends coincide, which the field must recognise as a sphere."""
        body = Body()
        body.add_chunk("lone", [1.0, 2.0, 3.0], radius=0.4)

        row = sdf.pack(body)[0]
        assert row[0:3] == pytest.approx(row[4:7])
        assert row[3] == pytest.approx(0.4)
        assert row[7] == pytest.approx(0.4)
        axis_length = np.linalg.norm(row[4:7] - row[0:3])
        assert axis_length < sdf.DEGENERATE_EPSILON

    def test_output_is_contiguous_float32(self):
        packed = sdf.pack(chain(5))
        assert packed.dtype == np.float32
        assert packed.flags["C_CONTIGUOUS"]
        assert packed.shape[1] == sdf.PRIMITIVE_STRIDE


class TestOverrides:
    def test_positions_override_is_used(self):
        """The surface must follow interpolated draw positions, not raw physics."""
        body = chain(2)
        moved = np.array([[0.0, 5.0, 0.0], [1.0, 5.0, 0.0]], dtype=np.float32)

        row = sdf.pack(body, positions=moved)[0]
        assert row[1] == pytest.approx(5.0)
        assert row[5] == pytest.approx(5.0)

    def test_radii_override_is_used(self):
        body = chain(2)
        squashed = np.array([0.1, 0.2], dtype=np.float32)

        row = sdf.pack(body, radii=squashed)[0]
        assert row[3] == pytest.approx(0.1)
        assert row[7] == pytest.approx(0.2)

    def test_mismatched_override_shapes_rejected(self):
        body = chain(3)
        with pytest.raises(ValueError):
            sdf.pack(body, positions=np.zeros((2, 3), dtype=np.float32))
        with pytest.raises(ValueError):
            sdf.pack(body, radii=np.zeros(2, dtype=np.float32))


class TestDimensions:
    def test_a_2d_body_widens_onto_the_z_plane(self):
        """The field always works in 3D; a planar rig sits at z=0."""
        body = Body(dim=2)
        a = body.add_chunk("a", [1.0, 2.0], radius=0.5)
        b = body.add_chunk("b", [3.0, 4.0], radius=0.5)
        body.add_link(a, b)

        row = sdf.pack(body)[0]
        assert row[0:3] == pytest.approx([1.0, 2.0, 0.0])
        assert row[4:7] == pytest.approx([3.0, 4.0, 0.0])


class TestRealRig:
    def test_the_chubby_worm_packs_without_double_counting(self):
        from mesh_xml.io import xml_legacy

        body = xml_legacy.read_legacy("tests/data/legacy_v1_chubby_worm.xml").body
        packed = sdf.pack(body)

        assert body.n_chunks == 10
        assert body.n_links == 9
        assert len(packed) == 9
        assert np.isfinite(packed).all()

    def test_every_chunk_is_covered_by_some_primitive(self):
        """A chunk absent from every primitive would vanish from the surface."""
        body = chain(6)
        packed = sdf.pack(body)
        ends = np.vstack([packed[:, 0:3], packed[:, 4:7]])

        for slot in range(body.n_chunks):
            centre = body.pos[slot]
            assert np.isclose(ends, centre).all(axis=1).any()


class TestMechanismLinks:
    """render=False carries force without flesh — the contract angle limits need."""

    def test_a_mechanism_link_emits_no_cone(self):
        body = chain(3)
        ids = body.chunk_ids()
        body.add_link(ids[0], ids[2], render=False)

        assert body.n_links == 3
        assert sdf.primitive_count(body) == 2

    def test_a_chunk_held_only_by_mechanism_links_keeps_its_sphere(self):
        """Held by braces, not fused into them."""
        body = Body()
        a = body.add_chunk("a", [0.0, 0.0, 0.0], radius=0.3)
        b = body.add_chunk("b", [1.0, 0.0, 0.0], radius=0.4)
        body.add_link(a, b, render=False)

        packed = sdf.pack(body)
        assert len(packed) == 2
        for row in packed:
            assert row[0:3] == pytest.approx(row[4:7])

    def test_rendered_geometry_is_unchanged_by_adding_a_brace(self):
        body = chain(4)
        ids = body.chunk_ids()
        before = sdf.pack(body).copy()
        body.add_link(ids[0], ids[2], render=False)
        body.add_link(ids[1], ids[3], render=False)

        after = sdf.pack(body)
        assert np.array_equal(np.sort(before, axis=0), np.sort(after, axis=0))


class TestBlendGroups:
    """One creature per connected component; the table describes the packing."""

    @staticmethod
    def _herd() -> Body:
        """Two linked worms with the link build order deliberately interleaved,
        so the group-major sort in pack() has real work to do."""
        body = Body()
        a = [body.add_chunk(f"a{i}", (i * 1.0, 0.0, 0.0), radius=0.4) for i in range(3)]
        b = [body.add_chunk(f"b{i}", (i * 1.0, 3.0, 0.0), radius=0.3) for i in range(2)]
        body.add_link(b[0], b[1])
        body.add_link(a[0], a[1])
        body.add_link(a[1], a[2])
        return body

    def test_pack_is_group_major(self):
        body = self._herd()
        packed = sdf.pack(body)

        assert len(packed) == 3
        # Creature A owns the first-seen slot, so its two cones come first
        # even though its links were added after B's.
        assert packed[0, 1] == pytest.approx(0.0)  # a-row y
        assert packed[1, 1] == pytest.approx(0.0)
        assert packed[2, 1] == pytest.approx(3.0)  # b-row y

    def test_the_table_describes_the_packing(self):
        body = self._herd()
        table = sdf.group_table(body)

        assert table.shape == (2, 8)
        assert table[:, 0].tolist() == [0.0, 2.0]  # starts
        assert table[:, 1].tolist() == [2.0, 1.0]  # counts
        assert table[:, 1].sum() == len(sdf.pack(body))

    def test_group_zero_wears_the_authored_colour(self):
        table = sdf.group_table(self._herd(), base_colour=(0.2, 0.5, 0.8))
        assert table[0, 4:7] == pytest.approx((0.2, 0.5, 0.8))
        assert not np.allclose(table[1, 4:7], table[0, 4:7])

    def test_a_mechanism_link_joins_identity_without_joining_flesh(self):
        """A knee brace is part of the creature it stops: one group, and the
        brace still emits no primitive."""
        body = self._herd()
        body.add_link(0, 3, render=False)

        table = sdf.group_table(body)
        assert table.shape == (1, 8)
        assert len(sdf.pack(body)) == 3

    def test_a_lone_chunk_is_its_own_creature(self):
        body = self._herd()
        body.add_chunk("pebble", (5.0, 5.0, 0.0), radius=0.2)

        table = sdf.group_table(body)
        assert table.shape == (3, 8)
        assert table[2, 1] == pytest.approx(1.0)

    def test_an_empty_body_has_an_empty_table(self):
        assert sdf.group_table(Body()).shape == (0, 8)


class TestGroupBounds:
    """The sphere must cover the group's whole surface, or the cull lies."""

    def test_the_sphere_covers_every_end_ball_plus_blend(self):
        body = TestBlendGroups._herd()
        primitives = sdf.pack(body)
        table = sdf.group_table(body)
        k = 0.35
        bounds = sdf.group_bounds(primitives, table, k=k)

        for group, row in enumerate(table):
            start, count = int(row[0]), int(row[1])
            centre, radius = bounds[group, 0:3], float(bounds[group, 3])
            for prim in primitives[start : start + count]:
                for end, r in ((prim[0:3], prim[3]), (prim[4:7], prim[7])):
                    reach = float(np.linalg.norm(end - centre)) + float(r)
                    assert reach + k <= radius + 1e-5

    def test_the_lower_bound_never_exceeds_the_field(self):
        """|p - c| - r <= field everywhere is the cull's entire correctness."""
        from mesh_xml.render.field import field as cpu_field

        body = TestBlendGroups._herd()
        primitives = sdf.pack(body)
        table = sdf.group_table(body)
        k = 0.4
        bounds = sdf.group_bounds(primitives, table, k=k)
        points = np.random.default_rng(11).uniform(-4.0, 7.0, size=(200, 3))

        for group, row in enumerate(table):
            start, count = int(row[0]), int(row[1])
            rows = primitives[start : start + count]
            d = cpu_field(points, rows, k)
            lower = np.linalg.norm(points - bounds[group, 0:3], axis=1) - bounds[group, 3]
            assert (lower <= d + 1e-5).all()

    def test_an_empty_group_gets_a_zero_sphere(self):
        table = np.array([[0.0, 0.0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        bounds = sdf.group_bounds(np.zeros((0, 8), dtype=np.float32), table)
        assert bounds.shape == (1, 8)
        assert bounds[0, 3] == 0.0


class TestAuthoredColour:
    def test_meta_colour_outranks_the_palette(self):
        body = TestBlendGroups._herd()
        first_a = body.chunk_ids()[0]
        body.meta_of(first_a)["colour"] = "0.9 0.1 0.2"

        table = sdf.group_table(body, base_colour=(0.2, 0.5, 0.8))
        assert table[0, 4:7] == pytest.approx((0.9, 0.1, 0.2))
        assert not np.allclose(table[1, 4:7], table[0, 4:7])

    def test_any_chunk_of_the_component_can_carry_it(self):
        body = TestBlendGroups._herd()
        # b-worm chunks are slots 3-4; paint the second one only.
        b_last = body.chunk_ids()[4]
        body.meta_of(b_last)["colour"] = "0 1 0"

        table = sdf.group_table(body)
        assert table[1, 4:7] == pytest.approx((0.0, 1.0, 0.0))

    def test_garbage_meta_falls_back_to_the_palette(self):
        body = TestBlendGroups._herd()
        body.meta_of(body.chunk_ids()[0])["colour"] = "salmon"

        table = sdf.group_table(body, base_colour=(0.2, 0.5, 0.8))
        assert table[0, 4:7] == pytest.approx((0.2, 0.5, 0.8))

    def test_values_clamp_into_unit_range(self):
        body = TestBlendGroups._herd()
        body.meta_of(body.chunk_ids()[0])["colour"] = "2.0 -1.0 0.5"

        table = sdf.group_table(body)
        assert table[0, 4:7] == pytest.approx((1.0, 0.0, 0.5))


class TestPackAttributes:
    """The weighted skin's table (docs/TEXTURES.md): paired with pack row
    for row, and empty exactly when the flat path would draw the same
    picture — that emptiness is what makes "unpainted means unchanged"
    structural."""

    def test_a_bare_body_packs_empty(self):
        assert sdf.pack_attributes(chain(3)).shape == (0, 8)

    def test_a_uniformly_painted_creature_packs_empty(self):
        body = chain(3)
        for cid in body.chunk_ids():
            body.meta_of(cid)[sdf.COLOUR_META] = "0.9 0.1 0.1"
        assert sdf.pack_attributes(body).shape == (0, 8)

    def test_two_colours_arm_the_table(self):
        body = chain(3)
        body.meta_of(body.chunk_ids()[0])[sdf.COLOUR_META] = "1 0 0"
        body.meta_of(body.chunk_ids()[1])[sdf.COLOUR_META] = "0 1 0"
        rows = sdf.pack_attributes(body)
        assert rows.shape == (sdf.primitive_count(body), 8)

    def test_rows_pair_the_primitives_end_for_end(self):
        body = chain(3)
        by_x = {0.0: (1.0, 0.0, 0.0), 1.0: (0.0, 1.0, 0.0), 2.0: (0.0, 0.0, 1.0)}
        for i, cid in enumerate(body.chunk_ids()):
            body.meta_of(cid)[sdf.COLOUR_META] = " ".join(
                str(v) for v in by_x[float(i)]
            )

        prims = sdf.pack(body)
        attrs = sdf.pack_attributes(body)
        for prim, attr in zip(prims, attrs):
            assert tuple(attr[0:3]) == by_x[float(prim[0])]
            assert tuple(attr[4:7]) == by_x[float(prim[4])]
            assert attr[3] == 0.0 and attr[7] == 0.0, "pattern lanes reserved"

    def test_a_lonely_chunk_wears_its_colour_on_both_ends(self):
        body = chain(2)
        body.meta_of(body.chunk_ids()[0])[sdf.COLOUR_META] = "1 0 0"
        body.meta_of(body.chunk_ids()[1])[sdf.COLOUR_META] = "0 1 0"
        body.add_chunk("free", [5.0, 0.0, 0.0], radius=0.4, meta={sdf.COLOUR_META: "0 0 1"})

        prims = sdf.pack(body)
        attrs = sdf.pack_attributes(body)
        lonely = next(
            i for i, prim in enumerate(prims) if float(prim[0]) == 5.0
        )
        assert tuple(attrs[lonely][0:3]) == (0.0, 0.0, 1.0)
        assert tuple(attrs[lonely][4:7]) == (0.0, 0.0, 1.0)

    def test_a_pattern_arms_the_table_over_uniform_colours(self):
        """A striped creature varies by definition."""
        body = chain(3)
        for cid in body.chunk_ids():
            body.meta_of(cid)[sdf.PATTERN_META] = "stripes"

        rows = sdf.pack_attributes(body)
        assert rows.shape == (sdf.primitive_count(body), 8)
        assert (rows[:, 3] == 1.0).all(), "the id rides the a texel's lane"
        assert (rows[:, 7] == sdf.PATTERN_DEFAULTS["stripes"]).all(), (
            "an unset parameter takes the pattern's default"
        )

    def test_pattern_lanes_are_per_component(self):
        body = Body()
        for creature, (name, param) in enumerate(
            (("stripes", "3.0"), ("countershade", "0.5"))
        ):
            ids = [
                body.add_chunk(
                    f"w{creature}c{i}",
                    [float(i), creature * 5.0, 0.0],
                    radius=0.3,
                    meta={sdf.PATTERN_META: name, sdf.PATTERN_PARAM_META: param},
                )
                for i in range(3)
            ]
            for a, b in zip(ids, ids[1:]):
                body.add_link(a, b)

        rows = sdf.pack_attributes(body)
        assert rows.shape[0] == 4, "two links per creature, group-major"
        assert tuple(rows[:, 3]) == (1.0, 1.0, 2.0, 2.0)
        assert tuple(rows[:, 7]) == (3.0, 3.0, 0.5, 0.5)

    def test_pattern_of_reads_like_the_colour_reader(self):
        body = chain(3)
        ids = body.chunk_ids()
        assert sdf.pattern_of(body, ids) == ("none", 0.0)

        body.meta_of(ids[0])[sdf.PATTERN_META] = "tartan"  # not in the vocabulary
        body.meta_of(ids[1])[sdf.PATTERN_META] = "stripes"
        body.meta_of(ids[1])[sdf.PATTERN_PARAM_META] = "garbage"
        name, param = sdf.pattern_of(body, ids)
        assert name == "stripes", "unknown names are passed over, not raised on"
        assert param == sdf.PATTERN_DEFAULTS["stripes"], "garbage takes the default"

        body.meta_of(ids[1])[sdf.PATTERN_PARAM_META] = "6.5"
        assert sdf.pattern_of(body, ids) == ("stripes", 6.5)

    def test_the_fallback_is_the_palette_not_the_flat_colour(self):
        """Painting a head must not repaint the body through the fallback.

        The flat path colours a whole creature by its first authored chunk;
        the weighted skin's unpainted chunks fall back to the *palette*
        colour instead, which is what makes painting one chunk a statement
        about one chunk.
        """
        base = (0.2, 0.4, 0.8)
        unpainted = chain(3)
        palette = sdf.group_table(unpainted, base)[0, 4:7]

        body = chain(3)
        body.meta_of(body.chunk_ids()[2])[sdf.COLOUR_META] = "1 0 0"
        attrs = sdf.pack_attributes(body, base)
        assert attrs.shape[0] == 2
        first_link = attrs[0]
        assert first_link[0:3] == pytest.approx(palette)
        assert first_link[4:7] == pytest.approx(palette)
        second_link = attrs[1]
        assert second_link[4:7] == pytest.approx((1.0, 0.0, 0.0))
