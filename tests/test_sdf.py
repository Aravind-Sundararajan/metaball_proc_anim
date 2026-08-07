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
