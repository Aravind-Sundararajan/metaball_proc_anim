"""Regression tests for the core body graph.

Several of these encode defects that were reproduced against the old
``ArmatureManager`` before the rewrite; they are named after the section of
``docs/REFACTOR_PLAN.md`` that documents them so the connection survives.
"""

import numpy as np
import pytest

from mesh_xml.core import Body, LinkType


def square_with_diagonal() -> tuple[Body, list[int]]:
    """Four chunks in a ring plus one diagonal — the smallest cyclic body."""
    body = Body()
    ids = [
        body.add_chunk("a", [0, 0, 0]),
        body.add_chunk("b", [1, 0, 0]),
        body.add_chunk("c", [1, 1, 0]),
        body.add_chunk("d", [0, 1, 0]),
    ]
    body.add_link(ids[0], ids[1])
    body.add_link(ids[1], ids[2])
    body.add_link(ids[2], ids[3])
    body.add_link(ids[3], ids[0])
    body.add_link(ids[0], ids[2])
    return body, ids


class TestLinkIdentity:
    def test_reversed_duplicate_link_is_a_noop(self):
        """Defect 2.1: (A,B) then (B,A) made A its own ancestor and emptied the export."""
        body = Body()
        a = body.add_chunk("a", [0, 0, 0])
        b = body.add_chunk("b", [1, 0, 0])

        assert body.add_link(a, b) is True
        assert body.add_link(b, a) is False
        assert body.n_links == 1

    def test_link_lookup_is_undirected(self):
        body = Body()
        a = body.add_chunk("a", [0, 0, 0])
        b = body.add_chunk("b", [1, 0, 0])
        body.add_link(a, b)

        assert body.has_link(a, b)
        assert body.has_link(b, a)
        assert body.remove_link(b, a) is True
        assert body.n_links == 0

    def test_self_link_rejected(self):
        body = Body()
        a = body.add_chunk("a", [0, 0, 0])
        with pytest.raises(ValueError):
            body.add_link(a, a)

    def test_rest_defaults_to_current_separation(self):
        body = Body()
        a = body.add_chunk("a", [0, 0, 0])
        b = body.add_chunk("b", [3, 4, 0])
        body.add_link(a, b)

        assert next(body.links()).rest == pytest.approx(5.0)


class TestCycles:
    def test_cycles_are_legal(self):
        """Defect 2.2: closing any loop was rejected, so no girdle or ribcage was expressible."""
        body, _ = square_with_diagonal()
        assert body.n_links == 5

    def test_spanning_forest_is_total(self):
        """Every chunk appears exactly once however cyclic the graph is."""
        body, ids = square_with_diagonal()
        children, roots, extra = body.spanning_forest()

        visited = []

        def walk(node):
            visited.append(node)
            for child in children[node]:
                walk(child)

        for root in roots:
            walk(root)

        assert sorted(visited) == sorted(ids)
        assert len(visited) == len(set(visited))

    def test_non_tree_edges_are_reported_not_dropped(self):
        body, _ = square_with_diagonal()
        children, roots, extra = body.spanning_forest()

        tree_edges = sum(len(kids) for kids in children.values())
        assert tree_edges + len(extra) == body.n_links

    def test_disconnected_components_each_get_a_root(self):
        body = Body()
        a = body.add_chunk("a", [0, 0, 0])
        b = body.add_chunk("b", [1, 0, 0])
        c = body.add_chunk("c", [9, 9, 9])
        body.add_link(a, b)

        children, roots, extra = body.spanning_forest()
        assert len(roots) == 2
        assert c in roots

    def test_preferred_root_is_honoured(self):
        body, ids = square_with_diagonal()
        children, roots, extra = body.spanning_forest(roots=[ids[2]])
        assert roots[0] == ids[2]


class TestIdentity:
    def test_duplicate_names_are_allowed(self):
        """Defect 2.6: two spline generations both emitted Spline_1..N and overwrote silently."""
        body = Body()
        first = body.add_chunk("Spline_1", [0, 0, 0])
        second = body.add_chunk("Spline_1", [5, 0, 0])

        assert first != second
        assert body.n_chunks == 2
        assert body.chunk(first).pos != body.chunk(second).pos

    def test_rename_does_not_disturb_links(self):
        body = Body()
        a = body.add_chunk("a", [0, 0, 0])
        b = body.add_chunk("b", [1, 0, 0])
        body.add_link(a, b)

        body.rename(a, "spine")
        assert body.name_of(a) == "spine"
        assert body.has_link(a, b)

    def test_ids_are_stable_across_removal(self):
        body = Body()
        ids = [body.add_chunk(f"c{i}", [i, 0, 0]) for i in range(5)]
        body.remove_chunk(ids[1])

        for kept in (ids[0], ids[2], ids[3], ids[4]):
            assert body.has_chunk(kept)
            assert body.chunk(kept).name == f"c{kept}"
        assert not body.has_chunk(ids[1])

    def test_unknown_id_raises(self):
        body = Body()
        with pytest.raises(KeyError):
            body.chunk(99)


class TestRemoval:
    def test_removing_a_chunk_removes_incident_links(self):
        body, ids = square_with_diagonal()
        body.remove_chunk(ids[0])

        assert body.n_chunks == 3
        assert body.n_links == 2
        for link in body.links():
            assert ids[0] not in (link.a, link.b)

    def test_link_slots_stay_valid_after_swap_with_last(self):
        """Removal relocates the last chunk, so link slot references must follow it."""
        body, ids = square_with_diagonal()
        body.remove_chunk(ids[1])

        assert all(0 <= s < body.n_chunks for s in body.link_a)
        assert all(0 <= s < body.n_chunks for s in body.link_b)

        expected = {(min(link.a, link.b), max(link.a, link.b)) for link in body.links()}
        assert expected == {
            (min(ids[2], ids[3]), max(ids[2], ids[3])),
            (min(ids[3], ids[0]), max(ids[3], ids[0])),
            (min(ids[0], ids[2]), max(ids[0], ids[2])),
        }

    def test_removing_last_chunk_is_handled(self):
        body = Body()
        a = body.add_chunk("a", [0, 0, 0])
        b = body.add_chunk("b", [1, 0, 0])
        body.add_link(a, b)
        body.remove_chunk(b)

        assert body.n_chunks == 1
        assert body.n_links == 0
        assert body.chunk(a).name == "a"


class TestStorage:
    def test_arrays_survive_capacity_growth(self):
        """np.resize repeats data when growing; the live prefix must still be intact."""
        body = Body()
        ids = [body.add_chunk(f"c{i}", [i, i * 2, i * 3], radius=0.1 * (i + 1)) for i in range(40)]

        assert body.n_chunks == 40
        for i, cid in enumerate(ids):
            chunk = body.chunk(cid)
            assert chunk.pos == pytest.approx((i, i * 2, i * 3))
            assert chunk.radius == pytest.approx(0.1 * (i + 1), rel=1e-5)

    def test_links_survive_capacity_growth(self):
        body = Body()
        ids = [body.add_chunk(f"c{i}", [i, 0, 0]) for i in range(40)]
        for i in range(39):
            assert body.add_link(ids[i], ids[i + 1]) is True

        assert body.n_links == 39
        assert all(link.rest == pytest.approx(1.0) for link in body.links())

    def test_pos_property_is_a_live_view(self):
        body = Body()
        a = body.add_chunk("a", [0, 0, 0])
        body.pos[0] += np.array([1, 2, 3], dtype=np.float32)

        assert body.chunk(a).pos == pytest.approx((1, 2, 3))

    def test_set_pos_clears_implicit_velocity(self):
        body = Body()
        a = body.add_chunk("a", [0, 0, 0])
        body.pos[0] = [5, 0, 0]
        body.set_pos(a, [9, 0, 0])

        assert body.pos[0] == pytest.approx(body.prev_pos[0])

    def test_pinned_chunk_has_zero_inverse_mass(self):
        body = Body()
        a = body.add_chunk("anchor", [0, 0, 0], pinned=True)
        b = body.add_chunk("free", [1, 0, 0], mass=2.0)

        assert body.inv_mass[body.slot(a)] == 0.0
        assert body.inv_mass[body.slot(b)] == pytest.approx(0.5)

    def test_dim_is_respected(self):
        body = Body(dim=2)
        a = body.add_chunk("a", [1, 2])
        assert body.chunk(a).pos == pytest.approx((1, 2))
        with pytest.raises(ValueError):
            body.add_chunk("bad", [1, 2, 3])


class TestValidation:
    @pytest.mark.parametrize("radius", [0.0, -1.0])
    def test_non_positive_radius_rejected(self, radius):
        body = Body()
        with pytest.raises(ValueError):
            body.add_chunk("a", [0, 0, 0], radius=radius)

    def test_elasticity_bounds_enforced(self):
        body = Body()
        a = body.add_chunk("a", [0, 0, 0])
        b = body.add_chunk("b", [1, 0, 0])
        with pytest.raises(ValueError):
            body.add_link(a, b, elasticity=1.5)

    def test_link_type_round_trips(self):
        body = Body()
        a = body.add_chunk("a", [0, 0, 0])
        b = body.add_chunk("b", [1, 0, 0])
        body.add_link(a, b, type=LinkType.PULL)

        assert next(body.links()).type is LinkType.PULL
