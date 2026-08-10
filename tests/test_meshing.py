"""Tests for polygonisation and mesh export.

A mesher has no useful regression test. "The sphere still has 19232 vertices" is
true of a correct mesh and of a subtly wrong one, and the number changes for
reasons that are not defects. So almost nothing here compares against a stored
value. It compares against things known independently of the code:

* a sphere's vertices against ``|p - c| - r``, and the mesh's enclosed volume
  against ``4/3 pi r**3``, which the mesher has no way of knowing;
* the surface's topology against Euler's formula and against every directed edge
  appearing exactly once, which is what closed and consistently wound *means*;
* the normals against the field's own gradient, by finite difference through the
  surface rather than by re-reading :func:`~mesh_xml.render.field.normal`;
* the ``k`` bound the watertightness argument rests on, against randomly
  generated primitive configurations.

The convergence tests matter more than the absolute ones. A mesher can be wrong
in a way that a single resolution hides — a half-cell bias reads as a slightly
small sphere and nothing else — and only shows up as an error that stops
falling when the cells shrink.

``RuntimeWarning`` is an error for the whole module, for the same reason it is
in ``tests/test_field.py``: the crossing interpolation divides by a difference
of field values and the STL facet normal divides by a length, and a NaN in
either produces a file that opens and is wrong rather than one that fails.
"""

from __future__ import annotations

import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from mesh_xml.io.mesh_export import write_mesh, write_obj, write_stl
from mesh_xml.io.xml_legacy import load_legacy
from mesh_xml.render import sdf
from mesh_xml.render.field import field, round_cone
from mesh_xml.render.meshing import Mesh, polygonise

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

WORM = Path(__file__).resolve().parent / "data" / "legacy_v1_chubby_worm.xml"

#: Blend radius the worm is meshed at throughout. Large enough that the chunks
#: fuse into one surface rather than reading as beads, which is the case the
#: mesher actually has to handle.
WORM_BLEND = 0.25


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def ball(centre, radius: float) -> np.ndarray:
    """One degenerate round cone, which is what :func:`sdf.pack` emits for a sphere."""
    return np.array([[*centre, radius, *centre, radius]], dtype=np.float64)


def worm_primitives() -> np.ndarray:
    return sdf.pack(load_legacy(WORM))


def corners(mesh: Mesh) -> np.ndarray:
    """``(F, 3, 3)`` — the three vertex positions of every face."""
    return mesh.vertices[mesh.faces]


def enclosed_volume(mesh: Mesh) -> float:
    """Signed volume by the divergence theorem, ``sum (v0 x v1) . v2 / 6``.

    Positive only when the winding is outward, so this doubles as an orientation
    check: a mesh wound the wrong way reports its volume as negative rather than
    as nearly right.
    """
    v = corners(mesh)
    return float(np.einsum("ij,ij->i", np.cross(v[:, 0], v[:, 1]), v[:, 2]).sum() / 6.0)


def face_areas(mesh: Mesh) -> np.ndarray:
    v = corners(mesh)
    return 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)


def directed_edges(mesh: Mesh) -> np.ndarray:
    """``(3F, 2)`` — every face's three edges, keeping their direction."""
    f = mesh.faces
    return np.concatenate([f[:, (0, 1)], f[:, (1, 2)], f[:, (2, 0)]])


def component_count(mesh: Mesh) -> int:
    """Connected components of the face graph, by union-find over face corners."""
    parent = list(range(len(mesh.vertices)))

    def root(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b, c in mesh.faces:
        parent[root(int(b))] = parent[root(int(c))] = root(int(a))
    return len({root(i) for i in range(len(mesh.vertices))})


def radial_error(mesh: Mesh, centre, radius: float) -> np.ndarray:
    """How far each vertex is from an analytic sphere, signed."""
    return np.linalg.norm(mesh.vertices - np.asarray(centre), axis=1) - radius


def parse_obj(path: Path) -> tuple[int, int, int]:
    """Count ``v``, ``vn`` and ``f`` records, without using our own reader.

    Deliberately naive — split on whitespace and look at the keyword. A writer
    bug that a lenient parser would forgive is exactly what this has to catch.
    """
    counts = {"v": 0, "vn": 0, "f": 0}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and parts[0] in counts:
            assert len(parts) == 4, f"expected 3 operands, got {line!r}"
            counts[parts[0]] += 1
    return counts["v"], counts["vn"], counts["f"]


# ----------------------------------------------------------------------
# against an analytic sphere
# ----------------------------------------------------------------------


class TestAnalyticSphere:
    def test_every_vertex_is_within_one_cell_diagonal_of_the_surface(self):
        """The bound the whole method rests on.

        A surface-net vertex is the mean of crossings on one cell's edges, so it
        lies in a cell the surface passes through and cannot be further from
        that surface than the cell's diameter. Anything further means the vertex
        was placed in the wrong cell.
        """
        centre, radius = (0.3, -0.7, 1.1), 1.7
        mesh = polygonise(ball(centre, radius), 0.0, 48)

        assert np.abs(radial_error(mesh, centre, radius)).max() < mesh.cell * np.sqrt(3)

    def test_the_error_is_far_inside_that_bound_in_practice(self):
        """The diagonal is the guarantee; a tenth of a cell is the behaviour."""
        mesh = polygonise(ball((0.0, 0.0, 0.0), 1.0), 0.0, 48)

        assert np.abs(radial_error(mesh, (0, 0, 0), 1.0)).max() < 0.1 * mesh.cell

    def test_volume_converges_on_four_thirds_pi_r_cubed(self):
        radius = 1.3
        exact = 4.0 / 3.0 * np.pi * radius**3

        errors = [
            abs(enclosed_volume(polygonise(ball((0, 0, 0), radius), 0.0, res)) - exact)
            / exact
            for res in (16, 32, 64)
        ]

        assert errors == sorted(errors, reverse=True), errors
        assert errors[-1] < 0.005, errors

    def test_higher_resolution_reduces_the_error(self):
        """Second order, so each doubling should take roughly a quarter off.

        Asserting only that it falls would pass a mesher that had converged to
        the wrong surface; asserting the rate is what says the cells are what
        the error is made of.
        """
        centre, radius = (0.0, 0.0, 0.0), 1.0
        errors = [
            np.abs(radial_error(polygonise(ball(centre, radius), 0.0, res), centre, radius)).max()
            for res in (16, 32, 64)
        ]

        assert errors == sorted(errors, reverse=True), errors
        for coarse, fine in zip(errors, errors[1:]):
            assert fine < 0.35 * coarse, errors

    def test_the_sphere_is_not_biased_inward_or_outward(self):
        """A half-cell placement bug shows here and nowhere else.

        Mean radial error near zero says the vertices straddle the true surface.
        A mesher that consistently rounded to one side would still pass the
        cell-diagonal bound and still converge — just to a sphere of the wrong
        radius.
        """
        mesh = polygonise(ball((0, 0, 0), 1.0), 0.0, 32)

        assert abs(radial_error(mesh, (0, 0, 0), 1.0).mean()) < 0.02 * mesh.cell


# ----------------------------------------------------------------------
# topology
# ----------------------------------------------------------------------


class TestWatertight:
    @pytest.mark.parametrize("resolution", [17, 32, 48])
    def test_a_sphere_has_no_boundary_edges(self, resolution: int):
        """Every undirected edge is shared by exactly two faces.

        This is what closed means, and it is checked at an odd resolution too
        because an even one puts the sphere's centre on a lattice point and can
        hide an off-by-one that only bites when the surface is off-lattice.
        """
        mesh = polygonise(ball((0, 0, 0), 1.0), 0.0, resolution)
        edges = np.sort(directed_edges(mesh), axis=1)

        _, counts = np.unique(edges, axis=0, return_counts=True)
        assert set(counts.tolist()) == {2}

    def test_a_sphere_is_consistently_wound(self):
        """Every directed edge appears once, so no two faces disagree on outside."""
        mesh = polygonise(ball((0, 0, 0), 1.0), 0.0, 32)

        _, counts = np.unique(directed_edges(mesh), axis=0, return_counts=True)
        assert set(counts.tolist()) == {1}

    def test_a_sphere_has_the_euler_characteristic_of_a_sphere(self):
        mesh = polygonise(ball((0, 0, 0), 1.0), 0.0, 32)
        edges = np.unique(np.sort(directed_edges(mesh), axis=1), axis=0)

        assert len(mesh.vertices) - len(edges) + len(mesh.faces) == 2

    def test_the_worm_comes_out_closed(self):
        mesh = polygonise(worm_primitives(), WORM_BLEND, 64)
        edges = np.sort(directed_edges(mesh), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)

        assert mesh.closed
        assert mesh.open_edges == 0
        assert set(counts.tolist()) == {2}

    def test_the_fold_never_sinks_more_than_k_below_the_hard_minimum(self):
        """The premise of the watertightness argument, checked rather than assumed.

        :func:`~mesh_xml.render.field.bounds` inflates by ``k`` and warns that a
        pile-up could in principle sink ``(P - 1) * k / 4``. It cannot: past a
        separation of ``k`` the smooth minimum is the hard one, so the inflation
        converges instead of accumulating. If this ever fails, the sampled box
        is too small and holes become reachable.
        """
        rng = np.random.default_rng(7)

        for _ in range(40):
            count = int(rng.integers(1, 14))
            prims = np.concatenate(
                [
                    rng.normal(0.0, 1.2, (count, 3)),
                    rng.uniform(0.05, 0.9, (count, 1)),
                    rng.normal(0.0, 1.2, (count, 3)),
                    rng.uniform(0.05, 0.9, (count, 1)),
                ],
                axis=1,
            )
            k = float(rng.uniform(0.0, 3.0))
            points = rng.normal(0.0, 2.5, (2000, 3))

            hard = np.min([round_cone(points, p) for p in prims], axis=0)
            assert (hard - field(points, prims, k)).max() <= k + 1e-9


class TestFaces:
    def test_every_index_addresses_a_real_vertex(self):
        mesh = polygonise(worm_primitives(), WORM_BLEND, 48)

        assert mesh.faces.min() >= 0
        assert mesh.faces.max() < len(mesh.vertices)

    def test_no_face_repeats_a_vertex(self):
        mesh = polygonise(worm_primitives(), WORM_BLEND, 48)
        a, b, c = mesh.faces.T

        assert np.all((a != b) & (b != c) & (c != a))

    @pytest.mark.parametrize("resolution", [16, 32, 48])
    def test_no_face_has_zero_area(self, resolution: int):
        """Slivers are what a repair pass and a slicer both choke on.

        The margin the cell vertex is held back by keeps distinct cells' vertices
        distinct; collinearity of three of them would still be degenerate, and
        this is the check that it does not happen.
        """
        mesh = polygonise(worm_primitives(), WORM_BLEND, resolution)
        areas = face_areas(mesh)

        assert areas.min() > 0.0
        assert areas.min() > 1e-4 * mesh.cell**2


# ----------------------------------------------------------------------
# connectivity
# ----------------------------------------------------------------------


class TestConnectivity:
    def test_two_blended_spheres_are_one_surface(self):
        """They do not touch; the blend is the only reason they join.

        ``k = 0.6`` is exactly the tangency for this pair — the field is zero at
        the midpoint and the neck has no thickness at all — so the joined case
        is asked for well past it, where the answer is about the surface rather
        than about which side of zero a rounding error fell on.
        """
        apart = np.concatenate([ball((-1.15, 0, 0), 1.0), ball((1.15, 0, 0), 1.0)])

        assert component_count(polygonise(apart, 0.0, 48)) == 2
        assert component_count(polygonise(apart, 1.2, 48)) == 1

    def test_two_far_apart_spheres_stay_two_surfaces(self):
        far = np.concatenate([ball((-5.0, 0, 0), 1.0), ball((5.0, 0, 0), 1.0)])
        mesh = polygonise(far, 0.3, 48)

        assert component_count(mesh) == 2
        assert mesh.closed

    def test_a_blended_pair_encloses_more_than_two_spheres_would(self):
        """One component is a count; the neck between them is the geometry.

        A mesher that merged the two components by mistake — a stray index, a
        shared vertex where there should be two — would satisfy the count and
        enclose no more volume than the balls it started from. The smooth
        minimum only ever inflates, so a genuine join has to show up as extra
        volume, and by more than the ~1% a lattice this coarse loses.
        """
        apart = np.concatenate([ball((-1.15, 0, 0), 1.0), ball((1.15, 0, 0), 1.0)])
        two_balls = 2.0 * (4.0 / 3.0 * np.pi)

        assert enclosed_volume(polygonise(apart, 1.2, 48)) > 1.03 * two_balls


# ----------------------------------------------------------------------
# normals
# ----------------------------------------------------------------------


class TestNormals:
    def test_there_is_one_per_vertex_and_all_are_unit_length(self):
        mesh = polygonise(worm_primitives(), WORM_BLEND, 48)

        assert mesh.normals.shape == mesh.vertices.shape
        assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0)

    def test_the_field_increases_along_them(self):
        """Outward, measured on the field itself rather than trusted.

        A step of a quarter cell each way has to cross the surface, so the
        difference is positive at every vertex if the normals point out and
        negative at every vertex if the whole mesh is inverted.
        """
        prims = worm_primitives()
        mesh = polygonise(prims, WORM_BLEND, 48)
        step = 0.25 * mesh.cell

        ahead = field(mesh.vertices + step * mesh.normals, prims, WORM_BLEND)
        behind = field(mesh.vertices - step * mesh.normals, prims, WORM_BLEND)

        assert np.all(ahead > behind)

    def test_they_match_the_analytic_sphere_normal(self):
        """The field's gradient is exact, so this is a hard tolerance, not a soft one.

        Face-averaged normals on the same mesh would be off by degrees at this
        resolution — which is the entire reason they are not used.
        """
        mesh = polygonise(ball((0, 0, 0), 1.0), 0.0, 32)
        radial = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1)[:, None]

        assert np.einsum("ij,ij->i", mesh.normals, radial).min() > 0.9999

    def test_they_are_not_face_normals_in_disguise(self):
        """A coarse mesh's faces are visibly not tangent to the sphere.

        If the normals were derived from the faces they would carry the
        polygonisation's faceting; the point is that they do not.
        """
        mesh = polygonise(ball((0, 0, 0), 1.0), 0.0, 12)
        v = corners(mesh)
        geometric = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        geometric /= np.linalg.norm(geometric, axis=1)[:, None]
        radial = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1)[:, None]

        exact = np.einsum("ij,ij->i", mesh.normals, radial).min()
        faceted = np.einsum("ij,ij->i", geometric, radial[mesh.faces[:, 0]]).min()
        assert exact > 0.999 > faceted


# ----------------------------------------------------------------------
# degenerate input
# ----------------------------------------------------------------------


class TestDegenerate:
    def test_no_primitives_gives_an_empty_mesh(self):
        """A body mid-edit exports an empty file rather than raising."""
        mesh = polygonise(np.zeros((0, 8)), 0.3, 32)

        assert len(mesh.vertices) == len(mesh.normals) == len(mesh.faces) == 0
        assert mesh.faces.shape == (0, 3)
        assert mesh.closed

    def test_a_body_with_no_extent_gives_an_empty_mesh(self):
        """Zero radius and no blend: the zero set is a point, and a point is no surface."""
        mesh = polygonise(ball((1.0, 2.0, 3.0), 0.0), 0.0, 32)

        assert len(mesh.faces) == 0

    def test_an_empty_mesh_still_writes_a_file(self, tmp_path: Path):
        mesh = polygonise(np.zeros((0, 8)), 0.3, 32)
        path = write_obj(tmp_path / "empty.obj", mesh.vertices, mesh.faces, mesh.normals)

        assert parse_obj(path) == (0, 0, 0)

    def test_a_malformed_primitive_array_is_rejected(self):
        with pytest.raises(ValueError, match="shape"):
            polygonise(np.zeros((4, 7)), 0.0, 16)

    @pytest.mark.parametrize("resolution", [0, -1])
    def test_a_useless_resolution_is_rejected(self, resolution: int):
        with pytest.raises(ValueError, match="resolution"):
            polygonise(ball((0, 0, 0), 1.0), 0.0, resolution)

    def test_one_cell_of_resolution_does_not_crash(self):
        """The smallest lattice the padding permits. It need not be a good mesh."""
        mesh = polygonise(ball((0, 0, 0), 1.0), 0.0, 1)

        assert mesh.closed


# ----------------------------------------------------------------------
# export
# ----------------------------------------------------------------------


class TestObj:
    def test_round_trip_preserves_the_counts(self, tmp_path: Path):
        mesh = polygonise(worm_primitives(), WORM_BLEND, 32)
        path = write_obj(tmp_path / "worm.obj", mesh.vertices, mesh.faces, mesh.normals)

        assert parse_obj(path) == (
            len(mesh.vertices),
            len(mesh.normals),
            len(mesh.faces),
        )

    def test_round_trip_preserves_the_positions(self, tmp_path: Path):
        """Counts alone would pass a writer that transposed or truncated."""
        mesh = polygonise(ball((0.3, -0.7, 1.1), 1.7), 0.0, 16)
        path = write_obj(tmp_path / "ball.obj", mesh.vertices, mesh.faces, mesh.normals)

        read = np.array(
            [
                [float(x) for x in line.split()[1:]]
                for line in path.read_text().splitlines()
                if line.startswith("v ")
            ]
        )
        assert np.allclose(read, mesh.vertices, rtol=1e-8, atol=1e-12)

    def test_face_indices_are_one_based(self, tmp_path: Path):
        """Off by one here rotates a corner of every triangle and looks plausible."""
        mesh = polygonise(ball((0, 0, 0), 1.0), 0.0, 12)
        path = write_obj(tmp_path / "ball.obj", mesh.vertices, mesh.faces, mesh.normals)

        indices = np.array(
            [
                [int(token.split("//")[0]) for token in line.split()[1:]]
                for line in path.read_text().splitlines()
                if line.startswith("f ")
            ]
        )
        assert np.array_equal(indices, mesh.faces + 1)
        assert indices.min() == 1

    def test_a_face_names_its_own_vertex_normal(self, tmp_path: Path):
        mesh = polygonise(ball((0, 0, 0), 1.0), 0.0, 12)
        path = write_obj(tmp_path / "ball.obj", mesh.vertices, mesh.faces, mesh.normals)

        tokens = [
            token
            for line in path.read_text().splitlines()
            if line.startswith("f ")
            for token in line.split()[1:]
        ]
        assert all(t.split("//")[0] == t.split("//")[1] for t in tokens)

    def test_faces_have_no_normal_slot_when_no_normals_are_given(self, tmp_path: Path):
        mesh = polygonise(ball((0, 0, 0), 1.0), 0.0, 12)
        path = write_obj(tmp_path / "ball.obj", mesh.vertices, mesh.faces)

        assert "//" not in path.read_text()
        assert parse_obj(path) == (len(mesh.vertices), 0, len(mesh.faces))

    def test_vertex_colours_ride_the_v_lines(self, tmp_path: Path):
        """The widely read `v x y z r g b` extension: six numbers when
        colours are given, the classic three when not — an unpainted
        export stays byte-identical."""
        mesh = polygonise(ball((0, 0, 0), 1.0), 0.0, 12)
        colours = np.tile((0.9, 0.2, 0.1), (len(mesh.vertices), 1))
        painted = write_obj(
            tmp_path / "painted.obj",
            mesh.vertices,
            mesh.faces,
            mesh.normals,
            colours=colours,
        )
        v_lines = [
            line for line in painted.read_text().splitlines() if line.startswith("v ")
        ]
        assert all(len(line.split()) == 7 for line in v_lines)
        read = np.array([[float(x) for x in line.split()[4:]] for line in v_lines])
        assert np.allclose(read, colours)

        plain = write_obj(
            tmp_path / "plain.obj", mesh.vertices, mesh.faces, mesh.normals
        )
        assert all(
            len(line.split()) == 4
            for line in plain.read_text().splitlines()
            if line.startswith("v ")
        )

    def test_colours_clamp_to_the_extensions_domain(self, tmp_path: Path):
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]])
        path = write_obj(
            tmp_path / "hot.obj",
            vertices,
            faces,
            colours=np.array([[2.0, -1.0, 0.5]] * 3),
        )
        read = [
            [float(x) for x in line.split()[4:]]
            for line in path.read_text().splitlines()
            if line.startswith("v ")
        ]
        assert read[0] == [1.0, 0.0, 0.5]

    def test_mismatched_colours_are_refused(self, tmp_path: Path):
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]])
        with pytest.raises(ValueError, match="one RGB per vertex"):
            write_obj(
                tmp_path / "bad.obj", vertices, faces, colours=np.zeros((2, 3))
            )

    def test_stl_accepts_and_ignores_colours(self, tmp_path: Path):
        mesh = polygonise(ball((0, 0, 0), 1.0), 0.0, 12)
        colours = np.tile((0.9, 0.2, 0.1), (len(mesh.vertices), 1))
        painted = write_stl(
            tmp_path / "painted.stl",
            mesh.vertices,
            mesh.faces,
            colours=colours,
        )
        plain = write_stl(tmp_path / "plain.stl", mesh.vertices, mesh.faces)
        assert painted.read_bytes() == plain.read_bytes()


class TestStl:
    def test_the_file_is_exactly_the_size_the_format_says(self, tmp_path: Path):
        """80-byte header, a count, then 50 bytes a facet. Padding here is corruption."""
        mesh = polygonise(ball((0, 0, 0), 1.0), 0.0, 16)
        path = write_stl(tmp_path / "ball.stl", mesh.vertices, mesh.faces, mesh.normals)

        assert path.stat().st_size == 84 + 50 * len(mesh.faces)

    def test_the_header_declares_the_facet_count(self, tmp_path: Path):
        mesh = polygonise(ball((0, 0, 0), 1.0), 0.0, 16)
        path = write_stl(tmp_path / "ball.stl", mesh.vertices, mesh.faces, mesh.normals)
        raw = path.read_bytes()

        assert struct.unpack("<I", raw[80:84])[0] == len(mesh.faces)

    def test_the_header_does_not_begin_with_solid(self, tmp_path: Path):
        """That word is the only thing that tells binary STL from ASCII apart."""
        mesh = polygonise(ball((0, 0, 0), 1.0), 0.0, 16)
        path = write_stl(tmp_path / "ball.stl", mesh.vertices, mesh.faces)

        assert not path.read_bytes().lstrip().startswith(b"solid")

    def test_facet_normals_agree_with_the_winding(self, tmp_path: Path):
        """STL's normal and its vertex order can disagree; a slicer trusts one of them."""
        mesh = polygonise(ball((0, 0, 0), 1.0), 0.0, 16)
        path = write_stl(tmp_path / "ball.stl", mesh.vertices, mesh.faces, mesh.normals)
        raw = path.read_bytes()

        facets = np.frombuffer(
            raw[84:],
            dtype=np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")]),
        )
        edge = facets["v"][:, 1:] - facets["v"][:, :1]
        geometric = np.cross(edge[:, 0], edge[:, 1])
        assert np.all(np.einsum("ij,ij->i", geometric, facets["n"]) > 0.0)

    def test_it_is_written_little_endian_regardless_of_host(self, tmp_path: Path):
        """A vertex at a known coordinate, read back byte for byte."""
        vertices = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]])
        path = write_stl(tmp_path / "tri.stl", vertices, np.array([[0, 1, 2]]))
        raw = path.read_bytes()

        assert struct.unpack("<3f", raw[96:108]) == (1.0, 0.0, 0.0)


class TestWriterIsIndependentOfTheMesher:
    """Hand-built geometry, never near :mod:`~mesh_xml.render.meshing`."""

    TETRAHEDRON = (
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
    )

    def test_a_tetrahedron_writes_and_reads_back(self, tmp_path: Path):
        vertices, faces = self.TETRAHEDRON
        path = write_obj(tmp_path / "tet.obj", vertices, faces)

        assert parse_obj(path) == (4, 0, 4)

    def test_normals_need_not_come_from_a_field(self, tmp_path: Path):
        vertices, faces = self.TETRAHEDRON
        normals = np.full((4, 3), 3.0**-0.5)
        path = write_obj(tmp_path / "tet.obj", vertices, faces, normals)

        assert parse_obj(path) == (4, 4, 4)

    def test_an_out_of_range_face_index_is_refused(self, tmp_path: Path):
        """OBJ has no bounds check, so this is the last place it can be caught."""
        vertices, _ = self.TETRAHEDRON

        with pytest.raises(ValueError, match="face indices"):
            write_obj(tmp_path / "bad.obj", vertices, np.array([[0, 1, 4]]))

    def test_a_negative_face_index_is_refused(self, tmp_path: Path):
        """Numpy would happily index backwards and write a plausible wrong mesh."""
        vertices, _ = self.TETRAHEDRON

        with pytest.raises(ValueError, match="face indices"):
            write_obj(tmp_path / "bad.obj", vertices, np.array([[0, 1, -1]]))

    def test_a_normal_per_face_instead_of_per_vertex_is_refused(self, tmp_path: Path):
        vertices, faces = self.TETRAHEDRON

        with pytest.raises(ValueError, match="one per vertex"):
            write_obj(tmp_path / "bad.obj", vertices, faces, np.zeros((4, 4)))

    @pytest.mark.parametrize("bad", [np.zeros((4, 2)), np.zeros(4)])
    def test_geometry_that_is_not_three_dimensional_is_refused(
        self, tmp_path: Path, bad: np.ndarray
    ):
        with pytest.raises(ValueError, match="shape"):
            write_obj(tmp_path / "bad.obj", bad, np.zeros((0, 3), dtype=int))

    def test_the_suffix_chooses_the_writer(self, tmp_path: Path):
        vertices, faces = self.TETRAHEDRON

        assert write_mesh(tmp_path / "t.obj", vertices, faces).read_bytes()[:1] == b"#"
        assert write_mesh(tmp_path / "t.STL", vertices, faces).stat().st_size == 284

    def test_an_unknown_suffix_is_refused(self, tmp_path: Path):
        vertices, faces = self.TETRAHEDRON

        with pytest.raises(ValueError, match="no writer"):
            write_mesh(tmp_path / "t.gltf", vertices, faces)


# ----------------------------------------------------------------------
# purity
# ----------------------------------------------------------------------


class TestPurity:
    """The mesher is meant to run on a worker thread, so it has to be pure.

    Purity here is not an aesthetic: the export path is expected to be handed to
    a background thread while the viewport keeps drawing from the same primitive
    array. Any cache, any scratch buffer above function scope, any write into
    the caller's array, and the two would race.
    """

    def test_it_does_not_touch_the_array_it_was_given(self):
        prims = worm_primitives()
        before = prims.copy()

        polygonise(prims, WORM_BLEND, 32)

        assert np.array_equal(prims, before)

    def test_the_same_arguments_give_the_same_mesh(self):
        prims = worm_primitives()
        first = polygonise(prims, WORM_BLEND, 32)
        second = polygonise(prims, WORM_BLEND, 32)

        assert np.array_equal(first.vertices, second.vertices)
        assert np.array_equal(first.faces, second.faces)

    def test_concurrent_calls_agree_with_serial_ones(self):
        """Four resolutions at once, against the same four run one after another."""
        prims = worm_primitives()
        resolutions = (24, 32, 40, 48)

        def mesh(resolution: int) -> Mesh:
            return polygonise(prims, WORM_BLEND, resolution)

        serial = [mesh(r) for r in resolutions]
        with ThreadPoolExecutor(max_workers=4) as pool:
            parallel = list(pool.map(mesh, resolutions))

        for one, other in zip(serial, parallel):
            assert np.array_equal(one.vertices, other.vertices)
            assert np.array_equal(one.normals, other.normals)
            assert np.array_equal(one.faces, other.faces)


# ----------------------------------------------------------------------
# cost
# ----------------------------------------------------------------------


class TestCost:
    def test_a_real_rig_meshes_in_seconds_not_minutes(self, capsys):
        """The number this reports is the point; the assertion is only a floor.

        A per-cell field call instead of a batched one costs about 280 times as
        much per point, which turns this from under a second into most of an
        hour. The threshold is loose enough not to be flaky on a slow machine
        and tight enough that losing the vectorisation cannot pass.
        """
        prims = worm_primitives()
        timings = {}

        for resolution in (32, 64, 96, 128):
            start = time.perf_counter()
            mesh = polygonise(prims, WORM_BLEND, resolution)
            timings[resolution] = time.perf_counter() - start
            assert mesh.closed

        with capsys.disabled():
            print(f"\n  chubby_worm.xml, {len(prims)} primitives, k={WORM_BLEND}")
            for resolution, elapsed in timings.items():
                print(f"    resolution {resolution:4d}  {elapsed:7.3f} s")

        assert timings[128] < 30.0, timings

    def test_cost_grows_with_the_lattice_and_not_faster(self, capsys):
        """Doubling resolution is eight times the samples. Much worse means a loop crept in."""
        prims = worm_primitives()

        start = time.perf_counter()
        polygonise(prims, WORM_BLEND, 32)
        coarse = time.perf_counter() - start

        start = time.perf_counter()
        polygonise(prims, WORM_BLEND, 64)
        fine = time.perf_counter() - start

        with capsys.disabled():
            print(f"\n  resolution 32 -> 64 cost ratio {fine / max(coarse, 1e-6):.1f}x")

        assert fine < 40.0 * max(coarse, 1e-3)
