"""Tests for the CPU reference metaball field.

This module is the oracle a GLSL raymarcher will be validated against, so these
tests are not regression guards — they are the evidence that the oracle is
itself correct. Several therefore check against something computed a completely
different way rather than against a stored number:

* a degenerate primitive against ``|p - c| - r``,
* a uniform-radius primitive against a segment-distance formula,
* an arbitrary round cone against a densely sampled union of the balls it is
  defined as,
* the surface radius near each end against a bisection on the field itself.

``RuntimeWarning`` is an error for the whole module. Half the reason
:mod:`mesh_xml.render.field` is shaped the way it is — the degeneracy branches,
the guarded normalizations, the ``k == 0`` early return — is to keep NaNs and
divide-by-zeros out of a field a shader will be diffed against, and a test
suite that merely tolerated them would not be testing that.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from mesh_xml.io import load_legacy
from mesh_xml.render import sdf
from mesh_xml.core.distance import smooth_min_weight
from mesh_xml.render.field import (
    SURFACE_EPSILON,
    albedo,
    bounds,
    field,
    normal,
    raymarch,
    round_cone,
    smooth_min,
)

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

WORM = Path(__file__).resolve().parent / "data" / "legacy_v1_chubby_worm.xml"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def cone(a, ra: float, b, rb: float) -> np.ndarray:
    return np.array([*a, ra, *b, rb], dtype=np.float64)


def ball(centre, radius: float) -> np.ndarray:
    return cone(centre, radius, centre, radius)


def at(point, primitives, k: float = 0.0) -> float:
    """The field at one point, as a plain float."""
    return float(field(np.asarray(point, dtype=float)[None], primitives, k)[0])


def segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unsigned distance to the segment ``ab``, computed independently.

    A round cone with ``ra == rb`` is a capsule, and a capsule's SDF is this
    minus the radius. Deriving it here rather than reusing anything from the
    module under test is the point.
    """
    ba = b - a
    h = np.clip(((points - a) @ ba) / (ba @ ba), 0.0, 1.0)
    return np.linalg.norm(points - (a + h[:, None] * ba), axis=1)


def union_of_balls(points: np.ndarray, prim: np.ndarray, samples: int = 40001):
    """Independent oracle: a round cone as the union of its interpolated balls.

    A round cone *is* the union over ``t`` of balls at ``lerp(a, b, t)`` with
    radius ``lerp(ra, rb, t)``, and the signed distance to a union of balls is
    the minimum of the per-ball signed distances. Outside that is the definition
    of distance-to-a-union; inside it holds too because the interior minimiser
    over ``t`` is exactly the tangency the closed form solves for analytically.

    So this converges to the true SDF from above at ``O(dt**2)``, which at forty
    thousand samples is far below any tolerance used here — and it shares no
    algebra whatsoever with Quilez's three-branch formulation.
    """
    a, ra, b, rb = prim[0:3], prim[3], prim[4:7], prim[7]
    s = np.linspace(0.0, 1.0, samples)
    centres = a + s[:, None] * (b - a)
    radii = ra + s * (rb - ra)
    d2 = (
        (points * points).sum(axis=1)[:, None]
        - 2.0 * points @ centres.T
        + (centres * centres).sum(axis=1)[None, :]
    )
    return (np.sqrt(np.maximum(d2, 0.0)) - radii[None, :]).min(axis=1)


def surface_radius(primitives, centre, direction, k: float = 0.0, far: float = 50.0):
    """Distance from ``centre`` out to the surface, by bisection on the field.

    Reads the surface off the field rather than off the formula that produced
    it, so a taper that exists only in the algebra cannot pass.
    """
    unit = np.asarray(direction, dtype=float)
    unit = unit / np.linalg.norm(unit)
    centre = np.asarray(centre, dtype=float)
    lo, hi = 0.0, far
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if at(centre + mid * unit, primitives, k) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def analytic_sphere_t(origin, direction, centre, radius: float) -> float:
    """Nearest ray-sphere intersection distance, from the quadratic."""
    origin = np.asarray(origin, dtype=float)
    unit = np.asarray(direction, dtype=float)
    unit = unit / np.linalg.norm(unit)
    oc = origin - np.asarray(centre, dtype=float)
    half_b = unit @ oc
    c = oc @ oc - radius * radius
    return -half_b - math.sqrt(half_b * half_b - c)


def unit_directions(n: int, seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


@pytest.fixture(scope="module")
def worm() -> tuple[np.ndarray, np.ndarray]:
    """``examples/chubby_worm.xml`` packed the way the renderer would pack it."""
    body = load_legacy(WORM)
    centres = np.zeros((body.n_chunks, 3))
    centres[:, : body.dim] = body.pos
    return sdf.pack(body), centres


# ----------------------------------------------------------------------
# round_cone — against three independent ground truths
# ----------------------------------------------------------------------


class TestRoundConeGroundTruth:
    def test_a_degenerate_primitive_is_exactly_a_sphere(self):
        """``pack`` emits ``a == b`` for every unlinked chunk, so this is normal
        traffic and 'exactly' is the right word — not 'to within a tolerance'."""
        centre = np.array([1.0, -2.0, 0.5])
        radius = 0.75
        points = np.random.default_rng(1).normal(scale=2.0, size=(200, 3))

        got = round_cone(points, ball(centre, radius))

        expected = np.linalg.norm(points - centre, axis=1) - radius
        assert got == pytest.approx(expected, abs=1e-12)

    def test_a_uniform_radius_primitive_is_exactly_a_capsule(self):
        a = np.array([-1.0, 0.0, 0.25])
        b = np.array([2.0, 1.0, -0.5])
        radius = 0.4
        points = np.random.default_rng(2).normal(scale=2.0, size=(200, 3))

        got = round_cone(points, cone(a, radius, b, radius))

        assert got == pytest.approx(segment_distance(points, a, b) - radius, abs=1e-12)

    @pytest.mark.parametrize("trial", range(8))
    def test_matches_the_union_of_balls_it_is_defined_as(self, trial):
        """Random cones against the brute-force oracle. The axis is built longer
        than the radius difference on purpose, so every trial exercises the
        three-branch formula rather than the containment fallback, which
        :class:`TestRoundConeDegeneracies` covers on its own terms."""
        rng = np.random.default_rng(100 + trial)
        ra, rb = rng.uniform(0.05, 1.0), rng.uniform(0.05, 1.0)
        axis = rng.normal(size=3)
        axis *= (abs(ra - rb) + rng.uniform(0.1, 2.0)) / np.linalg.norm(axis)
        a = rng.normal(scale=1.5, size=3)
        prim = cone(a, ra, a + axis, rb)
        points = rng.normal(scale=2.0, size=(40, 3))

        assert round_cone(points, prim) == pytest.approx(
            union_of_balls(points, prim), abs=1e-6
        )

    def test_an_endpoint_reports_minus_its_own_radius(self):
        """The centre of an end ball is exactly its radius inside the surface.
        This is also the point the exact formula is most likely to divide by
        zero at, so it doubles as a degeneracy probe."""
        a, b = np.array([0.0, 0.0, 0.0]), np.array([4.0, 0.0, 0.0])
        prim = cone(a, 1.0, b, 0.25)

        assert round_cone(np.array([a, b]), prim) == pytest.approx([-1.0, -0.25])


class TestRoundConeTaper:
    """The taper is the whole reason the primitive is a cone and not a capsule."""

    A = np.array([0.0, 0.0, 0.0])
    B = np.array([4.0, 0.0, 0.0])
    RA = 1.0
    RB = 0.25

    @property
    def prim(self) -> np.ndarray:
        return cone(self.A, self.RA, self.B, self.RB)

    def test_the_surface_radius_at_each_end_is_that_end_s_radius(self):
        """Sampled over the spherical cap at each end, where the surface is
        exactly that end ball. The cap on ``a`` runs out to ``n . u == rr/l``,
        so the hemisphere facing away from ``b`` is safely inside it, and the
        cap on ``b`` starts there."""
        prims = self.prim[None]
        axis = (self.B - self.A) / np.linalg.norm(self.B - self.A)
        cap_edge = (self.RA - self.RB) / np.linalg.norm(self.B - self.A)
        dirs = unit_directions(300, seed=3)

        a_cap = dirs[dirs @ axis <= -0.1]
        b_cap = dirs[dirs @ axis >= cap_edge + 0.1]
        assert len(a_cap) > 20 and len(b_cap) > 20

        for n in a_cap:
            assert surface_radius(prims, self.A, n) == pytest.approx(self.RA, abs=1e-9)
        for n in b_cap:
            assert surface_radius(prims, self.B, n) == pytest.approx(self.RB, abs=1e-9)

    def test_a_uniform_capsule_would_get_both_ends_wrong(self):
        """The discriminating assertion. A capsule of any single radius puts the
        two tips symmetrically; a cone must not."""
        prims = self.prim[None]
        axis = (self.B - self.A) / np.linalg.norm(self.B - self.A)
        mean = 0.5 * (self.RA + self.RB)

        a_end = surface_radius(prims, self.A, -axis)
        b_end = surface_radius(prims, self.B, axis)

        assert a_end - b_end == pytest.approx(self.RA - self.RB, abs=1e-9)
        assert a_end != pytest.approx(mean, abs=1e-3)
        assert b_end != pytest.approx(mean, abs=1e-3)

    def test_the_far_tips_lie_on_the_surface(self):
        axis = (self.B - self.A) / np.linalg.norm(self.B - self.A)
        tips = np.array([self.A - self.RA * axis, self.B + self.RB * axis])

        assert round_cone(tips, self.prim) == pytest.approx([0.0, 0.0], abs=1e-12)

    def test_the_lateral_surface_is_a_straight_taper(self):
        """Between the caps the surface is the cone tangent to both end balls,
        which is *not* the radius lerped along the axis. At the midpoint the two
        differ by a factor of ``1/cos(alpha)`` — small, but it is the entire
        difference between the exact formulation and the cheap approximation,
        and it is what a shader diff would surface as a seam."""
        prims = self.prim[None]
        length = np.linalg.norm(self.B - self.A)
        sin_alpha = (self.RA - self.RB) / length
        cos_alpha = math.sqrt(1.0 - sin_alpha * sin_alpha)
        mean = 0.5 * (self.RA + self.RB)

        got = surface_radius(prims, 0.5 * (self.A + self.B), [0.0, 1.0, 0.0])

        assert got == pytest.approx(mean / cos_alpha, abs=1e-9)
        assert got != pytest.approx(mean, abs=1e-4)


class TestRoundConeDegeneracies:
    def test_coincident_ends_with_unequal_radii_give_the_larger_sphere(self):
        centre = np.array([0.5, 0.0, -1.0])
        prim = cone(centre, 0.3, centre, 0.9)
        points = np.random.default_rng(4).normal(size=(50, 3))

        got = round_cone(points, prim)

        assert got == pytest.approx(np.linalg.norm(points - centre, axis=1) - 0.9)

    @pytest.mark.parametrize("swap", [False, True])
    def test_a_contained_end_sphere_gives_the_containing_one(self, swap):
        """``|ra - rb| >= |b - a|`` makes the discriminant negative and the exact
        formula take the square root of it. A large chunk with a small one linked
        close by produces exactly this in an ordinary rig, so it cannot be left
        to NaN."""
        big, small = np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])
        prim = cone(small, 0.1, big, 2.0) if swap else cone(big, 2.0, small, 0.1)
        points = np.random.default_rng(5).normal(scale=2.0, size=(80, 3))

        got = round_cone(points, prim)

        assert got == pytest.approx(np.linalg.norm(points - big, axis=1) - 2.0)

    def test_a_zero_radius_end_is_a_sharp_cone_tip(self):
        a, b = np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])
        prim = cone(a, 0.5, b, 0.0)

        assert float(round_cone(b[None], prim)[0]) == pytest.approx(0.0, abs=1e-12)
        assert float(round_cone(a[None], prim)[0]) == pytest.approx(-0.5)

    @pytest.mark.parametrize("bad", [np.zeros(7), np.zeros((2, 8)), np.zeros(9)])
    def test_a_malformed_record_is_rejected(self, bad):
        with pytest.raises(ValueError):
            round_cone(np.zeros((1, 3)), bad)

    def test_points_must_be_three_dimensional(self):
        with pytest.raises(ValueError):
            round_cone(np.zeros((4, 2)), ball(np.zeros(3), 1.0))


# ----------------------------------------------------------------------
# smooth_min
# ----------------------------------------------------------------------


class TestSmoothMin:
    @pytest.mark.parametrize("k", [0.05, 0.25, 1.0, 7.5])
    def test_blending_a_field_with_itself_costs_exactly_k_over_four(self, k):
        """The relation ``sdf.pack`` is built around: duplicated geometry does
        not cancel, it silently inflates the surface by ``k/4``. That is why a
        chunk contributes a primitive only when nothing links it."""
        d = np.array([-3.0, -0.5, 0.0, 0.25, 12.0])

        assert smooth_min(d, d, k) == pytest.approx(d - k / 4.0, abs=1e-12)

    def test_zero_k_is_a_plain_min(self):
        rng = np.random.default_rng(6)
        a, b = rng.normal(size=500), rng.normal(size=500)

        assert smooth_min(a, b, 0.0) == pytest.approx(np.minimum(a, b))

    def test_it_degrades_to_min_as_k_shrinks(self):
        rng = np.random.default_rng(7)
        a, b = rng.normal(size=500), rng.normal(size=500)

        errors = [
            float(np.abs(smooth_min(a, b, k) - np.minimum(a, b)).max())
            for k in (1.0, 1e-2, 1e-6, 1e-12)
        ]

        assert errors == sorted(errors, reverse=True)
        assert errors[-1] < 1e-12

    def test_it_is_commutative(self):
        rng = np.random.default_rng(8)
        a, b = rng.normal(size=500), rng.normal(size=500)

        assert smooth_min(a, b, 0.3) == pytest.approx(smooth_min(b, a, 0.3))

    @pytest.mark.parametrize("k", [0.1, 0.5, 2.0])
    def test_it_is_bracketed_by_min_and_min_minus_k_over_four(self, k):
        """The bracket :func:`bounds` relies on: smoothing only ever inflates."""
        rng = np.random.default_rng(9)
        a, b = rng.normal(size=2000), rng.normal(size=2000)

        got = smooth_min(a, b, k)

        assert np.all(got <= np.minimum(a, b) + 1e-12)
        assert np.all(got >= np.minimum(a, b) - k / 4.0 - 1e-12)

    def test_inside_the_blend_it_matches_the_closed_form(self):
        """``(a + b)/2 - (a - b)**2/(4k) - k/4`` is the form the Lipschitz
        derivation in the module docstring differentiates. If the implementation
        and that form ever part company, the safety argument goes with it."""
        k = 0.75
        a = np.linspace(-1.0, 1.0, 501)
        b = a + np.linspace(-k, k, 501)

        expected = 0.5 * (a + b) - (a - b) ** 2 / (4.0 * k) - k / 4.0
        assert smooth_min(a, b, k) == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
    def test_an_unusable_k_is_rejected(self, bad):
        with pytest.raises(ValueError):
            smooth_min(np.zeros(3), np.zeros(3), bad)


# ----------------------------------------------------------------------
# field
# ----------------------------------------------------------------------


class TestFieldFold:
    def test_one_primitive_is_that_primitive_untouched(self):
        """The observable consequence of seeding the fold with the first
        primitive rather than a sentinel: a lone primitive keeps its own surface
        instead of losing ``k/4`` to a blend against the seed."""
        prim = ball(np.array([0.0, 0.0, 0.0]), 1.0)
        points = np.random.default_rng(10).normal(scale=2.0, size=(100, 3))

        for k in (0.0, 0.5, 4.0):
            assert field(points, prim[None], k) == pytest.approx(
                round_cone(points, prim), abs=1e-12
            )

    def test_k_zero_is_the_hard_union(self, worm):
        prims, _ = worm
        points = np.random.default_rng(11).uniform(-1.0, 6.0, size=(400, 3))

        expected = np.min([round_cone(points, p) for p in prims], axis=0)
        assert field(points, prims, 0.0) == pytest.approx(expected, abs=1e-12)

    def test_the_fold_is_order_dependent_for_positive_k(self, worm):
        """Not a defect — a contract. The shader has to fold in the same order
        ``sdf.pack`` emitted, or the two surfaces differ by a fraction of ``k``
        wherever three primitives blend at once."""
        prims, _ = worm
        points = np.random.default_rng(12).uniform(-1.0, 6.0, size=(400, 3))

        forward = field(points, prims, 0.4)
        reversed_ = field(points, prims[::-1], 0.4)

        assert np.abs(forward - reversed_).max() > 1e-6

    def test_the_fold_is_order_independent_at_k_zero(self, worm):
        prims, _ = worm
        points = np.random.default_rng(13).uniform(-1.0, 6.0, size=(400, 3))

        assert field(points, prims[::-1], 0.0) == pytest.approx(
            field(points, prims, 0.0), abs=1e-12
        )

    def test_the_leading_shape_is_preserved(self):
        prims = ball(np.zeros(3), 1.0)[None]

        assert field(np.zeros((7, 4, 3)), prims).shape == (7, 4)
        assert field(np.zeros((5, 3)), prims).shape == (5,)
        assert field(np.zeros(3), prims).shape == ()

    def test_an_empty_primitive_set_is_infinitely_far(self):
        empty = np.zeros((0, 8))

        assert np.all(np.isposinf(field(np.zeros((4, 3)), empty, 0.5)))


class TestFieldBlending:
    """Two separated spheres, which is the smallest scene where blending shows."""

    LEFT = np.array([-1.0, 0.0, 0.0])
    RIGHT = np.array([1.0, 0.0, 0.0])
    RADIUS = 1.2

    @property
    def prims(self) -> np.ndarray:
        return np.array([ball(self.LEFT, self.RADIUS), ball(self.RIGHT, self.RADIUS)])

    def test_k_zero_is_exactly_the_hard_union(self):
        points = np.random.default_rng(14).uniform(-3.0, 3.0, size=(500, 3))

        expected = np.minimum(
            np.linalg.norm(points - self.LEFT, axis=1) - self.RADIUS,
            np.linalg.norm(points - self.RIGHT, axis=1) - self.RADIUS,
        )
        assert field(points, self.prims, 0.0) == pytest.approx(expected, abs=1e-12)

    def test_blending_pulls_the_field_below_the_hard_union(self):
        points = np.random.default_rng(15).uniform(-3.0, 3.0, size=(500, 3))
        hard = field(points, self.prims, 0.0)

        blended = field(points, self.prims, 0.6)

        assert np.all(blended <= hard + 1e-12)
        between = np.abs(points[:, 0]) < 0.2
        assert np.all(blended[between] < hard[between] - 1e-9)

    def test_the_neck_thickens_monotonically_with_k(self):
        """The one knob the refactor plan calls the difference between 'spheres'
        and 'creature'. Measured off the field by bisection, not predicted."""
        necks = [
            surface_radius(self.prims, np.zeros(3), [0.0, 1.0, 0.0], k=k)
            for k in (0.0, 0.1, 0.2, 0.4, 0.8, 1.6)
        ]

        assert necks == sorted(necks)
        assert all(b > a + 1e-6 for a, b in zip(necks, necks[1:]))

    def test_a_hard_union_neck_is_the_intersection_circle(self):
        """Anchors the k=0 end of that sequence to a number derived by hand."""
        expected = math.sqrt(self.RADIUS**2 - 1.0)

        got = surface_radius(self.prims, np.zeros(3), [0.0, 1.0, 0.0], k=0.0)

        assert got == pytest.approx(expected, abs=1e-9)

    def test_separated_spheres_stay_separated_at_k_zero(self):
        far = np.array([ball(np.array([-3.0, 0.0, 0.0]), 1.0), ball(np.zeros(3), 1.0)])

        assert at([-1.5, 0.0, 0.0], far, 0.0) > 0.0
        assert at([-1.5, 0.0, 0.0], far, 2.0) < at([-1.5, 0.0, 0.0], far, 0.0)


class TestFieldIsLipschitz:
    """The property that makes sphere tracing legal, checked rather than assumed."""

    @pytest.mark.parametrize("k", [0.0, 0.15, 0.4, 1.0, 3.0])
    def test_no_central_difference_exceeds_unit_slope(self, worm, k):
        prims, _ = worm
        lo, hi = bounds(prims, k)
        points = np.random.default_rng(16).uniform(lo - 1.0, hi + 1.0, size=(3000, 3))
        h = 1e-4

        slopes = []
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = h
            ahead = field(points + offset, prims, k)
            behind = field(points - offset, prims, k)
            slopes.append((ahead - behind) / (2.0 * h))
        gradient = np.stack(slopes, axis=1)

        assert np.abs(gradient).max() <= 1.0 + 1e-9
        assert np.linalg.norm(gradient, axis=1).max() <= 1.0 + 1e-6

    def test_the_field_never_overstates_the_distance_to_its_own_surface(self, worm):
        """The consequence sphere tracing actually uses: stepping by the field
        value can never step through the surface."""
        prims, _ = worm
        k = 0.35
        rng = np.random.default_rng(17)
        lo, hi = bounds(prims, k)
        origins = rng.uniform(lo - 2.0, hi + 2.0, size=(300, 3))
        outside = origins[field(origins, prims, k) > 0.0]
        steps = field(outside, prims, k)
        dirs = unit_directions(len(outside), seed=18)

        arrived = field(outside + dirs * steps[:, None], prims, k)

        assert np.all(arrived >= -1e-9)


# ----------------------------------------------------------------------
# normal
# ----------------------------------------------------------------------


class TestNormal:
    def test_a_sphere_normal_is_radial(self):
        """The tolerance is not slack: a central difference on a curved surface
        carries an ``O(eps * curvature)`` bias, so ``eps`` trades truncation
        against cancellation and cannot be driven to zero."""
        centre = np.array([1.0, 2.0, -1.0])
        prims = ball(centre, 0.8)[None]
        points = centre + unit_directions(100, seed=19) * 0.8

        got = normal(points, prims, 0.0)

        assert got == pytest.approx((points - centre) / 0.8, abs=1e-3)

    def test_normals_are_unit_length(self, worm):
        prims, centres = worm
        rng = np.random.default_rng(20)
        points = centres[rng.integers(0, len(centres), 400)] + rng.normal(
            scale=0.6, size=(400, 3)
        )

        lengths = np.linalg.norm(normal(points, prims, 0.3), axis=1)

        assert lengths == pytest.approx(np.ones(400), abs=1e-9)

    def test_the_field_increases_along_the_normal(self, worm):
        """'Outward' with no reference to which formula produced it."""
        prims, centres = worm
        k = 0.3
        rng = np.random.default_rng(21)
        points = centres[rng.integers(0, len(centres), 300)] + rng.normal(
            scale=0.8, size=(300, 3)
        )
        n = normal(points, prims, k)
        here = field(points, prims, k)

        step = 1e-3
        assert np.all(field(points + n * step, prims, k) > here)
        assert np.all(field(points - n * step, prims, k) < here)

    def test_the_shape_matches_the_points(self):
        prims = ball(np.zeros(3), 1.0)[None]

        assert normal(np.ones((6, 2, 3)), prims).shape == (6, 2, 3)

    def test_a_vanished_gradient_gives_zero_rather_than_nan(self):
        """The centre of a sphere has no normal. Reporting the zero vector is a
        answer a caller can test for; a NaN silently poisons a shading pass."""
        prims = ball(np.zeros(3), 1.0)[None]

        got = normal(np.zeros((1, 3)), prims)

        assert np.all(np.isfinite(got))
        assert got == pytest.approx(np.zeros((1, 3)))

    def test_an_empty_primitive_set_gives_no_normal(self):
        got = normal(np.ones((3, 3)), np.zeros((0, 8)), 0.5)

        assert np.all(np.isfinite(got))
        assert got.shape == (3, 3)

    @pytest.mark.parametrize("bad", [0.0, -1e-3, float("inf")])
    def test_an_unusable_eps_is_rejected(self, bad):
        with pytest.raises(ValueError):
            normal(np.zeros((1, 3)), ball(np.zeros(3), 1.0)[None], eps=bad)


# ----------------------------------------------------------------------
# raymarch
# ----------------------------------------------------------------------


class TestRaymarch:
    CENTRE = np.array([0.0, 0.0, 5.0])
    RADIUS = 1.0

    @property
    def prims(self) -> np.ndarray:
        return ball(self.CENTRE, self.RADIUS)[None]

    def test_it_hits_a_known_sphere_at_the_analytic_distance(self):
        origins = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 5.0],
                [0.0, -4.0, 8.0],
                [0.0, 0.0, 12.0],
            ]
        )
        directions = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.3, 0.0, 5.0],
                [-0.2, 0.15, 5.0],
                [-1.0, 0.0, 0.0],
                [0.0, 4.0, -3.0],
                [0.0, 0.0, -1.0],
            ]
        )

        t, hit = raymarch(
            origins,
            directions,
            self.prims,
            0.0,
            max_steps=512,
            surface_epsilon=1e-7,
        )

        expected = [
            analytic_sphere_t(o, d, self.CENTRE, self.RADIUS)
            for o, d in zip(origins, directions)
        ]
        assert np.all(hit)
        assert t == pytest.approx(expected, abs=1e-5)

    def test_the_head_on_case_is_the_hand_computed_four(self):
        t, hit = raymarch(
            np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), self.prims, 0.0
        )

        assert bool(hit[0])
        assert float(t[0]) == pytest.approx(4.0, abs=1e-3)

    @pytest.mark.parametrize(
        "direction",
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, -1.0]],
    )
    def test_it_misses_when_aimed_away(self, direction):
        _, hit = raymarch(np.zeros((1, 3)), np.array([direction]), self.prims, 0.0)

        assert not bool(hit[0])

    @pytest.mark.parametrize("offset", [1.2, 1.02, 1.001])
    def test_a_grazing_ray_misses(self, offset):
        """Sphere tracing crawls past a near miss in tiny steps. It must still
        come out a miss, whether it escapes or exhausts its step budget."""
        origin = np.array([[0.0, offset, 0.0]])

        _, hit = raymarch(
            origin, np.array([[0.0, 0.0, 1.0]]), self.prims, 0.0, max_distance=20.0
        )

        assert not bool(hit[0])

    def test_the_hit_point_sits_just_outside_the_surface(self):
        """A relaxed step can never overshoot an exact SDF, so the landing point
        is strictly outside and within one epsilon of the surface."""
        origins = np.zeros((40, 3))
        directions = self.CENTRE + unit_directions(40, seed=22) * 0.5

        t, hit = raymarch(origins, directions, self.prims, 0.0)

        landed = field(origins[hit] + _unit(directions[hit]) * t[hit, None], self.prims)
        assert np.all(landed > 0.0)
        assert np.all(landed < SURFACE_EPSILON)

    def test_an_unrelaxed_step_does_not_tunnel(self, worm):
        """``step_relaxation`` defaults below 1 for float safety, not because the
        maths needs it. If a full step could tunnel through the surface, the
        1-Lipschitz claim the whole marcher rests on would be false — so this
        marches a blended body at 1.0 and checks it lands on the near side."""
        prims, _ = worm
        k = 0.4
        origins = np.zeros((60, 3))
        origins[:, 0] = 2.5
        origins[:, 2] = 25.0
        directions = np.zeros((60, 3))
        directions[:, 0] = np.linspace(-1.2, 1.2, 60)
        directions[:, 2] = -25.0

        t, hit = raymarch(
            origins, directions, prims, k, max_distance=60.0, step_relaxation=1.0
        )

        assert hit.any()
        landings = origins[hit] + _unit(directions[hit]) * t[hit, None]
        assert np.all(field(landings, prims, k) > -SURFACE_EPSILON)
        assert np.all(field(landings, prims, k) < SURFACE_EPSILON)

    def test_a_ray_starting_inside_reports_zero(self):
        t, hit = raymarch(
            self.CENTRE[None], np.array([[0.0, 0.0, 1.0]]), self.prims, 0.0
        )

        assert bool(hit[0])
        assert float(t[0]) == 0.0

    def test_one_origin_broadcasts_against_a_grid_of_directions(self):
        """A pinhole camera, which is the only way this is ever called."""
        directions = np.zeros((8, 6, 3))
        directions[..., 2] = 5.0
        directions[..., 0] = np.linspace(-1.0, 1.0, 8)[:, None]
        directions[..., 1] = np.linspace(-1.0, 1.0, 6)[None, :]

        t, hit = raymarch(np.zeros(3), directions, self.prims, 0.0)

        assert t.shape == (8, 6) and hit.shape == (8, 6)
        assert hit.any() and not hit.all()

    def test_t_never_exceeds_max_distance(self):
        t, hit = raymarch(
            np.zeros((1, 3)),
            np.array([[0.0, 0.0, 1.0]]),
            self.prims,
            0.0,
            max_distance=2.0,
        )

        assert not bool(hit[0])
        assert float(t[0]) <= 2.0

    def test_a_blended_body_is_reachable(self, worm):
        prims, centres = worm
        origin = np.array([2.5, 0.0, 30.0])
        direction = np.array([0.0, 0.0, -1.0])

        t, hit = raymarch(origin, direction, prims, 0.3, max_distance=60.0)

        assert bool(hit)
        assert at(origin + direction * float(t), prims, 0.3) == pytest.approx(
            0.0, abs=SURFACE_EPSILON
        )

    def test_an_empty_primitive_set_hits_nothing(self):
        t, hit = raymarch(np.zeros((5, 3)), np.ones((5, 3)), np.zeros((0, 8)), 0.5)

        assert not hit.any()
        assert np.all(np.isfinite(t))

    def test_a_zero_length_direction_is_rejected(self):
        with pytest.raises(ValueError):
            raymarch(np.zeros((1, 3)), np.zeros((1, 3)), self.prims, 0.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_steps": 0},
            {"max_distance": 0.0},
            {"surface_epsilon": 0.0},
            {"step_relaxation": 0.0},
            {"step_relaxation": 1.5},
        ],
    )
    def test_unusable_march_parameters_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            raymarch(
                np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), self.prims, 0.0, **kwargs
            )


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ----------------------------------------------------------------------
# bounds
# ----------------------------------------------------------------------


class TestBounds:
    def test_a_sphere_box_is_its_radius(self):
        lo, hi = bounds(ball(np.array([1.0, -2.0, 3.0]), 0.5)[None], 0.0)

        assert lo == pytest.approx([0.5, -2.5, 2.5])
        assert hi == pytest.approx([1.5, -1.5, 3.5])

    def test_it_is_inflated_by_k(self):
        prims = ball(np.zeros(3), 1.0)[None]

        lo, hi = bounds(prims, 0.25)

        assert lo == pytest.approx([-1.25] * 3)
        assert hi == pytest.approx([1.25] * 3)

    def test_it_covers_both_ends_of_a_cone(self):
        prim = cone([0.0, 0.0, 0.0], 0.1, [2.0, 0.0, 0.0], 1.5)

        lo, hi = bounds(prim[None], 0.0)

        assert lo == pytest.approx([-0.1, -1.5, -1.5])
        assert hi == pytest.approx([3.5, 1.5, 1.5])

    def test_an_empty_primitive_set_is_a_degenerate_box(self):
        lo, hi = bounds(np.zeros((0, 8)), 0.5)

        assert lo == pytest.approx(np.zeros(3))
        assert hi == pytest.approx(np.zeros(3))

    @pytest.mark.parametrize("k", [0.0, 0.3, 1.0])
    def test_the_whole_surface_stays_inside_the_box(self, worm, k):
        """The property meshing needs. Sampled over all six faces, since a face
        the surface leaked through is exactly the bug this catches.

        Containment is closed, not strict: at ``k == 0`` the box is tight and
        the surface touches it at the tip of the outermost chunk, where the
        field is exactly zero.
        """
        prims, _ = worm
        lo, hi = bounds(prims, k)

        grid = np.linspace(0.0, 1.0, 15)
        u, v = np.meshgrid(grid, grid, indexing="ij")
        face = np.stack([u.ravel(), v.ravel()], axis=1)
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            for side in (lo[axis], hi[axis]):
                points = np.empty((len(face), 3))
                points[:, axis] = side
                for column, other in enumerate(others):
                    points[:, other] = lo[other] + face[:, column] * (
                        hi[other] - lo[other]
                    )
                assert np.all(field(points, prims, k) >= 0.0)

    def test_a_tight_box_is_tight(self, worm):
        """The other half of containment: an inflated-by-nothing box actually
        touches the surface, so :func:`bounds` is not passing by being loose."""
        prims, _ = worm
        _, hi = bounds(prims, 0.0)

        assert at([hi[0], 0.0, 0.0], prims, 0.0) == pytest.approx(0.0, abs=1e-7)


# ----------------------------------------------------------------------
# round trip through sdf.pack
# ----------------------------------------------------------------------


class TestPackedBody:
    def test_the_worm_packs_to_one_primitive_per_link(self, worm):
        prims, centres = worm

        assert prims.shape == (9, sdf.PRIMITIVE_STRIDE)
        assert len(centres) == 10

    @pytest.mark.parametrize("k", [0.0, 0.2, 0.6])
    def test_every_chunk_centre_is_strictly_inside(self, worm, k):
        prims, centres = worm

        assert np.all(field(centres, prims, k) < 0.0)

    @pytest.mark.parametrize("k", [0.0, 0.2, 0.6])
    def test_the_surface_is_connected_along_every_link(self, worm, k):
        """Links carry the geometry, so the material between two linked chunks
        has to be solid. Spheres alone would leave gaps and read as beads."""
        prims, _ = worm
        a, b = prims[:, 0:3], prims[:, 4:7]
        along = np.linspace(0.0, 1.0, 21)[:, None, None]
        samples = (a + along * (b - a)).reshape(-1, 3)

        assert np.all(field(samples, prims, k) < 0.0)

    def test_a_point_far_outside_the_box_is_outside(self, worm):
        """Outside, the fold is sandwiched between the hard union and the hard
        union less ``k/4`` per fold step. Both halves matter: the upper bound is
        what makes a blended body never smaller than an unblended one, the lower
        is where :func:`bounds`' inflation figure comes from."""
        prims, _ = worm
        k = 0.6
        _, hi = bounds(prims, k)
        near, far = hi + 1.0, hi + 10.0

        outer = at(far, prims, k)
        hard = float(np.min([round_cone(far[None], p)[0] for p in prims]))

        assert at(near, prims, k) > 0.0
        assert outer > at(near, prims, k)
        assert hard - (len(prims) - 1) * k / 4.0 <= outer <= hard + 1e-12

    def test_an_unlinked_chunk_still_gets_a_surface(self):
        """``pack`` emits a degenerate cone for it, which only exists as a
        surface because :func:`round_cone` branches to a sphere."""
        from mesh_xml.core import Body

        body = Body()
        body.add_chunk("lonely", [5.0, 0.0, 0.0], radius=0.4)
        joined = body.add_chunk("a", [0.0, 0.0, 0.0], radius=0.3)
        other = body.add_chunk("b", [1.0, 0.0, 0.0], radius=0.3)
        body.add_link(joined, other)
        prims = sdf.pack(body)

        assert len(prims) == 2
        assert at([5.0, 0.0, 0.0], prims, 0.2) < 0.0
        assert at([5.0, 0.39, 0.0], prims, 0.0) < 0.0
        assert at([5.0, 0.41, 0.0], prims, 0.0) > 0.0

    def test_blending_grows_the_worm_rather_than_shrinking_it(self, worm):
        prims, _ = worm
        points = np.random.default_rng(23).uniform(-2.0, 7.0, size=(2000, 3))

        hard = field(points, prims, 0.0)
        soft = field(points, prims, 0.5)

        assert np.all(soft <= hard + 1e-12)
        assert np.count_nonzero(soft < 0.0) > np.count_nonzero(hard < 0.0)


# ----------------------------------------------------------------------
# nothing produces a NaN, anywhere
# ----------------------------------------------------------------------


NASTY = {
    "coincident spheres": np.array([ball(np.zeros(3), 0.5), ball(np.zeros(3), 0.5)]),
    "coincident cones": np.array(
        [
            cone([0.0, 0.0, 0.0], 0.4, [1.0, 0.0, 0.0], 0.2),
            cone([0.0, 0.0, 0.0], 0.4, [1.0, 0.0, 0.0], 0.2),
        ]
    ),
    "zero radius sphere": ball(np.zeros(3), 0.0)[None],
    "zero radius at one end": cone([0.0, 0.0, 0.0], 0.5, [1.0, 0.0, 0.0], 0.0)[None],
    "zero radius at both ends": cone([0.0, 0.0, 0.0], 0.0, [1.0, 0.0, 0.0], 0.0)[None],
    "contained end sphere": cone([0.0, 0.0, 0.0], 2.0, [0.1, 0.0, 0.0], 0.05)[None],
    "tangent end spheres": cone([0.0, 0.0, 0.0], 1.0, [1.0, 0.0, 0.0], 0.0)[None],
    "sub-epsilon separation": cone([0.0, 0.0, 0.0], 0.5, [1e-9, 0.0, 0.0], 0.25)[None],
    "huge radius ratio": cone([0.0, 0.0, 0.0], 1e-6, [1.0, 0.0, 0.0], 1e3)[None],
}

PROBES = np.concatenate(
    [
        np.zeros((1, 3)),
        np.array([[1.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1e-9, 0.0, 0.0]]),
        np.array([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
        np.random.default_rng(24).normal(scale=1.5, size=(60, 3)),
    ]
)


@pytest.mark.parametrize("name", sorted(NASTY))
@pytest.mark.parametrize("k", [0.0, 1e-9, 0.3, 100.0])
def test_no_degenerate_configuration_produces_a_nan(name, k):
    """Every entry here is reachable: ``pack`` emits coincident ends for unlinked
    chunks, an editor can put two chunks on top of each other, and a big chunk
    beside a small one makes the exact formula's discriminant negative. The
    module-level ``error::RuntimeWarning`` filter means an intermediate divide by
    zero fails this even when the final number comes out finite."""
    prims = NASTY[name]

    assert np.all(np.isfinite(field(PROBES, prims, k)))
    assert np.all(np.isfinite(normal(PROBES, prims, k)))
    assert np.all(np.isfinite(bounds(prims, k)[0]))
    assert np.all(np.isfinite(bounds(prims, k)[1]))

    t, hit = raymarch(
        np.full((len(PROBES), 3), 8.0), -PROBES - 8.0, prims, k, max_distance=40.0
    )
    assert np.all(np.isfinite(t))
    assert hit.dtype == bool


def test_points_exactly_on_the_primitive_data_are_finite():
    """Cone endpoints, axis points and the exact centre are where the scaled
    formulation cancels hardest."""
    prim = cone([0.0, 0.0, 0.0], 0.75, [2.0, 0.0, 0.0], 0.25)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-0.75, 0.0, 0.0],
            [2.25, 0.0, 0.0],
            [0.0, 0.75, 0.0],
        ]
    )

    got = round_cone(points, prim)

    assert np.all(np.isfinite(got))
    assert got[0] == pytest.approx(-0.75)
    assert got[1] == pytest.approx(-0.25)
    assert got[3] == pytest.approx(0.0, abs=1e-12)
    assert got[4] == pytest.approx(0.0, abs=1e-12)


class TestBlendGroups:
    """Smooth minimum within a group, hard minimum across; grouping is exact."""

    @staticmethod
    def _sphere(centre, radius) -> list[float]:
        x, y, z = centre
        return [x, y, z, radius, x, y, z, radius]

    def test_one_group_is_the_global_fold_bit_for_bit(self):
        prims = np.array(
            [self._sphere((i * 0.8, 0.0, 0.0), 0.5) for i in range(4)],
            dtype=np.float32,
        )
        points = np.random.default_rng(4).uniform(-2.0, 5.0, size=(64, 3))

        whole = field(points, prims, 0.3)
        grouped = field(points, prims, 0.3, groups=np.array([[0, 4]]))

        assert np.array_equal(grouped, whole)

    def test_groups_compose_by_hard_min_of_their_own_folds(self):
        prims = np.array(
            [self._sphere((i * 0.8, 0.0, 0.0), 0.5) for i in range(5)],
            dtype=np.float32,
        )
        points = np.random.default_rng(5).uniform(-2.0, 6.0, size=(64, 3))
        split = 2

        grouped = field(points, prims, 0.3, groups=np.array([[0, split], [split, 5 - split]]))
        manual = np.minimum(
            field(points, prims[:split], 0.3), field(points, prims[split:], 0.3)
        )

        assert np.array_equal(grouped, manual)

    def test_two_creatures_touching_stay_two_skins(self):
        """The weld is the artifact this exists to remove: at the midpoint of
        a near-touching pair, the global smooth minimum digs a connecting
        neck; grouped, the field is the honest hard minimum."""
        prims = np.array(
            [self._sphere((0.0, 0.0, 0.0), 0.5), self._sphere((1.05, 0.0, 0.0), 0.5)],
            dtype=np.float32,
        )
        midpoint = np.array([[0.525, 0.0, 0.0]])

        welded = float(field(midpoint, prims, 0.4)[0])
        skinned = float(field(midpoint, prims, 0.4, groups=np.array([[0, 1], [1, 1]]))[0])

        assert skinned == pytest.approx(0.025, abs=1e-6)
        assert welded < skinned - 0.05

    def test_an_empty_group_is_ignored(self):
        prims = np.array([self._sphere((0.0, 0.0, 0.0), 0.5)], dtype=np.float32)
        point = np.array([[2.0, 0.0, 0.0]])

        d = field(point, prims, 0.2, groups=np.array([[0, 1], [1, 0]]))
        assert float(d[0]) == pytest.approx(1.5, abs=1e-6)


class TestSmoothMinWeight:
    """The weight-returning smooth minimum: same value, plus the h the
    colour fold blends by."""

    def test_the_value_is_smooth_min_bit_for_bit(self):
        a = np.linspace(-1.0, 1.0, 41)
        b = 0.3 - a
        for k in (0.0, 0.05, 0.4):
            value, h = smooth_min_weight(a, b, k)
            assert np.array_equal(value, smooth_min(a, b, k))
            assert ((h >= 0.0) & (h <= 1.0)).all()

    def test_far_apart_the_nearer_side_owns_the_weight(self):
        value, h = smooth_min_weight(np.array([-5.0]), np.array([5.0]), 0.3)
        assert float(value[0]) == -5.0 and float(h[0]) == 1.0
        value, h = smooth_min_weight(np.array([5.0]), np.array([-5.0]), 0.3)
        assert float(value[0]) == -5.0 and float(h[0]) == 0.0

    def test_k_zero_is_a_step_with_ties_to_a(self):
        _, h = smooth_min_weight(np.array([1.0]), np.array([1.0]), 0.0)
        assert float(h[0]) == 1.0, "the accumulated side of a fold wins ties"
        _, h = smooth_min_weight(np.array([2.0]), np.array([1.0]), 0.0)
        assert float(h[0]) == 0.0


class TestAlbedo:
    """The colour oracle: the distance fold re-run carrying colour, blended
    by the same h — sd_group_albedo's float64 twin."""

    @staticmethod
    def _sphere_row(centre, radius):
        return [*centre, radius, *centre, radius]

    @staticmethod
    def _red_blue_cone():
        prims = np.array([[0.0, 0.0, 0.0, 0.3, 2.0, 0.0, 0.0, 0.3]])
        attrs = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        return prims, attrs

    def test_uniform_attributes_are_exact(self):
        """Equal colours make every mix an identity — no float wiggle at all."""
        prims = np.array(
            [
                [0.0, 0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.3],
                [1.0, 0.0, 0.0, 0.3, 2.0, 0.0, 0.0, 0.3],
            ]
        )
        colour = (0.82, 0.55, 0.42)
        attrs = np.tile([*colour, 0.0], (2, 2)).reshape(2, 8)
        points = np.stack(
            np.meshgrid(
                np.linspace(-0.5, 2.5, 7),
                np.linspace(-0.6, 0.6, 5),
                [0.0, 0.31],
            ),
            axis=-1,
        ).reshape(-1, 3)

        out = albedo(points, prims, attrs, 0.25)
        assert np.array_equal(out, np.broadcast_to(colour, out.shape))

    def test_ends_wear_their_own_colours(self):
        prims, attrs = self._red_blue_cone()
        out = albedo(
            np.array([[0.0, 0.35, 0.0], [2.0, 0.35, 0.0], [1.0, 0.35, 0.0]]),
            prims,
            attrs,
            0.0,
        )
        assert out[0] == pytest.approx((1.0, 0.0, 0.0))
        assert out[1] == pytest.approx((0.0, 0.0, 1.0))
        assert out[2] == pytest.approx((0.5, 0.0, 0.5))

    def test_the_axial_lerp_clamps_past_the_ends(self):
        prims, attrs = self._red_blue_cone()
        out = albedo(np.array([[-3.0, 0.0, 0.0], [5.0, 0.0, 0.0]]), prims, attrs, 0.0)
        assert out[0] == pytest.approx((1.0, 0.0, 0.0))
        assert out[1] == pytest.approx((0.0, 0.0, 1.0))

    def test_the_blend_weight_mixes_neighbours(self):
        prims = np.array(
            [
                self._sphere_row((0.0, 0.0, 0.0), 0.3),
                self._sphere_row((1.0, 0.0, 0.0), 0.3),
            ]
        )
        attrs = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        out = albedo(
            np.array([[0.5, 0.0, 0.0], [0.02, 0.4, 0.0]]), prims, attrs, 0.5
        )
        assert out[0] == pytest.approx((0.5, 0.0, 0.5)), "equidistant is half"
        assert out[1][0] > 0.9, "beside one sphere its colour dominates"

    def test_k_zero_switches_at_the_nearer_primitive(self):
        prims = np.array(
            [
                self._sphere_row((0.0, 0.0, 0.0), 0.3),
                self._sphere_row((1.0, 0.0, 0.0), 0.3),
            ]
        )
        attrs = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        out = albedo(
            np.array([[0.2, 0.0, 0.0], [0.8, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            prims,
            attrs,
            0.0,
        )
        assert out[0] == pytest.approx((1.0, 0.0, 0.0))
        assert out[1] == pytest.approx((0.0, 0.0, 1.0))
        assert out[2] == pytest.approx((1.0, 0.0, 0.0)), "ties go to the seed"

    def test_groups_pick_the_nearest_creature(self):
        prims = np.array(
            [
                self._sphere_row((0.0, 0.0, 0.0), 0.4),
                self._sphere_row((5.0, 0.0, 0.0), 0.4),
            ]
        )
        attrs = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        groups = np.array([[0, 1], [1, 1]])
        out = albedo(
            np.array([[0.6, 0.0, 0.0], [4.5, 0.0, 0.0]]),
            prims,
            attrs,
            0.3,
            groups=groups,
        )
        assert out[0] == pytest.approx((1.0, 0.0, 0.0))
        assert out[1] == pytest.approx((0.0, 0.0, 1.0))

    def test_mismatched_attributes_are_refused(self):
        prims, attrs = self._red_blue_cone()
        with pytest.raises(ValueError, match="row for row"):
            albedo(np.zeros((1, 3)), prims, attrs[:0], 0.2)

    def test_empty_primitives_yield_zeros(self):
        out = albedo(np.zeros((4, 3)), np.zeros((0, 8)), np.zeros((0, 8)), 0.2)
        assert out.shape == (4, 3) and not out.any()
