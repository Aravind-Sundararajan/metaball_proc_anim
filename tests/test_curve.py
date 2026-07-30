"""Tests for :mod:`mesh_xml.core.curve`.

The four defect classes from the refactor plan's §2.5 each get a named
regression test that reproduces the *old* behaviour alongside the new one, so a
future regression fails loudly rather than drifting quietly.

Where a test needs to know what a cubic Bezier is, it computes one from scratch
rather than importing the module's helpers. An oracle that shares code with the
thing under test cannot catch the bug this module exists to fix.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from mesh_xml.core import curve as curve_module
from mesh_xml.core.curve import X_EPSILON, Curve1D, Key


def _bezier_point(t, p0, p1, p2, p3):
    """Independent cubic Bezier evaluation, returning ``(x, y)``."""
    mt = 1.0 - t
    return tuple(
        mt**3 * a + 3.0 * mt**2 * t * b + 3.0 * mt * t**2 * c + t**3 * d
        for a, b, c, d in zip(p0, p1, p2, p3)
    )


def _reference_t(x, p0, p1, p2, p3):
    """Independent parameter solve: 200 rounds of plain bisection on x(t)."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _bezier_point(mid, p0, p1, p2, p3)[0] < x:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _asymmetric_curve():
    """Handles deliberately nowhere near the x-thirds, in both directions."""
    return Curve1D(
        [
            Key(0.0, 0.0, out_handle=(0.9, 0.0)),
            Key(1.0, 1.0, in_handle=(0.95, 0.0), out_handle=(1.05, 1.4)),
            Key(2.0, 0.5, in_handle=(1.2, 2.0)),
        ]
    )


# ----------------------------------------------------------------------
# defect 1 — the sampled curve must equal the drawn curve
# ----------------------------------------------------------------------


def test_segment_uses_the_next_keys_in_handle_not_its_out_handle():
    curve = _asymmetric_curve()
    keys = curve.keys

    p0, p1, p2, p3 = curve.segment_control_points(0)

    assert p0 == keys[0].pos
    assert p1 == keys[0].out_handle
    assert p2 == keys[1].in_handle
    assert p3 == keys[1].pos
    assert p2 != keys[1].out_handle


def test_sample_matches_the_drawn_control_polygon():
    curve = _asymmetric_curve()

    for index in range(len(curve) - 1):
        control = curve.segment_control_points(index)
        for t in np.linspace(0.0, 1.0, 401):
            x, y = _bezier_point(float(t), *control)
            assert curve.sample(x) == pytest.approx(y, abs=1e-12)


def test_pairing_with_the_next_out_handle_would_give_a_different_answer():
    """The exact §2.5 defect: ``handles[i + 1]`` was the wrong handle."""
    curve = _asymmetric_curve()
    keys = curve.keys

    p0, p1, p3 = keys[0].pos, keys[0].out_handle, keys[1].pos
    correct_p2 = keys[1].in_handle
    buggy_p2 = keys[1].out_handle

    x = 0.7
    correct = _bezier_point(
        _reference_t(x, p0, p1, correct_p2, p3), p0, p1, correct_p2, p3
    )[1]
    buggy = _bezier_point(_reference_t(x, p0, p1, buggy_p2, p3), p0, p1, buggy_p2, p3)[
        1
    ]

    assert curve.sample(x) == pytest.approx(correct, abs=1e-12)
    assert abs(correct - buggy) > 0.1


# ----------------------------------------------------------------------
# defect 2 — x is not the Bezier parameter
# ----------------------------------------------------------------------


def test_x_fraction_is_not_the_bezier_parameter():
    p0, p1, p2, p3 = (0.0, 0.0), (0.9, 0.0), (0.95, 0.0), (1.0, 1.0)
    curve = Curve1D([Key(0.0, 0.0, out_handle=p1), Key(1.0, 1.0, in_handle=p2)])

    x = 0.5
    naive = _bezier_point(x, p0, p1, p2, p3)[1]
    solved = _bezier_point(_reference_t(x, p0, p1, p2, p3), p0, p1, p2, p3)[1]

    assert curve.sample(x) == pytest.approx(solved, abs=1e-12)
    assert abs(curve.sample(x) - naive) > 0.1


def test_sample_agrees_with_high_resolution_parametric_evaluation():
    curve = Curve1D(
        [
            Key(-3.0, 1.25, out_handle=(-2.98, 4.0)),
            Key(5.5, -2.0, in_handle=(-1.0, -9.0), out_handle=(5.6, 1.0)),
            Key(9.0, 9.0, in_handle=(8.99, 9.0)),
        ]
    )

    for index in range(len(curve) - 1):
        control = curve.segment_control_points(index)
        for t in np.linspace(0.0, 1.0, 1001):
            x, y = _bezier_point(float(t), *control)
            assert curve.sample(x) == pytest.approx(y, abs=1e-12)


def test_bisection_fallback_stays_accurate():
    """Handles crowded against the far endpoint defeat Newton on half the range.

    ``x'(t)`` is near zero around the root here, so Newton either overshoots
    ``[0, 1]`` or fails to shrink its step, and the solve falls through to
    bisection. That path must land in the same place.
    """
    control = ((0.0, 0.0), (0.999, 0.0), (0.9995, 0.0), (1.0, 1.0))
    curve = Curve1D(
        [Key(0.0, 0.0, out_handle=control[1]), Key(1.0, 1.0, in_handle=control[2])]
    )

    for t in np.linspace(0.0, 1.0, 1001):
        x, y = _bezier_point(float(t), *control)
        assert curve.sample(x) == pytest.approx(y, abs=1e-10)

    ys = curve.sample_many(np.linspace(0.0, 1.0, 501))
    assert np.all(np.isfinite(ys))
    assert np.all(np.diff(ys) >= -1e-12)
    assert curve.sample(0.5) < 0.01


def test_linear_handles_make_the_parameter_the_x_fraction():
    """The one case the old code got right must still be right."""
    curve = Curve1D.linear(0.0, 0.0, 3.0, 6.0)
    for x in np.linspace(0.0, 3.0, 97):
        assert curve.sample(float(x)) == pytest.approx(2.0 * x, abs=1e-12)


# ----------------------------------------------------------------------
# defect 3 — handles follow their key
# ----------------------------------------------------------------------


def test_move_key_preserves_handle_offsets_exactly():
    curve = Curve1D(
        [
            Key(0.0, 0.0),
            Key(2.0, 3.0, in_handle=(1.5, 4.0), out_handle=(2.5, 1.0)),
            Key(8.0, 0.0),
        ]
    )
    before = curve.keys[1]

    curve.move_key(1, 7.0, 11.0)
    after = curve.keys[1]

    assert after.pos == (7.0, 11.0)
    assert after.in_offset == before.in_offset == (-0.5, 1.0)
    assert after.out_offset == before.out_offset == (0.5, -2.0)
    assert after.in_handle == (6.5, 12.0)
    assert after.out_handle == (7.5, 9.0)


def test_move_key_is_not_a_no_op_on_the_handles():
    """``setPos(pos + (handle - pos))`` left handles exactly where they were."""
    curve = Curve1D(
        [
            Key(0.0, 0.0),
            Key(2.0, 3.0, in_handle=(1.5, 4.0), out_handle=(2.5, 1.0)),
            Key(8.0, 0.0),
        ]
    )
    before = curve.keys[1]

    curve.move_key(1, 7.0, 11.0)
    after = curve.keys[1]

    assert after.in_handle != before.in_handle
    assert after.out_handle != before.out_handle


def test_move_key_changes_the_sampled_curve():
    curve = Curve1D([Key(0.0, 0.0), Key(1.0, 1.0, in_handle=(0.5, 2.0)), Key(2.0, 0.0)])
    before = curve.sample_many(np.linspace(0.0, 2.0, 51))

    curve.move_key(1, 1.0, -4.0)
    after = curve.sample_many(np.linspace(0.0, 2.0, 51))

    assert not np.allclose(before, after)


def test_move_key_returns_the_index_after_reordering():
    curve = Curve1D([Key(0.0, 0.0), Key(1.0, 1.0), Key(2.0, 2.0)])

    index = curve.move_key(1, 3.0, 5.0)

    assert index == 2
    assert [k.x for k in curve.keys] == [0.0, 2.0, 3.0]
    assert curve.keys[2].y == 5.0


def test_auto_handles_stay_auto_through_a_move():
    curve = Curve1D([Key(0.0, 0.0), Key(1.0, 1.0), Key(2.0, 2.0)])

    curve.move_key(1, 1.5, 4.0)

    assert curve.keys[1].in_handle is None
    assert curve.keys[1].out_handle is None


# ----------------------------------------------------------------------
# defect 4 — no stacked keys
# ----------------------------------------------------------------------


def test_repeated_add_key_never_stacks():
    """The old ``add_point`` always inserted at x=150, piling keys up."""
    curve = Curve1D.linear(0.0, 0.0, 1.0, 1.0)

    for _ in range(5):
        curve.add_key()

    xs = [k.x for k in curve.keys]
    assert len(xs) == 7
    assert len(set(xs)) == 7
    assert xs == sorted(xs)
    assert all(b - a >= X_EPSILON for a, b in zip(xs, xs[1:]))


def test_add_key_inside_the_domain_does_not_change_the_shape():
    curve = _asymmetric_curve()
    probes = np.linspace(-0.5, 2.5, 601)
    before = curve.sample_many(probes)

    index = curve.add_key(0.37)

    assert index == 1
    assert len(curve) == 4
    assert curve.keys[1].x == 0.37
    np.testing.assert_allclose(curve.sample_many(probes), before, atol=1e-12)


def test_add_key_lands_on_the_curve_it_split():
    curve = _asymmetric_curve()
    expected = curve.sample(1.6)

    index = curve.add_key(1.6)

    assert curve.keys[index].y == pytest.approx(expected, abs=1e-12)


def test_add_key_with_an_explicit_y_moves_the_curve_there():
    curve = Curve1D.linear(0.0, 0.0, 2.0, 2.0)

    index = curve.add_key(1.0, 5.0)

    assert index == 1
    assert curve.keys[1].pos == (1.0, 5.0)
    assert curve.sample(1.0) == pytest.approx(5.0)


def test_add_key_outside_the_domain_extends_it():
    curve = Curve1D.linear(0.0, 1.0, 1.0, 2.0)

    index = curve.add_key(4.0)

    assert index == 2
    assert curve.domain() == (0.0, 4.0)
    assert curve.keys[2].y == pytest.approx(2.0)


def test_add_key_on_an_empty_curve():
    curve = Curve1D()

    index = curve.add_key()

    assert index == 0
    assert curve.keys[0].pos == (0.0, 0.0)


def test_coincident_keys_are_separated_not_dropped():
    curve = Curve1D([Key(0.0, 0.0), Key(1.0, 1.0), Key(1.0, 5.0), Key(2.0, 2.0)])

    xs = [k.x for k in curve.keys]
    assert len(curve) == 4
    assert all(b > a for a, b in zip(xs, xs[1:]))
    assert xs[2] - xs[1] == pytest.approx(X_EPSILON, rel=1e-6)
    assert np.all(np.isfinite(curve.sample_many(np.linspace(-1.0, 3.0, 1001))))


def test_moving_a_key_onto_another_separates_them():
    curve = Curve1D([Key(0.0, 0.0), Key(1.0, 1.0), Key(2.0, 2.0)])

    index = curve.move_key(1, 2.0, 9.0)

    xs = [k.x for k in curve.keys]
    assert index == 1
    assert curve.keys[1].y == 9.0
    assert all(b > a for a, b in zip(xs, xs[1:]))


def test_separation_holds_at_large_x_where_epsilon_rounds_away():
    big = 1e12
    curve = Curve1D([Key(big, 0.0), Key(big, 1.0)])

    xs = [k.x for k in curve.keys]
    assert len(curve) == 2
    assert xs[1] > xs[0]
    assert np.isfinite(curve.sample(big))


# ----------------------------------------------------------------------
# invariants
# ----------------------------------------------------------------------


def test_keys_are_sorted_on_construction():
    curve = Curve1D([Key(5.0, 1.0), Key(-2.0, 2.0), Key(0.0, 3.0)])

    assert [k.x for k in curve.keys] == [-2.0, 0.0, 5.0]
    assert [k.y for k in curve.keys] == [2.0, 3.0, 1.0]


def test_sampling_outside_the_domain_clamps_to_the_end_values():
    curve = Curve1D([Key(1.0, 4.0, out_handle=(1.2, 9.0)), Key(3.0, -7.0)])

    assert curve.sample(-100.0) == 4.0
    assert curve.sample(1.0) == 4.0
    assert curve.sample(3.0) == -7.0
    assert curve.sample(100.0) == -7.0


def test_sampling_at_key_x_returns_the_key_y_exactly():
    curve = _asymmetric_curve()

    for key in curve.keys:
        assert curve.sample(key.x) == key.y


def test_sample_many_agrees_with_looped_sample_bit_for_bit():
    curve = _asymmetric_curve()
    xs = np.linspace(-1.0, 3.0, 733)

    vector = curve.sample_many(xs)
    looped = np.array([curve.sample(float(x)) for x in xs])

    assert np.array_equal(vector, looped)


def test_sample_many_preserves_input_shape():
    curve = _asymmetric_curve()
    xs = np.linspace(0.0, 2.0, 12).reshape(3, 4)

    ys = curve.sample_many(xs)

    assert ys.shape == (3, 4)
    assert ys.dtype == np.float64
    assert ys[1, 2] == curve.sample(float(xs[1, 2]))


def test_sample_many_accepts_lists_and_empty_input():
    curve = _asymmetric_curve()

    assert curve.sample_many([0.0, 1.0, 2.0]).tolist() == [0.0, 1.0, 0.5]
    assert curve.sample_many([]).shape == (0,)


def test_domain():
    assert Curve1D().domain() == (0.0, 0.0)
    assert Curve1D([Key(3.0, 7.0)]).domain() == (3.0, 3.0)
    assert Curve1D.linear(-1.0, 0.0, 4.0, 0.0).domain() == (-1.0, 4.0)


def test_remove_key():
    curve = Curve1D([Key(0.0, 0.0), Key(1.0, 1.0), Key(2.0, 2.0)])

    curve.remove_key(1)

    assert [k.x for k in curve.keys] == [0.0, 2.0]
    with pytest.raises(IndexError):
        curve.remove_key(7)


def test_set_handles_and_revert_to_auto():
    curve = Curve1D.linear(0.0, 0.0, 1.0, 1.0)

    curve.set_handles(0, out_handle=(0.9, 0.0))
    assert curve.keys[0].out_handle == (0.9, 0.0)
    assert curve.sample(0.5) < 0.5

    curve.set_handles(0, out_handle=None)
    assert curve.keys[0].out_handle is None
    assert curve.sample(0.5) == pytest.approx(0.5)


def test_keys_property_is_a_defensive_copy():
    curve = Curve1D.linear(0.0, 0.0, 1.0, 1.0)

    curve.keys.append(Key(0.5, 99.0))

    assert len(curve) == 2


def test_key_is_immutable():
    key = Key(0.0, 0.0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        key.x = 5.0  # type: ignore[misc]


def test_segment_control_points_rejects_bad_indices():
    with pytest.raises(IndexError):
        Curve1D().segment_control_points(0)
    with pytest.raises(IndexError):
        Curve1D([Key(0.0, 0.0)]).segment_control_points(0)
    with pytest.raises(IndexError):
        Curve1D.linear().segment_control_points(1)


# ----------------------------------------------------------------------
# y_range
# ----------------------------------------------------------------------


def test_y_range_is_off_by_default():
    curve = Curve1D([Key(0.0, -5.0), Key(1.0, 5.0)])

    assert curve.y_range is None
    assert curve.sample(0.0) == -5.0


def test_y_range_clamps_output_without_touching_the_keys():
    curve = Curve1D([Key(0.0, -5.0), Key(1.0, 5.0)], y_range=(0.1, 3.0))

    ys = curve.sample_many(np.linspace(-1.0, 2.0, 201))

    assert np.all(ys >= 0.1)
    assert np.all(ys <= 3.0)
    assert curve.keys[0].y == -5.0

    curve.y_range = None
    assert curve.sample(0.0) == -5.0


def test_y_range_rejects_an_inverted_interval():
    with pytest.raises(ValueError):
        Curve1D.linear(y_range=(2.0, 1.0))


# ----------------------------------------------------------------------
# degenerate curves
# ----------------------------------------------------------------------


def test_zero_keys_is_the_zero_curve():
    curve = Curve1D()

    assert len(curve) == 0
    assert curve.keys == []
    assert curve.sample(123.0) == 0.0
    assert curve.sample_many([-1.0, 0.0, 1.0]).tolist() == [0.0, 0.0, 0.0]


def test_one_key_is_constant():
    curve = Curve1D([Key(3.0, 7.0, in_handle=(0.0, 0.0), out_handle=(9.0, 9.0))])

    assert curve.sample(-10.0) == 7.0
    assert curve.sample(3.0) == 7.0
    assert curve.sample(10.0) == 7.0


def test_two_coincident_keys_still_sample():
    curve = Curve1D([Key(2.0, 1.0), Key(2.0, 6.0)])

    assert len(curve) == 2
    assert curve.sample(1.0) == 1.0
    assert curve.sample(2.0) == 1.0
    assert curve.sample(5.0) == 6.0


def test_constant_constructor():
    curve = Curve1D.constant(2.5)

    assert len(curve) == 1
    assert curve.sample(-99.0) == 2.5
    assert curve.sample(99.0) == 2.5


def test_linear_constructor_is_exact_at_the_endpoints():
    curve = Curve1D.linear(2.0, 10.0, 6.0, -10.0)

    assert curve.sample(2.0) == 10.0
    assert curve.sample(6.0) == -10.0
    assert curve.sample(4.0) == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------------
# purity
# ----------------------------------------------------------------------


def _module_ast():
    return ast.parse(Path(curve_module.__file__).read_text(encoding="utf-8"))


def test_module_imports_nothing_heavier_than_numpy():
    imported = set()
    for node in ast.walk(_module_ast()):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported == {"__future__", "math", "dataclasses", "typing", "numpy"}


def test_module_carries_no_pixel_to_radius_mapping():
    """No trace of the ``2.0 - (y / 180) * 1.9`` conversion.

    That expression appeared in eight places in the widget being replaced. It
    encodes both a pixel scale and a radius codomain, neither of which a general
    curve has any business knowing. Numeric literals are read off the parsed
    module so that prose explaining the removal does not trip the check.
    """
    literals = {
        node.value
        for node in ast.walk(_module_ast())
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
    }

    assert 180 not in literals
    assert 1.9 not in literals
