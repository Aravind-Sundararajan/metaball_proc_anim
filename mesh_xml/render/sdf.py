"""Packing a body into signed-distance primitives.

This module owns the *format* the metaball surface is built from. The CPU
reference field and the GPU raymarcher both consume it, and they have to agree
exactly or the viewport stops matching what gets exported.

Two decisions are load-bearing.

**Everything is a round cone.** A round cone — a capsule with a different radius
at each end — is the only primitive emitted. A sphere is the degenerate case
where both ends coincide. One record type means the shader has a single loop
with no per-primitive branching and no type tag to unpack, and it means chunk
radii can vary along a chain without the surface stepping between segments the
way uniform capsules would.

**Links carry the geometry, not chunks.** A round cone spanning two chunks
already contains both end spheres, so emitting a sphere *and* its cones would
overlap them. That is not free: the smooth minimum of a shape with itself is
``d - k/4``, so duplicated geometry silently inflates the surface by a fraction
of the blend radius. Chunks therefore contribute a primitive only when nothing
links them.

That removes a *third* overlapping copy at every joint; it does not remove the
first two, and it cannot. A chunk of degree two is an endpoint of both its cones
and each cone contains its whole end sphere, so the field at that chunk's centre
is ``-r - k/4`` exactly, not ``-r``. The practical consequence, which anyone
tuning radii against a visual target will hit: **authored radii read about
``k/4`` fat wherever a chunk has two or more links, so a body beads slightly at
its joints.** Degree-one tips are unaffected. Compensating for it belongs in the
authoring layer, not here — the packing has to stay a faithful description of
the rig.

The whole point of building the surface from links rather than chunks is that
spheres alone read as beads on a string. ``examples/character_armature.xml`` was
hand-authored with ``type="capsule"`` for exactly this reason, and the old
importer threw that away.
"""

from __future__ import annotations

import colorsys

import numpy as np

from mesh_xml.core.body import Body
from mesh_xml.core.distance import DEGENERATE_EPSILON, PRIMITIVE_STRIDE

__all__ = [
    "COLOUR_META",
    "DEGENERATE_EPSILON",
    "PRIMITIVE_STRIDE",
    "group_bounds",
    "group_table",
    "pack",
    "pack_attributes",
    "primitive_count",
]

#: Hue step between blend groups, as a fraction of the wheel: the golden-ratio
#: conjugate, which is what keeps any number of groups maximally spread without
#: knowing the count in advance.
_GOLDEN_HUE = 0.61803398875

#: Chunk-meta key an authored creature colour lives under, as "r g b" floats
#: in [0, 1]. Meta because a creature has no object of its own — it is a
#: connected component — and meta is the one per-chunk store that is opaque to
#: the simulation and round-trips through every format era.
COLOUR_META = "colour"

# Canonical home of the record format and its two degeneracies is
# core/distance.py, since the world's terrain shares the same primitives.
# Re-exported here so every existing consumer keeps its import path.




def primitive_count(body: Body) -> int:
    """How many primitives :func:`pack` will emit for ``body``.

    Only rendered links count. A mechanism link — an angle stop compiled to a
    PUSH brace across a joint — constrains without carrying flesh, and a chunk
    whose every link is mechanism still emits its own sphere: it is held by
    braces, not fused into them.
    """
    linked = set()
    rendered = 0
    for link in body.links():
        if not link.render:
            continue
        rendered += 1
        linked.add(link.a)
        linked.add(link.b)
    return rendered + (body.n_chunks - len(linked))


def pack(
    body: Body,
    positions: np.ndarray | None = None,
    radii: np.ndarray | None = None,
) -> np.ndarray:
    """Build the ``(P, 8) float32`` primitive array for a body.

    ``positions`` and ``radii`` override the body's own, which is how the
    smoothing layer feeds in interpolated draw positions and squashed radii:
    the surface should follow what the viewer sees, not the raw physics.
    Both are indexed by slot, matching :attr:`Body.pos`.

    The array is contiguous and ready to upload; the GPU path should not need to
    touch it again.
    """
    pos = body.pos if positions is None else np.asarray(positions, dtype=np.float32)
    rad = body.radius if radii is None else np.asarray(radii, dtype=np.float32)

    if pos.shape != (body.n_chunks, body.dim):
        raise ValueError(
            f"positions must have shape ({body.n_chunks}, {body.dim}), got {pos.shape}"
        )
    if rad.shape != (body.n_chunks,):
        raise ValueError(f"radii must have shape ({body.n_chunks},), got {rad.shape}")

    count = primitive_count(body)
    out = np.zeros((count, PRIMITIVE_STRIDE), dtype=np.float32)
    if count == 0:
        return out

    rendered = body.link_render
    n_rendered = int(rendered.sum())
    if n_rendered:
        a = body.link_a[rendered]
        b = body.link_b[rendered]
        out[:n_rendered, 0:3] = _to_vec3(pos[a], body.dim)
        out[:n_rendered, 3] = rad[a]
        out[:n_rendered, 4:7] = _to_vec3(pos[b], body.dim)
        out[:n_rendered, 7] = rad[b]

    # Membership in the surface is decided by rendered links alone: a chunk
    # held only by mechanism links is not fused into them and keeps its sphere.
    linked = set()
    for link in body.links():
        if not link.render:
            continue
        linked.add(body.slot(link.a))
        linked.add(body.slot(link.b))

    row = n_rendered
    for slot in range(body.n_chunks):
        if slot in linked:
            continue
        centre = _to_vec3(pos[slot : slot + 1], body.dim)[0]
        out[row, 0:3] = centre
        out[row, 3] = rad[slot]
        out[row, 4:7] = centre
        out[row, 7] = rad[slot]
        row += 1

    # Rows sort group-major so :func:`group_table` can describe each creature
    # as one contiguous range. The sort is stable, so within a group the order
    # stays links-then-lonely-chunks exactly as built — and a single-component
    # body sorts to the identity, byte for byte the array this function has
    # always produced. Cross-group order was never a contract: groups combine
    # by the hard minimum, which is associative.
    return out[np.argsort(_row_groups(body), kind="stable")]


def group_table(
    body: Body,
    base_colour: tuple[float, float, float] = (0.82, 0.55, 0.42),
) -> np.ndarray:
    """The blend groups of a packed body: ``(G, 8) float32``, one creature each.

    Row layout mirrors the primitive record so the same texture machinery
    carries it: texel one is ``[start, count, 0, 0]`` — the contiguous range of
    :func:`pack` rows belonging to the group — and texel two is
    ``[r, g, b, 0]``, the group's surface colour. Group 0 wears ``base_colour``
    exactly, so a single-creature document keeps the colour its author picked;
    later groups rotate the hue by the golden-ratio step, which spreads any
    number of creatures without a palette being authored anywhere.

    Groups are connected components over *all* links, mechanism braces
    included: a knee stop is part of the creature it stops, even though it
    carries no flesh.

    An authored colour outranks the palette: the first chunk of a component
    (in slot order) carrying parseable ``COLOUR_META`` — ``"r g b"`` floats —
    colours the whole creature. Unparseable meta falls back to the palette
    rather than raising, because a colour is a preference and a broken one
    must not make a document unrenderable.
    """
    rows = _row_groups(body)
    if rows.size == 0:
        return np.zeros((0, PRIMITIVE_STRIDE), dtype=np.float32)

    counts = np.bincount(rows)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    out = np.zeros((len(counts), PRIMITIVE_STRIDE), dtype=np.float32)
    out[:, 0] = starts
    out[:, 1] = counts

    labels = _component_labels(body)
    for group in range(len(counts)):
        authored = _authored_colour(body, np.nonzero(labels == group)[0])
        out[group, 4:7] = (
            authored
            if authored is not None
            else _palette_colour(base_colour, group)
        )
    return out


def pack_attributes(
    body: Body,
    base_colour: tuple[float, float, float] = (0.82, 0.55, 0.42),
) -> np.ndarray:
    """Per-primitive end colours: ``(P, 8) float32``, row *i* skinning
    :func:`pack` row *i*.

    The weighted skin's data (``docs/TEXTURES.md``): each record carries its
    primitive's two end colours — ``[ra_r, ra_g, ra_b, 0, rb_r, rb_g,
    rb_b, 0]``, fourth lanes reserved for the pattern vocabulary — built in
    :func:`pack`'s own row order and sorted by the same stable key, so the
    two tables pair by construction whatever the topology. A chunk's colour
    is its own parseable :data:`COLOUR_META`, falling back to its group's
    *palette* colour — the golden-angle hue by group index, deliberately not
    the flat path's authored-first group colour, so painting one chunk of a
    creature does not silently repaint the rest through the fallback.

    **A body whose skin would look exactly like the flat path packs to an
    empty ``(0, 8)`` table.** That is the contract that makes "unpainted
    means unchanged" structural rather than numerical: with nothing
    uploaded the shader shades by the flat group colour on the code path it
    always ran, and no colour fold can wiggle a uniform document by even a
    ulp of float arithmetic.
    """
    count = primitive_count(body)
    if count == 0:
        return np.zeros((0, PRIMITIVE_STRIDE), dtype=np.float32)

    labels = _component_labels(body)
    n_groups = int(labels.max()) + 1

    flat = np.zeros((n_groups, 3), dtype=np.float32)
    palette = np.zeros((n_groups, 3), dtype=np.float32)
    for group in range(n_groups):
        palette[group] = _palette_colour(base_colour, group)
        authored = _authored_colour(body, np.nonzero(labels == group)[0])
        flat[group] = authored if authored is not None else palette[group]

    per_slot = np.zeros((body.n_chunks, 3), dtype=np.float32)
    for slot in range(body.n_chunks):
        own = _chunk_colour(body, slot)
        per_slot[slot] = own if own is not None else palette[labels[slot]]

    if np.array_equal(per_slot, flat[labels]):
        return np.zeros((0, PRIMITIVE_STRIDE), dtype=np.float32)

    out = np.zeros((count, PRIMITIVE_STRIDE), dtype=np.float32)
    rendered = body.link_render
    n_rendered = int(rendered.sum())
    if n_rendered:
        out[:n_rendered, 0:3] = per_slot[body.link_a[rendered]]
        out[:n_rendered, 4:7] = per_slot[body.link_b[rendered]]

    linked = set()
    for link in body.links():
        if not link.render:
            continue
        linked.add(body.slot(link.a))
        linked.add(body.slot(link.b))
    row = n_rendered
    for slot in range(body.n_chunks):
        if slot in linked:
            continue
        out[row, 0:3] = per_slot[slot]
        out[row, 4:7] = per_slot[slot]
        row += 1

    return out[np.argsort(_row_groups(body), kind="stable")]


def _palette_colour(
    base_colour: tuple[float, float, float], group: int
) -> tuple[float, float, float]:
    """Group ``n``'s automatic colour: the base hue walked ``n`` golden steps."""
    hue, saturation, value = colorsys.rgb_to_hsv(*base_colour)
    return colorsys.hsv_to_rgb(
        (hue + group * _GOLDEN_HUE) % 1.0, saturation, value
    )


def _chunk_colour(body: Body, slot: int) -> tuple[float, float, float] | None:
    """One chunk's authored colour, or ``None`` — the single parse both the
    flat path and the weighted skin read, so they cannot disagree on what a
    meta string means."""
    raw = body.meta_of(body.id_at(int(slot))).get(COLOUR_META)
    if not raw:
        return None
    try:
        parts = [float(part) for part in str(raw).split()]
    except ValueError:
        return None
    if len(parts) == 3:
        return tuple(min(max(part, 0.0), 1.0) for part in parts)
    return None


def _authored_colour(
    body: Body, slots: np.ndarray
) -> tuple[float, float, float] | None:
    """The component's authored colour, or ``None`` for the palette."""
    for slot in slots:
        colour = _chunk_colour(body, int(slot))
        if colour is not None:
            return colour
    return None


def group_bounds(
    primitives: np.ndarray, table: np.ndarray, k: float = 0.0
) -> np.ndarray:
    """A bounding sphere per blend group: ``(G, 8) float32``, ``[cx,cy,cz,r]``.

    The sphere covers every end ball of the group's primitives — a round cone
    is the convex hull of its end balls — inflated by ``k``, which
    :func:`~mesh_xml.render.field.bounds` proves covers the smooth-min
    inflation for any primitive count. That makes ``|p - c| - r`` a true lower
    bound on the group's field everywhere, which is the property the shader's
    cull rests on: a group whose lower bound cannot beat the running minimum
    can be skipped without changing the result by even a ulp, because the
    skipped fold is never evaluated at all.

    Recomputed per frame from the packed positions — creatures move — and
    padded to the stride-8 record so the same texture machinery carries it.
    """
    prims = np.asarray(primitives, dtype=np.float32)
    out = np.zeros((len(table), PRIMITIVE_STRIDE), dtype=np.float32)
    for group, row in enumerate(np.asarray(table)):
        start, count = int(row[0]), int(row[1])
        rows = prims[start : start + count]
        if len(rows) == 0:
            continue
        ends = np.vstack((rows[:, 0:3], rows[:, 4:7]))
        radii = np.concatenate((rows[:, 3], rows[:, 7]))
        lo = (ends - radii[:, None]).min(axis=0)
        hi = (ends + radii[:, None]).max(axis=0)
        centre = (lo + hi) * 0.5
        reach = np.linalg.norm(ends - centre, axis=1) + radii
        out[group, 0:3] = centre
        # Rounded outward by a handful of float32 ulps, because the cull's
        # byte-exactness claim is about *computed* values: at k = 0 the
        # shader's length()-based lower bound can land a few ulps above the
        # group's computed field, and a bound that overstates by even one ulp
        # can cull a group that would have lowered the running minimum. The
        # padding is ~1e-6 relative — invisible to cull effectiveness.
        radius = np.float32(float(reach.max()) + float(k))
        out[group, 3] = radius * np.float32(1.0 + 8.0 * np.finfo(np.float32).eps)
    return out


def _row_groups(body: Body) -> np.ndarray:
    """The component index of each :func:`pack` row, in the pre-sort build order.

    Components are found over every link — mechanism included — because
    identity is physical connectivity, not flesh. Component numbering is by
    first-seen slot, which makes the labels deterministic for a given body.
    """
    labels = _component_labels(body)
    rendered = body.link_render
    link_groups = labels[body.link_a[rendered]]

    linked = set()
    for link in body.links():
        if not link.render:
            continue
        linked.add(body.slot(link.a))
        linked.add(body.slot(link.b))
    lonely_groups = [
        labels[slot] for slot in range(body.n_chunks) if slot not in linked
    ]
    return np.concatenate(
        (link_groups, np.asarray(lonely_groups, dtype=np.int64))
    ).astype(np.int64)


def _component_labels(body: Body) -> np.ndarray:
    """``(N,) int64`` component index per slot, numbered by first-seen slot."""
    n = body.n_chunks
    labels = np.full(n, -1, dtype=np.int64)
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for a, b in zip(body.link_a, body.link_b):
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))

    group = 0
    for seed in range(n):
        if labels[seed] >= 0:
            continue
        stack = [seed]
        labels[seed] = group
        while stack:
            node = stack.pop()
            for neighbour in adjacency[node]:
                if labels[neighbour] < 0:
                    labels[neighbour] = group
                    stack.append(neighbour)
        group += 1
    return labels


def _to_vec3(points: np.ndarray, dim: int) -> np.ndarray:
    """Widen a 2-D body into the 3-D the field always works in, on the z=0 plane."""
    if dim == 3:
        return points.astype(np.float32, copy=False)
    widened = np.zeros((len(points), 3), dtype=np.float32)
    widened[:, :dim] = points
    return widened
