"""Per-screen-tile group shortlists: the raymarcher's broad phase.

The bound cull made each field evaluation skip groups that cannot win, but
every pixel still *asks* every group, every step. This module answers most of
those questions once per frame instead: the screen divides into fixed tiles,
and each tile records — as a bitmask — which groups' bounding spheres its
rays can come near at all. The fragment shader then iterates only the bits
that are set, which for a spread herd is one or two groups per tile instead
of all of them.

The prepass runs on the CPU, and that is the point rather than a concession.
This renderer's GL 3.3 target has no compute stage, which looked like a
blocker only while the prepass was imagined GPU-side; but the CPU already
owns every per-frame derived artefact — packing, draw positions, bounding
spheres — and a few thousand tile masks over at most 128 groups is numpy
housekeeping beside them. The masks ride an ``RGBA32UI`` integer texture,
which is core since GL 3.0.

The exactness contract, stated plainly
--------------------------------------
Tile culling is **surface-exact, not byte-exact** — the one optimisation in
this renderer that is not bit-identical, and the reason is fundamental
rather than sloppy: a signed-distance *value* far from every surface
legitimately depends on every group, so any scheme that drops far groups
changes the march's step sizes in empty space, and the final ``t`` of a hit
drifts by float dust even though it lands on the same surface. What is
preserved, and what the tests assert: every ray hits exactly the surface it
would have hit (a hit needs the field under ``surface_epsilon``, which
requires a group the mask kept — the slack guarantees it), the winning group
and colour at every hit are unchanged, and ``t`` moves by less than the
march's own tolerance. Anything needing bit-exact frames — the CPU-oracle
comparisons — simply renders without a tile texture and takes the unchanged
path.

Conservatism is the load-bearing property: a mask that includes too much
merely wastes what the dynamic cull would have skipped anyway, while a mask
that excludes an influencing group clips a silhouette. Every geometric test
here therefore rounds toward inclusion — spheres are inflated by ``slack``
on top of the blend inflation the bounds already carry, projected extents
are taken at box corners, and any sphere the camera sits inside or near
simply lands in every tile.
"""

from __future__ import annotations

import math

import numpy as np

from mesh_xml.render.gl.camera import OrbitCamera, viewport_aspect

__all__ = ["MAX_GROUPS", "TILE_SIZE", "tile_masks"]

#: Screen-space tile edge, in pixels. Small enough that a tile's rays form a
#: narrow cone, large enough that the mask texture stays a few kilobytes.
TILE_SIZE = 16

#: Bits available per tile: an RGBA32UI texel. Scenes with more groups fall
#: back to the untiled path rather than to a wrong image.
MAX_GROUPS = 128

#: World-units margin added to every bound before projection, over and above
#: the blend inflation the bounds already include. It must cover everything a
#: near-surface evaluation reaches past the bound itself: the march's growing
#: surface tolerance (``surface_epsilon * t``, about 0.02 at a 200-unit far
#: plane), the normal estimator's taps, and float error in this module's own
#: float64 arithmetic. Half a unit is orders of magnitude above all three and
#: still far below any creature's bound radius.
SLACK = 0.5


def tile_masks(
    bounds: np.ndarray,
    camera: OrbitCamera,
    width: int,
    height: int,
    *,
    tile_size: int = TILE_SIZE,
    slack: float = SLACK,
) -> np.ndarray | None:
    """Per-tile group bitmasks for one frame: ``(rows, cols, 4) uint32``.

    ``bounds`` is :func:`mesh_xml.render.sdf.group_bounds` output — one
    ``[cx, cy, cz, r]`` sphere per group, already inflated by the blend
    radius. ``width`` and ``height`` are the framebuffer's device pixels,
    matching what ``gl_FragCoord`` will report; row 0 of the result is the
    *bottom* tile row for the same reason.

    Returns ``None`` when tiling cannot apply — no groups, more groups than
    the mask has bits, or a degenerate viewport — and the caller renders
    untiled, which is always correct.
    """
    count = len(bounds)
    if count == 0 or count > MAX_GROUPS or width <= 0 or height <= 0:
        return None

    cols = math.ceil(width / tile_size)
    rows = math.ceil(height / tile_size)
    masks = np.zeros((rows, cols, 4), dtype=np.uint32)

    centres = np.asarray(bounds[:, 0:3], dtype=np.float64)
    radii = np.asarray(bounds[:, 3], dtype=np.float64) + float(slack)

    # View-space coordinates come from the SAME float32 quantities the shader
    # builds its rays from — camera.position and camera.basis() — never from
    # view_matrix(), which is a differently-rounded float32 reduction of the
    # same rotation. The two roundings drift apart by ~2^-24 of the coordinate
    # magnitude, which passes a fixed slack once a rig sits millions of units
    # from the origin; found by an adversarial sweep, reproduced at |coords|
    # of 3e6, and retired by construction here: the eye below is bit-for-bit
    # the ray origin, and the basis rows are bit-identical to the view
    # rotation, so the remaining float64 arithmetic on exact float32 inputs
    # sits comfortably inside any sane slack at any coordinate magnitude.
    eye = np.asarray(camera.position, dtype=np.float64)
    right, up, forward = (v.astype(np.float64) for v in camera.basis())
    rel = centres - eye
    x = rel @ right
    y = rel @ up
    depth = rel @ forward  # positive is ahead

    aspect = viewport_aspect(width, height)
    if camera.orthographic:
        half_y = camera.ortho_height * 0.5
        half_x = half_y * aspect
        lo_x = (x - radii) / half_x
        hi_x = (x + radii) / half_x
        lo_y = (y - radii) / half_y
        hi_y = (y + radii) / half_y
    else:
        focal = 1.0 / camera.tan_half_fov
        # Extremes of x/z over the sphere's bounding box are at box corners,
        # and the box contains the sphere, so the corner ratios bound the true
        # silhouette. Guarded below for spheres near the eye plane, where the
        # near denominator crosses zero and the rect stops meaning anything.
        near_z = depth - radii
        far_z = depth + radii
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios_x = np.stack(
                [(x - radii) / near_z, (x - radii) / far_z,
                 (x + radii) / near_z, (x + radii) / far_z]
            )
            ratios_y = np.stack(
                [(y - radii) / near_z, (y - radii) / far_z,
                 (y + radii) / near_z, (y + radii) / far_z]
            )
        lo_x = (focal / aspect) * ratios_x.min(axis=0)
        hi_x = (focal / aspect) * ratios_x.max(axis=0)
        lo_y = focal * ratios_y.min(axis=0)
        hi_y = focal * ratios_y.max(axis=0)

    # Every evaluated point sits on a forward ray, so a sphere strictly behind
    # the eye plane by more than its (inflated) radius can never come within
    # slack of one. Anything nearer than that — the camera inside or beside a
    # creature — floods every tile, because its field reaches the very first
    # step of every ray in the frame.
    behind = depth < -radii
    everywhere = ~behind & (depth <= radii)

    for group in range(count):
        if behind[group]:
            continue
        word, bit = group >> 5, np.uint32(1 << (group & 31))
        if everywhere[group]:
            masks[:, :, word] |= bit
            continue

        px_lo = (lo_x[group] + 1.0) * 0.5 * width
        px_hi = (hi_x[group] + 1.0) * 0.5 * width
        py_lo = (lo_y[group] + 1.0) * 0.5 * height
        py_hi = (hi_y[group] + 1.0) * 0.5 * height
        if px_hi < 0.0 or px_lo > width or py_hi < 0.0 or py_lo > height:
            continue

        c0 = max(0, int(px_lo) // tile_size)
        c1 = min(cols - 1, int(px_hi) // tile_size)
        r0 = max(0, int(py_lo) // tile_size)
        r1 = min(rows - 1, int(py_hi) // tile_size)
        masks[r0 : r1 + 1, c0 : c1 + 1, word] |= bit

    return masks
