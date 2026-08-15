"""Regenerate ``mesh_xml/editor/icon.png``.

Two metaballs mid-blend, drawn by the CPU field itself — the app's one-glyph
summary, generated rather than authored so it is literally the thing the app
computes. Run it after the theme accent moves::

    python scripts/make_icon.py

This was prose in a design document until that document was deleted for having
drifted from the code. A recipe nobody can execute is a recipe that rots, so it
is a script now: the numbers below are the only copy of them that exists, and
running the file is how you find out whether they still work.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mesh_xml.editor import theme
from mesh_xml.render import field

#: Image is square and this many pixels on a side.
SIZE = 256

#: The blend radius the two balls fold at — chosen so the neck reads as one
#: body rather than two touching spheres.
BLEND = 0.55

#: Two degenerate round-cone records (a == b, so each is a sphere), in the
#: [-1, 1] square the image spans.
BALLS = (
    ((-0.32, -0.18), 0.42),
    ((0.34, 0.26), 0.30),
)

#: The bottom of the vertical gradient — the accent's "deepened form". See
#: :func:`render` for why this is a literal rather than a theme constant.
DEEP = "#d6811b"

TARGET = Path(__file__).resolve().parents[1] / "mesh_xml" / "editor" / "icon.png"


def _hex_to_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.array([int(value[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def render(size: int = SIZE) -> np.ndarray:
    """``(size, size, 4) uint8`` RGBA, straight from ``field.field``."""
    axis = np.linspace(-1.0, 1.0, size)
    x, y = np.meshgrid(axis, -axis)
    points = np.stack([x, y, np.zeros_like(x)], axis=-1).reshape(-1, 3)

    primitives = np.zeros((len(BALLS), 8), dtype=np.float64)
    for row, ((cx, cy), radius) in enumerate(BALLS):
        primitives[row, 0:3] = (cx, cy, 0.0)
        primitives[row, 3] = radius
        primitives[row, 4:7] = (cx, cy, 0.0)
        primitives[row, 7] = radius

    distance = field.field(points, primitives, k=BLEND).reshape(size, size)

    # Alpha from the signed distance, smoothed over two pixels so the edge is
    # antialiased by the field rather than by a filter over a hard mask.
    feather = 2.0 * (2.0 / size)
    alpha = np.clip(0.5 - distance / feather, 0.0, 1.0)

    # A vertical gradient over the whole image, from the lit accent at the top
    # to its deepened form at the bottom. DEEP is not a theme constant — the
    # original generator computed it ad hoc and only the phrase "its deepened
    # form" survived, so this value was recovered by fitting the ramp off the
    # shipped icon. It reproduces it to under a level per channel.
    lit = _hex_to_rgb(theme.ACCENT)
    deep = _hex_to_rgb(DEEP)
    ramp = np.linspace(0.0, 1.0, size)[:, None]
    rgb = lit[None, None, :] * (1.0 - ramp[..., None]) + deep[None, None, :] * ramp[..., None]

    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[..., :3] = np.round(np.broadcast_to(rgb, (size, size, 3))).astype(np.uint8)
    out[..., 3] = np.round(alpha * 255.0).astype(np.uint8)
    return out


def main() -> None:
    import imageio.v3 as iio

    iio.imwrite(TARGET, render())
    print(f"wrote {TARGET} ({SIZE}x{SIZE})")


if __name__ == "__main__":
    main()
