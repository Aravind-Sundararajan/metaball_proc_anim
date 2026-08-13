"""The look record: everything the shaded frame is, that the field is not.

One dataclass for the key light, the fill it stands in, and the palette's
base — the fields ``docs/LIGHTING.md`` designed together because they only
mean anything together: the background colours are also the ambient's
hues, the light colour is also the glow's, and a file that carried half
the record would state a look nobody ever saw.

It lives in this package, not beside the GL code that consumes it,
because ``io`` reads and writes it: the render package's own rule — "Only
the gl/ subpackage may import OpenGL" — is what keeps a GL-less machine
able to import the format module at all, and this module holds to it by
importing nothing beyond the standard library. ``MetaballSettings``
*composes* an instance rather than mirroring these fields, so the record
the file round-trips and the record the frame renders are the same
object and cannot drift (the twin-spelling rejection, SETTINGS.md §2).

Every default below is load-bearing: an absent ``<look>`` section means
this exact record, which reproduces the pre-arc render's four authored
values (light direction, the three colours) and lands its exposure
approximately — the old bounce term is deleted, not defaulted, so a
pre-arc document reopens exposure-similar rather than pixel-identical
(LIGHTING.md §5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Look"]

_COLOUR_FIELDS = (
    "light_colour",
    "surface_colour",
    "terrain_colour",
    "background_colour",
    "background_top",
)

_STRENGTH_FIELDS = (
    "light_strength",
    "ambient_strength",
    "ao_strength",
    "glow_strength",
)


@dataclass
class Look:
    """The authored look: one key light, the environment, the palette's base.

    Strengths are levels; colours are chromaticity, conventionally in
    ``0..1`` (the panel's swatches speak that range) but deliberately
    uncapped — a component above 1 overdrives on purpose, and the frame
    clips per-channel exactly as ``damping="1.02"`` rings (LIGHTING.md
    §4 names the artefact). ``surface_colour`` is data as well as a
    uniform: it seeds the group table and the unpainted-chunk fallback,
    which is how it reaches OBJ vertex colours.
    """

    light_direction: tuple[float, float, float] = (0.4, 0.8, 0.45)
    light_colour: tuple[float, float, float] = (1.0, 1.0, 1.0)
    light_strength: float = 1.0
    ambient_strength: float = 0.35
    ao_strength: float = 0.5
    glow_strength: float = 0.35
    surface_colour: tuple[float, float, float] = (0.82, 0.55, 0.42)
    terrain_colour: tuple[float, float, float] = (0.36, 0.44, 0.31)
    background_colour: tuple[float, float, float] = (0.11, 0.12, 0.14)
    background_top: tuple[float, float, float] = (0.15, 0.17, 0.21)

    def validate(self) -> "Look":
        """Refuse what breaks downstream; leave taste to the author.

        The house rule (SETTINGS.md §4): every float finite — a NaN here
        poisons every pixel rather than every position; the direction
        non-zero, because the shader normalizes it and zero has no
        direction to state; colours and strengths non-negative — negative
        light is not a tuning, it is a typo (the ``iterations`` ruling).
        No ceilings anywhere. Raises ``ValueError``; callers with their
        own error vocabulary (the format reader) wrap it. Returns the
        record so a read can validate in one breath.
        """
        for name in ("light_direction", *_COLOUR_FIELDS):
            value = getattr(self, name)
            if len(value) != 3:
                raise ValueError(f"{name} must have 3 components, got {len(value)}")
            for component in value:
                if not math.isfinite(component):
                    raise ValueError(f"{name} must be finite, got {value}")
        if not any(component != 0.0 for component in self.light_direction):
            raise ValueError("light_direction must be non-zero")
        for name in _COLOUR_FIELDS:
            value = getattr(self, name)
            if any(component < 0.0 for component in value):
                raise ValueError(f"{name} components must be non-negative, got {value}")
        for name in _STRENGTH_FIELDS:
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        return self
