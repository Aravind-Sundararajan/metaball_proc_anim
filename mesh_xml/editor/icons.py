"""The icon vocabulary: one name per concept, one place to change it.

Every glyph in the editor is Font Awesome 6 Free, resolved through
``qtawesome``, and every one of them is named here rather than at its use
site. That is the same rule the palette follows in :mod:`mesh_xml.editor.theme`
and for the same reason: a concept that picks its own glyph in four places
grows four glyphs, and the drift is invisible until somebody notices the walk
verb is a boot in the toolbar and a running figure in the dock.

Icons are tinted from the theme, so a palette change carries them. They are
also cached by ``qtawesome`` itself, so asking repeatedly is cheap.

The choices lean on what the app *means* rather than on generic tooling
metaphors: chunks are a node graph because that is literally what a body is,
creatures are a paw, the world is a mountain, the gait is a walking figure.
A metaphor an author has to learn twice is worse than a word.
"""

from __future__ import annotations

from PySide6.QtGui import QIcon

from mesh_xml.editor import theme

__all__ = ["PANEL", "icon", "panel_icon"]

#: Concept to Font Awesome name. Panels first, then the verbs the toolbar and
#: the docks share — a verb appearing in both must appear once here.
NAMES: dict[str, str] = {
    # panels
    "rooms": "fa6s.door-open",
    "chunks": "fa6s.circle-nodes",
    "creatures": "fa6s.paw",
    "controllers": "fa6s.person-walking",
    "surface": "fa6s.droplet",
    "lighting": "fa6s.lightbulb",
    "solver": "fa6s.sliders",
    "terrain": "fa6s.mountain",
    "smoothing": "fa6s.wave-square",
    "curve": "fa6s.bezier-curve",
    # verbs
    "play": "fa6s.play",
    "pause": "fa6s.pause",
    "step": "fa6s.forward-step",
    "rest": "fa6s.rotate-left",
    "add": "fa6s.plus",
    "autorig": "fa6s.wand-magic-sparkles",
    "legs": "fa6s.shoe-prints",
    "frame": "fa6s.expand",
    "hint": "fa6s.circle-question",
}

#: Dock title to concept, so the rail can be built from the panel table
#: without every call site restating the mapping.
PANEL = {
    "Rooms": "rooms",
    "Chunks": "chunks",
    "Creatures": "creatures",
    "Controllers": "controllers",
    "Surface": "surface",
    "Lighting": "lighting",
    "Solver": "solver",
    "Terrain": "terrain",
    "Smoothing": "smoothing",
    "Curve": "curve",
}


def icon(concept: str, colour: str | None = None) -> QIcon:
    """The glyph for ``concept``, tinted from the theme.

    Unknown concepts raise rather than falling back to a blank icon: a
    silently empty button is a control an author cannot find, and the typo
    that caused it would survive every test that only checks the button
    exists.
    """
    try:
        name = NAMES[concept]
    except KeyError:
        raise KeyError(
            f"no icon named {concept!r}; add it to mesh_xml.editor.icons.NAMES"
        ) from None
    import qtawesome as qta

    return qta.icon(
        name,
        color=colour or theme.DIM,
        color_active=theme.INK,
        color_disabled=theme.QUIET,
    )


def panel_icon(title: str, colour: str | None = None) -> QIcon | None:
    """The glyph for a dock title, or ``None`` for a panel with no entry."""
    concept = PANEL.get(title)
    return None if concept is None else icon(concept, colour)
