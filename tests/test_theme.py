"""Tests for the editor's skin: provenance first, application second.

The theme's whole argument is that it invents nothing — every colour is
the viewport's own (docs/CHROME.md). That claim is enforced here rather
than remembered: if someone retunes the overlay's selection orange or the
raymarcher's background, the matching chrome constant fails loudly instead
of drifting into a second brand.
"""

from __future__ import annotations

import pytest
from PySide6.QtGui import QColor, QPalette

context = pytest.importorskip("mesh_xml.render.gl.context")

try:
    APPLICATION = context.ensure_application()
except Exception as exc:  # pragma: no cover - headless box with no platform plugin
    pytest.skip(f"no Qt platform available: {exc}", allow_module_level=True)

from mesh_xml.editor import theme  # noqa: E402
from mesh_xml.editor import viewport  # noqa: E402
from mesh_xml.render.gl.metaball import MetaballSettings  # noqa: E402


class TestProvenance:
    def test_the_accent_is_the_selection_orange(self):
        selected = viewport.SELECTED_COLOUR
        assert QColor(theme.ACCENT).getRgb()[:3] == (
            selected.red(),
            selected.green(),
            selected.blue(),
        )

    def test_the_ink_is_the_chunk_dot_ivory(self):
        chunk = viewport.CHUNK_COLOUR
        assert QColor(theme.INK).getRgb()[:3] == (
            chunk.red(),
            chunk.green(),
            chunk.blue(),
        )

    def test_the_info_blue_is_the_link_stroke(self):
        link = viewport.LINK_COLOUR
        assert QColor(theme.INFO).getRgb()[:3] == (
            link.red(),
            link.green(),
            link.blue(),
        )

    def test_the_field_is_the_raymarchers_background(self):
        background = MetaballSettings().background_colour
        expected = tuple(round(channel * 255.0) for channel in background)
        assert QColor(theme.FIELD).getRgb()[:3] == expected


class TestApply:
    def test_apply_dresses_the_application(self):
        theme.apply(APPLICATION)
        try:
            palette = APPLICATION.palette()
            assert (
                palette.color(QPalette.ColorRole.Highlight).name()
                == QColor(theme.ACCENT).name()
            )
            assert (
                palette.color(QPalette.ColorRole.Base).name()
                == QColor(theme.FIELD).name()
            )
            assert "QGroupBox" in APPLICATION.styleSheet()
        finally:
            # Other test modules share this QApplication and met it stock;
            # the theme must not leak into their assumptions.
            APPLICATION.setStyleSheet("")
            APPLICATION.setPalette(QPalette())
