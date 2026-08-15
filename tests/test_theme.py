"""Tests for the editor's skin: provenance first, application second.

The theme's whole argument is that it invents nothing — every colour is
the viewport's own. That claim is enforced here rather
than remembered: if someone retunes the overlay's selection orange or the
raymarcher's background, the matching chrome constant fails loudly instead
of drifting into a second brand.
"""

from __future__ import annotations

import re
from pathlib import Path

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
        background = MetaballSettings().look.background_colour
        expected = tuple(round(channel * 255.0) for channel in background)
        assert QColor(theme.FIELD).getRgb()[:3] == expected


class TestMarks:
    """The glyphs the stylesheet can only reach as files.

    Claiming a widget in a style sheet makes Qt stop drawing that widget's
    sub-controls, which is how the editor shipped numeric fields with no
    stepper arrows, combo boxes with no caret, and checkboxes that were a
    hollow outline in a panel and a black square in a table. Putting them back
    means pointing ``image:`` at a file, and a file is exactly where a colour
    goes to drift away from the palette — so the files are generated, and
    these tests are what makes "generated from the palette" checkable.
    """

    def test_every_image_the_stylesheet_asks_for_exists(self):
        """A mark renamed out from under a rule is invisible, not loud.

        Qt does not warn on a ``url()`` that resolves to nothing; it draws
        empty space, which is precisely the bug this whole mechanism exists to
        undo. So the sheet's own references are resolved here instead.
        """
        referenced = re.findall(r"url\(([^)]+)\)", theme.QSS)
        assert referenced, "no marks referenced — the sub-controls are blank again"
        missing = [path for path in referenced if not Path(path).is_file()]
        assert missing == [], f"stylesheet points at files that do not exist: {missing}"

    def test_the_marks_are_drawn_in_the_palettes_own_colours(self):
        """Recolour DIM and the arrows must follow, or the palette has a leak."""
        marks = {name: Path(path).read_text(encoding="utf-8") for name, path in theme.MARKS.items()}
        assert theme.DIM.lower() in marks["up"].lower()
        assert theme.DIM.lower() in marks["down"].lower()
        assert theme.QUIET.lower() in marks["up_off"].lower()
        assert theme.QUIET.lower() in marks["down_off"].lower()
        # The tick sits on the accent, so it is inked in the accent's own
        # foreground rather than in INK, which would vanish against it.
        assert theme.ON_ACCENT.lower() in marks["tick"].lower()

    def test_the_stylesheet_redraws_what_it_took_over(self):
        """The regression guard for the original defect, stated as a list."""
        for sub_control in (
            "QSpinBox::up-arrow",
            "QSpinBox::down-arrow",
            "QDoubleSpinBox::up-arrow",
            "QDoubleSpinBox::down-arrow",
            "QComboBox::down-arrow",
            "QCheckBox::indicator",
            "QTableWidget::indicator",
        ):
            assert sub_control in theme.QSS, f"{sub_control} is unstyled and so unpainted"


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
