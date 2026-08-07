"""Procedural controllers. Each contributes force; none writes positions the
solver owns. :mod:`drive` moves pinned anchors — inverse mass zero, so the
solver has no claim on them — which is the one stated exception."""
