# Chrome: the editor wears the viewport's own palette

**Status:** design — package 1 in progress
**Provenance:** user-requested (2026-08-10), after a session of actually
using the app: "I think you need some ui/ux chrome, like a modern app."
**North star:** the app already has a visual identity — the render's dark
field, the selection orange, the link blue, the chunk-dot ivory. The chrome
does not invent a brand; it adopts the one the viewport has been drawing
all along, so the panels stop looking like a dialog that wandered in front
of a render.

---

## 1. Where the shell already is

Stock Qt widgets in the platform's default light style, wrapped around a
deliberately dark viewport. Every panel is honest and information-dense —
the notes, tooltips and scope labels carry real knowledge — but the frame
around them says "settings dialog, 2005": platform-grey docks, default
buttons, a light table grid beside a black render. The mismatch is the
whole problem; the content is not.

## 2. The design

* **One palette, sourced from the render.** Text is the overlay's chunk
  ivory; the accent is the selection orange (what the app already means by
  "chosen"); secondary/info is the link blue; every surface tone is a step
  off the raymarcher's background. The constants live in
  ``editor/theme.py`` with their sources named, and a test asserts they
  match the viewport's own values — provenance enforced, not remembered.
* **Fusion as the base, palette plus stylesheet on top.** Fusion is the
  one Qt style that obeys a dark palette everywhere (indicators, sliders,
  menus); the stylesheet then handles what palettes cannot — group-box
  framing, dock titles, tab bars, slim scrollbars, focus rings in the
  accent.
* **Dark only.** The viewport defines the identity and it is dark; a light
  mode would be a second brand to maintain against the render it frames.
* **Applied at the entrypoint, not in the library.** ``theme.apply(app)``
  runs in ``main()``; tests and embedders get unstyled widgets unless they
  ask, so no test starts asserting against a stylesheet by accident.

## 3. Work packages

1. **The skin.** ``editor/theme.py`` — palette, stylesheet, ``apply`` —
   wired into ``main()``. Acceptance: the app reads as one artifact
   (before/after screenshots), the provenance test holds the chrome to the
   viewport's constants, and the full suite passes untouched because
   nothing but ``main()`` opts in.
2. **The toolbar.** The transport (Play/Step/Rest) and the core authoring
   gestures (Add, Fit Autorig, Rig Legs) as a toolbar with standard
   shortcuts shown — today they live behind menus and per-panel buttons a
   new user has to discover panel by panel.
3. **The details.** Consistent panel margins and note styling, dock title
   icons or glyphs, empty-state text given one voice, the window title and
   an app icon.

## 4. Not in scope

A custom frameless title bar (platform pain for chrome's sake). Light
mode. Animation and transition effects. Redesigning any panel's content or
information architecture — the arc styles what exists.
