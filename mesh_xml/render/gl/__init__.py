"""Core-profile OpenGL 3.3 scaffolding. The only place in the tree that imports GL.

Nothing is imported at package level. :mod:`mesh_xml.render.gl.camera` is pure
numpy and has to stay importable on a machine with no GL library at all — that
is what makes its ray construction testable headless, and eager imports here
would drag PyOpenGL in behind it and defeat the whole arrangement. Attribute
access resolves lazily instead, so ``from mesh_xml.render.gl import OrbitCamera``
still works and still costs nothing extra.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Re-exports, spelled `as` so a type checker sees the names without ruff
    # reading them as dead imports. Runtime resolution is __getattr__'s.
    from mesh_xml.render.gl.camera import CameraPose as CameraPose
    from mesh_xml.render.gl.camera import OrbitCamera as OrbitCamera
    from mesh_xml.render.gl.camera import viewport_aspect as viewport_aspect
    from mesh_xml.render.gl.context import FullscreenTriangle as FullscreenTriangle
    from mesh_xml.render.gl.context import RaymarchView as RaymarchView
    from mesh_xml.render.gl.context import (
        configure_surface_format as configure_surface_format,
    )
    from mesh_xml.render.gl.context import ensure_application as ensure_application
    from mesh_xml.render.gl.context import gl_available as gl_available
    from mesh_xml.render.gl.program import AssembledSource as AssembledSource
    from mesh_xml.render.gl.program import ShaderCompileError as ShaderCompileError
    from mesh_xml.render.gl.program import ShaderError as ShaderError
    from mesh_xml.render.gl.program import ShaderLinkError as ShaderLinkError
    from mesh_xml.render.gl.program import ShaderProgram as ShaderProgram
    from mesh_xml.render.gl.program import ShaderSourceError as ShaderSourceError
    from mesh_xml.render.gl.program import assemble as assemble

_MODULES = {
    "AssembledSource": "program",
    "CameraPose": "camera",
    "FullscreenTriangle": "context",
    "OrbitCamera": "camera",
    "RaymarchView": "context",
    "ShaderCompileError": "program",
    "ShaderError": "program",
    "ShaderLinkError": "program",
    "ShaderProgram": "program",
    "ShaderSourceError": "program",
    "assemble": "program",
    "configure_surface_format": "context",
    "ensure_application": "context",
    "gl_available": "context",
    "viewport_aspect": "camera",
}

#: Derived, never hand-listed: a second copy of these names could drift from
#: the table that actually resolves them, and the drift would only show as an
#: import that works but is undocumented, or a documented name that does not.
__all__ = sorted(_MODULES)


def __getattr__(name: str) -> object:
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)


def __dir__() -> list[str]:
    return list(__all__)
