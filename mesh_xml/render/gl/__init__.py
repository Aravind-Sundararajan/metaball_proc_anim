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
    from mesh_xml.render.gl.camera import CameraPose, OrbitCamera, viewport_aspect
    from mesh_xml.render.gl.context import (
        FullscreenTriangle,
        RaymarchView,
        configure_surface_format,
        ensure_application,
        gl_available,
    )
    from mesh_xml.render.gl.program import (
        AssembledSource,
        ShaderCompileError,
        ShaderError,
        ShaderLinkError,
        ShaderProgram,
        ShaderSourceError,
        assemble,
    )

__all__ = [
    "AssembledSource",
    "CameraPose",
    "FullscreenTriangle",
    "OrbitCamera",
    "RaymarchView",
    "ShaderCompileError",
    "ShaderError",
    "ShaderLinkError",
    "ShaderProgram",
    "ShaderSourceError",
    "assemble",
    "configure_surface_format",
    "ensure_application",
    "gl_available",
    "viewport_aspect",
]

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


def __getattr__(name: str) -> object:
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)


def __dir__() -> list[str]:
    return sorted(__all__)
