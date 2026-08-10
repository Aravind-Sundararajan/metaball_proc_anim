"""Serialization. Depends on core; nothing in core depends on this.

:mod:`~mesh_xml.io.xml_v2` is the format to write. :mod:`~mesh_xml.io.xml_legacy`
only reads the two pre-versioned ``<armature>`` eras and never writes them.
:mod:`~mesh_xml.io.mesh_export` writes polygon meshes out of the project
entirely, and imports nothing that produces them.
"""

from mesh_xml.io.mesh_export import write_mesh, write_obj, write_stl
from mesh_xml.io.scene_xml import (
    SceneRef,
    is_manifest,
    read_scene,
    save_scene,
)
from mesh_xml.io.xml_legacy import (
    CoordMode,
    LegacyChunkMeta,
    LegacyFormatError,
    LegacyImport,
    detect_coords,
    load_legacy,
    read_legacy,
)
from mesh_xml.io.xml_v2 import (
    FORMAT_VERSION,
    READABLE_VERSIONS,
    Rig,
    SchemaError,
    load,
    read,
    save,
)

__all__ = [
    "FORMAT_VERSION",
    "READABLE_VERSIONS",
    "Rig",
    "SceneRef",
    "SchemaError",
    "is_manifest",
    "read_scene",
    "save_scene",
    "save",
    "load",
    "read",
    "CoordMode",
    "LegacyChunkMeta",
    "LegacyFormatError",
    "LegacyImport",
    "detect_coords",
    "load_legacy",
    "read_legacy",
    "write_mesh",
    "write_obj",
    "write_stl",
]
