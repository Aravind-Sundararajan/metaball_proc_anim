"""Serialization. Depends on core; nothing in core depends on this.

:mod:`~mesh_xml.io.xml_v2` is the format to write. :mod:`~mesh_xml.io.xml_legacy`
only reads the two pre-versioned ``<armature>`` eras and never writes them.
"""

from mesh_xml.io.xml_legacy import (
    CoordMode,
    LegacyChunkMeta,
    LegacyFormatError,
    LegacyImport,
    detect_coords,
    load_legacy,
    read_legacy,
)
from mesh_xml.io.xml_v2 import FORMAT_VERSION, SchemaError, load, save

__all__ = [
    "FORMAT_VERSION",
    "SchemaError",
    "save",
    "load",
    "CoordMode",
    "LegacyChunkMeta",
    "LegacyFormatError",
    "LegacyImport",
    "detect_coords",
    "load_legacy",
    "read_legacy",
]
