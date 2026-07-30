"""Pure simulation core. No Qt, no OpenGL, no I/O — import nothing heavier than numpy."""

from mesh_xml.core.body import Body, Chunk, Link, LinkType

__all__ = ["Body", "Chunk", "Link", "LinkType"]
