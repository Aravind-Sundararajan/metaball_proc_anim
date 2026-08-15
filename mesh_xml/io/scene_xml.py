"""The scene manifest: which rooms, by name, from which files.

A scene is several room files plus the fact of their assembly, and the
assembly was the part that evaporated on close: doors persist per room,
but nothing recorded that ``den.xml`` and ``meadow.xml`` were ever one
world. The manifest is that record and deliberately nothing more — no
geometry, no settings, no state; every substantive fact stays in the room
files, which this package does not touch.

The shape::

    <scene name="world">
        <room name="den" src="den.xml"/>
        <room name="meadow" src="meadow.xml"/>
    </scene>

``src`` is relative to the manifest (written POSIX-style, read either
way), so a scene folder travels whole. The manifest's ``name`` wins over
whatever the room file's own body says, because the manifest is the
scene's addressing — the names doors compiled against — and two copies of
one file in one scene must answer to two names.

This module reads and writes the record only; resolving ``src`` against
real files and building rooms is the caller's business, which keeps the
one editor-facing question — "what do I do when the file is missing?" —
at the layer that can name the room in its answer.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mesh_xml.io.xml_v2 import SchemaError

__all__ = ["SceneRef", "is_manifest", "read_scene", "save_scene"]


@dataclass(frozen=True)
class SceneRef:
    """One manifest row: the room's scene name and where its file is.

    ``src`` is kept exactly as written — relative to the manifest — so a
    round trip through read and save moves no room to a new address.
    """

    name: str
    src: str


def root_tag(path: str | Path) -> str | None:
    """The file's root tag, or ``None`` if it has none this reader can see.

    A sniff, not a validation: it answers "which reader owns this file" and
    nothing else, and an unreadable or unparseable file simply has no tag —
    the opener that then runs will produce the real error.
    """
    try:
        for _event, element in ET.iterparse(Path(path), events=("start",)):
            return element.tag
    except (OSError, ET.ParseError):
        return None
    return None


def is_manifest(path: str | Path) -> bool:
    """Whether the file's root tag says ``<scene>``."""
    return root_tag(path) == "scene"


def save_scene(
    path: str | Path, name: str, rooms: Sequence[tuple[str, str | Path]]
) -> None:
    """Write the manifest: the scene's name and one ``<room>`` per entry.

    ``rooms`` pairs each room's name with its file. Paths are written
    relative to the manifest where the filesystem allows it (POSIX
    separators, so the file reads the same on every platform) and
    absolute where it does not — a room on another drive is unusual but
    not a reason to refuse the save.
    """
    path = Path(path)
    root = ET.Element("scene", {"name": name})
    for room_name, room_path in rooms:
        room_path = Path(room_path)
        try:
            src = Path(os.path.relpath(room_path, start=path.parent))
        except ValueError:
            src = room_path
        ET.SubElement(
            root, "room", {"name": room_name, "src": src.as_posix()}
        )
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(path, encoding="unicode", xml_declaration=True)


def read_scene(path: str | Path) -> tuple[str, list[SceneRef]]:
    """Read a manifest into its name and room references, validating shape.

    Duplicate room names are refused here, before any room file is read:
    names are the scene's addressing, and :class:`~mesh_xml.scene.scene.Scene`
    would refuse the second one anyway — this just says so while it can
    still name the manifest line's own vocabulary rather than a
    half-built scene.
    """
    path = Path(path)
    root = ET.parse(path).getroot()
    if root.tag != "scene":
        raise SchemaError(
            f"{path} is not a scene manifest; its root is <{root.tag}>"
        )
    name = root.get("name") or "scene"

    refs: list[SceneRef] = []
    seen: set[str] = set()
    for child in root:
        if child.tag != "room":
            raise SchemaError(f"unexpected <{child.tag}> inside <scene>")
        room_name = child.get("name")
        src = child.get("src")
        if not room_name:
            raise SchemaError("a <room> entry is missing its name")
        if not src:
            raise SchemaError(f"room {room_name!r} is missing its src")
        if room_name in seen:
            raise SchemaError(f"the manifest lists {room_name!r} twice")
        seen.add(room_name)
        refs.append(SceneRef(name=room_name, src=src))
    return name, refs
