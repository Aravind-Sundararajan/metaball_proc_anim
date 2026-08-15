"""The scene tier: rooms, scenes, abstraction, doors, and the crossings.

Rooms tick alone or under a scene's one clock, drop to anchors-and-brains and
realize back, and creatures walk out of named doors to arrive whole in the room
the door promised. The editor's binding of all of it lives above this tier, not
in it — nothing here imports Qt, and a test holds it to that.
"""

from mesh_xml.scene.door import Door
from mesh_xml.scene.room import Room
from mesh_xml.scene.scene import Scene
from mesh_xml.scene.transfer import move_creature

__all__ = ["Door", "Room", "Scene", "move_creature"]
