"""The scene tier: rooms, scenes, abstraction, doors, and the crossings.

The simulation side of docs/ROOMS.md is complete through package 5: rooms
tick alone or under a scene's one clock, drop to anchors-only and realize
back, and creatures walk out of named doors and arrive whole in the room
the door promised. What remains is the editor's binding of all of it —
package 6, the rooms dock — which lives above this tier, not in it.
"""

from mesh_xml.scene.door import Door
from mesh_xml.scene.room import Room
from mesh_xml.scene.scene import Scene
from mesh_xml.scene.transfer import move_creature

__all__ = ["Door", "Room", "Scene", "move_creature"]
