#!/usr/bin/env python3
"""
Armature Manager - Handles bone data and XML export
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom


class ArmatureManager:
    def __init__(self):
        self.bones = {}
        self.connections = []

    def add_bone(self, name, position, radius):
        """Add a new bone/sphere to the armature"""
        self.bones[name] = {
            'position': position,
            'radius': radius,
            'rotation': [0.0, 0.0, 0.0],
            'scale': [1.0, 1.0, 1.0]
        }

    def remove_bone(self, name):
        """Remove a bone from the armature"""
        if name in self.bones:
            del self.bones[name]

        # Remove connections involving this bone
        self.connections = [conn for conn in self.connections
                            if name not in conn]

    def connect_bones(self, bone1_name, bone2_name):
        """Connect two bones, preventing closed loops"""
        if bone1_name in self.bones and bone2_name in self.bones:
            # Check if this connection would create a closed loop
            if self._would_create_loop(bone1_name, bone2_name):
                print(
                    f"Warning: Cannot connect {bone1_name} to {bone2_name} - would create a closed loop")
                return False

            connection = (bone1_name, bone2_name)
            if connection not in self.connections:
                self.connections.append(connection)
                return True
        return False

    def disconnect_bones(self, bone1_name, bone2_name):
        """Disconnect two bones"""
        connection = (bone1_name, bone2_name)
        if connection in self.connections:
            self.connections.remove(connection)

    def get_bones(self):
        """Get all bones"""
        return self.bones.copy()

    def get_connections(self):
        """Get all connections"""
        return self.connections.copy()

    def get_bone_connections(self, bone_name):
        """Get all connections involving a specific bone"""
        connections = []
        for connection in self.connections:
            if bone_name in connection:
                connections.append(connection)
        return connections

    def get_bone_position(self, bone_name):
        """Get position of a specific bone"""
        if bone_name in self.bones:
            return self.bones[bone_name]['position']
        return None

    def set_bone_position(self, bone_name, position):
        """Set position of a specific bone"""
        if bone_name in self.bones:
            self.bones[bone_name]['position'] = position

    def update_bone_position(self, bone_name, position):
        """Update position of a specific bone (alias for set_bone_position)"""
        self.set_bone_position(bone_name, position)

    def set_bone_radius(self, bone_name, radius):
        """Set radius of a specific bone"""
        if bone_name in self.bones:
            self.bones[bone_name]['radius'] = radius

    def get_bone_radius(self, bone_name):
        """Get radius of a specific bone"""
        if bone_name in self.bones:
            return self.bones[bone_name]['radius']
        return None

    def update_bone_radius(self, bone_name, radius):
        """Update radius of a specific bone (alias for set_bone_radius)"""
        self.set_bone_radius(bone_name, radius)

    def rename_bone(self, old_name, new_name):
        """Rename a bone and update all connections"""
        if old_name not in self.bones:
            return False

        if new_name in self.bones:
            return False  # New name already exists

        # Get the bone data
        bone_data = self.bones[old_name]

        # Remove old bone
        del self.bones[old_name]

        # Add new bone with same data
        self.bones[new_name] = bone_data

        # Update all connections
        new_connections = []
        for connection in self.connections:
            if connection[0] == old_name:
                new_connections.append((new_name, connection[1]))
            elif connection[1] == old_name:
                new_connections.append((connection[0], new_name))
            else:
                new_connections.append(connection)

        self.connections = new_connections
        return True

    def _would_create_loop(self, bone1_name, bone2_name):
        """Check if connecting bone1 to bone2 would create a closed loop"""
        # Create a temporary connection to test
        temp_connections = self.connections.copy()
        temp_connections.append((bone1_name, bone2_name))

        # Build adjacency list from connections (bidirectional for cycle detection)
        adjacency = {}
        for bone_name in self.bones.keys():
            adjacency[bone_name] = []

        for conn in temp_connections:
            parent, child = conn
            if parent in adjacency:
                adjacency[parent].append(child)
            if child in adjacency:
                # Add reverse edge for cycle detection
                adjacency[child].append(parent)

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle_dfs(node, parent=None):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in adjacency.get(node, []):
                # Skip the edge back to parent (undirected graph cycle detection)
                if neighbor == parent:
                    continue

                if neighbor not in visited:
                    if has_cycle_dfs(neighbor, node):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        # Check all nodes for cycles
        for node in adjacency.keys():
            if node not in visited:
                if has_cycle_dfs(node):
                    return True

        return False

    def clear(self):
        """Clear all bones and connections"""
        self.bones.clear()
        self.connections.clear()

    def clear_connections(self):
        """Clear all connections but keep bones"""
        self.connections.clear()

    def export_xml(self, filename):
        """Export armature to XML file with relative positioning"""
        # Create root element
        root = ET.Element("armature")

        # Build bone hierarchy from connections
        bone_hierarchy = self._build_hierarchy()

        # Add bones to XML (root bones have no parent, so parent_position=None)
        for bone_name, children in bone_hierarchy.items():
            if bone_name in self.bones:
                bone_elem = self._create_bone_element(
                    bone_name, children, parent_position=None)
                root.append(bone_elem)

        # Create pretty XML
        xml_str = minidom.parseString(
            ET.tostring(root)).toprettyxml(indent="    ")

        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(xml_str)

    def _build_hierarchy(self):
        """Build bone hierarchy from connections"""
        hierarchy = {}

        # Initialize all bones as root nodes
        for bone_name in self.bones.keys():
            hierarchy[bone_name] = []

        # Build parent-child relationships from connections
        for bone1, bone2 in self.connections:
            # bone1 is parent, bone2 is child
            if bone1 in hierarchy:
                hierarchy[bone1].append(bone2)

        # Remove children from root level
        children_to_remove = set()
        for children in hierarchy.values():
            for child in children:
                children_to_remove.add(child)

        for child in children_to_remove:
            if child in hierarchy:
                del hierarchy[child]

        return hierarchy

    def _create_bone_element(self, bone_name, children, parent_position=None):
        """Create XML element for a bone with relative positioning"""
        bone_data = self.bones[bone_name]
        global_pos = bone_data['position']

        # Calculate relative position
        if parent_position is not None:
            relative_pos = [
                global_pos[0] - parent_position[0],
                global_pos[1] - parent_position[1],
                global_pos[2] - parent_position[2]
            ]
        else:
            # Root bone - position is already relative to world origin
            relative_pos = global_pos

        # Create bone element
        bone_elem = ET.Element("bone")
        bone_elem.set("name", bone_name)

        # Set relative position
        bone_elem.set(
            "position", f"{relative_pos[0]:.3f},{relative_pos[1]:.3f},{relative_pos[2]:.3f}")

        # Set rotation
        rot = bone_data['rotation']
        bone_elem.set("rotation", f"{rot[0]:.3f},{rot[1]:.3f},{rot[2]:.3f}")

        # Set scale
        scale = bone_data['scale']
        bone_elem.set("scale", f"{scale[0]:.3f},{scale[1]:.3f},{scale[2]:.3f}")

        # Add mesh element
        mesh_elem = ET.SubElement(bone_elem, "mesh")
        mesh_elem.set("type", "sphere")
        mesh_elem.set("radius", f"{bone_data['radius']:.3f}")
        mesh_elem.set("material", "body")

        # Add child bones recursively with this bone's global position as parent
        for child_name in children:
            if child_name in self.bones:
                # Get the children of this child
                child_children = []
                for conn in self.connections:
                    if conn[0] == child_name:
                        child_children.append(conn[1])

                child_elem = self._create_bone_element(
                    child_name, child_children, global_pos)
                bone_elem.append(child_elem)

        return bone_elem

    def import_xml(self, filename):
        """Import armature from XML file with relative positioning"""
        try:
            tree = ET.parse(filename)
            root = tree.getroot()

            self.clear()

            # Parse only the bone elements (children of armature root)
            for bone_elem in root.findall("bone"):
                self._parse_bone_element(bone_elem, parent_position=None)

            return True
        except Exception as e:
            print(f"Error importing XML: {e}")
            return False

    def _parse_bone_element(self, bone_elem, parent_name=None, parent_position=None):
        """Parse a bone element from XML with relative positioning"""
        bone_name = bone_elem.get("name", "Unknown")

        # Parse relative position
        pos_str = bone_elem.get("position", "0,0,0")
        relative_position = [float(x) for x in pos_str.split(",")]

        # Convert to global position
        if parent_position is not None:
            global_position = [
                relative_position[0] + parent_position[0],
                relative_position[1] + parent_position[1],
                relative_position[2] + parent_position[2]
            ]
        else:
            # Root bone - relative position is already global
            global_position = relative_position

        # Parse rotation
        rot_str = bone_elem.get("rotation", "0,0,0")
        rotation = [float(x) for x in rot_str.split(",")]

        # Parse scale
        scale_str = bone_elem.get("scale", "1,1,1")
        scale = [float(x) for x in scale_str.split(",")]

        # Get radius from mesh element
        radius = 0.5  # default
        mesh_elem = bone_elem.find("mesh")
        if mesh_elem is not None:
            radius = float(mesh_elem.get("radius", 0.5))

        # Add bone with global position
        self.bones[bone_name] = {
            'position': global_position,
            'radius': radius,
            'rotation': rotation,
            'scale': scale
        }

        # Connect to parent if exists
        if parent_name:
            self.connect_bones(parent_name, bone_name)

        # Parse children with this bone's global position as parent
        for child_elem in bone_elem.findall("bone"):
            self._parse_bone_element(child_elem, bone_name, global_position)
