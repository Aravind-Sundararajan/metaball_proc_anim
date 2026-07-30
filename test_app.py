#!/usr/bin/env python3
"""
Test script for Mesh XML Editor functionality
"""

from armature_manager import ArmatureManager


def test_armature_manager():
    print("Testing Armature Manager...")

    # Create armature manager
    am = ArmatureManager()

    # Add some bones
    am.add_bone("Root", [0.0, 0.0, 0.0], 0.5)
    am.add_bone("Segment1", [0.0, 1.0, 0.0], 0.3)
    am.add_bone("Segment2", [0.0, 2.0, 0.0], 0.3)
    am.add_bone("Head", [0.0, 3.0, 0.0], 0.4)

    # Connect bones
    am.connect_bones("Root", "Segment1")
    am.connect_bones("Segment1", "Segment2")
    am.connect_bones("Segment2", "Head")

    # Print bones
    print("Bones:", am.get_bones())
    print("Connections:", am.get_connections())

    # Export to XML
    am.export_xml("test_armature.xml")
    print("Exported to test_armature.xml")

    # Test import
    am2 = ArmatureManager()
    if am2.import_xml("test_armature.xml"):
        print("Import successful!")
        print("Imported bones:", am2.get_bones())
        print("Imported connections:", am2.get_connections())
    else:
        print("Import failed!")

    print("Test completed!")


if __name__ == "__main__":
    test_armature_manager()
