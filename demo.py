#!/usr/bin/env python3
"""
Demo script showing how to create an armature programmatically
"""

from armature_manager import ArmatureManager


def create_simple_armature():
    """Create a simple armature similar to the character_armature.xml"""

    # Create armature manager
    am = ArmatureManager()

    # Create bones in a chain (like the character armature)
    am.add_bone("Root", [0.0, 0.0, 0.0], 0.15)
    am.add_bone("Segment1", [0.0, 1.0, 0.0], 0.15)
    am.add_bone("Segment2", [0.0, 2.0, 0.0], 0.15)
    am.add_bone("Segment3", [0.0, 3.0, 0.0], 0.15)
    am.add_bone("Segment4", [0.0, 4.0, 0.0], 0.15)
    am.add_bone("Segment5", [0.0, 5.0, 0.0], 0.15)
    am.add_bone("Head", [0.0, 6.0, 0.0], 0.2)

    # Connect bones in a chain
    am.connect_bones("Root", "Segment1")
    am.connect_bones("Segment1", "Segment2")
    am.connect_bones("Segment2", "Segment3")
    am.connect_bones("Segment3", "Segment4")
    am.connect_bones("Segment4", "Segment5")
    am.connect_bones("Segment5", "Head")

    return am


def create_branching_armature():
    """Create an armature with branching structure"""

    am = ArmatureManager()

    # Create main spine
    am.add_bone("Spine", [0.0, 0.0, 0.0], 0.2)
    am.add_bone("Neck", [0.0, 1.0, 0.0], 0.15)
    am.add_bone("Head", [0.0, 2.0, 0.0], 0.25)

    # Create left arm
    am.add_bone("LeftShoulder", [-1.0, 1.0, 0.0], 0.15)
    am.add_bone("LeftArm", [-1.5, 1.0, 0.0], 0.12)
    am.add_bone("LeftForearm", [-2.0, 1.0, 0.0], 0.1)
    am.add_bone("LeftHand", [-2.5, 1.0, 0.0], 0.08)

    # Create right arm
    am.add_bone("RightShoulder", [1.0, 1.0, 0.0], 0.15)
    am.add_bone("RightArm", [1.5, 1.0, 0.0], 0.12)
    am.add_bone("RightForearm", [2.0, 1.0, 0.0], 0.1)
    am.add_bone("RightHand", [2.5, 1.0, 0.0], 0.08)

    # Connect spine
    am.connect_bones("Spine", "Neck")
    am.connect_bones("Neck", "Head")

    # Connect left arm
    am.connect_bones("Neck", "LeftShoulder")
    am.connect_bones("LeftShoulder", "LeftArm")
    am.connect_bones("LeftArm", "LeftForearm")
    am.connect_bones("LeftForearm", "LeftHand")

    # Connect right arm
    am.connect_bones("Neck", "RightShoulder")
    am.connect_bones("RightShoulder", "RightArm")
    am.connect_bones("RightArm", "RightForearm")
    am.connect_bones("RightForearm", "RightHand")

    return am


def main():
    print("Creating simple armature...")
    simple_am = create_simple_armature()
    simple_am.export_xml("simple_armature.xml")
    print("Simple armature exported to simple_armature.xml")

    print("\nCreating branching armature...")
    branching_am = create_branching_armature()
    branching_am.export_xml("branching_armature.xml")
    print("Branching armature exported to branching_armature.xml")

    print("\nDemo completed! You can now:")
    print("1. Run 'poetry run python main.py' to open the GUI")
    print("2. Import the generated XML files")
    print("3. Create your own armatures interactively")


if __name__ == "__main__":
    main()
