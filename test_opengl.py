#!/usr/bin/env python3
"""
Test OpenGL functionality
"""

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtCore import QTimer
from opengl_widget import OpenGLWidget
from armature_manager import ArmatureManager


class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenGL Test")
        self.setGeometry(100, 100, 800, 600)

        # Create armature manager
        am = ArmatureManager()

        # Add some test spheres
        am.add_bone("Sphere1", [0.0, 0.0, 0.0], 0.5)
        am.add_bone("Sphere2", [2.0, 0.0, 0.0], 0.3)
        am.add_bone("Sphere3", [0.0, 2.0, 0.0], 0.4)
        am.add_bone("Sphere4", [-2.0, 0.0, 0.0], 0.6)

        # Connect some spheres
        am.connect_bones("Sphere1", "Sphere2")
        am.connect_bones("Sphere1", "Sphere3")
        am.connect_bones("Sphere1", "Sphere4")

        # Create OpenGL widget
        self.opengl_widget = OpenGLWidget(am)

        # Set up layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.opengl_widget)

        # Timer for continuous rendering
        self.timer = QTimer()
        self.timer.timeout.connect(self.opengl_widget.update)
        self.timer.start(16)  # ~60 FPS


def main():
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
