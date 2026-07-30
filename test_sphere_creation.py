#!/usr/bin/env python3
"""
Test sphere creation functionality
"""

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PySide6.QtCore import QTimer
from opengl_widget import OpenGLWidget
from armature_manager import ArmatureManager


class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sphere Creation Test")
        self.setGeometry(100, 100, 800, 600)

        # Create armature manager
        am = ArmatureManager()

        # Create OpenGL widget
        self.opengl_widget = OpenGLWidget(am, self)

        # Set up layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Add instructions
        instructions = QLabel(
            "Right-click in the 3D view to create spheres. Check the console for position feedback.")
        layout.addWidget(instructions)

        layout.addWidget(self.opengl_widget)

        # Timer for continuous rendering
        self.timer = QTimer()
        self.timer.timeout.connect(self.opengl_widget.update)
        self.timer.start(16)  # ~60 FPS

    def get_current_radius(self):
        """Get the current radius from the UI"""
        return 0.5  # Default radius for testing


def main():
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
