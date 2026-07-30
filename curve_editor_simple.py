#!/usr/bin/env python3
"""
Simple Curve Editor using PyQtGraph - Working Version
"""

import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox, QApplication
from PySide6.QtCore import Signal
import numpy as np
import sys


class SimpleCurveEditor(QWidget):
    """Simple curve editor using PyQtGraph"""

    curve_changed = Signal(list)  # List of (x, y) points

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)

        # Create the plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')  # White background
        self.plot_widget.setLabel('left', 'Radius Multiplier')
        self.plot_widget.setLabel('bottom', 'Position')
        self.plot_widget.setXRange(0, 1)
        self.plot_widget.setYRange(0, 2)
        self.plot_widget.showGrid(x=True, y=True)

        # Set fixed size to prevent cutoff
        self.plot_widget.setMinimumSize(300, 200)
        self.plot_widget.setMaximumHeight(250)

        # Add the plot to layout
        layout.addWidget(self.plot_widget)

        # Control buttons
        button_layout = QHBoxLayout()

        self.add_point_btn = QPushButton("Add Point")
        self.add_point_btn.clicked.connect(self.add_point)
        button_layout.addWidget(self.add_point_btn)

        self.remove_point_btn = QPushButton("Remove Point")
        self.remove_point_btn.clicked.connect(self.remove_point)
        button_layout.addWidget(self.remove_point_btn)

        layout.addLayout(button_layout)

        # Initialize curve data
        self.points = [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0)]
        self.plot_curve()

    def plot_curve(self):
        """Plot the curve with control points"""
        # Clear previous plots
        self.plot_widget.clear()

        if len(self.points) < 2:
            return

        # Extract x and y coordinates
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]

        # Plot the curve line
        self.plot_widget.plot(x_coords, y_coords,
                              pen=pg.mkPen('g', width=3),
                              symbol='o',
                              symbolBrush='y',
                              symbolSize=10)

        # Force redraw
        self.plot_widget.update()

    def add_point(self):
        """Add a new control point"""
        # Add point at middle
        new_x = 0.5
        new_y = 1.0
        self.points.append((new_x, new_y))
        self.points.sort(key=lambda p: p[0])  # Sort by x
        self.plot_curve()
        self.curve_changed.emit(self.points)

    def remove_point(self):
        """Remove the last control point"""
        if len(self.points) > 2:
            self.points.pop()
            self.plot_curve()
            self.curve_changed.emit(self.points)

    def get_radius_at_position(self, t):
        """Get radius multiplier at position t (0-1) along the curve"""
        if len(self.points) < 2:
            return 1.0

        # Find the segment containing t
        for i in range(len(self.points) - 1):
            x1, y1 = self.points[i]
            x2, y2 = self.points[i + 1]

            if x1 <= t <= x2:
                # Linear interpolation
                if x2 == x1:
                    return y1
                ratio = (t - x1) / (x2 - x1)
                return y1 + ratio * (y2 - y1)

        # If t is outside the curve, return the closest endpoint
        if t <= self.points[0][0]:
            return self.points[0][1]
        else:
            return self.points[-1][1]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimpleCurveEditor()
    window.setWindowTitle("Simple Curve Editor")
    window.resize(400, 300)
    window.show()
    sys.exit(app.exec())
