#!/usr/bin/env python3
"""
Simple Curve Editor using PyQtGraph
"""

import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox
from PySide6.QtCore import Signal
import numpy as np


class CurveEditorPyQtGraph(QWidget):
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
        self.make_points_draggable()

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
        curve_plot = self.plot_widget.plot(x_coords, y_coords,
                                           pen=pg.mkPen('g', width=3),
                                           symbol=None)  # No symbols on line

        # Add control points as draggable items
        for i, (x, y) in enumerate(self.points):
            point = pg.ScatterPlotItem([x], [y],
                                       pen=pg.mkPen('k', width=2),
                                       brush=pg.mkBrush('y'),
                                       size=15)
            point.sigClicked.connect(
                lambda item, points, index=i: self.on_point_clicked(index))
            self.plot_widget.addItem(point)

        # Force redraw
        self.plot_widget.update()

    def on_point_clicked(self, index):
        """Handle point click"""
        print(f"Point {index} clicked at {self.points[index]}")

    def make_points_draggable(self):
        """Make all points clickable"""
        # Clear existing points
        for item in self.plot_widget.items():
            if isinstance(item, pg.ScatterPlotItem):
                self.plot_widget.removeItem(item)

        # Add clickable points
        for i, (x, y) in enumerate(self.points):
            point = pg.ScatterPlotItem([x], [y],
                                       pen=pg.mkPen('k', width=2),
                                       brush=pg.mkBrush('y'),
                                       size=15)
            point.sigClicked.connect(
                lambda item, points, index=i: self.on_point_clicked(index))

            self.plot_widget.addItem(point)

    def add_point(self):
        """Add a new control point"""
        # Add point at middle
        new_x = 0.5
        new_y = 1.0
        self.points.append((new_x, new_y))
        self.points.sort(key=lambda p: p[0])  # Sort by x
        self.make_points_draggable()
        self.plot_curve()
        self.curve_changed.emit(self.points)

    def remove_point(self):
        """Remove the last control point"""
        if len(self.points) > 2:
            self.points.pop()
            self.make_points_draggable()
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


class SplineGeneratorPyQtGraph(QWidget):
    """Spline generator using PyQtGraph curve editor"""

    spline_generated = Signal(list)  # List of (position, radius) tuples

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)

        # Spline parameters
        params_group = QGroupBox("Spline Parameters")
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(2)

        # Number of spheres
        spheres_layout = QHBoxLayout()
        spheres_layout.addWidget(QLabel("Number of Spheres:"))
        self.sphere_count_spinbox = QSpinBox()
        self.sphere_count_spinbox.setRange(2, 50)
        self.sphere_count_spinbox.setValue(10)
        spheres_layout.addWidget(self.sphere_count_spinbox)
        params_layout.addLayout(spheres_layout)

        # Base radius
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Base Radius:"))
        self.base_radius_spinbox = QDoubleSpinBox()
        self.base_radius_spinbox.setRange(0.1, 2.0)
        self.base_radius_spinbox.setValue(0.5)
        self.base_radius_spinbox.setSingleStep(0.1)
        self.base_radius_spinbox.setDecimals(2)
        radius_layout.addWidget(self.base_radius_spinbox)
        params_layout.addLayout(radius_layout)

        # Spline length
        length_layout = QHBoxLayout()
        length_layout.addWidget(QLabel("Spline Length:"))
        self.spline_length_spinbox = QDoubleSpinBox()
        self.spline_length_spinbox.setRange(1.0, 20.0)
        self.spline_length_spinbox.setValue(5.0)
        self.spline_length_spinbox.setSingleStep(0.5)
        self.spline_length_spinbox.setDecimals(2)
        length_layout.addWidget(self.spline_length_spinbox)
        params_layout.addLayout(length_layout)

        # Curve type
        curve_layout = QHBoxLayout()
        curve_layout.addWidget(QLabel("Curve Type:"))
        self.curve_type_spinbox = QSpinBox()
        self.curve_type_spinbox.setRange(0, 3)
        self.curve_type_spinbox.setValue(0)
        self.curve_type_spinbox.setToolTip(
            "0=Linear, 1=Sine, 2=Spiral, 3=Custom")
        curve_layout.addWidget(self.curve_type_spinbox)
        params_layout.addLayout(curve_layout)

        layout.addWidget(params_group)

        # Curve editor
        curve_group = QGroupBox("Radius Curve Editor")
        curve_layout = QVBoxLayout(curve_group)
        curve_layout.setSpacing(2)

        self.curve_editor = CurveEditorPyQtGraph()
        self.curve_editor.curve_changed.connect(self.on_curve_changed)
        curve_layout.addWidget(self.curve_editor)

        layout.addWidget(curve_group)

        # Generate button
        self.generate_btn = QPushButton("Generate Spline")
        self.generate_btn.clicked.connect(self.generate_spline)
        layout.addWidget(self.generate_btn)

    def on_curve_changed(self, points):
        """Handle curve changes"""
        print(f"Curve updated with {len(points)} points")

    def generate_spline(self):
        """Generate the spline based on current parameters"""
        sphere_count = self.sphere_count_spinbox.value()
        base_radius = self.base_radius_spinbox.value()
        spline_length = self.spline_length_spinbox.value()
        curve_type = self.curve_type_spinbox.value()

        # Generate positions along the spline
        positions = []
        for i in range(sphere_count):
            t = i / (sphere_count - 1)  # Parameter from 0 to 1

            # Generate position based on curve type
            if curve_type == 0:  # Linear
                x = t * spline_length
                y = 0.0
                z = 0.0
            elif curve_type == 1:  # Sine wave
                x = t * spline_length
                y = 0.0
                z = np.sin(t * 2 * np.pi) * spline_length * 0.3
            elif curve_type == 2:  # Spiral
                angle = t * 2 * np.pi * 2
                radius = t * spline_length * 0.5
                x = radius * np.cos(angle)
                y = 0.0
                z = radius * np.sin(angle)
            else:  # Custom (linear for now)
                x = t * spline_length
                y = 0.0
                z = 0.0

            positions.append([x, y, z])

        # Generate radii using curve editor
        radii = []
        for i in range(sphere_count):
            t = i / (sphere_count - 1)
            radius_multiplier = self.curve_editor.get_radius_at_position(t)
            radius = base_radius * radius_multiplier
            radii.append(radius)

        # Emit the spline data
        spline_data = list(zip(positions, radii))
        self.spline_generated.emit(spline_data)

        print(f"Generated spline with {sphere_count} spheres")
        for i, ((pos, radius)) in enumerate(spline_data):
            print(f"  Sphere {i}: pos={pos}, radius={radius:.3f}")


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = SplineGeneratorPyQtGraph()
    window.show()
    sys.exit(app.exec())
