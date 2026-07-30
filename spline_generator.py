#!/usr/bin/env python3
"""
Spline Generator with Curve Editor for Radius Control
"""

import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox,
                               QSlider, QGraphicsView, QGraphicsScene, QGraphicsItem,
                               QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem,
                               QSizePolicy)
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QPen, QBrush, QColor, QPainter, QPainterPath, QFont
import math


class CurvePoint(QGraphicsEllipseItem):
    """Draggable point on the curve editor"""

    def __init__(self, x, y, index, parent=None):
        super().__init__(-5, -5, 10, 10, parent)
        self.index = index
        self.setPos(x, y)
        self.setBrush(QBrush(QColor(255, 255, 0)))  # Yellow
        self.setPen(QPen(QColor(0, 0, 0), 2))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.point_moved_callback = None

    def set_moved_callback(self, callback):
        """Set callback for when point is moved"""
        self.point_moved_callback = callback

    def mouseMoveEvent(self, event):
        """Handle mouse move during dragging"""
        # Get the new position from the event
        new_pos = event.scenePos()

        # Constrain to view bounds
        new_pos.setX(max(0, min(self.scene().width(), new_pos.x())))
        new_pos.setY(max(0, min(self.scene().height(), new_pos.y())))

        # Lock start and end points horizontally
        if self.index == 0:  # First point
            new_pos.setX(0)
        elif hasattr(self, 'is_last_point') and self.is_last_point:  # Last point
            new_pos.setX(self.scene().width())

        # Set the constrained position
        self.setPos(new_pos)

        # Call the callback with the new position
        if self.point_moved_callback:
            self.point_moved_callback(self.index, new_pos.x(), new_pos.y())


class CurveEditor(QGraphicsView):
    """Graph curve editor for radius control"""

    curve_changed = Signal(list)  # List of (x, y) points

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Disable scroll bars
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Curve points (x, y) where x is position along spline (0-1), y is radius multiplier (0-2)
        # Default: constant radius
        self.points = [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0)]
        self.point_items = []

        # Grid settings
        self.grid_size = 40
        self.view_width = 280
        self.view_height = 120

        # Set size policy to expand
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

        self.setup_scene()
        self.update_curve()

    def resizeEvent(self, event):
        """Handle resize events to update view dimensions"""
        super().resizeEvent(event)
        # Update view dimensions based on new size
        self.view_width = self.width() - 20  # Leave some margin
        self.view_height = self.height() - 20
        self.setup_scene()
        self.update_curve()
        self.update_point_positions()

    def setup_scene(self):
        """Setup the scene with grid and labels"""
        self.scene.clear()
        self.point_items.clear()

        # Set scene rect to match view exactly
        self.scene.setSceneRect(0, 0, self.view_width, self.view_height)

        # Draw grid
        self.draw_grid()

        # Draw axis labels
        self.draw_labels()

        # Create draggable points
        for i, (x, y) in enumerate(self.points):
            scene_x = x * self.view_width
            scene_y = (2.0 - y) * self.view_height / \
                2.0  # Convert 0-2 range to 0-height
            point = CurvePoint(scene_x, scene_y, i)
            point.set_moved_callback(self.on_point_moved)

            # Mark the last point
            if i == len(self.points) - 1:
                point.is_last_point = True

            self.scene.addItem(point)
            self.point_items.append(point)

    def draw_grid(self):
        """Draw grid lines"""
        pen = QPen(QColor(100, 100, 100), 1)

        # Vertical lines (position)
        for i in range(0, self.view_width + 1, self.grid_size):
            line = self.scene.addLine(i, 0, i, self.view_height, pen)

        # Horizontal lines (radius)
        for i in range(0, self.view_height + 1, self.grid_size):
            line = self.scene.addLine(0, i, self.view_width, i, pen)

    def draw_labels(self):
        """Draw axis labels"""
        font = QFont("Arial", 8)

        # X-axis labels (position) - position below the graph
        for i in range(0, 6):
            x = i * self.view_width / 5
            label = self.scene.addText(f"{i/5:.1f}", font)
            label.setPos(x - 10, self.view_height + 2)

        # Y-axis labels (radius multiplier) - position to the left with more space
        for i in range(0, 6):
            y = i * self.view_height / 5
            radius_val = 2.0 - (i * 2.0 / 5)  # 2.0 to 0.0
            label = self.scene.addText(f"{radius_val:.1f}", font)
            label.setPos(-30, y - 5)

    def on_point_moved(self, index, x, y):
        """Handle point movement"""
        # Normalize coordinates
        norm_x = max(0.0, min(1.0, x / self.view_width))
        # Convert scene Y back to normalized Y (0-2)
        norm_y = max(0.0, min(2.0, 2.0 - (y / self.view_height) * 2.0))

        # Update point
        self.points[index] = (norm_x, norm_y)

        # Sort points by x coordinate
        self.points.sort(key=lambda p: p[0])

        # Update point indices
        for i, point in enumerate(self.point_items):
            point.index = i

        # Update curve and reposition all points to ensure alignment
        self.update_curve()
        self.update_point_positions()
        self.curve_changed.emit(self.points)

    def update_point_positions(self):
        """Update the positions of all control points to match current view dimensions"""
        for i, (x, y) in enumerate(self.points):
            if i < len(self.point_items):
                point = self.point_items[i]
                scene_x = x * self.view_width
                scene_y = (1.0 - y) * self.view_height
                point.setPos(scene_x, scene_y)

                # Update the last point flag
                point.is_last_point = (i == len(self.points) - 1)

    def update_curve(self):
        """Update the curve visualization"""
        # Remove old curve
        for item in self.scene.items():
            if isinstance(item, QGraphicsLineItem) and item.data(0) == "curve":
                self.scene.removeItem(item)

        # Draw new curve
        if len(self.points) >= 2:
            pen = QPen(QColor(0, 255, 0), 3)  # Green curve

            for i in range(len(self.points) - 1):
                x1, y1 = self.points[i]
                x2, y2 = self.points[i + 1]

                # Convert to scene coordinates
                scene_x1 = x1 * self.view_width
                scene_y1 = (1.0 - y1) * self.view_height
                scene_x2 = x2 * self.view_width
                scene_y2 = (1.0 - y2) * self.view_height

                line = self.scene.addLine(
                    scene_x1, scene_y1, scene_x2, scene_y2, pen)
                line.setData(0, "curve")  # Mark as curve line

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

    def add_point(self):
        """Add a new control point"""
        # Add point at middle of curve
        new_x = 0.5
        new_y = 1.0

        self.points.append((new_x, new_y))
        self.points.sort(key=lambda p: p[0])

        # Recreate scene
        self.setup_scene()
        self.update_curve()
        self.update_point_positions()
        self.curve_changed.emit(self.points)

    def remove_point(self):
        """Remove the last control point (keep at least 2)"""
        if len(self.points) > 2:
            self.points.pop()
            self.setup_scene()
            self.update_curve()
            self.curve_changed.emit(self.points)


class SplineGenerator(QWidget):
    """Spline generator widget with curve editor"""

    spline_generated = Signal(list)  # List of (position, radius) tuples

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(5)  # Reduce spacing
        layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins

        # Spline parameters
        params_group = QGroupBox("Spline Parameters")
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(2)  # Very compact spacing

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
        curve_layout.setSpacing(2)  # Very compact spacing

        # Curve editor
        self.curve_editor = CurveEditor()
        self.curve_editor.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        curve_layout.addWidget(self.curve_editor)

        # Curve controls
        curve_controls_layout = QHBoxLayout()

        self.add_point_btn = QPushButton("Add Point")
        self.add_point_btn.clicked.connect(self.curve_editor.add_point)
        curve_controls_layout.addWidget(self.add_point_btn)

        self.remove_point_btn = QPushButton("Remove Point")
        self.remove_point_btn.clicked.connect(self.curve_editor.remove_point)
        curve_controls_layout.addWidget(self.remove_point_btn)

        curve_layout.addLayout(curve_controls_layout)
        layout.addWidget(curve_group)

        # Generate button
        self.generate_btn = QPushButton("Generate Spline")
        self.generate_btn.clicked.connect(self.generate_spline)
        layout.addWidget(self.generate_btn)

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
                z = math.sin(t * 2 * math.pi) * spline_length * 0.3
            elif curve_type == 2:  # Spiral
                angle = t * 2 * math.pi * 2
                radius = t * spline_length * 0.5
                x = radius * math.cos(angle)
                y = 0.0
                z = radius * math.sin(angle)
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
