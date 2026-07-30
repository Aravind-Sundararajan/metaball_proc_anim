import sys
import math
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QLabel, QSpinBox, QDoubleSpinBox, QPushButton,
                               QComboBox, QGraphicsView, QGraphicsScene, QGraphicsItem,
                               QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem,
                               QGraphicsPathItem, QSizePolicy)
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QPen, QBrush, QColor, QPainter, QFont, QPainterPath


class HandlePoint(QGraphicsEllipseItem):
    def __init__(self, x, y, radius=3, parent=None):
        super().__init__(-radius, -radius, radius*2, radius*2)
        self.setPos(x, y)
        self.setBrush(QBrush(QColor(255, 255, 100)))  # Yellow handles
        self.setPen(QPen(QColor(200, 200, 50), 1))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.radius = radius
        self.parent_editor = parent
        self.control_point = None  # Reference to the control point this handle belongs to
        self.is_in_handle = False  # True if this is the "in" handle, False if "out" handle

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        # Update the curve when handle moves
        if self.parent_editor:
            self.parent_editor.update_curve()


class ControlPoint(QGraphicsEllipseItem):
    def __init__(self, x, y, radius=5, parent=None):
        super().__init__(-radius, -radius, radius*2, radius*2)
        self.setPos(x, y)
        self.setBrush(QBrush(QColor(255, 100, 100)))
        self.setPen(QPen(QColor(200, 50, 50), 2))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.radius = radius
        self.is_start = False
        self.is_end = False
        self.parent_editor = parent

        # Handle points for controlling curve derivatives
        self.in_handle = None
        self.out_handle = None

    def set_start_point(self):
        self.is_start = True
        self.setBrush(QBrush(QColor(100, 255, 100)))
        self.setPen(QPen(QColor(50, 200, 50), 2))
        # Set cursor to indicate vertical-only movement
        self.setCursor(Qt.CursorShape.SizeVerCursor)

    def set_end_point(self):
        self.is_end = True
        self.setBrush(QBrush(QColor(100, 100, 255)))
        self.setPen(QPen(QColor(50, 50, 200), 2))
        # Set cursor to indicate vertical-only movement
        self.setCursor(Qt.CursorShape.SizeVerCursor)

    def mouseMoveEvent(self, event):
        # For start and end points, lock horizontal movement
        if self.is_start or self.is_end:
            # Call parent to handle the basic movement
            super().mouseMoveEvent(event)
            # Then lock the x position back to the original
            if self.is_start:
                self.setPos(0, self.pos().y())  # Lock to x=0
            elif self.is_end:
                self.setPos(300, self.pos().y())  # Lock to x=300
        else:
            # Normal movement for middle points
            super().mouseMoveEvent(event)

        # Move handles with the control point
        if self.in_handle:
            # Keep in_handle at a fixed offset from control point
            offset = self.in_handle.pos() - self.pos()
            self.in_handle.setPos(self.pos() + offset)
        if self.out_handle:
            # Keep out_handle at a fixed offset from control point
            offset = self.out_handle.pos() - self.pos()
            self.out_handle.setPos(self.pos() + offset)

        # Update the curve when control point moves
        if self.parent_editor:
            self.parent_editor.update_curve()


class SimpleCurveEditor(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Set size policy to let it size based on content
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Preferred)
        # Increase minimum height to accommodate axis labels
        self.setMinimumSize(350, 250)  # Increased height from 200 to 250
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Control points and handles
        self.control_points = []
        self.handle_lines = []
        self.curve_path = None

        # Setup scene
        self.setup_scene()
        self.create_default_points()

    def setup_scene(self):
        # Set scene rect with padding - expanded to accommodate axis titles outside graph area
        self.scene.setSceneRect(-50, -10, 360, 230)

        # Add grid lines
        pen = QPen(QColor(200, 200, 200), 1)

        # Vertical lines (position) - extend to show the full range
        for i in range(0, 12):  # Changed from 11 to 12 to include position 1.0
            x = i * 30
            self.scene.addLine(x, 0, x, 180, pen)

        # Horizontal lines (radius)
        for i in range(0, 7):
            y = i * 30
            self.scene.addLine(0, y, 300, y, pen)

        # Add labels
        font = QFont("Arial", 8)

        # X-axis labels (Position) - include the 1.0 label
        for i in range(0, 12):  # Changed from 11 to 12
            x = i * 30
            text = self.scene.addText(f"{i/10:.1f}", font)
            text.setPos(x - 10, 185)

        # Y-axis labels (Radius) - positioned to be fully visible
        for i in range(0, 7):
            y = i * 30
            # Calculate the actual radius value using a formula that gives positive values
            # Map y=0 to radius=2.0, y=180 to radius=0.1
            radius_value = 2.0 - (y / 180) * 1.9  # This gives range 2.0 to 0.1
            text = self.scene.addText(f"{radius_value:.1f}", font)
            text.setPos(-25, y - 8)  # Adjusted for compact scene

        # Axis titles - positioned outside the graph area
        title_font = QFont("Arial", 10, QFont.Weight.Bold)
        x_title = self.scene.addText("Position", title_font)
        # Below the x-axis numbers, outside graph area
        x_title.setPos(120, 210)

        y_title = self.scene.addText("Radius", title_font)
        # To the left of y-axis numbers, outside graph area
        y_title.setPos(-50, 80)
        y_title.setRotation(-90)  # Rotate to display vertically

    def create_default_points(self):
        # Clear existing points and handles
        for point in self.control_points:
            self.scene.removeItem(point)
            if point.in_handle:
                self.scene.removeItem(point.in_handle)
            if point.out_handle:
                self.scene.removeItem(point.out_handle)

        for line in self.handle_lines:
            self.scene.removeItem(line)

        if self.curve_path:
            self.scene.removeItem(self.curve_path)

        self.control_points.clear()
        self.handle_lines.clear()
        self.curve_path = None

        # Create default points
        points = [
            (0, 90),    # Start: position 0, radius 1.0
            (150, 60),  # Middle: position 0.5, radius 1.4
            (300, 90)   # End: position 1.0, radius 1.0
        ]

        for i, (x, y) in enumerate(points):
            point = ControlPoint(x, y, parent=self)
            if i == 0:
                point.set_start_point()
            elif i == len(points) - 1:
                point.set_end_point()

            # Create handle points
            if i > 0:  # Not the first point
                # In handle (for curve coming into this point)
                in_x = x - 30  # 30 pixels to the left
                in_y = y
                in_handle = HandlePoint(in_x, in_y, parent=self)
                in_handle.control_point = point
                in_handle.is_in_handle = True
                point.in_handle = in_handle
                self.scene.addItem(in_handle)

                # Connect handle to control point with a line
                handle_line = self.scene.addLine(in_x, in_y, x, y,
                                                 QPen(QColor(150, 150, 150), 1, Qt.PenStyle.DashLine))
                self.handle_lines.append(handle_line)

            if i < len(points) - 1:  # Not the last point
                # Out handle (for curve going out from this point)
                out_x = x + 30  # 30 pixels to the right
                out_y = y
                out_handle = HandlePoint(out_x, out_y, parent=self)
                out_handle.control_point = point
                out_handle.is_in_handle = False
                point.out_handle = out_handle
                self.scene.addItem(out_handle)

                # Connect handle to control point with a line
                handle_line = self.scene.addLine(out_x, out_y, x, y,
                                                 QPen(QColor(150, 150, 150), 1, Qt.PenStyle.DashLine))
                self.handle_lines.append(handle_line)

            self.control_points.append(point)
            self.scene.addItem(point)

        self.update_curve()

    def update_curve(self):
        # Remove old curve
        if self.curve_path:
            self.scene.removeItem(self.curve_path)

        if len(self.control_points) < 2:
            return

        # Sort points by x position
        sorted_points = sorted(self.control_points, key=lambda p: p.pos().x())

        # Create Bezier curve path
        path = QPainterPath()

        if len(sorted_points) == 2:
            # Simple line for 2 points
            p1 = sorted_points[0].pos()
            p2 = sorted_points[1].pos()
            path.moveTo(p1)
            path.lineTo(p2)
        else:
            # Cubic Bezier curve using handle points
            path.moveTo(sorted_points[0].pos())

            for i in range(len(sorted_points) - 1):
                current_point = sorted_points[i]
                next_point = sorted_points[i + 1]

                # Get control points from handles
                if current_point.out_handle and next_point.in_handle:
                    # Use the actual handle positions
                    cp1 = current_point.out_handle.pos()
                    cp2 = next_point.in_handle.pos()
                    path.cubicTo(cp1, cp2, next_point.pos())
                else:
                    # Fallback to linear if no handles
                    path.lineTo(next_point.pos())

        # Create and add the curve path
        pen = QPen(QColor(100, 100, 255), 3)
        self.curve_path = QGraphicsPathItem(path)
        self.curve_path.setPen(pen)
        self.scene.addItem(self.curve_path)

        # Update handle lines
        self.update_handle_lines()

    def update_handle_lines(self):
        # Remove old handle lines
        for line in self.handle_lines:
            self.scene.removeItem(line)
        self.handle_lines.clear()

        # Draw new handle lines
        for point in self.control_points:
            if point.in_handle:
                line = self.scene.addLine(
                    point.in_handle.pos().x(), point.in_handle.pos().y(),
                    point.pos().x(), point.pos().y(),
                    QPen(QColor(150, 150, 150), 1, Qt.PenStyle.DashLine)
                )
                self.handle_lines.append(line)

            if point.out_handle:
                line = self.scene.addLine(
                    point.out_handle.pos().x(), point.out_handle.pos().y(),
                    point.pos().x(), point.pos().y(),
                    QPen(QColor(150, 150, 150), 1, Qt.PenStyle.DashLine)
                )
                self.handle_lines.append(line)

    def add_point(self):
        # Add point at middle of current range
        if len(self.control_points) < 10:  # Limit number of points
            x = 150  # Middle position
            y = 90   # Middle radius

            point = ControlPoint(x, y, parent=self)

            # Create handle points
            in_x = x - 30
            in_y = y
            in_handle = HandlePoint(in_x, in_y, parent=self)
            in_handle.control_point = point
            in_handle.is_in_handle = True
            point.in_handle = in_handle
            self.scene.addItem(in_handle)

            out_x = x + 30
            out_y = y
            out_handle = HandlePoint(out_x, out_y, parent=self)
            out_handle.control_point = point
            out_handle.is_in_handle = False
            point.out_handle = out_handle
            self.scene.addItem(out_handle)

            self.control_points.append(point)
            self.scene.addItem(point)
            self.update_curve()

    def remove_point(self):
        # Remove last added point (not start or end)
        removable_points = [
            p for p in self.control_points if not p.is_start and not p.is_end]
        if removable_points:
            point = removable_points[-1]
            self.scene.removeItem(point)
            if point.in_handle:
                self.scene.removeItem(point.in_handle)
            if point.out_handle:
                self.scene.removeItem(point.out_handle)
            self.control_points.remove(point)
            self.update_curve()

    def get_radius_at_position(self, position):
        """Get radius multiplier at given position (0-1) using Bezier interpolation"""
        if not self.control_points or len(self.control_points) < 2:
            return 1.0

        # Convert position to scene coordinates
        scene_x = position * 300

        # Sort points by x position
        sorted_points = sorted(self.control_points, key=lambda p: p.pos().x())

        # If only 2 points, use linear interpolation
        if len(sorted_points) == 2:
            p1 = sorted_points[0]
            p2 = sorted_points[1]

            if p1.pos().x() <= scene_x <= p2.pos().x():
                t = (scene_x - p1.pos().x()) / (p2.pos().x() - p1.pos().x())
                # Convert y coordinates to radius values using positive-only formula
                radius1 = 2.0 - (p1.pos().y() / 180) * 1.9
                radius2 = 2.0 - (p2.pos().y() / 180) * 1.9
                radius = radius1 + t * (radius2 - radius1)
                return max(0.1, min(3.0, radius))

        # For 3+ points, use Bezier curve interpolation with handles
        points = [(p.pos().x(), p.pos().y()) for p in sorted_points]
        handles = []

        for point in sorted_points:
            if point.out_handle:
                handles.append((point.out_handle.pos().x(),
                               point.out_handle.pos().y()))
            else:
                handles.append((point.pos().x(), point.pos().y()))

        # Find the segment containing our x position
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            if x1 <= scene_x <= x2:
                # Calculate t parameter for this segment
                t = (scene_x - x1) / (x2 - x1) if x2 != x1 else 0

                # Use cubic Bezier interpolation with handles
                if i < len(handles) - 1:
                    cp1_x, cp1_y = handles[i]
                    cp2_x, cp2_y = handles[i + 1]

                    # Convert y coordinates to radius values using positive-only formula
                    radius1 = 2.0 - (y1 / 180) * 1.9
                    radius2 = 2.0 - (y2 / 180) * 1.9
                    cp1_radius = 2.0 - (cp1_y / 180) * 1.9
                    cp2_radius = 2.0 - (cp2_y / 180) * 1.9

                    # Cubic Bezier formula: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
                    radius = ((1-t)**3 * radius1 +
                              3*(1-t)**2 * t * cp1_radius +
                              3*(1-t) * t**2 * cp2_radius +
                              t**3 * radius2)
                else:
                    # Fallback to linear interpolation
                    radius1 = 2.0 - (y1 / 180) * 1.9
                    radius2 = 2.0 - (y2 / 180) * 1.9
                    radius = radius1 + t * (radius2 - radius1)

                return max(0.1, min(3.0, radius))

        # If outside range, return closest point
        if scene_x <= sorted_points[0].pos().x():
            return 2.0 - (sorted_points[0].pos().y() / 180) * 1.9
        else:
            return 2.0 - (sorted_points[-1].pos().y() / 180) * 1.9


class SplineGeneratorSimple(QWidget):
    # Signal for spline generation
    spline_generated = Signal(list)  # Emits list of (position, radius) tuples

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(5)

        # Title
        title = QLabel("Spline Generator")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Parameters
        param_layout = QVBoxLayout()

        # Number of spheres
        num_layout = QHBoxLayout()
        num_layout.addWidget(QLabel("Number of spheres:"))
        self.num_spheres = QSpinBox()
        self.num_spheres.setRange(2, 50)
        self.num_spheres.setValue(10)
        num_layout.addWidget(self.num_spheres)
        param_layout.addLayout(num_layout)

        # Base radius
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Base radius:"))
        self.base_radius = QDoubleSpinBox()
        self.base_radius.setRange(0.1, 2.0)
        self.base_radius.setSingleStep(0.1)
        self.base_radius.setValue(0.5)
        radius_layout.addWidget(self.base_radius)
        param_layout.addLayout(radius_layout)

        # Spline length
        length_layout = QHBoxLayout()
        length_layout.addWidget(QLabel("Spline length:"))
        self.spline_length = QDoubleSpinBox()
        self.spline_length.setRange(1.0, 20.0)
        self.spline_length.setSingleStep(0.5)
        self.spline_length.setValue(5.0)
        length_layout.addWidget(self.spline_length)
        param_layout.addLayout(length_layout)

        # Curve type
        curve_layout = QHBoxLayout()
        curve_layout.addWidget(QLabel("Curve type:"))
        self.curve_type = QComboBox()
        self.curve_type.addItems(["Linear", "Sine", "Spiral", "Custom"])
        curve_layout.addWidget(self.curve_type)
        param_layout.addLayout(curve_layout)

        layout.addLayout(param_layout)

        # Curve editor
        curve_label = QLabel("Radius Curve Editor:")
        curve_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(curve_label)

        self.curve_editor = SimpleCurveEditor()
        layout.addWidget(self.curve_editor)

        # Curve editor controls
        curve_controls = QHBoxLayout()
        self.add_point_btn = QPushButton("Add Point")
        self.add_point_btn.clicked.connect(self.curve_editor.add_point)
        curve_controls.addWidget(self.add_point_btn)

        self.remove_point_btn = QPushButton("Remove Point")
        self.remove_point_btn.clicked.connect(self.curve_editor.remove_point)
        curve_controls.addWidget(self.remove_point_btn)
        layout.addLayout(curve_controls)

        # Generate button
        self.generate_btn = QPushButton("Generate Spline")
        self.generate_btn.clicked.connect(self.generate_spline)
        layout.addWidget(self.generate_btn)

        self.setLayout(layout)

    def generate_spline(self):
        num_spheres = self.num_spheres.value()
        base_radius = self.base_radius.value()
        spline_length = self.spline_length.value()
        curve_type = self.curve_type.currentText()

        # Calculate positions and radii
        spline_data = []

        for i in range(num_spheres):
            # Position along spline (0 to 1)
            t = i / (num_spheres - 1) if num_spheres > 1 else 0.5

            # Calculate 3D position based on curve type
            # Note: This only affects the 3D spline path, not the radius curve editor
            if curve_type == "Linear":
                # Straight line along X axis
                x = t * spline_length
                y = 0
                z = 0
            elif curve_type == "Sine":
                # Sine wave pattern in Y direction
                x = t * spline_length
                y = math.sin(t * 2 * math.pi) * (spline_length * 0.3)
                z = 0
            elif curve_type == "Spiral":
                # Spiral pattern: circles in XY plane, rising in Z
                angle = t * 2 * math.pi * 2
                radius = t * (spline_length * 0.2)
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = t * spline_length * 0.5
            else:  # Custom
                # Same as Linear for now
                x = t * spline_length
                y = 0
                z = 0

            position = (x, y, z)

            # Calculate radius using curve editor
            radius_multiplier = self.curve_editor.get_radius_at_position(t)
            radius = base_radius * radius_multiplier

            spline_data.append((position, radius))

        # Emit signal with spline data
        self.spline_generated.emit(spline_data)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SplineGeneratorSimple()
    window.show()
    sys.exit(app.exec())
