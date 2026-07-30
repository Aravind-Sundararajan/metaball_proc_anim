#!/usr/bin/env python3
"""
OpenGL Widget for 3D rendering and interaction
"""

import numpy as np
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QMouseEvent, QWheelEvent, QVector3D, QMatrix4x4
from OpenGL.GL import *
from OpenGL.GLU import *
import math


class OpenGLWidget(QOpenGLWidget):
    sphere_created = Signal(str)
    bone_selected = Signal(str)
    bone_deleted = Signal(str)

    def __init__(self, armature_manager, main_window=None):
        super().__init__()
        self.armature_manager = armature_manager
        self.main_window = main_window

        # Camera settings
        self.camera_distance = 10.0
        self.camera_x_rotation = 0.0
        self.camera_y_rotation = 0.0

        # Mouse interaction
        self.last_mouse_pos = QPoint()
        self.mouse_pressed = False

        # Selection
        self.selected_bone = None
        self.hover_bone = None

        # Sphere creation
        self.sphere_counter = 0

        # Connection sequence tracking
        self.connection_sequence = []

        # Connection mode: "Direct Connect" or "Chain Connect"
        self.connection_mode = "Direct Connect"

        # Matrix for coordinate transformation
        self.projection = QMatrix4x4()
        self.view = QMatrix4x4()

        # No default spheres - user will create them

    def initializeGL(self):
        # Set clear color to dark gray
        glClearColor(0.2, 0.2, 0.2, 1.0)

        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        # Enable lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)

        # Set up lighting
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])

        # Set material properties
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)

        # Set up projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, width / height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

        # Update Qt projection matrix
        aspect = width / height if height else 1
        self.projection.setToIdentity()
        self.projection.perspective(45.0, aspect, 0.1, 100.0)

    def paintGL(self):
        try:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            # Set up camera
            glTranslatef(0.0, 0.0, -self.camera_distance)
            glRotatef(self.camera_x_rotation, 1.0, 0.0, 0.0)
            glRotatef(self.camera_y_rotation, 0.0, 1.0, 0.0)

            # Update Qt view matrix to match OpenGL camera
            self.view.setToIdentity()
            self.view.translate(0.0, 0.0, -self.camera_distance)
            self.view.rotate(self.camera_x_rotation, 1.0, 0.0, 0.0)
            self.view.rotate(self.camera_y_rotation, 0.0, 1.0, 0.0)

            # Draw coordinate axes
            self.draw_axes()

            # Draw ground grid
            self.draw_ground_grid()

            # Draw armature
            self.draw_armature()

        except Exception as e:
            print(f"OpenGL error in paintGL: {e}")

    def draw_axes(self):
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)

        # X axis (red)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(2.0, 0.0, 0.0)
        glEnd()

        # Y axis (green)
        glBegin(GL_LINES)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 2.0, 0.0)
        glEnd()

        # Z axis (blue)
        glBegin(GL_LINES)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 2.0)
        glEnd()

        glEnable(GL_LIGHTING)

    def draw_ground_grid(self):
        """Draw a grid on the ground plane for better spatial reference"""
        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        glColor3f(0.4, 0.4, 0.4)

        # Draw grid lines
        glBegin(GL_LINES)
        for i in range(-5, 6):
            # X lines
            glVertex3f(i, 0.0, -5.0)
            glVertex3f(i, 0.0, 5.0)
            # Z lines
            glVertex3f(-5.0, 0.0, i)
            glVertex3f(5.0, 0.0, i)
        glEnd()

        glEnable(GL_LIGHTING)

    def draw_armature(self):
        bones = self.armature_manager.get_bones()

        for bone_name, bone_data in bones.items():
            # Set color based on selection state
            if bone_name == self.selected_bone:
                glColor3f(1.0, 1.0, 0.0)  # Yellow for selected
            elif bone_name == self.hover_bone:
                glColor3f(0.0, 1.0, 1.0)  # Cyan for hover
            elif bone_name in self.connection_sequence:
                glColor3f(1.0, 0.5, 0.0)  # Orange for connection sequence
            else:
                glColor3f(0.7, 0.7, 0.7)  # Gray for normal

            # Draw sphere
            self.draw_sphere(bone_data['position'], bone_data['radius'])

        # Draw connections
        self.draw_connections()

    def draw_sphere(self, position, radius):
        glPushMatrix()
        glTranslatef(position[0], position[1], position[2])

        # Create sphere using GLU
        quad = gluNewQuadric()
        gluSphere(quad, radius, 16, 16)
        gluDeleteQuadric(quad)

        glPopMatrix()

    def draw_connections(self):
        connections = self.armature_manager.get_connections()

        # if connections:
        #    print(f"Drawing {len(connections)} connections: {connections}")

        glDisable(GL_LIGHTING)
        glLineWidth(5.0)  # Make lines thicker
        # Cyan for connections (visible but not too aggressive)
        glColor3f(0.0, 1.0, 1.0)

        for connection in connections:
            bone1_pos = self.armature_manager.get_bone_position(connection[0])
            bone2_pos = self.armature_manager.get_bone_position(connection[1])

            if bone1_pos and bone2_pos:
                # print(
                #    f"Drawing connection: {connection[0]} ({bone1_pos}) -> {connection[1]} ({bone2_pos})")
                glBegin(GL_LINES)
                glVertex3f(bone1_pos[0], bone1_pos[1], bone1_pos[2])
                glVertex3f(bone2_pos[0], bone2_pos[1], bone2_pos[2])
                glEnd()
            else:
                print(
                    f"Warning: Could not get positions for connection {connection}")

        glEnable(GL_LIGHTING)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.mouse_pressed = True
            self.last_mouse_pos = event.pos()

            # Check for bone selection
            bone_name = self.pick_bone(event.pos())
            if bone_name:
                # Handle shift-click for connecting bones
                if event.modifiers() & Qt.ShiftModifier:
                    self.handle_shift_click(bone_name)
                else:
                    # Normal selection
                    self.selected_bone = bone_name
                    self.bone_selected.emit(bone_name)
                self.update()

        elif event.button() == Qt.RightButton:
            # Create sphere at mouse position
            world_pos = self.screen_to_world(event.pos())
            self.create_sphere_at_position(world_pos)

    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self.delete_selected_sphere()
        else:
            super().keyPressEvent(event)

    def delete_selected_sphere(self):
        """Delete the currently selected sphere"""
        if self.selected_bone:
            # Store the bone name before deletion
            deleted_bone_name = self.selected_bone

            # Remove from connection sequence if present
            if self.selected_bone in self.connection_sequence:
                self.connection_sequence.remove(self.selected_bone)

            # Delete the bone from armature manager
            self.armature_manager.remove_bone(self.selected_bone)

            print(f"Deleted sphere '{deleted_bone_name}'")

            # Clear selection
            self.selected_bone = None
            self.hover_bone = None

            # Emit signals to update UI
            self.bone_selected.emit("")
            self.bone_deleted.emit(deleted_bone_name)

            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.mouse_pressed = False

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.mouse_pressed:
            delta = event.pos() - self.last_mouse_pos

            # Rotate camera
            self.camera_y_rotation += delta.x() * 0.5
            self.camera_x_rotation += delta.y() * 0.5

            # Clamp vertical rotation
            self.camera_x_rotation = max(-90.0,
                                         min(90.0, self.camera_x_rotation))

            self.last_mouse_pos = event.pos()
            self.update()
        else:
            # Check for hover
            bone_name = self.pick_bone(event.pos())
            if bone_name != self.hover_bone:
                self.hover_bone = bone_name
                self.update()

    def wheelEvent(self, event: QWheelEvent):
        # Zoom in/out
        delta = event.angleDelta().y()
        self.camera_distance -= delta * 0.01
        self.camera_distance = max(1.0, min(50.0, self.camera_distance))
        self.update()

    def pick_bone(self, screen_pos):
        """Simple picking - find bone closest to mouse ray using Qt matrices"""
        try:
            # Get ray from mouse position using the same method as screen_to_world
            ray_origin, ray_direction = self.get_ray_from_mouse(
                screen_pos.x(), screen_pos.y())

            # Check intersection with each bone
            bones = self.armature_manager.get_bones()
            closest_bone = None
            closest_distance = float('inf')

            for bone_name, bone_data in bones.items():
                bone_pos = QVector3D(*bone_data['position'])
                bone_radius = bone_data['radius']

                # Simple sphere-ray intersection using Qt vectors
                oc = ray_origin - bone_pos
                a = QVector3D.dotProduct(ray_direction, ray_direction)
                b = 2.0 * QVector3D.dotProduct(oc, ray_direction)
                c = QVector3D.dotProduct(oc, oc) - bone_radius * bone_radius

                discriminant = b * b - 4 * a * c
                if discriminant > 0:
                    t = (-b - math.sqrt(discriminant)) / (2.0 * a)
                    if t > 0 and t < closest_distance:
                        closest_distance = t
                        closest_bone = bone_name

            return closest_bone
        except Exception as e:
            print(f"Error in pick_bone: {e}")
            return None

    def screen_to_world(self, screen_pos):
        """Convert screen position to world position on ground plane using Qt matrices"""
        try:
            # Get ray from mouse position
            ray_origin, ray_direction = self.get_ray_from_mouse(
                screen_pos.x(), screen_pos.y())

            # Intersect with the XZ plane at Y=0
            plane_normal = QVector3D(0, 1, 0)
            plane_point = QVector3D(0, 0, 0)

            denom = QVector3D.dotProduct(plane_normal, ray_direction)
            if abs(denom) > 1e-6:
                t = QVector3D.dotProduct(
                    plane_point - ray_origin, plane_normal) / denom
                if t > 0:  # Only use intersection if it's in front of camera
                    intersection = ray_origin + ray_direction * t

                    # Clamp to reasonable bounds
                    x = max(-10.0, min(10.0, intersection.x()))
                    z = max(-10.0, min(10.0, intersection.z()))

                    result = [x, 0.0, z]
                    print(
                        f"Screen ({screen_pos.x()}, {screen_pos.y()}) -> World {result}")
                    return result

            # Fallback if intersection fails
            print(f"Ray-plane intersection failed, using fallback")
            return [0.0, 0.0, 0.0]

        except Exception as e:
            print(f"Error in screen_to_world: {e}")
            return [0.0, 0.0, 0.0]

    def get_ray_from_mouse(self, x, y):
        """Get ray origin and direction from mouse position using Qt matrices"""
        w, h = self.width(), self.height()
        ndc_x = (2.0 * x) / w - 1.0
        ndc_y = 1.0 - (2.0 * y) / h  # Flip Y axis for OpenGL

        ndc_near = QVector3D(ndc_x, ndc_y, -1.0)
        ndc_far = QVector3D(ndc_x, ndc_y, 1.0)

        inv_matrix = (self.projection * self.view).inverted()[0]

        world_near = inv_matrix.map(ndc_near)
        world_far = inv_matrix.map(ndc_far)

        ray_origin = world_near
        ray_direction = (world_far - world_near).normalized()
        return ray_origin, ray_direction

    def create_sphere(self, radius):
        """Create sphere at origin"""
        sphere_name = f"Sphere_{self.sphere_counter}"
        self.sphere_counter += 1

        self.armature_manager.add_bone(sphere_name, [0.0, 0.0, 0.0], radius)
        self.sphere_created.emit(sphere_name)
        self.update()

    def create_sphere_at_position(self, position):
        """Create sphere at specific position"""
        sphere_name = f"Sphere_{self.sphere_counter}"
        self.sphere_counter += 1

        # Use the radius from the UI if available, otherwise default to 0.5
        radius = 0.5
        if self.main_window:
            radius = self.main_window.get_current_radius()

        self.armature_manager.add_bone(sphere_name, position, radius)
        self.sphere_created.emit(sphere_name)

        # Select the newly created sphere
        self.selected_bone = sphere_name
        self.bone_selected.emit(sphere_name)

        print(
            f"Created sphere '{sphere_name}' at position {position} with radius {radius}")

        self.update()

    def clear_selection(self):
        """Clear current selection"""
        self.selected_bone = None
        self.hover_bone = None
        self.connection_sequence.clear()
        self.update()

    def set_connection_mode(self, mode):
        """Set the connection mode"""
        self.connection_mode = mode
        # Clear connection sequence when switching modes
        self.connection_sequence.clear()
        print(f"Connection mode set to: {mode}")

    def handle_shift_click(self, bone_name):
        """Handle shift-click for connecting bones"""
        if self.connection_mode == "Direct Connect":
            # Direct Connect mode: connect selected bone to clicked bone
            if self.selected_bone and self.selected_bone != bone_name:
                if self.armature_manager.connect_bones(
                        self.selected_bone, bone_name):
                    print(
                        f"Direct Connect: Connected {self.selected_bone} to {bone_name}")
                    # Keep the clicked bone selected for next connection
                    self.selected_bone = bone_name
                    self.bone_selected.emit(bone_name)
                else:
                    print(
                        f"Direct Connect: Failed to connect {self.selected_bone} to {bone_name} (would create loop)")
            else:
                # No bone selected yet, just select this one
                self.selected_bone = bone_name
                self.bone_selected.emit(bone_name)
                print(f"Direct Connect: Selected {bone_name} for connection")
        else:
            # Chain Connect mode: build sequence
            if bone_name not in self.connection_sequence:
                self.connection_sequence.append(bone_name)
                print(
                    f"Added {bone_name} to connection sequence: {self.connection_sequence}")

                # If we have at least 2 bones, create connections
                if len(self.connection_sequence) >= 2:
                    for i in range(len(self.connection_sequence) - 1):
                        bone1 = self.connection_sequence[i]
                        bone2 = self.connection_sequence[i + 1]
                        if self.armature_manager.connect_bones(bone1, bone2):
                            print(
                                f"Chain Connect: Connected {bone1} to {bone2}")
                        else:
                            print(
                                f"Chain Connect: Failed to connect {bone1} to {bone2} (would create loop)")

                    # Keep the last bone selected
                    self.selected_bone = bone_name
                    self.bone_selected.emit(bone_name)
            else:
                print(f"{bone_name} is already in the connection sequence")
