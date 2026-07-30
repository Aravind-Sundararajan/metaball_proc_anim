#!/usr/bin/env python3
"""
OpenGL Widget for 3D rendering and interaction - Simplified version
"""

import numpy as np
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QMouseEvent, QWheelEvent
from OpenGL.GL import *
from OpenGL.GLU import *
import math


class OpenGLWidget(QOpenGLWidget):
    sphere_created = Signal(str)
    bone_selected = Signal(str)

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

    def paintGL(self):
        try:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            # Set up camera
            glTranslatef(0.0, 0.0, -self.camera_distance)
            glRotatef(self.camera_x_rotation, 1.0, 0.0, 0.0)
            glRotatef(self.camera_y_rotation, 0.0, 1.0, 0.0)

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

        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        glColor3f(0.5, 0.5, 1.0)  # Blue for connections

        for connection in connections:
            bone1_pos = self.armature_manager.get_bone_position(connection[0])
            bone2_pos = self.armature_manager.get_bone_position(connection[1])

            if bone1_pos and bone2_pos:
                glBegin(GL_LINES)
                glVertex3f(bone1_pos[0], bone1_pos[1], bone1_pos[2])
                glVertex3f(bone2_pos[0], bone2_pos[1], bone2_pos[2])
                glEnd()

        glEnable(GL_LIGHTING)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.mouse_pressed = True
            self.last_mouse_pos = event.pos()

            # Check for bone selection
            bone_name = self.pick_bone(event.pos())
            if bone_name:
                self.selected_bone = bone_name
                self.bone_selected.emit(bone_name)
                self.update()

        elif event.button() == Qt.RightButton:
            # Create sphere at mouse position
            world_pos = self.screen_to_world(event.pos())
            self.create_sphere_at_position(world_pos)

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
        """Simple picking - find bone closest to mouse ray"""
        # Convert screen position to world ray
        viewport = glGetIntegerv(GL_VIEWPORT)
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)

        # Get near and far points
        near = gluUnProject(screen_pos.x(), viewport[3] - screen_pos.y(), 0.0,
                            modelview, projection, viewport)
        far = gluUnProject(screen_pos.x(), viewport[3] - screen_pos.y(), 1.0,
                           modelview, projection, viewport)

        if near and far:
            ray_direction = np.array(far) - np.array(near)
            ray_direction = ray_direction / np.linalg.norm(ray_direction)

            # Check intersection with each bone
            bones = self.armature_manager.get_bones()
            closest_bone = None
            closest_distance = float('inf')

            for bone_name, bone_data in bones.items():
                bone_pos = np.array(bone_data['position'])
                bone_radius = bone_data['radius']

                # Simple sphere-ray intersection
                oc = np.array(near) - bone_pos
                a = np.dot(ray_direction, ray_direction)
                b = 2.0 * np.dot(oc, ray_direction)
                c = np.dot(oc, oc) - bone_radius * bone_radius

                discriminant = b * b - 4 * a * c
                if discriminant > 0:
                    t = (-b - math.sqrt(discriminant)) / (2.0 * a)
                    if t > 0 and t < closest_distance:
                        closest_distance = t
                        closest_bone = bone_name

            return closest_bone
        return None

    def screen_to_world(self, screen_pos):
        """Convert screen position to world position on ground plane"""
        try:
            viewport = glGetIntegerv(GL_VIEWPORT)

            # Calculate normalized device coordinates (-1 to 1)
            ndc_x = (2.0 * screen_pos.x()) / viewport[2] - 1.0
            ndc_y = 1.0 - (2.0 * screen_pos.y()) / viewport[3]  # Invert Y

            # Calculate the view frustum size at the ground plane
            # Based on camera distance and field of view
            fov_rad = math.radians(45.0)  # Same as in gluPerspective
            half_height = self.camera_distance * math.tan(fov_rad / 2.0)
            aspect_ratio = viewport[2] / viewport[3]
            half_width = half_height * aspect_ratio

            # Map NDC to world coordinates at ground plane
            world_x = ndc_x * half_width
            world_z = ndc_y * half_height

            # Apply camera rotation transformation
            angle_rad = math.radians(self.camera_y_rotation)
            cos_angle = math.cos(angle_rad)
            sin_angle = math.sin(angle_rad)

            # Rotate the coordinates based on camera Y rotation
            rotated_x = world_x * cos_angle - world_z * sin_angle
            rotated_z = world_x * sin_angle + world_z * cos_angle

            # Clamp to reasonable bounds
            final_x = max(-10.0, min(10.0, rotated_x))
            final_z = max(-10.0, min(10.0, rotated_z))

            result = [final_x, 0.0, final_z]
            print(
                f"Screen ({screen_pos.x()}, {screen_pos.y()}) -> NDC ({ndc_x:.3f}, {ndc_y:.3f}) -> World {result}")
            return result

        except Exception as e:
            print(f"Error in screen_to_world: {e}")
            return [0.0, 0.0, 0.0]

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
        self.update()
