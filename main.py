#!/usr/bin/env python3
"""
Mesh XML Editor - A PySide6 application for creating 3D armatures
"""

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox, QTableView, QSplitter, QComboBox, QFileDialog, QMessageBox, QStyledItemDelegate, QListWidget, QTabWidget, QScrollArea
from PySide6.QtCore import Qt, QTimer, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QAction, QKeySequence
from opengl_widget import OpenGLWidget
from armature_manager import ArmatureManager
from spline_generator_simple import SplineGeneratorSimple as SplineGenerator


class BoneTableModel(QAbstractTableModel):
    """Table model for displaying and editing bone names"""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.bones = []
        self.update_bones()

    def update_bones(self):
        """Update the bone list from armature manager"""
        self.beginResetModel()
        self.bones = list(self.main_window.armature_manager.get_bones().keys())
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self.bones)

    def columnCount(self, parent=QModelIndex()):
        return 1  # Only one column for bone names

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        if role == Qt.DisplayRole or role == Qt.EditRole:
            return self.bones[index.row()]

        return None

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid() or role != Qt.EditRole:
            return False

        new_name = value.strip()
        old_name = self.bones[index.row()]

        if new_name == old_name:
            return True

        # Check if new name already exists
        if new_name in self.main_window.armature_manager.get_bones():
            QMessageBox.warning(
                self.main_window,
                "Rename Failed",
                f"Bone name '{new_name}' already exists. Please choose a different name."
            )
            return False

        # Rename the bone
        if self.main_window.armature_manager.rename_bone(old_name, new_name):
            self.bones[index.row()] = new_name

            # Update selected sphere reference if it was the renamed bone
            if self.main_window.selected_sphere == old_name:
                self.main_window.selected_sphere = new_name

            # Update OpenGL widget
            self.main_window.opengl_widget.update()

            print(f"Renamed bone from '{old_name}' to '{new_name}'")
            return True
        else:
            QMessageBox.warning(
                self.main_window,
                "Rename Failed",
                f"Failed to rename bone '{old_name}' to '{new_name}'"
            )
            return False

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable


class BoneNameDelegate(QStyledItemDelegate):
    """Custom delegate for editing bone names"""

    def createEditor(self, parent, option, index):
        """Create editor widget for inline editing"""
        from PySide6.QtWidgets import QLineEdit
        editor = QLineEdit(parent)
        editor.setFrame(False)
        return editor

    def setEditorData(self, editor, index):
        """Set the current bone name in the editor"""
        value = index.data()
        editor.setText(value)

    def setModelData(self, editor, model, index):
        """Handle the rename when editing is finished"""
        value = editor.text().strip()
        model.setData(index, value, Qt.EditRole)


class MeshXMLEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mesh XML Editor")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize armature manager
        self.armature_manager = ArmatureManager()

        # Track currently selected sphere
        self.selected_sphere = None
        self.updating_position = False  # Flag to prevent recursive updates
        self.updating_radius = False  # Flag to prevent recursive radius updates

        # Setup UI
        self.setup_ui()
        self.setup_menu()

        # Connect signals
        self.connect_signals()

        # Timer for continuous rendering
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.opengl_widget.update)
        self.render_timer.start(16)  # ~60 FPS

    def setup_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(5)  # Reduce spacing

        # Create scroll area for left panel
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Remove maximum width constraint to allow natural expansion

        # Sphere creation controls
        sphere_group = QGroupBox("Create Sphere")
        sphere_layout = QVBoxLayout(sphere_group)

        # Sphere radius
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Radius:"))
        self.radius_spinbox = QDoubleSpinBox()
        self.radius_spinbox.setRange(0.1, 5.0)
        self.radius_spinbox.setValue(0.5)
        self.radius_spinbox.setSingleStep(0.1)
        self.radius_spinbox.setDecimals(2)
        radius_layout.addWidget(self.radius_spinbox)
        sphere_layout.addLayout(radius_layout)

        # Create sphere button
        self.create_sphere_btn = QPushButton("Create Sphere")
        sphere_layout.addWidget(self.create_sphere_btn)

        left_layout.addWidget(sphere_group)

        # Position editing controls
        position_group = QGroupBox("Edit Position")
        position_layout = QVBoxLayout(position_group)

        # X coordinate
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X:"))
        self.x_spinbox = QDoubleSpinBox()
        self.x_spinbox.setRange(-20.0, 20.0)
        self.x_spinbox.setValue(0.0)
        self.x_spinbox.setSingleStep(0.1)
        self.x_spinbox.setDecimals(3)
        x_layout.addWidget(self.x_spinbox)
        position_layout.addLayout(x_layout)

        # Y coordinate
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.y_spinbox = QDoubleSpinBox()
        self.y_spinbox.setRange(-20.0, 20.0)
        self.y_spinbox.setValue(0.0)
        self.y_spinbox.setSingleStep(0.1)
        self.y_spinbox.setDecimals(3)
        y_layout.addWidget(self.y_spinbox)
        position_layout.addLayout(y_layout)

        # Z coordinate
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z:"))
        self.z_spinbox = QDoubleSpinBox()
        self.z_spinbox.setRange(-20.0, 20.0)
        self.z_spinbox.setValue(0.0)
        self.z_spinbox.setSingleStep(0.1)
        self.z_spinbox.setDecimals(3)
        z_layout.addWidget(self.z_spinbox)
        position_layout.addLayout(z_layout)

        left_layout.addWidget(position_group)

        # Armature controls
        armature_group = QGroupBox("Armature")
        armature_layout = QVBoxLayout(armature_group)

        # Connection mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Connection Mode:"))
        self.connection_mode_combo = QComboBox()
        self.connection_mode_combo.addItems(
            ["Direct Connect", "Chain Connect"])
        self.connection_mode_combo.currentTextChanged.connect(
            self.on_connection_mode_changed)
        mode_layout.addWidget(self.connection_mode_combo)
        armature_layout.addLayout(mode_layout)

        # Mode description
        self.mode_description = QLabel(
            "Direct Connect: Click bone, then Shift+Click another to connect")
        self.mode_description.setWordWrap(True)
        armature_layout.addWidget(self.mode_description)

        # Bone table
        self.bone_table = QTableView()
        self.bone_table.setSelectionBehavior(
            QTableView.SelectionBehavior.SelectRows)
        self.bone_table.setSelectionMode(
            QTableView.SelectionMode.MultiSelection)

        # Set up model and delegate
        self.bone_model = BoneTableModel(self)
        self.bone_delegate = BoneNameDelegate()
        self.bone_table.setModel(self.bone_model)
        self.bone_table.setItemDelegate(self.bone_delegate)

        # Configure table appearance
        self.bone_table.horizontalHeader().setStretchLastSection(True)
        self.bone_table.horizontalHeader().setVisible(False)
        self.bone_table.verticalHeader().setVisible(False)
        self.bone_table.setShowGrid(False)

        armature_layout.addWidget(QLabel("Bones:"))
        armature_layout.addWidget(self.bone_table)

        # Connect bones button
        self.connect_bones_btn = QPushButton("Connect Selected Bones")
        armature_layout.addWidget(self.connect_bones_btn)

        # Disconnect bones button
        self.disconnect_bones_btn = QPushButton("Disconnect Selected Bones")
        armature_layout.addWidget(self.disconnect_bones_btn)

        # Clear all connections button
        self.clear_connections_btn = QPushButton("Clear All Connections")
        armature_layout.addWidget(self.clear_connections_btn)

        # Clear selection button
        self.clear_selection_btn = QPushButton("Clear Selection")
        armature_layout.addWidget(self.clear_selection_btn)

        # Delete selected bone button
        self.delete_bone_btn = QPushButton("Delete Selected Bone")
        armature_layout.addWidget(self.delete_bone_btn)

        left_layout.addWidget(armature_group)

        # Spline Generator
        spline_group = QGroupBox("Spline Generator")
        spline_layout = QVBoxLayout(spline_group)

        self.spline_generator = SplineGenerator()
        # Remove maximum height constraint to allow natural expansion
        self.spline_generator.spline_generated.connect(
            self.on_spline_generated)
        spline_layout.addWidget(self.spline_generator)

        left_layout.addWidget(spline_group)

        # Export controls
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)

        self.export_btn = QPushButton("Export to XML")
        export_layout.addWidget(self.export_btn)

        left_layout.addWidget(export_group)

        # Right panel - OpenGL view
        self.opengl_widget = OpenGLWidget(self.armature_manager, self)
        self.opengl_widget.setFocusPolicy(
            Qt.StrongFocus)  # Allow focus for key events

        # Add panels to splitter
        splitter.addWidget(left_scroll)
        splitter.addWidget(self.opengl_widget)
        splitter.setSizes([300, 900])  # Initial sizes

    def setup_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)

        open_action = QAction("&Open", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)

        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        undo_action = QAction("&Undo", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("&Redo", self)
        redo_action.setShortcut(QKeySequence.Redo)
        redo_action.triggered.connect(self.redo)
        edit_menu.addAction(redo_action)

    def connect_signals(self):
        self.create_sphere_btn.clicked.connect(self.create_sphere)
        self.connect_bones_btn.clicked.connect(self.connect_bones)
        self.disconnect_bones_btn.clicked.connect(self.disconnect_bones)
        self.clear_connections_btn.clicked.connect(self.clear_all_connections)
        self.clear_selection_btn.clicked.connect(self.clear_selection)
        self.delete_bone_btn.clicked.connect(self.delete_selected_bone)
        self.export_btn.clicked.connect(self.export_xml)

        # Connect position spinbox signals
        self.x_spinbox.valueChanged.connect(self.on_position_changed)
        self.y_spinbox.valueChanged.connect(self.on_position_changed)
        self.z_spinbox.valueChanged.connect(self.on_position_changed)

        # Connect radius spinbox signal
        self.radius_spinbox.valueChanged.connect(self.on_radius_changed)

        # Connect OpenGL widget signals
        self.opengl_widget.sphere_created.connect(self.on_sphere_created)
        self.opengl_widget.bone_selected.connect(self.on_bone_selected)
        self.opengl_widget.bone_deleted.connect(self.on_bone_deleted)

        # Connect table signals
        self.bone_table.selectionModel().selectionChanged.connect(
            self.on_table_selection_changed)

        # Set initial connection mode
        self.opengl_widget.set_connection_mode("Direct Connect")

    def create_sphere(self):
        radius = self.radius_spinbox.value()
        self.opengl_widget.create_sphere(radius)

    def create_bone_at_position(self, x, y, z, radius, bone_name):
        """Create a bone at a specific position with given radius and name"""
        # Add bone to armature manager
        self.armature_manager.add_bone(bone_name, [x, y, z], radius)

        # Update the bone table model
        self.bone_model.update_bones()

        # Update OpenGL widget
        self.opengl_widget.update()

        print(
            f"Created bone '{bone_name}' at position [{x}, {y}, {z}] with radius {radius}")

    def get_current_radius(self):
        """Get the current radius from the UI"""
        return self.radius_spinbox.value()

    def connect_bones(self, bone1_name=None, bone2_name=None):
        """Connect two specific bones or selected bones from table"""
        if bone1_name and bone2_name:
            # Connect specific bones (used by spline generator)
            if self.armature_manager.connect_bones(bone1_name, bone2_name):
                self.opengl_widget.update()
                print(f"Connected {bone1_name} to {bone2_name}")
            else:
                print(
                    f"Failed to connect {bone1_name} to {bone2_name} (would create loop)")
        else:
            # Connect selected bones from table
            selected_rows = self.bone_table.selectionModel().selectedRows()
            if len(selected_rows) >= 2:
                # Connect all selected bones in pairs
                connected_count = 0
                for i in range(len(selected_rows) - 1):
                    bone1 = self.bone_model.data(selected_rows[i])
                    bone2 = self.bone_model.data(selected_rows[i + 1])
                    if self.armature_manager.connect_bones(bone1, bone2):
                        connected_count += 1
                    else:
                        print(
                            f"Failed to connect {bone1} to {bone2} (would create loop)")
                self.opengl_widget.update()
                print(
                    f"Connected {connected_count} out of {len(selected_rows)-1} bone pairs")
            else:
                print("Please select at least 2 bones to connect")

    def disconnect_bones(self):
        selected_rows = self.bone_table.selectionModel().selectedRows()
        if len(selected_rows) >= 2:
            # Disconnect all selected bones in pairs
            for i in range(len(selected_rows) - 1):
                bone1_name = self.bone_model.data(selected_rows[i])
                bone2_name = self.bone_model.data(selected_rows[i + 1])
                self.armature_manager.disconnect_bones(bone1_name, bone2_name)
            self.opengl_widget.update()
            print(f"Disconnected {len(selected_rows)} bones in sequence")
        else:
            print("Please select at least 2 bones to disconnect")

    def clear_all_connections(self):
        """Clear all bone connections"""
        self.armature_manager.clear_connections()
        self.opengl_widget.update()
        print("Cleared all bone connections")

    def delete_selected_bone(self):
        """Delete the currently selected bone"""
        if self.selected_sphere:
            # Store the bone name for printing
            bone_name = self.selected_sphere

            # Delete from armature manager
            self.armature_manager.remove_bone(self.selected_sphere)

            # Update the bone table model
            self.bone_model.update_bones()

            # Clear selection
            self.selected_sphere = None
            self.update_position_fields()

            # Update OpenGL widget
            self.opengl_widget.update()

            print(f"Deleted bone '{bone_name}'")
        else:
            print("No bone selected to delete")

    def clear_selection(self):
        self.bone_table.clearSelection()
        self.opengl_widget.clear_selection()
        self.selected_sphere = None
        self.update_position_fields()

    def on_sphere_created(self, sphere_name):
        self.bone_model.update_bones()

    def on_bone_deleted(self, bone_name):
        """Handle when a bone is deleted from OpenGL widget"""
        # Update the bone table model
        self.bone_model.update_bones()

        # Clear selection if the deleted bone was selected
        if self.selected_sphere == bone_name:
            self.selected_sphere = None
            self.update_position_fields()

    def on_spline_generated(self, spline_data):
        """Handle spline generation"""
        # Create spheres from spline data
        for i, (position, radius) in enumerate(spline_data):
            bone_name = f"Spline_{i+1}"
            self.armature_manager.add_bone(bone_name, position, radius)

        # Update the bone table model
        self.bone_model.update_bones()

        # Update OpenGL widget
        self.opengl_widget.update()

        # Connect spheres in sequence
        for i in range(len(spline_data) - 1):
            self.armature_manager.connect_bones(
                f"Spline_{i+1}", f"Spline_{i+2}")

        # Update OpenGL widget again after connections
        self.opengl_widget.update()

        print(f"Generated spline with {len(spline_data)} spheres")

    def on_bone_selected(self, bone_name):
        # Find and select the bone in the table
        for row in range(self.bone_model.rowCount()):
            if self.bone_model.data(self.bone_model.index(row, 0)) == bone_name:
                # Select the row
                self.bone_table.selectRow(row)
                # Scroll to make the item visible
                self.bone_table.scrollTo(self.bone_model.index(row, 0))
                break

        # Update position fields with selected sphere's position
        self.selected_sphere = bone_name
        self.update_position_fields()

    def on_table_selection_changed(self):
        """Handle when selection changes in the bone table"""
        selected_rows = self.bone_table.selectionModel().selectedRows()
        if selected_rows:
            # Update the selected sphere based on table selection
            self.selected_sphere = self.bone_model.data(selected_rows[0])
            self.update_position_fields()
        else:
            # No selection in table
            self.selected_sphere = None
            self.update_position_fields()

    def update_position_fields(self):
        """Update the position and radius spinboxes with the selected sphere's data"""
        if self.selected_sphere:
            position = self.armature_manager.get_bone_position(
                self.selected_sphere)
            radius = self.armature_manager.get_bone_radius(
                self.selected_sphere)

            if position:
                # Set flag to prevent recursive updates
                self.updating_position = True
                self.x_spinbox.setValue(position[0])
                self.y_spinbox.setValue(position[1])
                self.z_spinbox.setValue(position[2])
                self.updating_position = False

                # Enable position editing
                self.x_spinbox.setEnabled(True)
                self.y_spinbox.setEnabled(True)
                self.z_spinbox.setEnabled(True)

            if radius is not None:
                # Set flag to prevent recursive updates
                self.updating_radius = True
                self.radius_spinbox.setValue(radius)
                self.updating_radius = False

                # Enable radius editing
                self.radius_spinbox.setEnabled(True)

        else:
            # Disable editing when no sphere is selected
            self.x_spinbox.setEnabled(False)
            self.y_spinbox.setEnabled(False)
            self.z_spinbox.setEnabled(False)
            self.radius_spinbox.setEnabled(False)

    def on_position_changed(self):
        """Handle position spinbox value changes"""
        if self.selected_sphere and not self.updating_position:
            x = self.x_spinbox.value()
            y = self.y_spinbox.value()
            z = self.z_spinbox.value()

            self.armature_manager.update_bone_position(
                self.selected_sphere, [x, y, z])
            self.opengl_widget.update()
            print(
                f"Updated {self.selected_sphere} position to [{x}, {y}, {z}]")

    def on_connection_mode_changed(self, mode):
        """Handle connection mode changes"""
        self.opengl_widget.set_connection_mode(mode)

        # Update description
        if mode == "Direct Connect":
            self.mode_description.setText(
                "Direct Connect: Click bone, then Shift+Click another to connect")
        else:  # Chain Connect
            self.mode_description.setText(
                "Chain Connect: Shift+Click bones in sequence to create a chain")

    def on_radius_changed(self):
        """Handle radius spinbox value changes"""
        if self.selected_sphere and not self.updating_radius:
            radius = self.radius_spinbox.value()
            self.armature_manager.update_bone_radius(
                self.selected_sphere, radius)
            self.opengl_widget.update()
            print(f"Updated {self.selected_sphere} radius to {radius}")

    def export_xml(self):
        """Export armature to XML file with file dialog"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Armature to XML",
            "armature.xml",
            "XML Files (*.xml);;All Files (*)"
        )

        if filename:
            try:
                self.armature_manager.export_xml(filename)
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Armature exported successfully to:\n{filename}"
                )
                print(f"Exported armature to: {filename}")
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to export armature:\n{str(e)}"
                )
                print(f"Export failed: {e}")
        else:
            print("Export cancelled by user")

    def new_project(self):
        """Create a new project with confirmation"""
        if self.armature_manager.get_bones():
            # Ask for confirmation if there's existing data
            reply = QMessageBox.question(
                self,
                "New Project",
                "This will clear all current data. Are you sure?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                print("New project cancelled by user")
                return

        # Clear current project
        self.armature_manager.clear()
        self.bone_model.update_bones()
        self.selected_sphere = None
        self.update_position_fields()
        self.opengl_widget.update()
        print("New project created")

    def open_project(self):
        """Open project from XML file with file dialog"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "XML Files (*.xml);;All Files (*)"
        )

        if filename:
            try:
                # Clear current project
                self.armature_manager.clear()

                # Import from XML
                if self.armature_manager.import_xml(filename):
                    # Update bone table
                    self.bone_model.update_bones()

                    # Update OpenGL widget
                    self.opengl_widget.update()

                    QMessageBox.information(
                        self,
                        "Open Successful",
                        f"Project opened successfully from:\n{filename}"
                    )
                    print(f"Project opened from: {filename}")
                else:
                    QMessageBox.critical(
                        self,
                        "Open Failed",
                        f"Failed to open project from:\n{filename}"
                    )
                    print(f"Failed to open project from: {filename}")
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Open Failed",
                    f"Failed to open project:\n{str(e)}"
                )
                print(f"Open failed: {e}")
        else:
            print("Open cancelled by user")

    def save_project(self):
        """Save project to XML file with file dialog"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            "project.xml",
            "XML Files (*.xml);;All Files (*)"
        )

        if filename:
            try:
                self.armature_manager.export_xml(filename)
                QMessageBox.information(
                    self,
                    "Save Successful",
                    f"Project saved successfully to:\n{filename}"
                )
                print(f"Project saved to: {filename}")
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Save Failed",
                    f"Failed to save project:\n{str(e)}"
                )
                print(f"Save failed: {e}")
        else:
            print("Save cancelled by user")

    def undo(self):
        # TODO: Implement undo
        pass

    def redo(self):
        # TODO: Implement redo
        pass


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Mesh XML Editor")
    app.setApplicationVersion("0.1.0")

    window = MeshXMLEditor()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
