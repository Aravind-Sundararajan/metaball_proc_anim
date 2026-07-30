# Mesh XML Editor

A PySide6 application for creating 3D armatures with spheres and exporting them to XML format.

## Features

- **3D OpenGL View**: Interactive 3D viewport with camera controls
- **Sphere Creation**: Create spheres by clicking or using the UI controls
- **Bone Connections**: Connect spheres to create armature structures
- **XML Export**: Export armatures to XML format compatible with your existing structure
- **Real-time Rendering**: Smooth 60 FPS rendering with lighting and materials

## Installation

1. Make sure you have Python 3.10+ installed
2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

## Usage

### Running the Application

```bash
poetry run python main.py
```

### Controls

#### 3D Viewport
- **Left Mouse Button + Drag**: Rotate camera around the scene
- **Mouse Wheel**: Zoom in/out
- **Left Click on Sphere**: Select a bone/sphere
- **Right Click**: Create a new sphere at the clicked position

#### UI Controls
- **Radius SpinBox**: Set the radius for new spheres
- **Create Sphere Button**: Create a sphere at the origin
- **Bone List**: View and select all created bones
- **Connect Selected Bones**: Connect two selected bones
- **Clear Selection**: Clear current selection
- **Export to XML**: Save the armature to an XML file

### Creating an Armature

1. **Create Spheres**: Use the "Create Sphere" button or right-click in the 3D view
2. **Select Bones**: Click on spheres in the 3D view or select them from the bone list
3. **Connect Bones**: Select two bones and click "Connect Selected Bones"
4. **Export**: Click "Export to XML" to save your armature

### XML Format

The application exports XML files in the following format:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<armature>
    <bone name="Root" position="0.000,0.000,0.000" rotation="0.000,0.000,0.000" scale="1.000,1.000,1.000">
        <mesh type="sphere" radius="0.500" material="body" />
        <bone name="Child" position="1.000,0.000,0.000" rotation="0.000,0.000,0.000" scale="1.000,1.000,1.000">
            <mesh type="sphere" radius="0.300" material="body" />
        </bone>
    </bone>
</armature>
```

## Project Structure

- `main.py`: Main application window and UI setup
- `opengl_widget.py`: OpenGL rendering widget with 3D interaction
- `armature_manager.py`: Bone data management and XML import/export
- `character_armature.xml`: Example armature file

## Dependencies

- **PySide6**: Qt bindings for Python
- **PyOpenGL**: OpenGL bindings for Python
- **NumPy**: Numerical computing library

## Development

The application is built with:
- **PySide6** for the user interface
- **OpenGL** for 3D rendering
- **XML** for data export/import

### Key Components

1. **MeshXMLEditor**: Main application window with UI controls
2. **OpenGLWidget**: 3D rendering widget with mouse interaction
3. **ArmatureManager**: Data management for bones and connections

## License

This project is open source and available under the MIT License. 