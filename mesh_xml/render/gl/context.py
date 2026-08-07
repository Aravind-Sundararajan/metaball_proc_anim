"""The core-profile widget the raymarcher draws into.

.. warning::

   **QSurfaceFormat must be configured before QApplication is constructed.**

   .. code-block:: python

      from mesh_xml.render.gl.context import configure_surface_format

      configure_surface_format()      # first
      app = QApplication(sys.argv)    # second
      window = MainWindow()           # only now

   Get that order wrong and nothing raises. Qt hands out a compatibility
   context instead, the widget comes up, and the only symptom is that every
   ``#version 330 core`` shader fails to compile with a driver message about
   syntax — because on a compatibility profile ``in``/``out`` at global scope
   and the absence of ``gl_FragColor`` mean something else. Hours have been
   lost to that error message. :func:`ensure_application` exists so the
   ordering cannot be got wrong by accident.

This widget is the floor the renderer stands on and nothing more: a core-profile
surface, a vertex array object, one triangle covering the viewport, and a camera
wired to the mouse. The fragment shader supplied by the caller is the actual
renderer.

It draws and it does not mutate. No picking, no selection, no hover, no scene
editing — those need a document and a command stack, and a viewport that reaches
into the model directly is how a repaint ends up being the thing that changes it.
Phase 5 embeds this; it does not extend it.
"""

from __future__ import annotations

import math
import warnings
from typing import Final

from PySide6.QtCore import QElapsedTimer, QPoint, Qt, Signal
from PySide6.QtGui import QMouseEvent, QSurfaceFormat, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication

from mesh_xml.render.gl.camera import OrbitCamera, viewport_aspect
from mesh_xml.render.gl.program import (
    ShaderError,
    ShaderProgram,
    assemble,
    file_resolver,
)

#: Texture unit the coarse prepass's seed depths are bound to.
SEED_TEXTURE_UNIT = 5

#: Safety factor on the prepass cone radius, over the block half-diagonal it
#: must contain. Covers the chord-versus-parameter second-order mismatch
#: between rays of one bundle and plain float slack; oversizing only makes
#: the coarse pass stop earlier, which costs fine-pass steps, never
#: correctness.
_CONE_MARGIN = 1.2

__all__ = [
    "GL_MAJOR",
    "GL_MINOR",
    "DEFAULT_FRAGMENT_SHADER",
    "FullscreenTriangle",
    "RaymarchView",
    "VERTEX_SHADER",
    "configure_surface_format",
    "ensure_application",
    "gl_available",
]

GL_MAJOR: Final = 3
GL_MINOR: Final = 3

VERTEX_SHADER: Final = "fullscreen.vert"
DEFAULT_FRAGMENT_SHADER: Final = "rays.frag"

_available: bool | None = None


def configure_surface_format(
    samples: int = 0,
    depth_bits: int = 24,
    stencil_bits: int = 0,
) -> QSurfaceFormat:
    """Install a core-profile 3.3 default surface format process-wide.

    Must run **before** ``QApplication`` is constructed; see the module
    warning. Calling it afterwards emits a :class:`RuntimeWarning` rather than
    raising, because on some platforms a late call still takes effect for
    widgets created later and turning a maybe-fine situation into a crash is
    worse than saying so.

    ``samples`` stays at 0 by default. Multisampling costs a full extra shading
    pass per sample on a fullscreen fragment shader and buys nothing: there is
    no geometry to alias, and the surface edges come out of the SDF, which
    antialiases in the shader far more cheaply.
    """
    if QApplication.instance() is not None:
        warnings.warn(
            "configure_surface_format() called after QApplication was constructed; "
            "the core profile may not take effect and core-profile shaders will "
            "fail to compile. Call it before constructing QApplication.",
            RuntimeWarning,
            stacklevel=2,
        )

    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setVersion(GL_MAJOR, GL_MINOR)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(depth_bits)
    fmt.setStencilBufferSize(stencil_bits)
    fmt.setSamples(samples)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    QSurfaceFormat.setDefaultFormat(fmt)
    return fmt


def ensure_application(argv: list[str] | None = None) -> QApplication:
    """Return the QApplication, creating it *after* the surface format if absent.

    The single call site that cannot get the ordering wrong. Application entry
    points and tests should both go through it.
    """
    app = QApplication.instance()
    if app is None:
        configure_surface_format()
        app = QApplication(argv if argv is not None else [])
    return app  # type: ignore[return-value]


def gl_available() -> bool:
    """Whether this process can actually obtain a core-profile 3.3 context.

    Probes with an offscreen surface, so it neither needs nor flashes a window.
    Headless CI, a remote session without a GL driver, and software rasterisers
    that top out below 3.3 all answer ``False`` here, which is the hook the GL
    tests skip on. Cached: the answer cannot change within a process, and each
    probe costs a context creation.

    Constructs a QApplication if none exists, which is a real side effect but an
    unavoidable one — Qt cannot make a surface without one — and it constructs
    it through :func:`ensure_application`, so the probe also demonstrates the
    required ordering.
    """
    global _available
    if _available is not None:
        return _available

    _available = False
    try:
        from PySide6.QtGui import QOffscreenSurface, QOpenGLContext

        ensure_application()
        fmt = QSurfaceFormat.defaultFormat()

        surface = QOffscreenSurface()
        surface.setFormat(fmt)
        surface.create()
        if not surface.isValid():
            return _available

        context = QOpenGLContext()
        context.setFormat(fmt)
        if not context.create() or not context.makeCurrent(surface):
            return _available

        actual = context.format()
        version = (actual.majorVersion(), actual.minorVersion())
        core = actual.profile() == QSurfaceFormat.OpenGLContextProfile.CoreProfile
        context.doneCurrent()
        _available = core and version >= (GL_MAJOR, GL_MINOR)
    except Exception:  # noqa: BLE001 - any failure here means "no usable GL"
        _available = False
    return _available


class FullscreenTriangle:
    """The vertex array object and the three-vertex draw.

    Holds no buffer. The vertices come from ``gl_VertexID`` in
    ``shaders/fullscreen.vert``; the VAO exists purely because core profile
    removed the default one and refuses to draw without a bound object. A
    missing bind is an ``INVALID_OPERATION`` that draws nothing and, unless the
    error queue is polled, says nothing — a blank viewport with no diagnostic.
    """

    def __init__(self) -> None:
        self.id = 0

    def create(self) -> None:
        from OpenGL.GL import glGenVertexArrays

        self.id = int(glGenVertexArrays(1))

    def draw(self) -> None:
        from OpenGL.GL import GL_TRIANGLES, glBindVertexArray, glDrawArrays

        glBindVertexArray(self.id)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)

    def dispose(self) -> None:
        if not self.id:
            return
        from OpenGL.GL import glDeleteVertexArrays

        glDeleteVertexArrays(1, [self.id])
        self.id = 0


class RaymarchView(QOpenGLWidget):
    """A core-profile viewport that runs a caller-supplied fragment shader.

    The fragment shader is the entire renderer; this class only supplies the
    surface, the triangle, and the camera uniforms declared in
    ``shaders/camera.glsl``:

    ==========================  ====================================
    ``u_resolution``            framebuffer size in device pixels
    ``u_camera_position``       world-space eye
    ``u_camera_right/up/forward``  orthonormal view basis
    ``u_tan_half_fov``          half-height of the near plane / depth
    ``u_aspect``                width over height, 1.0 if degenerate
    ``u_time``                  seconds since the context came up
    ==========================  ====================================

    Anything else the shader needs goes in :attr:`uniforms`, which is applied
    after the built-ins each frame. That is the hook the primitive buffer from
    :mod:`mesh_xml.render.sdf` plugs into.
    """

    shader_compiled = Signal()
    shader_failed = Signal(str)
    camera_changed = Signal()

    def __init__(
        self,
        fragment_source: str | None = None,
        camera: OrbitCamera | None = None,
        parent: object = None,
    ) -> None:
        super().__init__(parent)
        self.camera = camera if camera is not None else OrbitCamera()
        self.uniforms: dict[str, object] = {}

        self._fragment_source = fragment_source
        self._fragment_name = "<caller>" if fragment_source else DEFAULT_FRAGMENT_SHADER
        self._program: ShaderProgram | None = None
        self._triangle = FullscreenTriangle()
        self._dirty = True
        self._framebuffer = (max(self.width(), 1), max(self.height(), 1))
        #: Coarse cone-prepass block size; 0 disables. Off by default so the
        #: CPU-oracle comparisons render the single-pass path bit-for-bit;
        #: the editor viewport turns it on.
        self.prepass_divisor: int = 0
        self._seed_fbo = 0
        self._seed_texture = 0
        self._seed_size = (0, 0)
        self._clock = QElapsedTimer()
        self._drag_button: Qt.MouseButton = Qt.MouseButton.NoButton
        self._last_pos = QPoint()

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    @property
    def fragment_source(self) -> str | None:
        """The caller's GLSL, or ``None`` while the placeholder shader is in use."""
        return self._fragment_source

    def set_fragment_source(self, source: str | None, name: str = "<caller>") -> None:
        """Swap the fragment shader. Compiles on the next paint.

        Deferred rather than immediate because compiling needs the context
        current, and the caller assembling a shader has no reason to know that.
        """
        self._fragment_source = source
        self._fragment_name = name if source is not None else DEFAULT_FRAGMENT_SHADER
        self._dirty = True
        self.update()

    def initializeGL(self) -> None:
        from OpenGL.GL import glClearColor, glDisable, GL_DEPTH_TEST

        self._clock.start()
        glClearColor(0.08, 0.08, 0.09, 1.0)

        # A fullscreen pass writes every pixel exactly once, in an order it
        # controls. Depth testing against a buffer nothing else writes can only
        # reject fragments that should have survived.
        glDisable(GL_DEPTH_TEST)

        self._triangle.create()
        self._rebuild()

        context = self.context()
        if context is not None:
            context.aboutToBeDestroyed.connect(self._release)

    def _device_size(self) -> tuple[int, int]:
        """The framebuffer's size in device pixels — the ``gl_FragCoord`` domain.

        Derived from the widget size and the device pixel ratio the same way
        Qt sizes the backing FBO, because nothing else is trustworthy:
        ``resizeGL``'s arguments were measured to arrive in *logical* pixels
        on this platform (an earlier comment here claimed the opposite), and
        in the never-shown ``grabFramebuffer`` path ``resizeGL`` never runs
        at all, leaving any cached size at its constructor-time snapshot.
        """
        ratio = self.devicePixelRatioF() or 1.0
        return (
            max(1, int(round(self.width() * ratio))),
            max(1, int(round(self.height() * ratio))),
        )

    def resizeGL(self, width: int, height: int) -> None:
        from OpenGL.GL import glViewport

        self._framebuffer = self._device_size()
        glViewport(0, 0, *self._framebuffer)

    def paintGL(self) -> None:
        from OpenGL.GL import GL_COLOR_BUFFER_BIT, glClear

        # Refreshed every paint rather than trusted from resize events: see
        # _device_size for the two ways a cached size goes stale.
        self._framebuffer = self._device_size()
        if self._dirty:
            self._rebuild()

        if self._program is None:
            glClear(GL_COLOR_BUFFER_BIT)
            return

        self._program.use()
        seeded = 0
        divisor = int(self.prepass_divisor or 0)
        if divisor > 1:
            seeded = self._render_seed_pass(divisor)

        glClear(GL_COLOR_BUFFER_BIT)
        self._apply_camera_uniforms()
        for name, value in self.uniforms.items():
            self._program.set_uniform(name, value)
        self._program.set_uniform("u_coarse", 0)
        self._program.set_uniform("u_seed_on", seeded)
        if seeded:
            self._program.set_uniform("u_seed", SEED_TEXTURE_UNIT)
            self._program.set_uniform("u_seed_divisor", divisor)
        self._triangle.draw()

    def _render_seed_pass(self, divisor: int) -> int:
        """Render the coarse cone-trace into the seed target. Returns 1 on use.

        The pass runs the same program with ``u_coarse`` set, at the
        framebuffer size divided by ``divisor``, into an R32F attachment the
        main pass reads back per block. The cone term covers the block's ray
        bundle: for perspective it grows with depth at the block's angular
        half-diagonal, for orthographic it is the constant half-diagonal in
        world units — the safety argument lives in metaball.frag beside the
        uniforms it feeds.
        """
        from OpenGL.GL import (
            GL_COLOR_BUFFER_BIT,
            GL_FRAMEBUFFER,
            GL_TEXTURE0,
            GL_TEXTURE_2D,
            glActiveTexture,
            glBindFramebuffer,
            glBindTexture,
            glClear,
            glViewport,
        )

        width, height = self._framebuffer
        coarse = (max(1, -(-width // divisor)), max(1, -(-height // divisor)))
        if not self._ensure_seed_target(coarse):
            return 0

        half_diagonal = 0.5 * math.sqrt(2.0) * divisor * _CONE_MARGIN
        if self.camera.orthographic:
            slope = 0.0
            radius0 = self.camera.ortho_height / max(height, 1) * half_diagonal
        else:
            slope = 2.0 * self.camera.tan_half_fov / max(height, 1) * half_diagonal
            radius0 = 0.0

        glBindFramebuffer(GL_FRAMEBUFFER, self._seed_fbo)
        glViewport(0, 0, *coarse)
        glClear(GL_COLOR_BUFFER_BIT)
        self._apply_camera_uniforms(resolution=coarse)
        for name, value in self.uniforms.items():
            self._program.set_uniform(name, value)
        self._program.set_uniform("u_coarse", 1)
        self._program.set_uniform("u_seed_on", 0)
        self._program.set_uniform("u_cone_slope", float(slope))
        self._program.set_uniform("u_cone_radius0", float(radius0))
        self._triangle.draw()

        glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())
        glViewport(0, 0, width, height)
        glActiveTexture(GL_TEXTURE0 + SEED_TEXTURE_UNIT)
        glBindTexture(GL_TEXTURE_2D, self._seed_texture)
        return 1

    def _ensure_seed_target(self, size: tuple[int, int]) -> bool:
        """Allocate (or resize) the R32F seed attachment. False if unusable."""
        from OpenGL.GL import (
            GL_CLAMP_TO_EDGE,
            GL_COLOR_ATTACHMENT0,
            GL_FLOAT,
            GL_FRAMEBUFFER,
            GL_FRAMEBUFFER_COMPLETE,
            GL_NEAREST,
            GL_R32F,
            GL_RED,
            GL_TEXTURE0,
            GL_TEXTURE_2D,
            GL_TEXTURE_MAG_FILTER,
            GL_TEXTURE_MIN_FILTER,
            GL_TEXTURE_WRAP_S,
            GL_TEXTURE_WRAP_T,
            glActiveTexture,
            glBindFramebuffer,
            glBindTexture,
            glCheckFramebufferStatus,
            glFramebufferTexture2D,
            glGenFramebuffers,
            glGenTextures,
            glTexImage2D,
            glTexParameteri,
        )

        if size == self._seed_size and self._seed_fbo:
            return True

        if self._seed_texture == 0:
            self._seed_texture = int(glGenTextures(1))
            glActiveTexture(GL_TEXTURE0 + SEED_TEXTURE_UNIT)
            glBindTexture(GL_TEXTURE_2D, self._seed_texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glActiveTexture(GL_TEXTURE0 + SEED_TEXTURE_UNIT)
        glBindTexture(GL_TEXTURE_2D, self._seed_texture)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_R32F, size[0], size[1], 0, GL_RED, GL_FLOAT, None
        )

        if self._seed_fbo == 0:
            self._seed_fbo = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, self._seed_fbo)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._seed_texture, 0
        )
        complete = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE
        glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())
        self._seed_size = size if complete else (0, 0)
        return complete

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._drag_button = event.button()
        self._last_pos = event.position().toPoint()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == self._drag_button:
            self._drag_button = Qt.MouseButton.NoButton

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_button == Qt.MouseButton.NoButton:
            return

        position = event.position().toPoint()
        delta = position - self._last_pos
        self._last_pos = position

        panning = self._drag_button == Qt.MouseButton.MiddleButton or bool(
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        )
        if panning:
            self.camera.pan(delta.x(), delta.y(), max(self.height(), 1))
        elif self._drag_button == Qt.MouseButton.LeftButton:
            self.camera.orbit(delta.x(), delta.y())
        else:
            return

        self.camera_changed.emit()
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        self.camera.zoom(event.angleDelta().y())
        self.camera_changed.emit()
        self.update()

    def _apply_camera_uniforms(
        self, resolution: tuple[int, int] | None = None
    ) -> None:
        assert self._program is not None
        width, height = resolution if resolution is not None else self._framebuffer
        right, up, forward = self.camera.basis()

        self._program.set_uniform("u_resolution", (float(width), float(height)))
        self._program.set_uniform("u_camera_position", self.camera.position)
        self._program.set_uniform("u_camera_right", right)
        self._program.set_uniform("u_camera_up", up)
        self._program.set_uniform("u_camera_forward", forward)
        self._program.set_uniform("u_tan_half_fov", self.camera.tan_half_fov)
        self._program.set_uniform("u_aspect", viewport_aspect(width, height))
        self._program.set_uniform("u_orthographic", 1 if self.camera.orthographic else 0)
        self._program.set_uniform("u_ortho_height", self.camera.ortho_height)
        self._program.set_uniform("u_time", self._clock.elapsed() / 1000.0)

    def _rebuild(self) -> None:
        """Compile and swap in the current shader. Requires the context current.

        A failed build keeps the previous program running and reports through
        :attr:`shader_failed`. Live shader editing is otherwise unusable: a
        transient syntax error in the middle of a keystroke would black out the
        viewport and take the working shader with it. The dirty flag clears
        either way, so one bad edit produces one error, not one per frame.
        """
        self._dirty = False
        resolve = file_resolver()
        try:
            vertex = assemble(resolve(VERTEX_SHADER), VERTEX_SHADER, resolve)
            if self._fragment_source is None:
                fragment_text = resolve(DEFAULT_FRAGMENT_SHADER)
            else:
                fragment_text = self._fragment_source
            fragment = assemble(fragment_text, self._fragment_name, resolve)
            program = ShaderProgram(vertex, fragment, name=self._fragment_name)
        except ShaderError as exc:
            self.shader_failed.emit(str(exc))
            return

        if self._program is not None:
            self._program.dispose()
        self._program = program
        self.shader_compiled.emit()

    def _release(self) -> None:
        from OpenGL.GL import glDeleteFramebuffers, glDeleteTextures

        self.makeCurrent()
        if self._seed_fbo:
            glDeleteFramebuffers(1, [self._seed_fbo])
            self._seed_fbo = 0
        if self._seed_texture:
            glDeleteTextures([self._seed_texture])
            self._seed_texture = 0
        self._seed_size = (0, 0)
        if self._program is not None:
            self._program.dispose()
            self._program = None
        self._triangle.dispose()
        self.doneCurrent()
