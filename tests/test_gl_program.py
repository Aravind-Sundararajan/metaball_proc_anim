"""Tests for shader assembly, diagnostics, and the core-profile scaffolding.

Split deliberately in two. Everything above ``TestLiveGL`` is pure string and
path work that runs anywhere — that is the whole reason source assembly and
error formatting were kept out of the GL calls. ``TestLiveGL`` needs a real
core-profile 3.3 context and skips when the machine cannot provide one, which
is the normal state of a headless runner.

The live half draws into an offscreen framebuffer rather than a widget: showing
a window in a test suite is antisocial, and the framebuffer path is what
actually needs proving anyway.
"""

import numpy as np
import pytest

from mesh_xml.render.gl.program import (
    SHADER_DIR,
    AssembledSource,
    ShaderCompileError,
    ShaderSourceError,
    assemble,
    build_error_message,
    error_lines,
    file_resolver,
    format_source,
)

NVIDIA_LOG = '0(42) : error C1503: undefined variable "smin"'
MESA_LOG = "0:42(15): error: no function with name 'smin'"
AMD_LOG = "ERROR: 0:42: 'smin' : no matching overloaded function found"

_TRIVIAL_VERTEX = AssembledSource(
    "#version 330 core\n"
    "void main() {\n"
    "    vec2 c = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));\n"
    "    gl_Position = vec4(c * 2.0 - 1.0, 0.0, 1.0);\n"
    "}\n",
    (),
    "trivial.vert",
)

_CONSTANT_FRAGMENT = AssembledSource(
    "#version 330 core\n"
    "uniform vec4 u_colour;\n"
    "out vec4 c;\n"
    "void main() { c = u_colour; }\n",
    (),
    "constant.frag",
)

_MATRIX_FRAGMENT = AssembledSource(
    "#version 330 core\n"
    "uniform mat4 u_matrix;\n"
    "out vec4 c;\n"
    "void main() { c = u_matrix * vec4(1.0); }\n",
    (),
    "matrix.frag",
)


class TestIncludeResolution:
    def test_a_source_without_includes_is_unchanged(self):
        result = assemble("void main() {}\n", "a.frag", {})
        assert result.text == "void main() {}"
        assert result.name == "a.frag"

    def test_an_include_is_spliced_in(self):
        result = assemble(
            '#version 330 core\n#include "sdf.glsl"\nvoid main() {}',
            "root.frag",
            {"sdf.glsl": "float sphere(vec3 p) { return length(p); }"},
        )
        assert "float sphere" in result.text
        assert "#include" not in result.text

    def test_includes_nest(self):
        result = assemble(
            '#include "a.glsl"',
            "root.frag",
            {"a.glsl": '#include "b.glsl"\nfloat a();', "b.glsl": "float b();"},
        )
        assert "float b();" in result.text
        assert "float a();" in result.text

    def test_angle_brackets_work_too(self):
        result = assemble("#include <a.glsl>", "root.frag", {"a.glsl": "float a();"})
        assert "float a();" in result.text

    @pytest.mark.parametrize(
        "line", ["  #include \"a.glsl\"", "#  include \"a.glsl\"", "#include\t\"a.glsl\""]
    )
    def test_whitespace_variants_are_recognised(self, line):
        assert "float a();" in assemble(line, "r", {"a.glsl": "float a();"}).text

    def test_a_trailing_include_is_not_a_directive(self):
        """Only a directive that starts its line counts, so a commented-out include
        survives into the source instead of silently expanding."""
        result = assemble('float x; // #include "a.glsl"', "r", {})
        assert '#include "a.glsl"' in result.text

    def test_a_repeated_include_is_dropped(self):
        """The diamond. Two headers pulling in one common unit is the normal case,
        and a second copy of its definitions fails to compile."""
        result = assemble(
            '#include "a.glsl"\n#include "b.glsl"',
            "root.frag",
            {
                "a.glsl": '#include "common.glsl"',
                "b.glsl": '#include "common.glsl"',
                "common.glsl": "float shared_helper();",
            },
        )
        assert result.text.count("float shared_helper();") == 1
        assert "already included" in result.text

    def test_a_cycle_is_reported_with_its_chain(self):
        with pytest.raises(ShaderSourceError) as excinfo:
            assemble(
                '#include "a.glsl"',
                "root.frag",
                {"a.glsl": '#include "b.glsl"', "b.glsl": '#include "a.glsl"'},
            )
        message = str(excinfo.value)
        assert "circular" in message
        assert "a.glsl" in message and "b.glsl" in message

    def test_self_inclusion_is_a_cycle(self):
        with pytest.raises(ShaderSourceError):
            assemble('#include "a.glsl"', "r", {"a.glsl": '#include "a.glsl"'})

    def test_a_missing_include_names_the_file_and_the_line(self):
        with pytest.raises(ShaderSourceError) as excinfo:
            assemble('float x;\n#include "gone.glsl"', "root.frag", {})
        message = str(excinfo.value)
        assert "gone.glsl" in message
        assert "root.frag:2" in message

    def test_the_file_resolver_refuses_to_escape_its_directory(self):
        resolve = file_resolver(SHADER_DIR)
        with pytest.raises(ShaderSourceError):
            resolve("../../../core/body.py")

    def test_the_shipped_shaders_assemble(self):
        resolve = file_resolver()
        fragment = assemble(resolve("rays.frag"), "rays.frag", resolve)

        assert fragment.text.startswith("#version 330 core")
        assert "camera_ray" in fragment.text
        assert "#include" not in fragment.text


class TestOrigins:
    def test_lines_map_back_to_the_file_they_came_from(self):
        result = assemble(
            '#version 330 core\n#include "sdf.glsl"\nvoid main() {}',
            "root.frag",
            {"sdf.glsl": "// header\nfloat sphere(vec3 p);"},
        )
        lines = result.text.splitlines()
        sphere_line = lines.index("float sphere(vec3 p);") + 1

        assert result.origin(1) == ("root.frag", 1)
        assert result.origin(sphere_line) == ("sdf.glsl", 2)
        assert result.origin(len(lines)) == ("root.frag", 3)

    def test_out_of_range_lines_return_none(self):
        result = assemble("float x;", "r", {})
        assert result.origin(0) is None
        assert result.origin(99) is None

    def test_there_is_exactly_one_origin_per_line(self):
        result = assemble(
            '#include "a.glsl"\nfloat x;',
            "r",
            {"a.glsl": "float a();\nfloat b();\nfloat c();"},
        )
        assert len(result.origins) == len(result.text.splitlines())


class TestLogParsing:
    @pytest.mark.parametrize("log", [NVIDIA_LOG, MESA_LOG, AMD_LOG])
    def test_every_vendor_dialect_yields_the_line(self, log):
        assert error_lines(log) == (42,)

    def test_multiple_errors_are_collected_and_sorted(self):
        log = "0(90) : error C1503: x\n0(12) : error C1503: y\n0(90) : error C0000: z"
        assert error_lines(log) == (12, 90)

    def test_an_unrecognised_log_yields_nothing(self):
        assert error_lines("something went wrong, good luck") == ()

    def test_an_empty_log_yields_nothing(self):
        assert error_lines("") == ()


class TestSourceFormatting:
    def test_lines_are_numbered(self):
        listing = format_source("a;\nb;\nc;")
        assert "1 | a;" in listing
        assert "3 | c;" in listing

    def test_highlighted_lines_are_marked(self):
        listing = format_source("a;\nb;\nc;", highlight=[2], context=None)
        assert ">>>" in listing
        assert [line for line in listing.splitlines() if line.startswith(">>>")] == [
            ">>> 2 | b;"
        ]

    def test_context_windows_elide_the_rest(self):
        """A three-hundred-line raymarch shader printed whole buries the error."""
        source = "\n".join(f"line{i};" for i in range(1, 101))
        listing = format_source(source, highlight=[50], context=2)

        assert "line50;" in listing
        assert "line48;" in listing
        assert "line1;" not in listing
        assert "..." in listing

    def test_without_highlights_the_whole_source_is_shown(self):
        source = "\n".join(f"line{i};" for i in range(1, 21))
        listing = format_source(source, highlight=[])
        assert "line1;" in listing
        assert "line20;" in listing

    def test_a_highlighted_include_line_names_its_origin(self):
        """The point of the origin map: line 4 of the assembled unit is line 2 of
        sdf.glsl, and only one of those two numbers is actionable."""
        assembled = assemble(
            '#version 330 core\nfloat pad;\n#include "sdf.glsl"\nvoid main() {}',
            "root.frag",
            {"sdf.glsl": "// header\nfloat broken(;"},
        )
        broken = assembled.text.splitlines().index("float broken(;") + 1
        listing = format_source(assembled, highlight=[broken])

        assert "sdf.glsl:2" in listing

    def test_out_of_range_highlights_are_ignored(self):
        assert ">>>" not in format_source("a;\nb;", highlight=[99])


class TestErrorMessage:
    def test_it_carries_the_log_and_the_offending_line(self):
        """Padded so the failing line lands on assembled line 42, which is the only
        number the driver knows and the one nobody can find by hand."""
        padding = 40
        assembled = assemble(
            '#version 330 core\n#include "sdf.glsl"\nvoid main() {}',
            "rays.frag",
            {"sdf.glsl": "\n" * padding + "float d = smin(a, b);"},
        )
        assert assembled.text.splitlines()[41] == "float d = smin(a, b);"

        message = build_error_message("fragment", NVIDIA_LOG, assembled)

        assert "rays.frag" in message
        assert "undefined variable" in message
        assert ">>> 42 | float d = smin(a, b);" in message
        assert f"sdf.glsl:{padding + 1}" in message

    def test_an_unparseable_log_still_dumps_the_source(self):
        message = build_error_message("vertex", "catastrophe", "void main() {}")
        assert "catastrophe" in message
        assert "void main() {}" in message

    def test_a_raw_string_source_is_accepted(self):
        message = build_error_message("fragment", NVIDIA_LOG, "x;\n" * 50)
        assert "fragment" in message


gl_reason = "no core-profile OpenGL 3.3 context available in this environment"


def _gl_available() -> bool:
    from mesh_xml.render.gl.context import gl_available

    return gl_available()


@pytest.fixture(scope="module")
def gl_context():
    """A current, window-less core-profile 3.3 context for the module."""
    if not _gl_available():
        pytest.skip(gl_reason)

    from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat

    from mesh_xml.render.gl.context import ensure_application

    ensure_application()
    fmt = QSurfaceFormat.defaultFormat()

    surface = QOffscreenSurface()
    surface.setFormat(fmt)
    surface.create()

    context = QOpenGLContext()
    context.setFormat(fmt)
    assert context.create()
    assert context.makeCurrent(surface)
    try:
        yield context
    finally:
        context.doneCurrent()


@pytest.mark.skipif(not _gl_available(), reason=gl_reason)
class TestLiveGL:
    def test_the_context_really_is_core_profile(self, gl_context):
        from PySide6.QtGui import QSurfaceFormat

        fmt = gl_context.format()
        assert (fmt.majorVersion(), fmt.minorVersion()) >= (3, 3)
        assert fmt.profile() == QSurfaceFormat.OpenGLContextProfile.CoreProfile

    def test_the_shipped_shaders_compile_and_link(self, gl_context):
        from mesh_xml.render.gl.context import DEFAULT_FRAGMENT_SHADER, VERTEX_SHADER
        from mesh_xml.render.gl.program import ShaderProgram

        program = ShaderProgram.from_files(VERTEX_SHADER, DEFAULT_FRAGMENT_SHADER)
        try:
            assert program.id != 0
            assert program.uniform("u_camera_forward") >= 0
        finally:
            program.dispose()

    def test_a_broken_shader_raises_with_the_log_and_the_source(self, gl_context):
        """The whole reason this module exists. 'compile failed' is not a diagnosis."""
        from mesh_xml.render.gl.program import ShaderProgram

        broken = "#version 330 core\nout vec4 c;\nvoid main() { c = undefined_thing; }"
        with pytest.raises(ShaderCompileError) as excinfo:
            ShaderProgram(_TRIVIAL_VERTEX, AssembledSource(broken, (), "broken.frag"))

        message = str(excinfo.value)
        assert "broken.frag" in message
        assert "undefined_thing" in message
        assert "driver log" in message
        assert ">>>" in message, "the driver named a line but it was not marked"

    def test_a_compile_failure_reports_the_included_file(self, gl_context):
        from mesh_xml.render.gl.program import ShaderProgram

        assembled = assemble(
            '#version 330 core\nout vec4 c;\n#include "bad.glsl"\n'
            "void main() { c = vec4(broken_helper()); }",
            "root.frag",
            {"bad.glsl": "// helper\nfloat broken_helper() { return nope; }"},
        )
        with pytest.raises(ShaderCompileError) as excinfo:
            ShaderProgram(_TRIVIAL_VERTEX, assembled)

        assert "bad.glsl:2" in str(excinfo.value)

    def test_missing_uniforms_are_reported_not_raised(self, gl_context):
        """A shader that does not use u_time must not crash the viewport."""
        from mesh_xml.render.gl.program import ShaderProgram

        program = ShaderProgram(_TRIVIAL_VERTEX, _CONSTANT_FRAGMENT)
        try:
            program.use()
            assert program.set_uniform("u_colour", np.array([1.0, 0.0, 0.0, 1.0])) is True
            assert program.set_uniform("u_never_declared", 1.0) is False
        finally:
            program.dispose()

    def test_a_matrix_uploads_transposed(self, gl_context):
        """Pins the one convention that would silently misplace the whole scene.

        Camera matrices are row-major in the ``M @ v`` sense and GL stores
        column-major, so glGetUniformfv must hand back the transpose. A
        confused transpose flag still produces a picture, just of somewhere
        else, which is the worst kind of bug to find later.
        """
        from OpenGL.GL import glGetUniformfv

        from mesh_xml.render.gl.camera import OrbitCamera
        from mesh_xml.render.gl.program import ShaderProgram

        matrix = OrbitCamera(distance=5.0, yaw=20.0, pitch=-10.0).view_matrix()
        program = ShaderProgram(_TRIVIAL_VERTEX, _MATRIX_FRAGMENT)
        try:
            program.use()
            assert program.set_uniform("u_matrix", matrix) is True

            readback = np.zeros(16, dtype=np.float32)
            glGetUniformfv(program.id, program.uniform("u_matrix"), readback)
        finally:
            program.dispose()

        assert readback.reshape(4, 4) == pytest.approx(matrix.T, abs=1e-6)

    def test_an_unsupported_uniform_shape_is_rejected(self, gl_context):
        from mesh_xml.render.gl.program import ShaderProgram

        program = ShaderProgram(_TRIVIAL_VERTEX, _MATRIX_FRAGMENT)
        try:
            program.use()
            with pytest.raises(TypeError, match="unsupported shape"):
                program.set_uniform("u_matrix", np.zeros((3, 3)))
        finally:
            program.dispose()

    def test_the_triangle_covers_every_pixel(self, gl_context):
        """A missing VAO draws nothing and reports nothing; a quad seam shows here."""
        from mesh_xml.render.gl.program import ShaderProgram

        program = ShaderProgram(_TRIVIAL_VERTEX, _CONSTANT_FRAGMENT)
        try:
            program.use()
            program.set_uniform("u_colour", np.array([1.0, 0.0, 0.5, 1.0]))
            pixels = _render(program, 32, 24)
        finally:
            program.dispose()

        assert (pixels[..., 0] == 255).all()
        assert (pixels[..., 1] == 0).all()
        assert pixels[..., 2] == pytest.approx(128, abs=1)

    def test_the_gpu_rays_match_the_cpu_rays(self, gl_context):
        """The integration this whole layer exists for.

        The reference field and the raymarcher are only comparable if both look
        through the same lens, so the shader's camera_ray and OrbitCamera's
        ray_directions have to agree. An 8-bit framebuffer bounds the check at
        about 1/255, which is still far tighter than any axis or sign error.
        """
        from mesh_xml.render.gl.camera import OrbitCamera
        from mesh_xml.render.gl.context import DEFAULT_FRAGMENT_SHADER, VERTEX_SHADER
        from mesh_xml.render.gl.program import ShaderProgram

        camera = OrbitCamera(distance=6.0, yaw=34.0, pitch=-21.0, target=(1.0, 0.5, -2.0))
        width, height = 40, 24
        _, expected = camera.ray_directions(width, height)
        right, up, forward = camera.basis()

        program = ShaderProgram.from_files(VERTEX_SHADER, DEFAULT_FRAGMENT_SHADER)
        try:
            program.use()
            program.set_uniform("u_resolution", (float(width), float(height)))
            program.set_uniform("u_camera_position", camera.position)
            program.set_uniform("u_camera_right", right)
            program.set_uniform("u_camera_up", up)
            program.set_uniform("u_camera_forward", forward)
            program.set_uniform("u_tan_half_fov", camera.tan_half_fov)
            program.set_uniform("u_aspect", width / height)
            pixels = _render(program, width, height)
        finally:
            program.dispose()

        actual = pixels[..., :3].astype(np.float64) / 255.0 * 2.0 - 1.0
        assert actual == pytest.approx(expected, abs=6e-3)


@pytest.mark.skipif(not _gl_available(), reason=gl_reason)
class TestSurfaceFormat:
    def test_configure_sets_a_core_profile_default(self):
        import warnings

        from PySide6.QtGui import QSurfaceFormat

        from mesh_xml.render.gl.context import configure_surface_format

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fmt = configure_surface_format()

        assert fmt.profile() == QSurfaceFormat.OpenGLContextProfile.CoreProfile
        assert (fmt.majorVersion(), fmt.minorVersion()) == (3, 3)
        assert fmt.samples() == 0, "multisampling costs a shading pass and buys nothing"
        assert QSurfaceFormat.defaultFormat().profile() == fmt.profile()

    def test_a_late_call_warns_loudly(self):
        """The silent failure this warning exists to catch: a compatibility context
        that misbehaves much later, at shader compile time, with a syntax error."""
        from mesh_xml.render.gl.context import (
            configure_surface_format,
            ensure_application,
        )

        ensure_application()
        with pytest.warns(RuntimeWarning, match="before constructing QApplication"):
            configure_surface_format()

    def test_the_view_can_be_constructed_without_being_shown(self):
        from mesh_xml.render.gl.context import RaymarchView, ensure_application

        ensure_application()
        view = RaymarchView()
        try:
            assert view.fragment_source is None
            view.set_fragment_source("#version 330 core\nout vec4 c;\nvoid main(){}")
            assert view.fragment_source is not None
            view.camera.orbit(20.0, 10.0)
            assert view.camera.yaw == pytest.approx(10.0)
        finally:
            view.deleteLater()


def _render(program, width: int, height: int) -> np.ndarray:
    """Draw the fullscreen triangle offscreen and read it back, rows bottom-up.

    glReadPixels returns the framebuffer in the same row order as
    OrbitCamera.ray_directions, which is exactly why that order was chosen.
    """
    from OpenGL.GL import (
        GL_COLOR_BUFFER_BIT,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        glClear,
        glClearColor,
        glReadPixels,
        glViewport,
    )
    from PySide6.QtOpenGL import QOpenGLFramebufferObject

    from mesh_xml.render.gl.context import FullscreenTriangle

    fbo = QOpenGLFramebufferObject(width, height)
    triangle = FullscreenTriangle()
    triangle.create()
    try:
        assert fbo.bind()
        glViewport(0, 0, width, height)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        program.use()
        triangle.draw()
        data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    finally:
        fbo.release()
        triangle.dispose()

    if isinstance(data, np.ndarray):
        raw = np.ascontiguousarray(data, dtype=np.uint8).ravel()
    else:
        raw = np.frombuffer(data, dtype=np.uint8)
    return raw.reshape(height, width, 4)
