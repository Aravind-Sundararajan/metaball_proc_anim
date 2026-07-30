"""Tests for the metaball raymarcher.

The tests that matter compare the shader against
:mod:`mesh_xml.render.field` on real geometry. That comparison is the only
reason the CPU reference exists — a GPU surface has no independent notion of
correct, and "it looked right" does not survive a refactor.

GL tests skip rather than fail where no core-profile context exists, so the pure
half still runs in CI. Check the report, not just the exit code.
"""

import numpy as np
import pytest

from mesh_xml.core import Body
from mesh_xml.render import field, sdf
from mesh_xml.render.gl import metaball, program

gl_context = pytest.importorskip("mesh_xml.render.gl.context")

needs_gl = pytest.mark.skipif(
    not gl_context.gl_available(),
    reason="no core-profile 3.3 context available",
)


def worm(n: int = 6, k_radius: float = 0.25) -> Body:
    body = Body()
    ids = [
        body.add_chunk(f"s{i}", [i * 0.8 - n * 0.4, np.sin(i * 0.7) * 0.4, 0.0], radius=k_radius + 0.04 * i)
        for i in range(n)
    ]
    for a, b in zip(ids, ids[1:]):
        body.add_link(a, b)
    return body


def assembled() -> program.AssembledSource:
    return program.assemble(
        metaball.fragment_source(), "metaball.frag", program.file_resolver()
    )


class TestSourceAssembly:
    def test_the_shader_assembles(self):
        source = assembled()
        assert "#version 330 core" == source.text.splitlines()[0]
        assert "sd_round_cone" in source.text
        assert "camera_ray" in source.text

    def test_every_uniform_the_settings_supply_is_declared(self):
        """A uniform the shader never declares is silently dropped, not an error."""
        text = assembled().text
        supplied = set(metaball.MetaballSettings().uniforms())
        supplied |= set(metaball.PrimitiveTexture().uniforms())

        for name in supplied:
            assert f" {name};" in text, f"{name} is set but not declared in the shader"

    def test_includes_are_not_duplicated(self):
        """camera.glsl and sdf.glsl both get pulled in; a second copy would not link."""
        text = assembled().text
        assert text.count("vec3 camera_ray(") == 1
        assert text.count("float sd_round_cone(") == 1

    def test_reserved_word_is_not_reintroduced(self):
        """`packed` is reserved in GLSL; naming a local that fails to compile."""
        text = assembled().text
        assert "vec3 packed" not in text
        assert "float packed" not in text


class TestDepthPacking:
    def test_round_trips_a_known_distance(self):
        t = 17.25
        far = 40.0
        normalised = t / far
        channels = np.modf(normalised * np.array([1.0, 255.0, 65025.0]))[0]
        channels -= channels[[1, 2, 2]] * np.array([1 / 255.0, 1 / 255.0, 0.0])

        recovered, hit = metaball.unpack_depth(channels[None, :], far)
        assert recovered[0] == pytest.approx(t, abs=1e-3)
        assert hit[0]

    def test_saturated_texel_is_the_miss_sentinel(self):
        _, hit = metaball.unpack_depth(np.full((1, 3), 255, dtype=np.uint8), 40.0)
        assert not hit[0]

    def test_accepts_uint8_and_normalised_input(self):
        raw = np.array([[128, 64, 32]], dtype=np.uint8)
        a, _ = metaball.unpack_depth(raw, 10.0)
        b, _ = metaball.unpack_depth(raw.astype(np.float64) / 255.0, 10.0)
        assert a == pytest.approx(b)


class TestSettings:
    def test_texels_per_primitive_matches_the_stride(self):
        assert metaball.TEXELS_PER_PRIMITIVE * 4 == sdf.PRIMITIVE_STRIDE

    def test_uniforms_are_plain_python_scalars(self):
        """numpy scalars reach glUniform as objects PyOpenGL will not accept."""
        for value in metaball.MetaballSettings().uniforms().values():
            assert isinstance(value, (int, float, tuple))


@needs_gl
class TestAgainstTheCpuReference:
    """The shader is only trustworthy where it agrees with field.py."""

    @staticmethod
    def render(primitives, camera, settings, size):
        from PySide6.QtGui import QImage

        width, height = size
        gl_context.ensure_application()
        view = gl_context.RaymarchView(camera=camera)
        view.set_fragment_source(metaball.fragment_source(), "metaball.frag")
        view.resize(width, height)

        failures: list[str] = []
        view.shader_failed.connect(failures.append)
        view.uniforms.update(settings.uniforms())
        view.grabFramebuffer()
        if failures:
            pytest.fail(failures[0])

        texture = metaball.PrimitiveTexture()
        view.makeCurrent()
        texture.upload(primitives)
        view.doneCurrent()
        view.uniforms.update(texture.uniforms())

        image = view.grabFramebuffer().convertToFormat(QImage.Format.Format_RGB888)
        buffer = np.frombuffer(memoryview(image.constBits()), np.uint8)
        rgb = buffer.reshape(image.height(), image.bytesPerLine() // 3, 3)[:, : image.width()]
        # QImage row 0 is the top; the camera's row 0 is the bottom.
        return rgb[::-1].astype(np.float64)

    def test_depth_matches_the_oracle(self):
        from mesh_xml.render.gl.camera import OrbitCamera

        body = worm()
        primitives = sdf.pack(body)
        blend = 0.3
        size = (160, 120)
        camera = OrbitCamera(distance=8.0, yaw=25.0, pitch=18.0)
        settings = metaball.MetaballSettings(blend=blend, debug_mode=2, max_distance=20.0)

        rgb = self.render(primitives, camera, settings, size)
        t_gpu, hit_gpu = metaball.unpack_depth(rgb, settings.max_distance)

        origin, directions = camera.ray_directions(*size)
        t_cpu, hit_cpu = field.raymarch(
            origin,
            directions,
            primitives,
            blend,
            max_distance=settings.max_distance,
            max_steps=settings.max_steps,
            surface_epsilon=settings.surface_epsilon,
            step_relaxation=settings.step_relaxation,
        )

        assert hit_gpu.sum() > 500, "the body should cover a meaningful part of the frame"

        # Silhouette pixels legitimately disagree: a ray grazing the surface is
        # decided by float32 on one side and float64 on the other.
        agreement = (hit_gpu == hit_cpu).mean()
        assert agreement > 0.99, f"hit masks agree on only {agreement:.3%}"

        shared = hit_gpu & hit_cpu
        error = np.abs(t_gpu[shared] - t_cpu[shared])
        assert np.percentile(error, 99) < 0.05
        assert error.mean() < 0.01

    def test_blending_thickens_the_surface(self):
        """k > 0 must move the surface outward, not merely smooth it."""
        from mesh_xml.render.gl.camera import OrbitCamera

        body = worm()
        primitives = sdf.pack(body)
        size = (128, 96)
        camera = OrbitCamera(distance=8.0, yaw=25.0, pitch=18.0)

        hits = []
        for blend in (0.0, 0.45):
            settings = metaball.MetaballSettings(
                blend=blend, debug_mode=2, max_distance=20.0
            )
            rgb = self.render(primitives, camera, settings, size)
            _, hit = metaball.unpack_depth(rgb, settings.max_distance)
            hits.append(int(hit.sum()))

        assert hits[1] > hits[0]

    def test_both_degenerate_primitives_render_without_nan(self):
        """Coincident ends and one ball containing the other, on the GPU."""
        from mesh_xml.render.gl.camera import OrbitCamera

        body = Body()
        lone = body.add_chunk("lone", [-1.2, 0.0, 0.0], radius=0.5)
        head = body.add_chunk("head", [1.0, 0.0, 0.0], radius=0.6)
        eye = body.add_chunk("eye", [1.1, 0.0, 0.0], radius=0.08)
        body.add_link(head, eye)
        assert lone is not None

        primitives = sdf.pack(body)
        axis = np.linalg.norm(primitives[0, 4:7] - primitives[0, 0:3])
        assert abs(primitives[0, 3] - primitives[0, 7]) >= axis, "containment case not built"

        camera = OrbitCamera(distance=7.0, yaw=15.0, pitch=10.0)
        settings = metaball.MetaballSettings(blend=0.25, debug_mode=1, max_distance=20.0)
        rgb = self.render(primitives, camera, settings, (128, 96))

        assert np.isfinite(rgb).all()
        # A NaN normal reaches the framebuffer as black; the lit body must not be.
        assert rgb.max() > 200.0

    def test_an_empty_body_renders_only_background(self):
        from mesh_xml.render.gl.camera import OrbitCamera

        settings = metaball.MetaballSettings(debug_mode=2, max_distance=20.0)
        rgb = self.render(sdf.pack(Body()), OrbitCamera(), settings, (64, 48))
        _, hit = metaball.unpack_depth(rgb, settings.max_distance)

        assert not hit.any()
