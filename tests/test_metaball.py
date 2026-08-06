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
from mesh_xml.core.terrain import Terrain
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


def floor_and_mound() -> Terrain:
    """A flat floor with a buried capsule rising out of it — both vocabularies."""
    return Terrain.flat(-0.8, blend=0.35).add_cone(
        (-0.6, -1.3, 0.0), 1.1, (1.5, -1.0, 0.0), 0.85
    )


def combined_raymarch(
    origins: np.ndarray,
    directions: np.ndarray,
    primitives: np.ndarray,
    blend: float,
    terrain: Terrain,
    *,
    max_steps: int,
    max_distance: float,
    surface_epsilon: float,
    step_relaxation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sphere-trace ``min(creature, terrain)`` — the composite the shader draws.

    A test-local twin of :func:`field.raymarch` rather than a library function,
    because the library keeps the two fields separate on purpose: the hard-min
    composite is the *frag shader's* move, and this file is where the shader is
    held to account. The loop shape is raymarch's verbatim; only the field
    evaluation changes.
    """
    origin, direction = np.broadcast_arrays(
        np.asarray(origins, np.float64), np.asarray(directions, np.float64)
    )
    shape = origin.shape[:-1]
    origin = origin.reshape(-1, 3)
    direction = direction.reshape(-1, 3)
    direction = direction / np.linalg.norm(direction, axis=1)[:, None]

    t = np.zeros(len(origin))
    hit = np.zeros(len(origin), dtype=bool)
    active = np.arange(len(origin))
    for _ in range(max_steps):
        if active.size == 0:
            break
        p = origin[active] + direction[active] * t[active, None]
        d = np.minimum(field.field(p, primitives, blend), terrain.distance(p))
        landed = d < surface_epsilon
        hit[active[landed]] = True
        t[active] += np.where(landed, 0.0, d * step_relaxation)
        active = active[~landed & (t[active] <= max_distance)]

    np.minimum(t, max_distance, out=t)
    return t.reshape(shape), hit.reshape(shape)


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
        supplied |= set(metaball.TerrainTexture().uniforms())

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


class TestTerrainPacking:
    """pack_terrain's row order IS the shader's fold order; pin the layout."""

    def test_planes_pack_first_then_cones_in_authoring_order(self):
        terrain = (
            Terrain(blend=0.4)
            .add_plane((0.0, 1.0, 0.0), -0.5)
            .add_plane((1.0, 0.0, 0.0), 2.0)
            .add_cone((1.0, 2.0, 3.0), 0.25, (4.0, 5.0, 6.0), 0.5)
        )

        rows = metaball.pack_terrain(terrain)

        assert rows.shape == (3, sdf.PRIMITIVE_STRIDE)
        assert rows.dtype == np.float32
        assert rows[0, :4] == pytest.approx([0.0, 1.0, 0.0, -0.5])
        assert rows[1, :4] == pytest.approx([1.0, 0.0, 0.0, 2.0])
        assert rows[2] == pytest.approx([1.0, 2.0, 3.0, 0.25, 4.0, 5.0, 6.0, 0.5])

    def test_plane_padding_is_zero_not_garbage(self):
        """The shader never fetches a plane row's second texel; keep it inert anyway."""
        rows = metaball.pack_terrain(Terrain().add_plane((0.0, 1.0, 0.0), 1.0))
        assert (rows[0, 4:] == 0.0).all()

    def test_none_and_empty_pack_to_zero_rows(self):
        assert metaball.pack_terrain(None).shape == (0, sdf.PRIMITIVE_STRIDE)
        assert metaball.pack_terrain(Terrain()).shape == (0, sdf.PRIMITIVE_STRIDE)

    def test_terrain_texture_uniforms_are_plain_python_scalars(self):
        for value in metaball.TerrainTexture().uniforms().values():
            assert isinstance(value, (int, float))

    def test_the_terrain_texture_shares_no_unit_with_the_creature(self):
        assert metaball.TERRAIN_TEXTURE_UNIT != metaball.PRIMITIVE_TEXTURE_UNIT
        creature = metaball.PrimitiveTexture().uniforms()["u_primitives"]
        world = metaball.TerrainTexture().uniforms()["u_terrain"]
        assert creature != world


@needs_gl
class TestAgainstTheCpuReference:
    """The shader is only trustworthy where it agrees with field.py."""

    @staticmethod
    def render(primitives, camera, settings, size, terrain=None):
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

        # terrain=None sets no terrain uniform at all — the shader's zero
        # defaults — which is exactly the path every terrain-unaware caller
        # runs; the bit-identity test below leans on that distinction.
        if terrain is not None:
            world = metaball.TerrainTexture()
            view.makeCurrent()
            world.upload_terrain(terrain)
            view.doneCurrent()
            view.uniforms.update(world.uniforms())

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

    # -- terrain: the composite against the combined oracle ---------------

    def test_depth_matches_the_oracle_with_terrain(self):
        """The packed depth reports the hard-min composite, marched the same way.

        With a floor under everything the classic silhouette disagreement
        changes shape: a grazing ray decided differently lands on the creature
        on one side and on the floor *behind* it on the other, so it shows up
        as a depth gap between two hits rather than as a hit-mask flip. The
        agreement number therefore counts a pixel as agreeing when both sides
        miss or both land within 5cm, which subsumes the old mask comparison.
        """
        from mesh_xml.render.gl.camera import OrbitCamera

        body = worm()
        primitives = sdf.pack(body)
        terrain = floor_and_mound()
        blend = 0.3
        size = (160, 120)
        # Pitch steep enough that every ray meets the floor inside
        # max_distance: no horizon band of escape-boundary pixels to argue over.
        camera = OrbitCamera(distance=8.0, yaw=25.0, pitch=40.0)
        settings = metaball.MetaballSettings(blend=blend, debug_mode=2, max_distance=25.0)

        rgb = self.render(primitives, camera, settings, size, terrain=terrain)
        t_gpu, hit_gpu = metaball.unpack_depth(rgb, settings.max_distance)

        origin, directions = camera.ray_directions(*size)
        t_cpu, hit_cpu = combined_raymarch(
            origin,
            directions,
            primitives,
            blend,
            terrain,
            max_distance=settings.max_distance,
            max_steps=settings.max_steps,
            surface_epsilon=settings.surface_epsilon,
            step_relaxation=settings.step_relaxation,
        )

        # The scene must be what it claims: the world catches nearly every ray,
        # and the creature alone accounts for only part of that.
        assert hit_cpu.mean() > 0.95, "the floor should catch nearly every ray"
        _, hit_creature = field.raymarch(
            origin,
            directions,
            primitives,
            blend,
            max_distance=settings.max_distance,
            max_steps=settings.max_steps,
        )
        assert hit_creature.sum() > 500, "the body should cover a meaningful part"
        assert hit_creature.mean() < 0.6, "terrain must contribute most of the hits"

        both_hit = hit_gpu & hit_cpu
        both_miss = ~hit_gpu & ~hit_cpu
        close = np.zeros(both_hit.shape, dtype=bool)
        close[both_hit] = np.abs(t_gpu[both_hit] - t_cpu[both_hit]) < 0.05
        agreement = (both_miss | close).mean()
        assert agreement > 0.99, f"depth agrees on only {agreement:.3%}"

        error = np.abs(t_gpu[both_hit] - t_cpu[both_hit])
        assert np.median(error) < 0.01

    def test_the_terrain_fold_order_is_part_of_the_surface(self):
        """Permuting terrain primitives is a different world.

        smooth_min is not associative, so Terrain.distance documents its fold
        order and the shader must reproduce it rather than sort, re-seed or
        reduce pairwise. Two cone orders rendering two different surfaces is
        what pins that: a shader folding in any canonicalised order would
        produce identical images and fail here.
        """
        from mesh_xml.render.gl.camera import OrbitCamera

        cone_a = ((-0.9, -1.2, 0.0), 1.0, (0.2, -0.7, 0.0), 0.7)
        cone_b = ((1.0, -1.3, 0.0), 1.1, (-0.1, -0.6, 0.0), 0.6)
        one = Terrain(blend=0.7).add_plane((0.0, 1.0, 0.0), -0.6)
        one.add_cone(*cone_a).add_cone(*cone_b)
        two = Terrain(blend=0.7).add_plane((0.0, 1.0, 0.0), -0.6)
        two.add_cone(*cone_b).add_cone(*cone_a)

        # The premise, on the CPU: the two orders disagree about the field
        # where all three primitives blend at once.
        rng = np.random.default_rng(11)
        probes = rng.normal(0.0, 1.0, (256, 3)) * np.array([1.2, 0.8, 0.8])
        assert np.abs(one.distance(probes) - two.distance(probes)).max() > 1e-3

        camera = OrbitCamera(distance=6.0, yaw=15.0, pitch=35.0)
        settings = metaball.MetaballSettings(debug_mode=2, max_distance=20.0)
        empty = sdf.pack(Body())
        size = (128, 96)

        t_one, hit_one = metaball.unpack_depth(
            self.render(empty, camera, settings, size, terrain=one), 20.0
        )
        t_two, hit_two = metaball.unpack_depth(
            self.render(empty, camera, settings, size, terrain=two), 20.0
        )

        shared = hit_one & hit_two
        assert shared.mean() > 0.9, "both worlds should fill the frame"
        assert np.abs(t_one[shared] - t_two[shared]).max() > 1e-3, (
            "reordering the cones left the surface unchanged; the shader is "
            "not folding in Terrain.distance's documented order"
        )

    def test_a_scene_without_terrain_is_unchanged_by_the_terrain_machinery(self):
        """One inert path, not two.

        The left render sets no terrain uniform at all — the path every
        terrain-unaware caller has always run, with the shader's zero
        defaults. The right one explicitly uploads an empty world with a
        non-zero blend. Byte equality says the terrain code is truly inert at
        count zero, and that adding it did not perturb the creature-only
        image: min(d, SDF_FAR) is d, bit for bit.
        """
        from mesh_xml.render.gl.camera import OrbitCamera

        primitives = sdf.pack(worm())
        camera = OrbitCamera(distance=8.0, yaw=25.0, pitch=18.0)
        settings = metaball.MetaballSettings(blend=0.3, max_distance=20.0)
        size = (128, 96)

        untouched = self.render(primitives, camera, settings, size)
        explicit = self.render(
            primitives, camera, settings, size, terrain=Terrain(blend=0.5)
        )

        assert np.array_equal(untouched, explicit)

    def test_the_terrain_wears_its_own_colour(self):
        """Per-hit winner: a floor pixel shades with u_terrain_colour and a
        creature pixel with u_surface_colour, inside one composited frame."""
        from mesh_xml.render.gl.camera import OrbitCamera

        primitives = sdf.pack(worm())
        terrain = floor_and_mound()
        camera = OrbitCamera(distance=8.0, yaw=25.0, pitch=40.0)
        settings = metaball.MetaballSettings(
            blend=0.3,
            max_distance=25.0,
            surface_colour=(0.9, 0.2, 0.2),
            terrain_colour=(0.1, 0.8, 0.2),
        )

        rgb = self.render(primitives, camera, settings, (160, 120), terrain=terrain)

        # Row 0 is the camera's bottom row: at this pitch, guaranteed ground.
        floor_patch = rgb[2:8, 70:90].reshape(-1, 3).mean(axis=0)
        assert floor_patch[1] > floor_patch[0] + 20.0, "the ground is not green"

        # The frame centre looks at the camera target, inside the worm.
        centre_patch = rgb[56:64, 76:84].reshape(-1, 3).mean(axis=0)
        assert centre_patch[0] > centre_patch[1] + 20.0, "the creature is not red"
