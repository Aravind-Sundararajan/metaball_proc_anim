#version 330 core

// Sphere-traced metaball surface, over the terrain it stands on.
//
// Mirrors mesh_xml.render.field.raymarch, including the step relaxation and the
// growing surface epsilon. The debug modes exist so the CPU reference can check
// this shader numerically rather than by eye: mode 2 writes the hit distance
// packed across all three channels, which recovers ~24 bits from an 8-bit
// framebuffer and is what tests/test_metaball.py reads back.
//
// The march runs against sd_scene — the hard minimum of creature and terrain,
// see sdf.glsl for why hard — so every debug mode, packed depth included,
// reports the composite; the CPU comparison marches the same minimum. With no
// terrain uploaded sd_scene is the creature field bit-for-bit.

#include "camera.glsl"
#include "sdf.glsl"

uniform float u_max_distance;
uniform int   u_max_steps;
uniform float u_surface_epsilon;
uniform float u_step_relaxation;
uniform int   u_debug_mode;      // 0 shaded, 1 normal, 2 packed depth, 3 step count, 4 albedo
uniform vec3  u_light_direction;
uniform vec3  u_light_colour;
uniform float u_light_strength;
uniform float u_ambient_strength;
uniform float u_ao_strength;
uniform float u_shadow_strength;
uniform float u_glow_strength;
uniform vec3  u_surface_colour;
uniform vec3  u_terrain_colour;
uniform vec3  u_background_colour;
uniform vec3  u_background_top;
uniform float u_normal_epsilon;

// The reference floor: an analytic plane the editor draws when no terrain is
// loaded. Never part of sd_scene — the march, the packed-depth debug output and
// the CPU comparison are all unaware of it, which is what keeps the oracle
// tests exact. Both uniforms default to 0, so a view that never sets them (the
// test harness, any standalone RaymarchView) renders exactly as before.
uniform int   u_floor_enabled;
uniform float u_floor_height;

// The coarse cone prepass. Pass one renders at a fraction of the resolution
// with u_coarse set: each coarse pixel sphere-traces while its distance still
// clears a cone wide enough to contain its whole block's ray bundle, and
// writes how far it provably got into an R32F target. Pass two seeds every
// full-resolution ray at its block's depth. The safety argument is the bound
// cull's, one step on: the field is 1-Lipschitz, every fine-ray point at
// parameter t lies within cone_radius(t) of the coarse point, so a coarse
// value above the cone radius proves the fine field positive — no fine ray
// can hit before the seed. Surface-exact, not byte-exact, exactly like the
// tile shortlist: empty-space step sizes change, the surfaces hit do not.
// cone_radius(t) = u_cone_radius0 + t * u_cone_slope covers both projections:
// perspective cones grow with t, orthographic bundles have constant width.
uniform int   u_coarse;
uniform float u_cone_slope;
uniform float u_cone_radius0;
uniform int   u_seed_on;
uniform sampler2D u_seed;
uniform int   u_seed_divisor;

float cone_radius(float t) {
    return u_cone_radius0 + t * u_cone_slope;
}

float trace_coarse(vec3 origin, vec3 direction) {
    float t = 0.0;
    for (int i = 0; i < u_max_steps; ++i) {
        float d = sd_scene(origin + direction * t);
        float slack = d - cone_radius(t);
        if (slack < u_surface_epsilon * max(1.0, t)) break;
        // The divisor keeps the invariant exact as the cone widens across
        // the step; at real cone slopes it costs a fraction of a percent.
        t += slack * u_step_relaxation / (1.0 + u_cone_slope);
        if (t > u_max_distance) return u_max_distance;
    }
    return t;
}

float seeded_start() {
    if (u_seed_on == 0) return 0.0;
    ivec2 block = ivec2(gl_FragCoord.xy) / max(u_seed_divisor, 1);
    block = clamp(block, ivec2(0), textureSize(u_seed, 0) - ivec2(1));
    return max(texelFetch(u_seed, block, 0).r, 0.0);
}

const float GRID_CELL       = 1.0;   // world units per minor cell
const float GRID_MAJOR      = 5.0;   // minor cells per major line
const float GRID_FADE_START = 18.0;  // eye distance where the grid starts to go
const float GRID_FADE_END   = 55.0;  // ... and where it has fully gone

// The shading model's baked constants (docs/LIGHTING.md 2) -- the
// vocabulary's opinion, like SDF_STRIPE_BAND, so light_strength = 1 lands
// the pre-arc diffuse level exactly.
const float DIFFUSE        = 0.75;  // key level at unit strength
const float AMBIENT_GROUND = 0.7;   // the fill's ground half sits below its sky half
const float GLOW_SKIN      = 0.5;   // how much of the rim the skin carries
const float AO_STEP        = 0.06;  // five taps reach 0.3 world units
const float AO_FALLOFF     = 0.7;   // per-tap weight decay
const float AO_GAIN        = 3.0;   // occlusion sum to darkening, before the knob
const float CHROMA_FLOOR   = 1e-3;  // near-black guard for the hue lift
const float CHROMA_FADE    = 0.04;  // the visible range over which black stops filling

// A backdrop colour's contribution to the fill: its hue lifted to unit
// brightness -- the swatches state the hue, u_ambient_strength states the
// level -- with the lift fading out across a visible near-black range, so a
// black sky fills nothing and approaches that boundary continuously rather
// than at a one-picker-quantum cliff.
vec3 chroma(vec3 c) {
    float level = max(max(c.r, c.g), c.b);
    return (c / max(level, CHROMA_FLOOR)) * smoothstep(0.0, CHROMA_FADE, level);
}

in  vec2 v_uv;
out vec4 frag_colour;

struct Hit {
    float t;
    bool  hit;
    int   steps;
};

Hit trace(vec3 origin, vec3 direction) {
    Hit result;
    result.t = seeded_start();
    result.hit = false;
    result.steps = 0;

    for (int i = 0; i < u_max_steps; ++i) {
        vec3 p = origin + direction * result.t;
        float d = sd_scene(p);
        result.steps = i + 1;

        // The tolerance grows with distance so a surface far from the camera is
        // not chased for the full step budget to satisfy a tolerance finer than
        // the pixel it lands in.
        if (d < u_surface_epsilon * max(1.0, result.t)) {
            result.hit = true;
            break;
        }
        result.t += d * u_step_relaxation;
        if (result.t > u_max_distance) break;
    }
    return result;
}

// Distance packed into three bytes, most significant first. The framebuffer is
// 8 bits per channel and a single channel would quantise depth to 1/255 of the
// far plane, which is far coarser than the disagreement worth detecting.
// `packed` is a GLSL reserved word; naming the local that is a compile error.
vec3 pack_depth(float t) {
    float normalised = clamp(t / u_max_distance, 0.0, 1.0);
    vec3 channels = fract(normalised * vec3(1.0, 255.0, 65025.0));
    channels -= channels.yzz * vec3(1.0 / 255.0, 1.0 / 255.0, 0.0);
    return channels;
}

// The shadow march's reach: it walks at most this far toward the light, and
// with penumbra factor 8 a surface farther than SHADOW_MAX_T / 8 from every
// point of the ray cannot bring the running min below one. Those two numbers
// are what make the early-out below exact rather than approximate.
const float SHADOW_MAX_T = 20.0;
const float SHADOW_REACH = SHADOW_MAX_T / 8.0;

// Whether any creature's bounding sphere, inflated by the penumbra reach,
// crosses the shadow ray. If none does, every field value along the march
// exceeds t/8, the running min never drops below one, and the march would
// return full light — so returning it without marching changes nothing.
// Without a valid bounds table the answer is "maybe", and the march runs.
bool maybe_shadowed(vec3 point, vec3 light) {
    if (u_group_count <= 0 || u_group_bounds_count != u_group_count) return true;
    for (int g = 0; g < u_group_count; ++g) {
        vec4 sphere = texelFetch(u_group_bounds, ivec2(0, g), 0);
        float reach = sphere.w + SHADOW_REACH;
        vec3 to = sphere.xyz - point;
        float along = dot(to, light);
        if (along < -reach || along > SHADOW_MAX_T + reach) continue;
        if (dot(to, to) - along * along <= reach * reach) return true;
    }
    return false;
}

// How much key reaches a receiver point — the reference floor, creature
// flesh and terrain alike take the one answer: a march toward the light
// through sd_field, never the composite, so the creature casts and the
// world only catches (docs/CAST_SHADOWS.md 3 — terrain has no bound for
// the gate to test). The running min of d/t is the classic penumbra
// estimate — distance the occluder clears the ray by, over how far along
// it sits. The tile mask stands down for the march and is restored after:
// shadow rays head for the light, not the camera, so this tile's mask
// says nothing about their occluders.
float key_shadow(vec3 point, vec3 light) {
    // key shadow -- begin. The markers are load-bearing: the
    // exact-skip test excises everything between them and asserts
    // shadow_strength 0 renders the same bytes as the deleted march.
    if (u_shadow_strength > 0.0 && maybe_shadowed(point, light)) {
        bool was_tiled = sdf_tiled;
        sdf_tiled = false;
        float t = 0.05;
        float shade = 1.0;
        for (int i = 0; i < 48; ++i) {
            float d = sd_field(point + light * t);
            if (d < 1e-3) { shade = 0.0; break; }
            shade = min(shade, 8.0 * d / t);
            t += clamp(d, 0.02, 0.5);
            if (t > SHADOW_MAX_T) break;
        }
        sdf_tiled = was_tiled;
        shade = clamp(shade, 0.0, 1.0);
        // The endpoint is exact by branch, not by arithmetic: GLSL
        // pins no rounding for a lerp, and the default must
        // reproduce a pre-shadow room's floor shadow bit for bit.
        return u_shadow_strength == 1.0
            ? shade
            : clamp(1.0 - u_shadow_strength * (1.0 - shade), 0.0, 1.0);
    }
    // key shadow -- end
    return 1.0;
}

// What colour the surface is at a hit, before any light touches it: the
// terrain's flat colour where the world won the hard min, else the
// creature's — its group's folded albedo when the attribute table is armed
// (docs/TEXTURES.md), its group's flat colour when not, and the single
// surface colour when no groups are uploaded at all. Debug mode 4 writes
// this raw, which is what lets the CPU oracle check colour numerically
// through the same readback the depth modes use, lighting never in frame.
vec3 scene_albedo(vec3 position) {
    vec3 creature;
    if (u_group_count > 0) {
        int group = sd_nearest_group(position);
        creature = (u_attribute_count > 0
                    && u_attribute_count == u_primitive_count)
            ? sd_group_albedo(position, group)
            : sd_group_colour(group);
    } else {
        creature = u_surface_colour;
    }
    return sd_terrain(position) < sd_field(position)
        ? u_terrain_colour
        : creature;
}

// Anti-aliased grid coverage at a floor point, in [0, 1]. Screen-space
// derivatives turn the line width into pixels, so lines stay one line wide at
// any distance and dissolve on their own where a cell is thinner than a pixel.
float grid_line(vec2 p, float cell) {
    vec2 coord = p / cell;
    vec2 pixels = fwidth(coord);
    vec2 dist = abs(fract(coord - 0.5) - 0.5) / max(pixels, vec2(1e-6));
    return 1.0 - smoothstep(0.6, 1.4, min(dist.x, dist.y));
}

void main() {
    vec3 direction = camera_ray(v_uv);
    vec3 origin = camera_origin(v_uv);

    // The coarse pass wants none of the fine pass's machinery: no tiles
    // (its pixels straddle tile boundaries), no floor, no shading — just
    // how far the whole block's bundle provably travels, written as depth.
    if (u_coarse != 0) {
        frag_colour = vec4(trace_coarse(origin, direction), 0.0, 0.0, 1.0);
        return;
    }

    sdf_arm_tiles(gl_FragCoord.xy);
    Hit result = trace(origin, direction);

    if (u_debug_mode == 2) {
        frag_colour = vec4(result.hit ? pack_depth(result.t) : vec3(1.0), 1.0);
        return;
    }
    if (u_debug_mode == 3) {
        frag_colour = vec4(vec3(float(result.steps) / float(u_max_steps)), 1.0);
        return;
    }
    // Raw albedo, unlit. A miss writes white, mode 2's sentinel convention.
    if (u_debug_mode == 4) {
        frag_colour = vec4(
            result.hit
                ? scene_albedo(origin + direction * result.t)
                : vec3(1.0),
            1.0
        );
        return;
    }

    // The backdrop: a vertical wash in screen space, deliberately not a sky
    // dome -- screen-keyed survives the orthographic toggle, and a two-stop
    // vertical lerp has no horizon line to tilt against the grid's own
    // (docs/LIGHTING.md 3). Debug mode 1 shares the miss branch below, so
    // its backdrop is the gradient too: eyeballed, never oracle-compared.
    vec3 sky = mix(u_background_colour, u_background_top, v_uv.y);

    // The reference floor draws only in the shaded mode: the debug modes are
    // the CPU oracle's view and the oracle marches sd_scene alone. The outer
    // branch is uniform, which is what keeps grid_line's derivatives defined.
    if (u_floor_enabled != 0 && u_debug_mode == 0) {
        float slope = direction.y;
        float floor_t = abs(slope) > 1e-6 ? (u_floor_height - origin.y) / slope : -1.0;
        vec3 point = origin + direction * max(floor_t, 0.0);

        float minor = grid_line(point.xz, GRID_CELL);
        float major = grid_line(point.xz, GRID_CELL * GRID_MAJOR);
        float fade = 1.0 - smoothstep(GRID_FADE_START, GRID_FADE_END, floor_t);

        bool wins = floor_t > 0.0
            && floor_t < u_max_distance
            && (!result.hit || floor_t < result.t);
        if (wins) {
            vec3 light = normalize(u_light_direction);
            float lit = mix(0.45, 1.0, key_shadow(point, light));
            vec3 base = mix(u_background_colour, vec3(1.0), 0.05);
            vec3 lined = base + vec3(0.06) * minor + vec3(0.10) * major;
            // The distance fade lands on the same sky sample the miss path
            // writes at this pixel; fading to the flat horizon instead
            // would seam the far grid against the gradient behind it.
            frag_colour = vec4(mix(sky, lined * lit, fade), 1.0);
            return;
        }
    }

    if (!result.hit) {
        frag_colour = vec4(sky, 1.0);
        return;
    }

    vec3 position = origin + direction * result.t;
    vec3 normal = sd_scene_normal(position, u_normal_epsilon);

    if (u_debug_mode == 1) {
        frag_colour = vec4(normal * 0.5 + 0.5, 1.0);
        return;
    }

    // The hard min against terrain has no blend weight to interpolate colour
    // by, so the hit belongs entirely to whichever field is closer at the
    // landing point — the same comparison min() made when the ray stopped;
    // scene_albedo owns that split and the creature's own colour ladder.
    vec3 albedo = scene_albedo(position);

    vec3 light = normalize(u_light_direction);
    float shadow = key_shadow(position, light);

    // One authored key, and the backdrop is the fill (docs/LIGHTING.md 2).
    // The fill is hemispheric on world-up -- the axis countershade keys on
    // -- and never tracks the key, so a countershaded belly reads as
    // pigment against quiet ground light instead of being cancelled twice.
    vec3 ambient = u_ambient_strength
        * mix(AMBIENT_GROUND * chroma(u_background_colour),
              chroma(u_background_top),
              0.5 + 0.5 * normal.y);
    vec3 key = u_light_colour * u_light_strength * DIFFUSE
        * max(dot(normal, light), 0.0);
    // The rim carries the key's colour through the skin: dark skins rim
    // dimmer, so a backlight cannot bleach the tones countershade authored.
    vec3 glow = u_light_colour * u_glow_strength
        * mix(vec3(1.0), albedo, GLOW_SKIN)
        * pow(1.0 - max(dot(normal, -direction), 0.0), 3.0);

    float ao = 1.0;
    // AO ladder -- begin. The markers are load-bearing: the exact-skip test
    // excises everything between them and asserts ao_strength == 0 renders
    // the same bytes, so the taps are provably the frame's only new field
    // evaluations and zero is a true off switch (a uniform branch, no
    // derivatives inside).
    if (u_ao_strength > 0.0) {
        // Taps leave the camera ray's cone, so the tile mask stands down
        // for the ladder and is restored after -- key_shadow's argument;
        // an occluder one tile over is real. The bounds cull stays live.
        bool was_tiled = sdf_tiled;
        sdf_tiled = false;
        float occ = 0.0;
        float fall = 1.0;
        for (int i = 1; i <= 5; ++i) {
            float h = float(i) * AO_STEP;
            float d = sd_scene(position + normal * h);
            // The reference floor stays outside sd_scene (march and oracle
            // exactness, the comment above); where it is actually drawn it
            // occludes here, so a foot planted on the grid darkens at the
            // contact. Modes 2 and 4 return long before this runs.
            if (u_floor_enabled != 0) {
                d = min(d, position.y + normal.y * h - u_floor_height);
            }
            occ += (h - d) * fall;
            fall *= AO_FALLOFF;
        }
        sdf_tiled = was_tiled;
        ao = clamp(1.0 - u_ao_strength * AO_GAIN * occ, 0.0, 1.0);
    }
    // AO ladder -- end

    // The occlusion authorities stay disjoint, one term each: AO scales the
    // fill and the glow, never the key -- occlusion is the environment
    // failing to reach a crease -- and the key failing to reach one is the
    // cast shadow, which gates the key alone (docs/CAST_SHADOWS.md 5).
    vec3 shaded = albedo * (ambient * ao + key * shadow) + glow * ao;
    frag_colour = vec4(shaded, 1.0);
}
