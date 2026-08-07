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
uniform int   u_debug_mode;      // 0 shaded, 1 normal, 2 packed depth, 3 step count
uniform vec3  u_light_direction;
uniform vec3  u_surface_colour;
uniform vec3  u_terrain_colour;
uniform vec3  u_background_colour;
uniform float u_normal_epsilon;

// The reference floor: an analytic plane the editor draws when no terrain is
// loaded. Never part of sd_scene — the march, the packed-depth debug output and
// the CPU comparison are all unaware of it, which is what keeps the oracle
// tests exact. Both uniforms default to 0, so a view that never sets them (the
// test harness, any standalone RaymarchView) renders exactly as before.
uniform int   u_floor_enabled;
uniform float u_floor_height;

const float GRID_CELL       = 1.0;   // world units per minor cell
const float GRID_MAJOR      = 5.0;   // minor cells per major line
const float GRID_FADE_START = 18.0;  // eye distance where the grid starts to go
const float GRID_FADE_END   = 55.0;  // ... and where it has fully gone

in  vec2 v_uv;
out vec4 frag_colour;

struct Hit {
    float t;
    bool  hit;
    int   steps;
};

Hit trace(vec3 origin, vec3 direction) {
    Hit result;
    result.t = 0.0;
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

// How much creature light reaches a floor point: a march toward the light
// through sd_field alone, so the world (absent when the floor draws) can never
// double-shadow. The running min of d/t is the classic penumbra estimate —
// distance the occluder clears the ray by, over how far along it sits.
float floor_shadow(vec3 point, vec3 light) {
    float t = 0.05;
    float shade = 1.0;
    for (int i = 0; i < 48; ++i) {
        float d = sd_field(point + light * t);
        if (d < 1e-3) return 0.0;
        shade = min(shade, 8.0 * d / t);
        t += clamp(d, 0.02, 0.5);
        if (t > 20.0) break;
    }
    return clamp(shade, 0.0, 1.0);
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
    Hit result = trace(origin, direction);

    if (u_debug_mode == 2) {
        frag_colour = vec4(result.hit ? pack_depth(result.t) : vec3(1.0), 1.0);
        return;
    }
    if (u_debug_mode == 3) {
        frag_colour = vec4(vec3(float(result.steps) / float(u_max_steps)), 1.0);
        return;
    }

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
            float lit = mix(0.45, 1.0, floor_shadow(point, light));
            vec3 base = mix(u_background_colour, vec3(1.0), 0.05);
            vec3 lined = base + vec3(0.06) * minor + vec3(0.10) * major;
            frag_colour = vec4(
                mix(u_background_colour, lined * lit, fade), 1.0
            );
            return;
        }
    }

    if (!result.hit) {
        frag_colour = vec4(u_background_colour, 1.0);
        return;
    }

    vec3 position = origin + direction * result.t;
    vec3 normal = sd_scene_normal(position, u_normal_epsilon);

    if (u_debug_mode == 1) {
        frag_colour = vec4(normal * 0.5 + 0.5, 1.0);
        return;
    }

    // The hard min has no blend weight to interpolate colour by, so the hit
    // belongs entirely to whichever field is closer at the landing point —
    // the same comparison min() made when the ray stopped. Absent terrain sits
    // at SDF_FAR and can never win, so the creature keeps its colour.
    vec3 albedo = sd_terrain(position) < sd_field(position)
        ? u_terrain_colour
        : u_surface_colour;

    vec3 light = normalize(u_light_direction);
    float lambert = max(dot(normal, light), 0.0);
    // A little bounce from below keeps the underside from going flat black,
    // which on a blobby silhouette reads as a hole rather than as shadow.
    float bounce = max(dot(normal, -light), 0.0) * 0.2;
    float rim = pow(1.0 - max(dot(normal, -direction), 0.0), 3.0) * 0.35;

    vec3 shaded = albedo * (0.25 + 0.75 * lambert + bounce) + rim;
    frag_colour = vec4(shaded, 1.0);
}
