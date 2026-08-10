// The metaball field, kept line-for-line with mesh_xml/core/distance.py
// (the primitive math), mesh_xml/render/field.py (the creature fold) and
// mesh_xml/core/terrain.py (the terrain fold).
//
// That correspondence is the whole point of having a CPU reference at all: it
// is only a check on this shader while the two compute the same thing. Any edit
// here needs the matching edit there, and tests/test_metaball.py compares them
// numerically on real rigs so a divergence fails rather than merely looks odd.
//
// Primitives arrive as an RGBA32F texture, two texels per primitive, matching
// the [ax,ay,az,ra, bx,by,bz,rb] stride-8 record that sdf.pack emits. The
// radii are interleaved precisely so each half is one texel fetch. A texture
// rather than a uniform array because 3.3 core has no shader storage buffers,
// and rather than a texture buffer because a plain sampler2D needs no extension
// and every 3.3 GPU allows at least 8192 rows.
//
// No #version here; included units splice in below the root shader's.

uniform sampler2D u_primitives;
uniform int   u_primitive_count;
uniform float u_blend;

// Squared, to compare against a squared axis length without a square root.
// Mirrors sdf.DEGENERATE_EPSILON.
const float SDF_DEGENERATE_EPSILON2 = 1e-12;

const float SDF_FAR = 1e20;

// Inigo Quilez's sdRoundCone, transcribed branch for branch including the l2
// scaling that lets all three cases share one square root.
//
// Two configurations have no cone and are answered with a sphere. Coincident
// ends, which pack emits for every unlinked chunk. And one end ball containing
// the other, |ra - rb| >= |b - a|, where the discriminant a2 goes negative --
// a big chunk linked to a small close one produces it in ordinary rigs, so
// guarding only the first branch renders NaN on real input.
float sd_round_cone(vec3 p, vec3 a, float ra, vec3 b, float rb) {
    vec3  ba = b - a;
    float l2 = dot(ba, ba);
    float rr = ra - rb;
    float a2 = l2 - rr * rr;

    if (l2 < SDF_DEGENERATE_EPSILON2 || a2 <= 0.0) {
        vec3  centre = (ra >= rb) ? a : b;
        float radius = max(ra, rb);
        return length(p - centre) - radius;
    }

    float il2 = 1.0 / l2;
    vec3  pa = p - a;
    float y = dot(pa, ba);
    float z = y - l2;

    vec3  perpendicular = pa * l2 - ba * y;
    float x2 = dot(perpendicular, perpendicular);
    float y2 = y * y * l2;
    float z2 = z * z * l2;

    float split = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > split) return sqrt(x2 + z2) * il2 - rb;
    if (sign(y) * a2 * y2 < split) return sqrt(x2 + y2) * il2 - ra;
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - ra;
}

// Polynomial smooth minimum. Its gradient is h*grad(a) + (1-h)*grad(b), a
// convex combination, so a field folded from exact primitives stays 1-Lipschitz
// and sphere tracing may step by the full field value without tunnelling.
float sdf_smooth_min(float a, float b, float k) {
    if (k <= 0.0) return min(a, b);
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// The same minimum, also reporting h — the convex weight of `a`. What the
// albedo fold blends colour by (docs/TEXTURES.md): one h, two folds, so the
// skin cannot disagree with the flesh about where a blend is. At k == 0 the
// weight is the step (a <= b) — ties go to the accumulated side — matching
// core/distance.py's smooth_min_weight exactly. The plain variant above
// stays untouched because the march calls it and owes nothing to colour.
float sdf_smooth_min_h(float a, float b, float k, out float h) {
    if (k <= 0.0) { h = (a <= b) ? 1.0 : 0.0; return min(a, b); }
    h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float sd_primitive(vec3 p, int index) {
    vec4 head = texelFetch(u_primitives, ivec2(0, index), 0);
    vec4 tail = texelFetch(u_primitives, ivec2(1, index), 0);
    return sd_round_cone(p, head.xyz, head.w, tail.xyz, tail.w);
}

// Blend groups: one creature per contiguous row range of the primitive
// texture, described by sdf.group_table's stride-8 rows on a third unit —
// texel one [start, count, 0, 0], texel two [r, g, b, 0]. Smooth minimum
// *within* a group, hard minimum *across* groups: two creatures touching are
// two skins, not one weld. With no groups uploaded (count 0) the fold below
// falls back to the single global chain, bit-exact with what it always was —
// which is what keeps the CPU-oracle comparisons running unchanged.
uniform sampler2D u_groups;
uniform int   u_group_count;

// One group's fold: array order, seeded with its first primitive — the same
// two contracts as the global fold, scoped to the group's rows. An empty
// group answers SDF_FAR so the hard min ignores it.
float sd_group(vec3 p, int group) {
    vec4 span = texelFetch(u_groups, ivec2(0, group), 0);
    int start = int(span.x);
    int last = start + int(span.y);
    if (last <= start) return SDF_FAR;
    float d = sd_primitive(p, start);
    for (int i = start + 1; i < last; ++i) {
        d = sdf_smooth_min(d, sd_primitive(p, i), u_blend);
    }
    return d;
}

// Per-group bounding spheres, [cx, cy, cz, r] in texel one of each stride-8
// row, recomputed by the CPU every frame the creatures move. r already covers
// the smooth-min inflation, so |p - c| - r is a true lower bound on the
// group's field, and a group whose lower bound cannot beat the running
// minimum is skipped without evaluating its fold — the skip changes nothing,
// not even a ulp, because the survivors' arithmetic is untouched. Culling
// arms only when the bounds table matches the group table row for row; a
// view that never uploads bounds folds every group, exactly as before.
uniform sampler2D u_group_bounds;
uniform int   u_group_bounds_count;

float sd_group_lower_bound(vec3 p, int group) {
    vec4 sphere = texelFetch(u_group_bounds, ivec2(0, group), 0);
    return length(p - sphere.xyz) - sphere.w;
}

// The broad phase: per-screen-tile group bitmasks built on the CPU
// (mesh_xml/render/tiles.py owns the geometry and the exactness contract —
// surface-exact, not byte-exact, and why that is fundamental). The root
// shader arms sdf_tile_mask from gl_FragCoord before tracing; the shadow
// march stands the mask down, because shadow rays leave the camera ray's
// cone and their occluders are not this tile's business. A fragment that
// never arms the mask folds every group, which is the always-correct path
// every test that wants bit-exactness renders through.
uniform usampler2D u_tiles;
uniform int u_tiles_on;
uniform int u_tile_size;

uvec4 sdf_tile_mask = uvec4(0xFFFFFFFFu);
bool  sdf_tiled = false;

void sdf_arm_tiles(vec2 frag_coord) {
    sdf_tiled = u_tiles_on != 0
        && u_tile_size > 0
        && u_group_count > 0
        && u_group_bounds_count == u_group_count;
    if (!sdf_tiled) return;
    ivec2 tile = ivec2(frag_coord) / u_tile_size;
    ivec2 grid = textureSize(u_tiles, 0);
    tile = clamp(tile, ivec2(0), grid - ivec2(1));
    sdf_tile_mask = texelFetch(u_tiles, tile, 0);
}

bool sdf_tile_has(int group) {
    return !sdf_tiled
        || (sdf_tile_mask[group >> 5] & (1u << uint(group & 31))) != 0u;
}

// Folded in array order and seeded with the first primitive, both of which are
// contracts rather than incidental: smooth_min is not associative for k > 0, so
// a different order is a different surface. Seeding with a large constant
// instead would blend against it and inflate the whole field. Across groups
// the hard min carries no such contract — it is associative.
float sd_field(vec3 p) {
    if (u_primitive_count == 0) return SDF_FAR;
    if (u_group_count <= 0) {
        float d = sd_primitive(p, 0);
        for (int i = 1; i < u_primitive_count; ++i) {
            d = sdf_smooth_min(d, sd_primitive(p, i), u_blend);
        }
        return d;
    }
    if (u_group_bounds_count != u_group_count) {
        float d = SDF_FAR;
        for (int g = 0; g < u_group_count; ++g) {
            d = min(d, sd_group(p, g));
        }
        return d;
    }

    // Nearest bound folds first, so the running minimum is tight before any
    // other group is considered and nearly everything else culls. Folding in
    // table order instead leaves the minimum loose while roughly ln(G)
    // record-setting groups fold — measured as most of the cull's win left
    // on the table. Order is free to choose: the cross-group hard minimum is
    // exact float min, associative and commutative, so any order is the same
    // image bit for bit.
    int first = -1;
    float nearest = SDF_FAR;
    for (int g = 0; g < u_group_count; ++g) {
        if (!sdf_tile_has(g)) continue;
        float lower = sd_group_lower_bound(p, g);
        if (lower < nearest) {
            nearest = lower;
            first = g;
        }
    }
    if (first < 0) return SDF_FAR;
    float d = sd_group(p, first);
    for (int g = 0; g < u_group_count; ++g) {
        if (g == first || !sdf_tile_has(g)) continue;
        if (sd_group_lower_bound(p, g) >= d) continue;
        d = min(d, sd_group(p, g));
    }
    return d;
}

// Which group's skin a point is nearest — what the shading looks up a colour
// by. Only meaningful at a hit, where exactly one group is at the surface.
// The same cull applies: a group that cannot beat the best cannot win it.
int sd_nearest_group(vec3 p) {
    bool cull = u_group_bounds_count == u_group_count;
    int first = 0;
    if (cull) {
        float nearest = SDF_FAR;
        for (int g = 0; g < u_group_count; ++g) {
            if (!sdf_tile_has(g)) continue;
            float lower = sd_group_lower_bound(p, g);
            if (lower < nearest) {
                nearest = lower;
                first = g;
            }
        }
    }
    int winner = first;
    float best = sd_group(p, first);
    for (int g = 0; g < u_group_count; ++g) {
        if (g == first) continue;
        if (!sdf_tile_has(g)) continue;
        if (cull && sd_group_lower_bound(p, g) >= best) continue;
        float d = sd_group(p, g);
        if (d < best) {
            best = d;
            winner = g;
        }
    }
    return winner;
}

vec3 sd_group_colour(int group) {
    return texelFetch(u_groups, ivec2(1, group), 0).rgb;
}

// Per-primitive end colours: the skin's counterpart of the primitive texture
// (docs/TEXTURES.md), one stride-8 row per primitive row on its own unit —
// [ra_r, ra_g, ra_b, 0][rb_r, rb_g, rb_b, 0], the fourth lanes reserved for
// the pattern vocabulary. Armed only when the count matches the primitive
// count; sdf.pack_attributes packs an EMPTY table for a body whose chunks
// all agree with their group colour, so an unpainted document shades by the
// flat group colour on the exact code path it always ran — byte-identity by
// construction, not by numerical luck. Colour folds once at the hit, never
// in the march: every march acceleration keeps its exactness contract
// because the thing it protects never reads this table.
uniform sampler2D u_attributes;
uniform int   u_attribute_count;

// One primitive's colour at p: end colours lerped by the orthogonal
// projection fraction along the primitive's own segment — deliberately not
// the cone maths' three-case parameter, because the projection is smooth,
// branchless, cheap to mirror in numpy, and equal to it wherever the two
// could visibly disagree. Degenerate records read their `a` end. CPU twin:
// field._primitive_albedo.
vec3 sd_primitive_albedo(vec3 p, int index) {
    vec4 head = texelFetch(u_primitives, ivec2(0, index), 0);
    vec4 tail = texelFetch(u_primitives, ivec2(1, index), 0);
    vec3 ca = texelFetch(u_attributes, ivec2(0, index), 0).rgb;
    vec3 cb = texelFetch(u_attributes, ivec2(1, index), 0).rgb;
    vec3 ba = tail.xyz - head.xyz;
    float l2 = dot(ba, ba);
    if (l2 < SDF_DEGENERATE_EPSILON2) return ca;
    float t = clamp(dot(p - head.xyz, ba) / l2, 0.0, 1.0);
    return mix(ca, cb, t);
}

// One group's skin at p: the distance fold re-run carrying colour beside it,
// blended by the h each smooth-min step already computed. Same seed, same
// array order — those are sd_group's contracts and colour inherits them.
// CPU twin: field.albedo.
vec3 sd_group_albedo(vec3 p, int group) {
    vec4 span = texelFetch(u_groups, ivec2(0, group), 0);
    int start = int(span.x);
    int last = start + int(span.y);
    if (last <= start) return vec3(0.0);
    float d = sd_primitive(p, start);
    vec3 colour = sd_primitive_albedo(p, start);
    for (int i = start + 1; i < last; ++i) {
        float h;
        d = sdf_smooth_min_h(d, sd_primitive(p, i), u_blend, h);
        colour = mix(sd_primitive_albedo(p, i), colour, h);
    }
    return colour;
}

// Central differences on a tetrahedron: four field taps instead of the six a
// per-axis difference needs.
vec3 sd_normal(vec3 p, float eps) {
    const vec2 e = vec2(1.0, -1.0);
    vec3 gradient = e.xyy * sd_field(p + e.xyy * eps)
                  + e.yyx * sd_field(p + e.yyx * eps)
                  + e.yxy * sd_field(p + e.yxy * eps)
                  + e.xxx * sd_field(p + e.xxx * eps);
    float len = length(gradient);
    return len > 0.0 ? gradient / len : vec3(0.0);
}

// ---------------------------------------------------------------------------
// Terrain: the world as a second field (docs/TERRAIN.md), kept line-for-line
// with Terrain.distance in mesh_xml/core/terrain.py.
//
// One texture holds both vocabularies, planes first: row i < plane count is a
// half-space whose first texel is [nx, ny, nz, offset] (the second texel is
// padding), and every later row is the same two-texel stride-8 round cone
// record the creature uses. Packing planes at stride 8 wastes 16 bytes per
// plane and buys the whole upload path: the texture is byte-compatible with
// the primitive texture above, so metaball.pack_terrain feeds the identical
// allocate/subimage machinery, and the row index *is* the fold index.
uniform sampler2D u_terrain;
uniform int   u_terrain_plane_count;
uniform int   u_terrain_cone_count;
uniform float u_terrain_blend;

float sd_terrain_primitive(vec3 p, int index) {
    vec4 head = texelFetch(u_terrain, ivec2(0, index), 0);
    if (index < u_terrain_plane_count) {
        // Half-space: dot(n, p) - offset with n unit length, which is what
        // makes offset a position along the normal and the distance metric.
        // CPU twin: the plane loop of Terrain.distance; unit length is
        // enforced by Terrain.add_plane, so no normalize here.
        return dot(p, head.xyz) - head.w;
    }
    vec4 tail = texelFetch(u_terrain, ivec2(1, index), 0);
    return sd_round_cone(p, head.xyz, head.w, tail.xyz, tail.w);
}

// The same two fold contracts as sd_field, against Terrain.distance's
// documented order: planes in authoring order, then cones in authoring order,
// seeded with the first primitive rather than a sentinel. The packed rows are
// already in that order, so one loop over rows is the CPU fold verbatim.
float sd_terrain(vec3 p) {
    int count = u_terrain_plane_count + u_terrain_cone_count;
    if (count == 0) return SDF_FAR;
    float d = sd_terrain_primitive(p, 0);
    for (int i = 1; i < count; ++i) {
        d = sdf_smooth_min(d, sd_terrain_primitive(p, i), u_terrain_blend);
    }
    return d;
}

// Creature against world is a *hard* minimum, never the smooth one: a foot
// rests on the ground, it does not blend into it and become it. Each field
// keeps its own blend radius strictly inside itself. With no terrain uploaded
// sd_terrain is SDF_FAR and min() returns the creature field bit-for-bit, so
// the terrain-free path is this one, unchanged.
float sd_scene(vec3 p) {
    return min(sd_field(p), sd_terrain(p));
}

// sd_normal on the composite. The hard min's gradient is the winning field's
// gradient, which is exactly what a foot pressed onto the floor should shade
// by; where terrain is absent every tap collapses to sd_field and this is
// sd_normal exactly.
vec3 sd_scene_normal(vec3 p, float eps) {
    const vec2 e = vec2(1.0, -1.0);
    vec3 gradient = e.xyy * sd_scene(p + e.xyy * eps)
                  + e.yyx * sd_scene(p + e.yyx * eps)
                  + e.yxy * sd_scene(p + e.yxy * eps)
                  + e.xxx * sd_scene(p + e.xxx * eps);
    float len = length(gradient);
    return len > 0.0 ? gradient / len : vec3(0.0);
}
