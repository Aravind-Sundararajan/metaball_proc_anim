// The metaball field, kept line-for-line with mesh_xml/render/field.py.
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

float sd_primitive(vec3 p, int index) {
    vec4 head = texelFetch(u_primitives, ivec2(0, index), 0);
    vec4 tail = texelFetch(u_primitives, ivec2(1, index), 0);
    return sd_round_cone(p, head.xyz, head.w, tail.xyz, tail.w);
}

// Folded in array order and seeded with the first primitive, both of which are
// contracts rather than incidental: smooth_min is not associative for k > 0, so
// a different order is a different surface. Seeding with a large constant
// instead would blend against it and inflate the whole field.
float sd_field(vec3 p) {
    if (u_primitive_count == 0) return SDF_FAR;
    float d = sd_primitive(p, 0);
    for (int i = 1; i < u_primitive_count; ++i) {
        d = sdf_smooth_min(d, sd_primitive(p, i), u_blend);
    }
    return d;
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
