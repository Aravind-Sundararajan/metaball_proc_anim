// Primary-ray construction, shared by every shader the viewport runs.
//
// This is the GPU half of OrbitCamera.ray_directions. The two must produce the
// same ray for the same pixel, because the CPU reference field is only a useful
// check on the raymarcher if both are looking through the same lens. Change one
// and change the other.
//
// v_uv runs 0..1 across the viewport with y up, matching gl_FragCoord and
// glReadPixels, and equals gl_FragCoord.xy / u_resolution. Its interpolation
// lands on fragment centres, which is the same half-pixel offset the CPU side
// samples at.
//
// No #version here. Included units are spliced in below the root shader's
// #version line, which must remain the first thing the compiler sees.

uniform vec2  u_resolution;
uniform vec3  u_camera_position;
uniform vec3  u_camera_right;
uniform vec3  u_camera_up;
uniform vec3  u_camera_forward;
uniform float u_tan_half_fov;
uniform float u_aspect;
uniform float u_time;
uniform int   u_orthographic;
uniform float u_ortho_height;

// The two projections put the per-pixel variation in different places:
// perspective fans the directions from one eye, orthographic tiles the origins
// across a plane and shares one direction. A shader therefore has to ask for
// both parts of the ray rather than assuming the eye is fixed.

vec3 camera_ray(vec2 uv) {
    if (u_orthographic != 0) return normalize(u_camera_forward);
    vec2 ndc = uv * 2.0 - 1.0;
    return normalize(u_camera_forward
                     + u_camera_right * (ndc.x * u_aspect * u_tan_half_fov)
                     + u_camera_up * (ndc.y * u_tan_half_fov));
}

vec3 camera_origin(vec2 uv) {
    if (u_orthographic == 0) return u_camera_position;
    vec2 ndc = uv * 2.0 - 1.0;
    float half_height = u_ortho_height * 0.5;
    return u_camera_position
           + u_camera_right * (ndc.x * u_aspect * half_height)
           + u_camera_up * (ndc.y * half_height);
}
