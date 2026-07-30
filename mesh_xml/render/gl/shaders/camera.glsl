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

vec3 camera_ray(vec2 uv) {
    vec2 ndc = uv * 2.0 - 1.0;
    return normalize(u_camera_forward
                     + u_camera_right * (ndc.x * u_aspect * u_tan_half_fov)
                     + u_camera_up * (ndc.y * u_tan_half_fov));
}
