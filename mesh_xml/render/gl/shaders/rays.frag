#version 330 core

// Placeholder until the raymarcher lands. It paints the primary ray direction,
// which makes the scaffolding self-checking: orbit the view and the colours
// have to sweep smoothly and consistently. A transposed matrix, a flipped row
// order, or an aspect applied to the wrong axis all show up here immediately
// rather than as a subtly wrong isosurface later.

#include "camera.glsl"

in vec2 v_uv;
out vec4 frag_colour;

void main() {
    frag_colour = vec4(camera_ray(v_uv) * 0.5 + 0.5, 1.0);
}
