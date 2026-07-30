#version 330 core

// The whole viewport from three vertices and no vertex buffer.
//
// gl_VertexID 0/1/2 give corners (0,0), (2,0), (0,2), which map to clip-space
// (-1,-1), (3,-1), (-1,3): one oversized triangle whose interior covers the
// screen. A quad would need two triangles, and fragments along their shared
// diagonal are rasterised into 2x2 quads by both, so that seam pays double
// shading cost and has discontinuous derivatives. For a raymarcher, where the
// fragment shader is the entire cost of the frame, that is not academic.
//
// A vertex array object must still be bound. Core profile deleted the default
// one, and glDrawArrays against object 0 is an INVALID_OPERATION that draws
// nothing and, unless the error queue is polled, says nothing.

out vec2 v_uv;

void main() {
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    v_uv = corner;
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
