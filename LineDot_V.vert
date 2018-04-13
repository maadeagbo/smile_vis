#version 430

layout (location = 0) in vec3 VertexPosition;
layout (location = 2) in vec2 VertexCoordinates;

uniform mat4 MVP;
uniform bool render_to_tex;
out vec2 out_uv;

void main() {
    gl_Position = MVP * vec4(VertexPosition, 1.0);
    if (render_to_tex) {
        out_uv = VertexCoordinates;
    }
}