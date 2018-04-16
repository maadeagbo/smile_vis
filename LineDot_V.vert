#version 430

layout (location = 0) in vec3 VertexPosition;
layout (location = 2) in vec2 VertexCoords;

uniform mat4 MVP;
uniform bool render_to_tex;
uniform bool send_to_back;
out vec2 out_uv;

void main() {
    gl_Position = MVP * vec4(VertexPosition, 1.0);
    if (render_to_tex) {
        out_uv = VertexCoords;
    }
    if (send_to_back) {
        gl_Position.z = gl_Position.w - 0.00001;
    }
}