#version 430

layout (location = 0) in vec3 VertexPosition;

uniform mat4 MV;
uniform mat4 Proj;

void main() {
    gl_Position = MV * vec4(VertexPosition, 1.f);
}