#version 430

layout( location = 0 ) out vec4 FragColor;
layout( location = 1 ) out vec4 OutColor;

uniform vec4 color;
uniform bool render_to_tex;
uniform sampler2D bound_tex;

in vec2 out_uv;

void main() {    
    if (render_to_tex) {
        OutColor = texture(bound_tex, out_uv);
    } else {
        OutColor = color;
    }
}