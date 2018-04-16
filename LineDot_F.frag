#version 430

layout( location = 0 ) out vec4 FragColor;
layout( location = 1 ) out vec4 OutColor;
layout( location = 2 ) out vec4 OutColor2;

uniform vec4 color;
uniform bool render_to_tex;
uniform sampler2D bound_tex;

in vec2 out_uv;

void main() {    
    if (!render_to_tex) {
        OutColor2 = color;
    } else {
        OutColor = texture(bound_tex, out_uv);
        //OutColor = vec4(1.0);
    }
}