#version 430

layout( location = 0 ) out vec4 FragColor;
layout( location = 1 ) out vec4 OutColor;
layout( location = 2 ) out vec4 OutColor2;

uniform vec4 color;

void main() {
    //OutColor = texture(Tex01, out_uv);
    OutColor2 = color;
}