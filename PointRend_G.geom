#version 430

layout( points ) in;
layout( triangle_strip, max_vertices = 4 ) out;

uniform float quad_h_width = 0.5;  // half width of quad
uniform mat4 Proj;

void main() {
  float W = quad_h_width;
	float H = quad_h_width;

  // points generated in triangle strip order
	// first point 
	gl_Position = Proj * (vec4(-W, -H, 0.0, 0.0) + gl_in[0].gl_Position);
	EmitVertex();
	// second point 
	gl_Position = Proj * (vec4(W, -H, 0.0, 0.0) + gl_in[0].gl_Position);
	EmitVertex();
	// third point 
	gl_Position = Proj * (vec4(-W, H, 0.0, 0.0) + gl_in[0].gl_Position);
	EmitVertex();
	// fourth point 
	gl_Position = Proj * (vec4(W, H, 0.0, 0.0) + gl_in[0].gl_Position);
	EmitVertex();

  EndPrimitive();
}