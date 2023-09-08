#version 330 core

uniform mat4 view;
uniform mat4 proj;

layout (location = 0) in vec3 aPos;
layout (location = 1) in float vertex_id;

out vec2 texel;
out float vert_id;


void main(){
   gl_Position = proj * view *  vec4(aPos, 1.0);
   vec3 vertex_pos = gl_Position.xyz / gl_Position.w;

   texel = aPos.xy;
   vert_id = vertex_id;
}