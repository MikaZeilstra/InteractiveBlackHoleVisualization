#version 330 core

uniform mat4 view;
uniform mat4 proj;

layout (location = 0) in vec3 aPos;

smooth out vec2 texel;


void main(){
   gl_Position = proj * view *  vec4(aPos, 1.0);
   vec3 vertex_pos = gl_Position.xyz / gl_Position.w;

   texel = aPos.xy;
}