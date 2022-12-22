#version 330 core

uniform mat4 view;
uniform mat4 proj;

layout (location = 0) in vec3 aPos;



void main(){
   gl_Position = proj * view *  vec4(aPos, 1.0);
   vec3 vertex_pos = gl_Position.xyz / gl_Position.w;

   vec4 BlackHole_left = vec4(1,0,0,1);
   vec4 BlackHoleRadiusProjected = proj * view * BlackHole_left;

   vec3 blackHole_radius = vertex_pos - BlackHoleRadiusProjected.xyz/BlackHoleRadiusProjected.w;

   gl_PointSize = 20;
}