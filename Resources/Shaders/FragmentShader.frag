#version 430 core

#define PI 3.1415926538

out vec4 FragColor;

in vec4 gl_FragCoord;

uniform sampler2D SkyTexture;

layout(location = 1) uniform int screen_width;
layout(location = 2) uniform int screen_height; 

layout(location = 3) uniform mat4 inv_project_matrix;
layout(location = 4) uniform vec3 camera_direction; 


float atan2(in float x, in float y)
{
    bool s = (abs(x) > abs(y));
    return mix(PI/2.0 - atan(x,y), atan(y,x), s);
}

void main()
{
    

    float center_frag_x = (float(gl_FragCoord.x) / float(screen_width)) - 0.5;
    float center_frag_y = (float(gl_FragCoord.y) / float(screen_height)) - 0.5;

    vec4 center_frag = vec4(center_frag_x,center_frag_y,0,1);
    vec4 world_space_pos = inv_project_matrix * center_frag;
    
    vec3 world_pos = world_space_pos.xyz / world_space_pos.w;

    normalize(world_pos);

    
    float phi_frag = atan2(center_frag_x,0.5);
    float theta_frag = -atan2(center_frag_y,0.5);

   //float phi_frag = atan2(world_pos.x,world_pos.z);
   //float theta_frag =   -atan2(world_pos.y,world_pos.z);



    float phi = camera_direction.z + phi_frag;
    float theta = mod(camera_direction.y + theta_frag , 2 * PI);
    
    //float phi = atan(center_frag_pos.x , 0.2 ) + camera_direction.z;
    //float theta = (-1 * atan( center_frag_pos.y,) + camera_direction.y;
    
    
    if(theta > PI){
        theta -= 2 * (theta-PI);
        phi += PI;
    }
    phi = mod(phi,2 * PI);
   
    
   // vec3 tex_val = vec3(phi/(2*PI), theta/(2 * PI),0);
    vec3 tex_val = texture(SkyTexture ,vec2( phi/ (2 * PI), theta / (PI))).xyz;
    FragColor = vec4(tex_val, 1.0f);
} 

