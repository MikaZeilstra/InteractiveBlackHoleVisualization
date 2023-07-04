#version 430 core

#define PI 3.1415926538

out vec4 FragColor;

in vec4 gl_FragCoord;

uniform sampler2D SkyTexture;

layout(location = 1) uniform int screen_width;
layout(location = 2) uniform int screen_height; 

layout(location = 3) uniform float fov;
layout(location = 4) uniform vec3 camera_direction;
layout(location = 5) uniform vec3 up_direction; 


float atan2(in float x, in float y)
{
    bool s = (abs(x) > abs(y));
    return mix(PI/2.0 - atan(x,y), atan(y,x), s);
}

void main()
{
    
    vec2  center_frag = {
        (((float(gl_FragCoord.x) / float(screen_width)) - 0.5) * 2),
        (((float(gl_FragCoord.y) / float(screen_width)) - 0.5) * 2)
    };

    vec3 right_direction = normalize(cross(up_direction,camera_direction));

    vec3 pixel_pos = normalize(camera_direction + 
        (center_frag.x * right_direction * fov)  + 
        (center_frag.y * up_direction * fov )
     ); 
    
    float theta = acos(pixel_pos.z);
    float phi = sign(pixel_pos.y) * acos(pixel_pos.x / sqrt((pixel_pos.x * pixel_pos.x) + (pixel_pos.y * pixel_pos.y)));

   // vec3 tex_val = vec3(theta/PI,phi/(2 * PI),0);//
    vec3 tex_val = texture(SkyTexture ,vec2( phi/ (2 * PI), theta / ( PI))).xyz;
    FragColor = vec4(tex_val, 1.0f);
} 

