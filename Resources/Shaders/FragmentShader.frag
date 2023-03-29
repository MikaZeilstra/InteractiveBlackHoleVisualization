#version 330 core
out vec4 FragColor;

in vec4 gl_FragCoord;

uniform sampler2D SkyTexture;

void main()
{
    vec3 tex_val = texture(SkyTexture, (gl_FragCoord.xy + 1)/2).xyz;
    FragColor = vec4(tex_val, 1.0f);
} 