#version 330 core
out vec4 FragColor;

in vec2 texel;

void main()
{
    if(dot(texel.xy,texel.xy) > 1){
        discard;
    }


    FragColor = vec4(0,0, 0.0f, 1.0f);
}