#version 330 core
out vec4 FragColor;


void main()
{
    vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
    if (dot(circCoord, circCoord) > 1.0) {
        discard;
    }

    FragColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
} 