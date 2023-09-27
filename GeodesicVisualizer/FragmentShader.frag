#version 330 core
out vec4 FragColor;

in float vert_id;


float near = 0.1; 
float far  = 100.0; 

float LinearizeDepth(float depth) 
{
    float z = depth * 2.0 - 1.0; // back to NDC 
    return (2.0 * near * far) / (far + near - z * (far - near));	
}

void main()
{
    //float depth = LinearizeDepth(gl_FragCoord.z) / far; // divide by far for demonstration
    float depth = vert_id;
    
    FragColor = vec4(depth , 0,1-depth, 1.0); 

    if(depth > 1.1){
        FragColor = vec4(0,1,0,1);
    }

    
} 