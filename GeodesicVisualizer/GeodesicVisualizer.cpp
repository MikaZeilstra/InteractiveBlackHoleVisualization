#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "glad.h"
#include "glfw3.h"

#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/mat4x4.hpp"

#include "../../InteractiveBlackHoleVisualization/Interactive Black-Hole Visualization/C++/Header files/IntegrationDefines.h"

#include "GeodesicVisualizer.h"
#include "camera.h"

#define GEODESICS_FILE "../Results/Geodesics/Grid_1_to_10_Spin_0.999_Rad_3_Inc_0.6pi_0.geo"

std::vector<float> geodesics;

GLuint VAO;
GLuint VBO;

GLuint BHVAO;
GLuint BHVBO;

std::vector<int> vertex_id;
std::vector<float> vertices;
std::vector<int> vertex_counts;
std::vector<int> vertex_starts;



std::string readFile(const char* filePath) {
    std::string content;
    std::ifstream fileStream(filePath, std::ios::in);


    if (!fileStream.is_open()) {
        std::cerr << "Could not read file " << filePath << ". File does not exist." << std::endl;
        return "";
    }

    std::string line = "";
    while (!fileStream.eof()) {
        std::getline(fileStream, line);
        content.append(line + "\n");
    }

    fileStream.close();
    return content;
}



int main()
{
	//Setup GLFW
	glfwInit();
	GLFWwindow* window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "LearnOpenGL", NULL, NULL);
    Camera camera(window);


	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	//Setup viewport after loading glad
	glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);

	//Turn off vsync
	glfwSwapInterval(0);

	//Setup opengl
	gladLoadGL();
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
	//Setup initial screen
	glfwSwapBuffers(window);


    cv::FileStorage fs(GEODESICS_FILE, cv::FileStorage::READ);

    fs["paths"] >> geodesics;

    fs.release();

    // Create Vertex Array Object
    
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    
    glLineWidth(5.0f);

    //Transfer point to VBO and transform them to cartesian coordinates
    int total_vertex_count = 0;
    for (int geo_index = 0; geo_index < geodesics.size()/ (MAXSTP / STEP_SAVE_INTERVAL); geo_index ++) {
        int step = 0;
        //Save the start of each new geodesic
        vertex_starts.push_back(total_vertex_count);
        //Continue pushing points untill we reached the maximum steps or all coordinates are zero
        while(step < MAXSTP * 3 && !(geodesics[geo_index + step] == 0 && geodesics[geo_index + step + 1] == 0 && geodesics[geo_index + step + 2] == 0)) {
            float theta = geodesics[geo_index * (MAXSTP / STEP_SAVE_INTERVAL) + step];
            float phi = geodesics[geo_index * (MAXSTP / STEP_SAVE_INTERVAL) + step + 1];
            float r = geodesics[geo_index * (MAXSTP / STEP_SAVE_INTERVAL) + step + 2];

            

            //Transform the coordinates to cartesian and save them
            vertices.push_back(r * sin(theta) * cos(phi));
            vertices.push_back(r * sin(theta) * sin(phi));
            vertices.push_back(r * cos(theta));

            vertices.push_back(step);

            //Go to the next step
            step += 3;
            total_vertex_count++;
        }
        //Save the amount of vertices in each geodesic
        vertex_counts.push_back(step/3);
    }
    //Upload buffer data and set VAO properties
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * total_vertex_count, vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(4 * sizeof(float)));
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);


  
    //Black hole quad
   

    std::vector<glm::vec3> vertices;

    vertices.resize(6);
    // first triangle
    vertices[0] = glm::vec3(-1, -1, 0.0f);
    vertices[1] = glm::vec3(1 ,-1, 0.0f); 
    vertices[2] = glm::vec3(-1, 1, 0.0f); 
    // second triangle
    vertices[3] = glm::vec3(-1, 1, 0.0f);
    vertices[4] = glm::vec3(1, 1, 0.0f); 
    vertices[5] = glm::vec3(1,-1, 0.0f); 

    GLuint bhvao;
    GLuint bhvbo;

    glGenVertexArrays(1, &bhvao);
    glBindVertexArray(bhvao);

    glGenBuffers(1, &bhvbo);
    glBindBuffer(GL_ARRAY_BUFFER, bhvbo);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);


    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);

    glBindVertexArray(0);



    //Compile vertexShader for geodesics
    std::string vertexShaderText = readFile("./VertexShader.vert");
    const char* vertexPointer = vertexShaderText.c_str();
    GLuint vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexPointer, NULL);
    glCompileShader(vertexShader);

    //Compile fragment shader for geodesics
    std::string fragShaderText = readFile("./FragmentShader.frag");
    const char* fragPointer = fragShaderText.c_str();
    GLuint fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragPointer, NULL);
    glCompileShader(fragmentShader);

    //Compile fragment shader for BH
    std::string BHfragShaderText = readFile("./BHFragmentShader.frag");
    const char* BHfragPointer = BHfragShaderText.c_str();
    GLuint BHfragmentShader;
    BHfragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(BHfragmentShader, 1, &BHfragPointer, NULL);
    glCompileShader(BHfragmentShader);

    //Create shader program for geodesics
    GLuint shaderProgram;
    shaderProgram = glCreateProgram();

    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    //Create shader program for BH
    GLuint BHshaderProgram;
    BHshaderProgram = glCreateProgram();

    glAttachShader(BHshaderProgram, vertexShader);
    glAttachShader(BHshaderProgram, BHfragmentShader);
    glLinkProgram(BHshaderProgram);
    

    glm::mat4 proj = glm::perspective(glm::radians(80.0f), 1.0f, 0.1f, 100.0f);

    GLuint  uniProj;
    GLuint  uniView;
	while (!glfwWindowShouldClose(window))
	{
        //Reset screen
        glClearColor(0.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


        //Draw Black hole
        glBindVertexArray(BHVAO);
        glBindBuffer(GL_ARRAY_BUFFER, BHVBO);

        glUseProgram(BHshaderProgram);

        uniProj = glGetUniformLocation(shaderProgram, "proj");
        glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));
        uniView = glGetUniformLocation(shaderProgram, "view");
        glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(camera.BHviewMatrix()));

        glBindVertexArray(bhvao);

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        glBindVertexArray(0);

 
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 2);

        //Draw geodesics
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        glUseProgram(shaderProgram);

        uniProj = glGetUniformLocation(shaderProgram, "proj");
        glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));

        uniView = glGetUniformLocation(shaderProgram, "view");
        glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(camera.viewMatrix()));

        glMultiDrawArrays(GL_LINE_STRIP, vertex_starts.data(), vertex_counts.data(), vertex_starts.size());

        glUseProgram(BHshaderProgram);

     
        



		glfwSwapBuffers(window);
		glfwPollEvents();
        camera.updateInput();
	}

	glfwTerminate();
}
