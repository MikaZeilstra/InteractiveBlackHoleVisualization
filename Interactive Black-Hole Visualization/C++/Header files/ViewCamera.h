#pragma once
#include <glm/matrix.hpp>

#define GLFW_INCLUDE_NONE
#include <glfw3.h>
#include "../../CUDA/Header files/Constants.cuh"

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void mouse_cursor_callback(GLFWwindow* window, double xpos, double ypos);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

class ViewCamera {
public:

    ViewCamera(GLFWwindow* pWindow);
    ViewCamera(GLFWwindow* pWindow, glm::vec3 CameraDirection, glm::vec3 up_direction, int window_width, int window_height, float Fov);

    GLFWwindow* get_window();

    glm::dvec2 previous_mouse_pos = {};

    //Direction of camera in spherical coordinates (R, Theta phi)
    glm::vec3 m_CameraDirection;
    glm::vec3 m_UpDirection;
    
    GLFWwindow* m_pWindow;
    glm::dvec2 m_prevCursorPos{ 0 };
    
    glm::mat4 inv_project_matrix;

    float m_CameraFov = {};

    bool mouse_pressed = false;

    void updateInput();
    void rotateYaw(float angle);
    void rotatePitch(float angle);
    void rotateRoll(float angle);

private:
  
    
};