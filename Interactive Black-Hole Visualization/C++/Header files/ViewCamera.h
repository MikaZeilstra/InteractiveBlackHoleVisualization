#pragma once
#include <glm/matrix.hpp>

#define GLFW_INCLUDE_NONE
#include <glfw3.h>
#include "../../CUDA/Header files/Constants.cuh"
#include "../Header files/Parameters.h"

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void mouse_cursor_callback(GLFWwindow* window, double xpos, double ypos);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

class ViewCamera {
public:
    ViewCamera(GLFWwindow* pWindow);
    ViewCamera(Parameters* param, glm::vec3 CameraDirection, glm::vec3 upDirection);

    GLFWwindow* get_window();

    glm::dvec2 previous_mouse_pos = {};

    //Direction of camera in spherical coordinates (R, Theta phi)
    glm::vec3 m_CameraDirection;
    glm::vec3 m_UpDirection;
    
    GLFWwindow* m_pWindow;
    glm::dvec2 m_prevCursorPos{ 0 };

    float m_CameraFov = 70;
    int screen_width = 0;
    int screen_height = 0;

    bool mouse_pressed = false;

    Parameters* m_param = nullptr;

    /// <summary>
    /// Returns the position of the camera
    /// </summary>
    /// <param name="angle">The number of the grid we want the position for. Ignored if movementmode != 1</param>
    double3 getCameraPos(int grid_nr);

    void updateInput();
    void rotateYaw(float angle);
    void rotatePitch(float angle);
    void rotateRoll(float angle);

    /// <summary>
    /// Updates camera Phi position does noting if movement mode != 2
    /// </summary>
    /// <param name="phiChange"></param>
    void movePhi(float phiChange);

    void set_window(GLFWwindow* window);

private:
    double3 world_pos;
    
};