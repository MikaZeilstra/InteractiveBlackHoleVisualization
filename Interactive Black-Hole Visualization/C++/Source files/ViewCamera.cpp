#pragma once
#include "../Header files/ViewCamera.h"
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtx/ma>

#define MOUSE_SENSITIVITY 0.01

ViewCamera::ViewCamera(GLFWwindow* pWindow)
    : m_pWindow(pWindow)
{}

ViewCamera::ViewCamera(GLFWwindow* pWindow, glm::vec3 CameraDirection,int window_width, int window_height, float Fov) {
    m_pWindow = pWindow;
    m_CameraDirection = CameraDirection;
    m_CameraFov = m_CameraFov;
    inv_project_matrix = glm::inverse(glm::perspective(Fov, window_width / (float)window_height, 1.0f, 10.0f));

}

GLFWwindow* ViewCamera::get_window() {
    return m_pWindow;
}

void ViewCamera::updateInput() {

    int state = glfwGetMouseButton(m_pWindow, GLFW_MOUSE_BUTTON_LEFT);

}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    ViewCamera* camera = (ViewCamera*)glfwGetWindowUserPointer(window);

    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {

        glfwGetCursorPos(window, &(camera->m_prevCursorPos.x), &(camera->m_prevCursorPos.y));
        camera->mouse_pressed = true;
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) {
        camera->mouse_pressed = false;
    }
}

void mouse_cursor_callback(GLFWwindow* window, double xpos, double ypos)
{
    ViewCamera* camera = (ViewCamera*)glfwGetWindowUserPointer(window);
    if (camera != nullptr) {
        //If the mouse button is down we allow the user to pan the image by dragging
        if (camera->mouse_pressed)
        {
            camera->m_CameraDirection.z += MOUSE_SENSITIVITY * (xpos - camera->previous_mouse_pos.x);
            camera->m_CameraDirection.y += MOUSE_SENSITIVITY * (ypos - camera->previous_mouse_pos.y);
        }

        camera->previous_mouse_pos = { xpos,ypos };
    }
}