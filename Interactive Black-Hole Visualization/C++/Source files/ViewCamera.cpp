#pragma once
#include "../Header files/ViewCamera.h"
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include "../Header files/Parameters.h"
//#include <glm/gtx/ma>

#define MOUSE_SENSITIVITY 0.01

ViewCamera::ViewCamera(GLFWwindow* pWindow)
    : m_pWindow(pWindow)
{}

ViewCamera::ViewCamera(Parameters* param, glm::vec3 CameraDirection, glm::vec3 upDirection) {
    m_CameraDirection = glm::normalize(CameraDirection);
    m_UpDirection = glm::normalize(upDirection);
    m_CameraFov = 1 / tan(param->fov);


    screen_width = param->windowWidth;
    screen_height = param->windowHeight;

    m_param = param;

    world_pos = {
        param->camRadiusFromTo.x,
        param->camInclinationFromTo.x,
        param->camPhiFromTo.x
    };
}

GLFWwindow* ViewCamera::get_window() {
    return m_pWindow;
}

void ViewCamera::updateInput() {

    int state = glfwGetMouseButton(m_pWindow, GLFW_MOUSE_BUTTON_LEFT);

}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    ViewCamera* camera = (ViewCamera*)glfwGetWindowUserPointer(window);

    if (key == GLFW_KEY_Q) {
        camera->rotateRoll(MOUSE_SENSITIVITY);
    }
    else if (key == GLFW_KEY_E)
    {
        camera->rotateRoll(-MOUSE_SENSITIVITY);
    }

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
            camera->rotateYaw(MOUSE_SENSITIVITY * -(xpos - camera->previous_mouse_pos.x));
            camera->rotatePitch(MOUSE_SENSITIVITY * -(ypos - camera->previous_mouse_pos.y));

        }

        camera->previous_mouse_pos = { xpos,ypos };
    }
}

void ViewCamera::rotateYaw(float angle)
{
    m_CameraDirection = glm::normalize(glm::angleAxis(angle, m_UpDirection) * m_CameraDirection);
    m_UpDirection = glm::normalize(glm::angleAxis(angle, m_UpDirection) * m_UpDirection);
}

void ViewCamera::rotatePitch(float angle)
{
    const glm::vec3 horAxis = glm::normalize(glm::cross(m_UpDirection, m_CameraDirection));

    m_CameraDirection = glm::normalize(glm::angleAxis(angle, horAxis) * m_CameraDirection);
    m_UpDirection = glm::normalize(glm::angleAxis(angle, horAxis) * m_UpDirection);
}

void ViewCamera::rotateRoll(float angle)
{
    m_UpDirection = glm::normalize(glm::angleAxis(angle, m_CameraDirection) * m_UpDirection);
}

void ViewCamera::set_window(GLFWwindow* window) {
    m_pWindow = window;
}

double3 ViewCamera::getCameraPos(int grid_nr) {
    if (m_param->movementMode == 0) {
        return world_pos;
    }
    else if (m_param->movementMode == 1) {
        return {
                    m_param->getRadius(grid_nr),
                    m_param->getInclination(grid_nr),
                    0
        };
    }
}