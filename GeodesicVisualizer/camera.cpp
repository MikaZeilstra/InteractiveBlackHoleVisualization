#pragma once
#include "camera.h"
#include "GeodesicVisualizer.h"


#include <glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <stdio.h>


Camera::Camera(GLFWwindow* pWindow)
    : m_pWindow(pWindow)
{}

Camera::Camera(GLFWwindow* pWindow, const glm::vec3& pos, const glm::vec3& forward)
    : m_position(pos)
    , m_forward(glm::normalize(forward))
    , m_pWindow(pWindow)
{
}

glm::vec3 Camera::cameraPos() const
{
    return m_position;
}

glm::mat4 Camera::viewMatrix() const
{
    return glm::lookAt(m_position, m_position + m_forward, m_up);
}

void Camera::rotateX(float angle)
{
    const glm::vec3 horAxis = glm::cross(s_yAxis, m_forward);

    m_forward = glm::normalize(glm::angleAxis(angle, horAxis) * m_forward);
    m_up = glm::normalize(glm::cross(m_forward, horAxis));
}

void Camera::rotateY(float angle)
{
    const glm::vec3 horAxis = glm::cross(s_yAxis, m_forward);

    m_forward = glm::normalize(glm::angleAxis(angle, s_yAxis) * m_forward);
    m_up = glm::normalize(glm::cross(m_forward, horAxis));
}

void Camera::updateInput()
{
    constexpr float moveSpeed = 0.0001f;
    constexpr float lookSpeed = 0.0035f;

    glm::vec3 localMoveDelta{ 0 };
    const glm::vec3 right = glm::normalize(glm::cross(m_forward, m_up));
    if (glfwGetKey(m_pWindow,GLFW_KEY_A) == GLFW_PRESS) {
        m_position -= moveSpeed * right;
    }
    if (glfwGetKey(m_pWindow, GLFW_KEY_D) == GLFW_PRESS) {
        m_position += moveSpeed * right;
    }
    if (glfwGetKey(m_pWindow, GLFW_KEY_W) == GLFW_PRESS)
        m_position += moveSpeed * m_forward;
    if (glfwGetKey(m_pWindow, GLFW_KEY_S) == GLFW_PRESS)
        m_position -= moveSpeed * m_forward;
    if (glfwGetKey(m_pWindow, GLFW_KEY_SPACE) == GLFW_PRESS)
        m_position += moveSpeed * m_up;
    if (glfwGetKey(m_pWindow, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        m_position -= moveSpeed * m_up;



     
    glm::dvec2 cursorPos = {};
    glfwGetCursorPos(m_pWindow, &(cursorPos.x), &(cursorPos.y));
    const glm::vec2 delta = lookSpeed * glm::vec2(cursorPos - m_prevCursorPos);
    m_prevCursorPos = cursorPos;

    if (glfwGetMouseButton(m_pWindow, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        if (delta.x != 0.0f)
            rotateY(delta.x);
        if (delta.y != 0.0f)
            rotateX(delta.y);
    }

}