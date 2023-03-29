#pragma once
#include "../Header files/ViewCamera.h"
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_transform.hpp>

ViewCamera::ViewCamera(GLFWwindow* pWindow)
    : m_pWindow(pWindow)
{}

ViewCamera::ViewCamera(GLFWwindow* pWindow, const glm::vec3& pos, const glm::vec3& forward, const glm::vec3& up)
    : m_position(pos)
    , m_forward(glm::normalize(forward))
    , m_pWindow(pWindow)
    , m_up(up)
{
}

GLFWwindow* ViewCamera::get_window() {
    return m_pWindow;
}

glm::vec3 ViewCamera::cameraPos() const
{
    return m_position;
}

glm::mat4 ViewCamera::viewMatrix() const
{
    return glm::lookAt(m_position, m_position + m_forward, m_up);
}

void ViewCamera::updateInput() {
};