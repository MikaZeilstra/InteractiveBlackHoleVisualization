#pragma once
#include <glm/matrix.hpp>

#define GLFW_INCLUDE_NONE
#include <glfw3.h>

class ViewCamera {
public:

    ViewCamera(GLFWwindow* pWindow);
    ViewCamera(GLFWwindow* pWindow, const glm::vec3& pos, const glm::vec3& forward, const glm::vec3& up);


    void updateInput();
    GLFWwindow* get_window();

    glm::vec3 cameraPos() const;
    glm::mat4 viewMatrix() const;

    glm::vec3 m_position{ 10,0,0 };
    glm::vec3 m_forward{ -1, 0, 0 };
    glm::vec3 m_up{ 0, 1, 0 };

private:


private:

    GLFWwindow* m_pWindow;
    glm::dvec2 m_prevCursorPos{ 0 };
};