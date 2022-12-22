#pragma once
#include <glfw3.h>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

class Camera {
public:
    
    Camera(GLFWwindow* pWindow);
    Camera(GLFWwindow* pWindow, const glm::vec3& pos, const glm::vec3& forward);


    void updateInput();

    glm::vec3 cameraPos() const;
    glm::mat4 viewMatrix() const;

    glm::vec3 m_position{ 10,0,0 };
    glm::vec3 m_forward{ -1, 0, 0};
    glm::vec3 m_up{ 0, 1, 0 };

private:
    void rotateX(float angle);
    void rotateY(float angle);

private:
    static constexpr glm::vec3 s_yAxis{ 0, 1, 0 };


    GLFWwindow* m_pWindow;
    glm::dvec2 m_prevCursorPos{ 0 };
};