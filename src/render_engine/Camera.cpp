#include "Camera.h"
#include <imgui.h>

void Camera::start()
{
    recalculateProjectionMatrix();
    recalculateViewMatrix();
}

void Camera::resize(glm::ivec2 windowSize)
{
    mAspectRatio = static_cast<float>(windowSize.x) / static_cast<float>(windowSize.y);
    recalculateProjectionMatrix();
}

void Camera::recalculateProjectionMatrix()
{
    mProjectionMatrix = glm::perspective(glm::radians(mFov), mAspectRatio, mZNear, mZFar);
}

void Camera::recalculateViewMatrix()
{
    mInverseViewMatrix = glm::translate(glm::mat4x4(1.0f), mPosition);
    mInverseViewMatrix = mInverseViewMatrix * glm::mat4_cast(mOrientation);
    mViewMatrix = glm::inverse(mInverseViewMatrix);
}

void Camera::drawGui()
{
    bool change = false;
    change |= ImGui::InputFloat("Fov", &mFov, 1.0f, 10.0f);
    change |= ImGui::SliderFloat("zNear", &mZNear, 0.01f, 5.0f);
    change |= ImGui::SliderFloat("zFar", &mZFar, 1.0f, 50.0f);
    if (change) recalculateProjectionMatrix();
}