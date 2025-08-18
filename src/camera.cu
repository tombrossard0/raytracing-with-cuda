#include "camera.cuh"

void Camera::updateCameraPosition(float yawDeg, float pitchDeg, float radius) {
    // Convert to radians
    float yawRad = yawDeg * M_PI / 180.0f;
    float pitchRad = pitchDeg * M_PI / 180.0f;

    // Spherical coordinates around center
    float x = center.x + radius * cosf(pitchRad) * cosf(yawRad);
    float y = center.y + radius * sinf(pitchRad);
    float z = center.z + radius * cosf(pitchRad) * sinf(yawRad);

    position = Vec3(x, y, z);
    forward = (center - position).normalize();
}
