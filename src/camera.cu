#include "camera.cuh"

void Camera::updateCameraPosition() {
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

void Camera::processInputs(InputManager inputManager, MouseState mouse, float deltaTime) {
    // Movements control
    float sensitivity = 0.2f;

    if (mouse.right.pressed) {
        yawDeg += mouse.dx * sensitivity;
        pitchDeg -= mouse.dy * sensitivity;
        if (pitchDeg > 89.0f) pitchDeg = 89.0f;
        if (pitchDeg < -89.0f) pitchDeg = -89.0f;
        SDL_SetRelativeMouseMode(SDL_TRUE);
    } else {
        SDL_SetRelativeMouseMode(SDL_FALSE);
    }

    radius -= 0.5f * mouse.wheel;
    if (radius < minRadius) radius = minRadius;
    if (radius > maxRadius) radius = maxRadius;

    float speed = 10.5f * deltaTime;
    float yawRad = yawDeg * M_PI / 180.0f;
    float pitchRad = pitchDeg * M_PI / 180.0f;

    Vec3 forward(cosf(pitchRad) * cosf(yawRad), sinf(pitchRad), cosf(pitchRad) * sinf(yawRad));
    forward = -forward.normalize();
    Vec3 right = -forward.cross(Vec3(0, 1, 0)).normalize();
    Vec3 up = right.cross(forward).normalize();

    if (inputManager.isKeyDown(SDL_SCANCODE_W)) center += forward * speed;
    if (inputManager.isKeyDown(SDL_SCANCODE_S)) center -= forward * speed;
    if (inputManager.isKeyDown(SDL_SCANCODE_A)) center += right * speed;
    if (inputManager.isKeyDown(SDL_SCANCODE_D)) center -= right * speed;
    if (inputManager.isKeyDown(SDL_SCANCODE_SPACE)) center += up * speed;
    if (inputManager.isKeyDown(SDL_SCANCODE_LCTRL)) center -= up * speed;
}
