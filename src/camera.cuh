#pragma once
#include "vec3.cuh"

#ifdef __CUDACC__
    #define HD __host__ __device__
#else
    #define HD
#endif

struct Camera {
    Vec3 center;
    Vec3 position;
    Vec3 forward;
    Vec3 up;
    float fov;
    float aspect;

    // Distance from the center
    float radius;

    // Camera angle
    float angleDeg;
    float yawDeg;
    float pitchDeg;

    float minRadius;
    float maxRadius;

    int maxBounces = 10;
    int numberOfRayPerPixel = 100;

    HD Camera(Vec3 c, Vec3 p, Vec3 f, Vec3 u, float _fov, float _aspect)
        : center(c), position(p), forward(f), up(u), fov(_fov), aspect(_aspect) {}

    void updateCameraPosition();
};
