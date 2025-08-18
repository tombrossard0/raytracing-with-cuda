#pragma once
#include "vec3.cuh"

struct Camera {
    Vec3 position;
    Vec3 forward;
    Vec3 up;
    float fov;
    float aspect;
    int maxBounces;

    __host__ __device__ Camera(Vec3 p, Vec3 f, Vec3 u, float _fov, float _aspect, int _maxBounces)
        : position(p), forward(f), up(u), fov(_fov), aspect(_aspect), maxBounces(_maxBounces) {}
};
