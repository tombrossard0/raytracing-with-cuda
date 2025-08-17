#pragma once
#include "vec3.h"

#ifdef __CUDACC__
    #define HD __host__ __device__
#else
    #define HD
#endif

struct Camera {
    Vec3 position;
    Vec3 forward;
    Vec3 up;
    float fov;
    float aspect;
    int maxBounces;

    HD Camera(Vec3 p, Vec3 f, Vec3 u, float _fov, float _aspect, int _maxBounces)
        : position(p), forward(f), up(u), fov(_fov), aspect(_aspect), maxBounces(_maxBounces) {}
};
