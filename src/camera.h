#pragma once
#include "vec3.h"

struct Camera
{
    Vec3 position;
    Vec3 forward;
    Vec3 up;
    float fov;
    float aspect;

    __host__ __device__ Camera(Vec3 p, Vec3 f, Vec3 u, float _fov, float _aspect)
        : position(p), forward(f), up(u), fov(_fov), aspect(_aspect) {}
};
