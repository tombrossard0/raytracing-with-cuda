#pragma once

#include "../ray.cuh"
#include "../vec3.cuh"
#include "../vec4.cuh"

__device__ bool intersectGrid(const Ray &ray, float &t);
__device__ Vec3 gridColor(const Vec3 &point, float gridSize = 1.0f, float lineWidth = 0.02f);
