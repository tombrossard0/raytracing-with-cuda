#pragma once

#include "../ray.cuh"
#include "../vec3.cuh"

__device__ Vec3 getEnvironmentLight(Ray ray);
