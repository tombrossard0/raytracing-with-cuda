#pragma once

#include "../vec3.cuh"

__device__ float randomValue(unsigned int &state);

__device__ Vec3 randomDirection(unsigned int &seed);
__device__ Vec3 randomHemisphereDirection(Vec3 normal, unsigned int &seed);
