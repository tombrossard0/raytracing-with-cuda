#pragma once

#include "../vec3.cuh"

__device__ __forceinline__ float randomValue(unsigned int &state) {
    state = (state ^ 61u) ^ (state >> 16);
    state *= 9u;
    state = state ^ (state >> 4);
    state *= 0x27d4eb2du;
    state = state ^ (state >> 15);
    // map to [0,1)
    return (state & 0x00FFFFFF) / 16777216.0f;
}

__device__ __forceinline__ float randomValueDistribution(unsigned int &state) {
    float theta = 2 * M_PI * randomValue(state);
    float rho = sqrt(-2 * log(randomValue(state)));
    return rho * cos(theta);
}

__device__ __forceinline__ Vec3 randomDirection(unsigned int &seed) {
    float x = randomValueDistribution(seed);
    float y = randomValueDistribution(seed);
    float z = randomValueDistribution(seed);

    Vec3 pointInCube = Vec3(x, y, z);
    return pointInCube.normalize();
}

// __device__ __forceinline__ Vec3 randomHemisphereDirection(const Vec3 &normal, unsigned int &seed) {
//     float u1 = randomValue(seed);
//     float u2 = randomValue(seed);

//     float r = sqrtf(u1);
//     float theta = 2.0f * M_PI * u2;

//     float x = r * cosf(theta);
//     float y = r * sinf(theta);
//     float z = sqrtf(1.0f - u1);

//     // Build an orthonormal basis around `normal`
//     Vec3 w = normal.normalize();
//     Vec3 a = fabs(w.x) > 0.1f ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
//     Vec3 v = w.cross(a).normalize();
//     Vec3 u = v.cross(w);

//     // Transform (x, y, z) from local space to world space
//     return (u * x + v * y + w * z).normalize();
// }

__device__ __forceinline__ float sign(float x) {
    return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
}

__device__ __forceinline__ Vec3 randomHemisphereDirection(const Vec3 &normal, unsigned int &seed) {
    Vec3 dir = randomDirection(seed);
    return dir * sign(normal.dot(dir));
}