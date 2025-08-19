#include "sampling.cuh"

__device__ float randomValue(unsigned int &state) {
    state = (state ^ 61u) ^ (state >> 16);
    state *= 9u;
    state = state ^ (state >> 4);
    state *= 0x27d4eb2du;
    state = state ^ (state >> 15);
    // map to [0,1)
    return (state & 0x00FFFFFF) / 16777216.0f;
}

__device__ Vec3 randomDirection(unsigned int &seed) {
    float x = randomValue(seed);
    float y = randomValue(seed);
    float z = randomValue(seed);
    Vec3 pointInCube = Vec3(x, y, z);
    return pointInCube.normalize();
}

__device__ Vec3 randomHemisphereDirection(Vec3 normal, unsigned int &seed) {
    Vec3 dir = randomDirection(seed);
    return dir * sinf(normal.dot(dir));
}
