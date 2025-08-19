#include "lighting.cuh"

__device__ Vec3 getEnvironmentLight(Ray ray) {
    // return Vec3(0, 0, 0.5f);
    return Vec3(0.5f * (ray.dir.x + 1.0f), 0.5f * (ray.dir.y + 1.0f), 0.5f * (ray.dir.z + 1.0f));
}
