#pragma once

#include "../camera.cuh"
#include "../ray.cuh"
#include "../scene.hpp"
#include "../sphere.cuh"
#include "../vec3.cuh"
#include "../vec4.cuh"

__device__ void computeCameraBasis(const Camera &cam, Vec3 &right, Vec3 &up);
__device__ HitInfo calculateRayCollision(Ray ray, const SceneProperties &sp);
__device__ Ray generateRay(float u, float v, const Camera &cam);
