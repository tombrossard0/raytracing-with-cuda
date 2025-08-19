#pragma once

#include "camera_utils.cuh"
#include "lighting.cuh"
#include "sampling.cuh"

__device__ Vec3 trace(Ray ray, unsigned int &seed, const SceneProperties &sp, float &dst);
