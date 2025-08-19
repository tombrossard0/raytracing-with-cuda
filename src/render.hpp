#pragma once

#include "camera.cuh"
#include "scene.hpp"
#include "sphere.cuh"
#include "vec3.cuh"

const int MAX_SPHERES = 64;

__global__ void render_scene(SceneProperties sp);
