#pragma once

#include "shaders/grid.cuh"
#include "shaders/tracer.cuh"

const int MAX_ENTITIES = 64;

__global__ void render_scene(SceneProperties sp);
