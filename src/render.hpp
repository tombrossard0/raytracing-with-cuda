#pragma once

#include "camera.cuh"
#include "sphere.hpp"
#include "vec3.hpp"

const int MAX_SPHERES = 64;

void render(Vec3 *fb, int width, int height, Sphere *sphere, int nSpheres, Camera *cam);
