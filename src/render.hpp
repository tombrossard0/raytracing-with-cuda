#pragma once

#include "camera.cuh"
#include "sphere.cuh"
#include "vec3.cuh"

const int MAX_SPHERES = 64;

void render(Vec3 *fb, int width, int height, Sphere *sphere, int nSpheres, Camera *cam);
