#pragma once

#include "vec3.h"
#include "sphere.h"
#include "camera.h"

const int MAX_SPHERES = 64;

void render(Vec3 *fb, int width, int height, Sphere *sphere, int nSpheres, Camera *cam);
