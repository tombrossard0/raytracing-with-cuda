#pragma once

#include "camera.h"
#include "sphere.h"
#include "vec3.h"

const int MAX_SPHERES = 64;

void render(Vec3 *fb, int width, int height, Sphere *sphere, int nSpheres, Camera *cam);
