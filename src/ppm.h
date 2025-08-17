#pragma once

#include "vec3.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

void savePPM(const std::string &filename, Vec3 *fb, int width, int height);
