#pragma once
#include <string>
#include "vec3.h"
#include <iostream>
#include <fstream>
#include <sstream>

void savePPM(const std::string &filename, Vec3 *fb, int width, int height);
