#include "ppm.h"

void savePPM(const std::string &filename, Vec3 *fb, int width, int height) {
    auto toByte = [](float x) {
        // gamma correction
        return static_cast<unsigned char>(255 * powf(fminf(fmaxf(x, 0.0f), 1.0f), 1.0f / 2.2f));
    };

    std::ofstream ofs(filename, std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; i++) {
        unsigned char r = toByte(fb[i].x);
        unsigned char g = toByte(fb[i].y);
        unsigned char b = toByte(fb[i].z);

        ofs.write(reinterpret_cast<char *>(&r), 1);
        ofs.write(reinterpret_cast<char *>(&g), 1);
        ofs.write(reinterpret_cast<char *>(&b), 1);
    }
}
