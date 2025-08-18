#pragma once

#include <cmath>

#ifdef __CUDACC__
    #define HD __host__ __device__
#else
    #define HD
#endif

struct Vec3 {
    float x, y, z;

    inline HD Vec3() : x(0), y(0), z(0) {}
    inline HD Vec3(float v) : x(v), y(v), z(v) {}
    inline HD Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    inline HD Vec3 operator-() const { return Vec3(-x, -y, -z); }

    inline HD Vec3 operator+(const Vec3 &b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
    inline HD Vec3 operator-(const Vec3 &b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
    inline HD Vec3 operator*(const Vec3 &b) const { return Vec3(x * b.x, y * b.y, z * b.z); }
    inline HD Vec3 operator*(float b) const { return Vec3(x * b, y * b, z * b); }
    inline HD friend Vec3 operator*(float a, const Vec3 &v) { return v * a; }
    inline HD Vec3 operator/(float b) const { return Vec3(x / b, y / b, z / b); }

    inline HD Vec3 &operator+=(const Vec3 &b) {
        x += b.x;
        y += b.y;
        z += b.z;
        return *this;
    }
    inline HD Vec3 &operator-=(const Vec3 &b) {
        x -= b.x;
        y -= b.y;
        z -= b.z;
        return *this;
    }
    inline HD Vec3 &operator*=(const Vec3 &b) {
        x *= b.x;
        y *= b.y;
        z *= b.z;
        return *this;
    }
    inline HD Vec3 &operator*=(float b) {
        x *= b;
        y *= b;
        z *= b;
        return *this;
    }
    inline HD Vec3 &operator/=(float b) {
        x /= b;
        y /= b;
        z /= b;
        return *this;
    }

    inline HD float dot(const Vec3 &b) const { return x * b.x + y * b.y + z * b.z; }
    inline HD Vec3 cross(const Vec3 &b) const {
        return Vec3(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
    }

    inline HD Vec3 normalize() const {
        float len = sqrtf(x * x + y * y + z * z);
        return len > 0 ? (*this) / len : Vec3(0, 0, 0);
    }
};
