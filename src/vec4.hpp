#pragma once

#include <cmath>

#ifdef __CUDACC__
    #define HD __host__ __device__
#else
    #define HD
#endif

struct Vec4 {
    float x, y, z, w;

    inline HD Vec4() : x(0), y(0), z(0), w(0) {}
    inline HD Vec4(float v) : x(v), y(v), z(v), w(v) {}
    inline HD Vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}

    inline HD Vec4 operator-() const { return Vec4(-x, -y, -z, -w); }

    inline HD Vec4 operator+(const Vec4 &b) const { return Vec4(x + b.x, y + b.y, z + b.z, w + b.w); }
    inline HD Vec4 operator-(const Vec4 &b) const { return Vec4(x - b.x, y - b.y, z - b.z, w - b.w); }
    inline HD Vec4 operator*(const Vec4 &b) const { return Vec4(x * b.x, y * b.y, z * b.z, w * b.w); }
    inline HD Vec4 operator*(float b) const { return Vec4(x * b, y * b, z * b, w * b); }
    inline HD friend Vec4 operator*(float a, const Vec4 &v) { return v * a; }
    inline HD Vec4 operator/(float b) const { return Vec4(x / b, y / b, z / b, w / b); }

    inline HD Vec4 &operator+=(const Vec4 &b) {
        x += b.x;
        y += b.y;
        z += b.z;
        w += b.w;
        return *this;
    }
    inline HD Vec4 &operator-=(const Vec4 &b) {
        x -= b.x;
        y -= b.y;
        z -= b.z;
        w -= b.w;
        return *this;
    }
    inline HD Vec4 &operator*=(const Vec4 &b) {
        x *= b.x;
        y *= b.y;
        z *= b.z;
        w *= b.w;
        return *this;
    }
    inline HD Vec4 &operator*=(float b) {
        x *= b;
        y *= b;
        z *= b;
        w *= b;
        return *this;
    }
    inline HD Vec4 &operator/=(float b) {
        x /= b;
        y /= b;
        z /= b;
        w /= b;
        return *this;
    }

    inline HD float dot(const Vec4 &b) const { return x * b.x + y * b.y + z * b.z + w * b.w; }

    inline HD Vec4 normalize() const {
        float len = sqrtf(x * x + y * y + z * z + w * w);
        return len > 0 ? (*this) / len : Vec4(0, 0, 0, 0);
    }
};
