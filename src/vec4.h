#pragma once
#include <cmath>

#ifdef __CUDACC__
  #define HD __host__ __device__
#else
  #define HD
#endif

struct Vec4
{
    float x, y, z, w;
    HD Vec4() : x(0), y(0), z(0), w(0) {}
    HD Vec4(float v) : x(v), y(v), z(v), w(v) {}
    HD Vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}

    HD Vec4 operator-() const { return Vec4(-x, -y, -z, -w); }

    HD Vec4 operator+(const Vec4 &b) const { return Vec4(x + b.x, y + b.y, z + b.z, w + b.w); }
    HD Vec4 operator-(const Vec4 &b) const { return Vec4(x - b.x, y - b.y, z - b.z, w - b.w); }
    HD Vec4 operator*(const Vec4 &b) const { return Vec4(x * b.x, y * b.y, z * b.z, w * b.w); }
    HD Vec4 operator*(float b) const { return Vec4(x * b, y * b, z * b, w * b); }
    HD friend Vec4 operator*(float a, const Vec4 &v) { return v * a; }
    HD Vec4 operator/(float b) const { return Vec4(x / b, y / b, z / b, w / b); }

    HD Vec4& operator+=(const Vec4 &b) { x += b.x; y += b.y; z += b.z; w += b.w; return *this; }
    HD Vec4& operator-=(const Vec4 &b) { x -= b.x; y -= b.y; z -= b.z; w -= b.w; return *this; }
    HD Vec4& operator*=(const Vec4 &b) { x *= b.x; y *= b.y; z *= b.z; w *= b.w; return *this; }
    HD Vec4& operator*=(float b) { x *= b; y *= b; z *= b; w *= b; return *this; }
    HD Vec4& operator/=(float b) { x /= b; y /= b; z /= b; w /= b; return *this; }

    HD float dot(const Vec4 &b) const { return x * b.x + y * b.y + z * b.z + w * b.w; }

    // Normalize
    HD Vec4 normalize() const {
        float len = sqrtf(x * x + y * y + z * z + w * w);
        return len > 0 ? (*this) / len : Vec4(0, 0, 0, 0);
    }
};
