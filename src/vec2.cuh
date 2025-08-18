#pragma once

#include <cmath>

#ifdef __CUDACC__
    #define HD __host__ __device__
#else
    #define HD
#endif

struct Vec2 {
    float x, y;

    inline HD Vec2() : x(0), y(0) {}
    inline HD Vec2(float v) : x(v), y(v) {}
    inline HD Vec2(float x_, float y_) : x(x_), y(y_) {}

    inline HD Vec2 operator-() const { return Vec2(-x, -y); }

    inline HD Vec2 operator+(const Vec2 &b) const { return Vec2(x + b.x, y + b.y); }
    inline HD Vec2 operator-(const Vec2 &b) const { return Vec2(x - b.x, y - b.y); }
    inline HD Vec2 operator*(const Vec2 &b) const { return Vec2(x * b.x, y * b.y); }
    inline HD Vec2 operator*(float b) const { return Vec2(x * b, y * b); }
    inline HD friend Vec2 operator*(float a, const Vec2 &v) { return v * a; }
    inline HD Vec2 operator/(float b) const { return Vec2(x / b, y / b); }

    inline HD Vec2 &operator+=(const Vec2 &b) {
        x += b.x;
        y += b.y;
        return *this;
    }
    inline HD Vec2 &operator-=(const Vec2 &b) {
        x -= b.x;
        y -= b.y;
        return *this;
    }
    inline HD Vec2 &operator*=(const Vec2 &b) {
        x *= b.x;
        y *= b.y;
        return *this;
    }
    inline HD Vec2 &operator*=(float b) {
        x *= b;
        y *= b;
        return *this;
    }
    inline HD Vec2 &operator/=(float b) {
        x /= b;
        y /= b;
        return *this;
    }

    inline HD float dot(const Vec2 &b) const { return x * b.x + y * b.y; }
    inline HD float cross(const Vec2 &b) const { return x * b.y - y * b.x; } // scalar in 2D
    inline HD float length() const { return sqrtf(x * x + y * y); }

    inline HD Vec2 normalize() const {
        float len = sqrtf(x * x + y * y);
        return len > 0 ? (*this) / len : Vec2(0, 0);
    }
};
