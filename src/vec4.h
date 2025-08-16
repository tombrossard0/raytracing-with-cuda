#pragma once
#include <cmath>

struct Vec4
{
    float x, y, z, m;
    __host__ __device__ Vec4() : x(0), y(0), z(0), m(0) {}
    __host__ __device__ Vec4(float x_, float y_, float z_, float m_) : x(x_), y(y_), z(z_), m(m_) {}

    __host__ __device__ Vec4 operator+(const Vec4 &b) const { return Vec4(x + b.x, y + b.y, z + b.z, m + b.m); }
    __host__ __device__ Vec4 operator-(const Vec4 &b) const { return Vec4(x - b.x, y - b.y, z - b.z, m - b.m); }
    __host__ __device__ Vec4 operator*(float b) const { return Vec4(x * b, y * b, z * b, m * b); }
    __host__ __device__ float dot(const Vec4 &b) const { return x * b.x + y * b.y + z * b.z + m * b.m; }
    __host__ __device__ Vec4 normalize() const
    {
        float mag = sqrtf(x * x + y * y + z * z + m * m);
        return (*this) * (1.0f / mag);
    }
};
