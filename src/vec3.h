#pragma once
#include <cmath>

struct Vec3
{
    float x, y, z;
    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    __host__ __device__ Vec3 operator+(const Vec3 &b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
    __host__ __device__ Vec3 operator-(const Vec3 &b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
    __host__ __device__ Vec3 operator*(Vec3 &b) const { return Vec3(x * b.x, y * b.y, z * b.z); }
    __host__ __device__ Vec3 operator*(float b) const { return Vec3(x * b, y * b, z * b); }
    __host__ __device__ Vec3 operator/(float b) const { return Vec3(x / b, y / b, z / b); }
    __host__ __device__ float dot(const Vec3 &b) const { return x * b.x + y * b.y + z * b.z; }
    __host__ __device__ Vec3 cross(const Vec3 &b) const {
        return Vec3(
            y * b.z - z * b.y,
            z * b.x - x * b.z,
            x * b.y - y * b.x
        );
    }
    __host__ __device__ Vec3 normalize() const
    {
        float mag = sqrtf(x * x + y * y + z * z);
        return (*this) * (1.0f / mag);
    }
};
