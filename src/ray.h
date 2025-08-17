#pragma once

#include "vec3.h"
#include "vec4.h"

struct RayTracingMaterial {
    Vec3 colour;
    Vec3 emissionColour;
    float emissionStrength;

    __host__ __device__ RayTracingMaterial(Vec3 _colour) : colour(_colour) {}
};

struct HitInfo {
    bool didHit;
    float dst;
    Vec3 hitPoint;
    Vec3 normal;
    RayTracingMaterial material;

    __host__ __device__ HitInfo()
        : didHit(false), dst(0), hitPoint(Vec3()), normal(Vec3()), material(Vec3()) {}

    __host__ __device__ HitInfo(bool _didHit, float _dst, Vec3 _hitPoint, Vec3 _normal)
        : didHit(_didHit), dst(_dst), hitPoint(_hitPoint), normal(_normal),
          material(RayTracingMaterial(Vec3())) {}

    __host__ __device__ HitInfo(bool _didHit, float _dst, Vec3 _hitPoint, Vec3 _normal,
                                RayTracingMaterial _material)
        : didHit(_didHit), dst(_dst), hitPoint(_hitPoint), normal(_normal), material(_material) {}
};

struct Ray {
    Vec3 origin, dir;
    __host__ __device__ Ray(Vec3 o, Vec3 d) : origin(o), dir(d) {}
};
