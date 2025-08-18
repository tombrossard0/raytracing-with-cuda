#pragma once

#include "vec3.hpp"
#include "vec4.hpp"

#ifdef __CUDACC__
    #define HD __host__ __device__
#else
    #define HD
#endif

struct RayTracingMaterial {
    Vec3 colour;
    Vec3 emissionColour;
    float emissionStrength;

    HD RayTracingMaterial(Vec3 _colour) : colour(_colour) {}
};

struct HitInfo {
    bool didHit;
    float dst;
    Vec3 hitPoint;
    Vec3 normal;
    RayTracingMaterial material;

    HD HitInfo() : didHit(false), dst(0), hitPoint(Vec3()), normal(Vec3()), material(Vec3()) {}

    HD HitInfo(bool _didHit, float _dst, Vec3 _hitPoint, Vec3 _normal)
        : didHit(_didHit), dst(_dst), hitPoint(_hitPoint), normal(_normal),
          material(RayTracingMaterial(Vec3())) {}

    HD HitInfo(bool _didHit, float _dst, Vec3 _hitPoint, Vec3 _normal, RayTracingMaterial _material)
        : didHit(_didHit), dst(_dst), hitPoint(_hitPoint), normal(_normal), material(_material) {}
};

struct Ray {
    Vec3 origin, dir;
    HD Ray(Vec3 o, Vec3 d) : origin(o), dir(d) {}
};
