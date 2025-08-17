#pragma once

#include "ray.h"
#include "vec3.h"

/**
 * @brief Simple sphere class for raytracing.
 */
struct Sphere {
    Vec3 center;
    float radius;
    RayTracingMaterial material;

    /**
     * @brief Construct a new Sphere object
     * @param c Center of the sphere
     * @param r Radius of the sphere
     */
    __host__ __device__ Sphere(Vec3 c, float r)
        : center(c), radius(r), material(RayTracingMaterial(Vec3(1, 0, 0))) {}

    /**
     * @brief Construct a new Sphere object
     * @param c Center of the sphere
     * @param r Radius of the sphere
     * @param m Material of the sphere
     */
    __host__ __device__ Sphere(Vec3 c, float r, Vec3 m) : center(c), radius(r), material(m) {}

    /**
     * @brief Check if a ray intersects the sphere
     * @param ray The ray to test
     * @param t Distance along the ray where intersection occurs
     * @return hitInfo
     */
    __host__ __device__ HitInfo intersect(const Ray &ray) const {
        HitInfo hitInfo;

        Vec3 oc = ray.origin - center;
        float a = ray.dir.dot(ray.dir);
        float b = 2.0f * oc.dot(ray.dir);
        float c = oc.dot(oc) - radius * radius;
        float discriminant = b * b - 4 * a * c;
        if (discriminant < 0) return hitInfo;

        hitInfo.dst = (-b - sqrtf(discriminant)) / (2.0f * a);
        hitInfo.didHit = hitInfo.dst > 0;
        hitInfo.material = material;
        hitInfo.hitPoint = ray.origin + ray.dir * hitInfo.dst;
        hitInfo.normal = (hitInfo.hitPoint - center).normalize();

        return hitInfo;
    }
};
