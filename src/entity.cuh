#pragma once

#include "ray.cuh"
#include "vec3.cuh"

#ifdef __CUDACC__
    #define HD __host__ __device__
#else
    #define HD
#endif

enum EntityType { SPHERE, MESH };

struct Triangle {
    Vec3 v0, v1, v2;
};

/**
 * @brief Simple entity class for raytracing.
 */
class Entity {
  public:
    EntityType type;
    Vec3 center;

    // Sphere
    float radius;
    RayTracingMaterial material;

    // Triangle
    int numTriangles;
    Triangle *triangles;

    HD Entity(EntityType t, Vec3 c, float r) : type(t), center(c), radius(r), material(1), numTriangles(0) {}
    HD Entity(EntityType t, Vec3 c, float r, Vec3 m)
        : type(t), center(c), radius(r), material(m), numTriangles(0) {}

    HD Entity(EntityType t, int numTriangles, Triangle *triangles)
        : type(t), center(), radius(), material(1), numTriangles(numTriangles), triangles(triangles) {}
    HD Entity(EntityType t, int numTriangles, Triangle *triangles, Vec3 m)
        : type(t), center(), radius(), material(m), numTriangles(numTriangles), triangles(triangles) {}

    HD HitInfo intersectSphere(const Ray &ray) const {
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

    HD HitInfo intersectTriangle(const Ray &ray, Triangle triangle) const {
        HitInfo hit{};

        const float EPS = 1e-6f;
        Vec3 v0v1 = (center + triangle.v1) - (center + triangle.v0);
        Vec3 v0v2 = (center + triangle.v2) - (center + triangle.v0);
        Vec3 pvec = ray.dir.cross(v0v2);
        float det = v0v1.dot(pvec);

        // if (fabs(det) < EPS) return hit; // parallel
        if (det < EPS) return hit; // cull backfaces: only if facing the ray

        float invDet = 1.0f / det;

        Vec3 tvec = ray.origin - (triangle.v0 + center);
        float u = tvec.dot(pvec) * invDet;
        if (u < 0 || u > 1) return hit;

        Vec3 qvec = tvec.cross(v0v1);
        float v = ray.dir.dot(qvec) * invDet;
        if (v < 0 || u + v > 1) return hit;

        float t = v0v2.dot(qvec) * invDet;
        if (t > EPS) {
            hit.didHit = true;
            hit.dst = t;
            hit.material = material;
            hit.hitPoint = ray.origin + ray.dir * t;
            hit.normal = v0v1.cross(v0v2).normalize();
        }
        return hit;
    }

    HD HitInfo intersect(const Ray &ray) const {
        switch (type) {
        case SPHERE:
            return intersectSphere(ray);
        case MESH:
            for (int i = 0; i < numTriangles; i++) {
                HitInfo hitInfo = intersectTriangle(ray, triangles[i]);
                if (hitInfo.didHit) { return hitInfo; }
            }

            return {};
        }
        return {};
    }
};
