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
    Vec3 size;
    RayTracingMaterial material;

    // Triangle
    int numTriangles;
    Triangle *triangles;

    HD Entity(EntityType t, Vec3 c, Vec3 r) : type(t), center(c), size(r), material(1), numTriangles(0) {}
    HD Entity(EntityType t, Vec3 c, Vec3 r, Vec3 m)
        : type(t), center(c), size(r), material(m), numTriangles(0) {}

    HD Entity(EntityType t, int numTriangles, Triangle *triangles, Vec3 size)
        : type(t), center(), size(size), material(1), numTriangles(numTriangles), triangles(triangles) {}
    HD Entity(EntityType t, int numTriangles, Triangle *triangles, Vec3 m, Vec3 size)
        : type(t), center(), size(size), material(m), numTriangles(numTriangles), triangles(triangles) {}

    HD HitInfo intersectSphere(const Ray &ray) const {
        HitInfo hitInfo;

        // Transform ray into scaled space
        Vec3 invScale = Vec3(1.0f / size.x, 1.0f / size.y, 1.0f / size.z);
        Vec3 ro = (ray.origin - center) * invScale; // scaled ray origin
        Vec3 rd = ray.dir * invScale;               // scaled ray dir
        rd = rd.normalize();

        // Sphere intersection in unit sphere space
        float a = rd.dot(rd);
        float b = 2.0f * ro.dot(rd);
        float c = ro.dot(ro) - 1.0f; // radius = 1
        float discriminant = b * b - 4 * a * c;
        if (discriminant < 0) return hitInfo;

        float t = (-b - sqrtf(discriminant)) / (2.0f * a);
        if (t <= 0) return hitInfo;

        Vec3 localHit = ro + rd * t; // hit point in scaled space

        hitInfo.didHit = true;
        hitInfo.dst = (localHit - ro).length(); // proper distance in scaled space
        hitInfo.material = material;

        // Transform back to world space
        hitInfo.hitPoint = ray.origin + ray.dir * hitInfo.dst;

        // Normal: transform & renormalize
        Vec3 n = localHit.normalize();
        n = (n * invScale).normalize(); // correct for scaling
        hitInfo.normal = n;

        return hitInfo;
    }

    HD HitInfo intersectTriangle(const Ray &ray, Triangle triangle) const {
        HitInfo hit{};

        const float EPS = 1e-6f;
        Vec3 v0v1 = (triangle.v1 - triangle.v0) * size;
        Vec3 v0v2 = (triangle.v2 - triangle.v0) * size;
        Vec3 pvec = ray.dir.cross(v0v2);
        float det = v0v1.dot(pvec);

        // if (fabs(det) < EPS) return hit; // parallel
        if (det < EPS) return hit; // cull backfaces: only if facing the ray

        float invDet = 1.0f / det;

        Vec3 tvec = ray.origin - (triangle.v0 + center) * size;
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
            HitInfo closestInfo;
            for (int i = 0; i < numTriangles; i++) {
                HitInfo hitInfo = intersectTriangle(ray, triangles[i]);
                if (hitInfo.didHit &&
                    ((closestInfo.didHit && hitInfo.dst < closestInfo.dst) || !closestInfo.didHit)) {
                    closestInfo = hitInfo;
                }
            }

            return closestInfo;
        }
        return {};
    }
};
