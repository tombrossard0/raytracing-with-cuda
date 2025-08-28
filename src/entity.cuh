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

struct Quat {
    float w, x, y, z;

    HD Quat(float w_ = 1, float x_ = 0, float y_ = 0, float z_ = 0) : w(w_), x(x_), y(y_), z(z_) {}

    // quaternion multiplication
    HD Quat operator*(const Quat &b) const {
        return Quat(w * b.w - x * b.x - y * b.y - z * b.z, w * b.x + x * b.w + y * b.z - z * b.y,
                    w * b.y - x * b.z + y * b.w + z * b.x, w * b.z + x * b.y - y * b.x + z * b.w);
    }

    HD Quat conjugate() const { return Quat(w, -x, -y, -z); }
};

inline HD Vec3 rotate(const Vec3 &v, const Quat &q) {
    Quat p(0, v.x, v.y, v.z);
    Quat res = q * p * q.conjugate();
    return Vec3(res.x, res.y, res.z);
}

inline HD Quat fromEuler(const Vec3 &deg) {
    // convert to radians
    Vec3 rad = deg * (M_PI / 180.0f);

    float cx = cosf(rad.x * 0.5f), sx = sinf(rad.x * 0.5f);
    float cy = cosf(rad.y * 0.5f), sy = sinf(rad.y * 0.5f);
    float cz = cosf(rad.z * 0.5f), sz = sinf(rad.z * 0.5f);

    Quat q;
    q.w = cx * cy * cz + sx * sy * sz;
    q.x = sx * cy * cz - cx * sy * sz;
    q.y = cx * sy * cz + sx * cy * sz;
    q.z = cx * cy * sz - sx * sy * cz;
    return q;
}

/**
 * @brief Simple entity class for raytracing.
 */
class Entity {
  public:
    EntityType type;
    Vec3 center;
    Vec3 size;
    RayTracingMaterial material;

    // Mesh only
    int numTriangles;
    Triangle *triangles;

    Vec3 rotationEuler;

    HD Entity(EntityType t, Vec3 c, Vec3 r) : type(t), center(c), size(r), material(1), numTriangles(0) {}
    HD Entity(EntityType t, Vec3 c, Vec3 r, Vec3 m)
        : type(t), center(c), size(r), material(m), numTriangles(0) {}

    HD Entity(EntityType t, int numTriangles, Triangle *triangles, Vec3 size)
        : type(t), center(), size(size), material(1), numTriangles(numTriangles), triangles(triangles) {}
    HD Entity(EntityType t, int numTriangles, Triangle *triangles, Vec3 m, Vec3 size)
        : type(t), center(), size(size), material(m), numTriangles(numTriangles), triangles(triangles) {}

    HD HitInfo intersectSphere(const Ray &ray) const {
        HitInfo hitInfo;

        Vec3 oc = ray.origin - center;
        float a = ray.dir.dot(ray.dir);
        float b = 2.0f * oc.dot(ray.dir);
        float c = oc.dot(oc) - size.x * size.x;
        float discriminant = b * b - 4 * a * c;
        if (discriminant < 0) return hitInfo;

        hitInfo.dst = (-b - sqrtf(discriminant)) / (2.0f * a);
        hitInfo.didHit = hitInfo.dst > 0;
        hitInfo.material = material;
        hitInfo.hitPoint = ray.origin + ray.dir * hitInfo.dst;
        hitInfo.normal = (hitInfo.hitPoint - center).normalize();

        return hitInfo;
    }

    HD HitInfo intersectTriangle(const Ray &ray, const Triangle &triangle) const {
        HitInfo hit{};
        const float EPS = 1e-6f;

        // Transform vertices by entity scale + rotation + translation
        Vec3 v0 = rotate(triangle.v0 * size, fromEuler(rotationEuler)) + center;
        Vec3 v1 = rotate(triangle.v1 * size, fromEuler(rotationEuler)) + center;
        Vec3 v2 = rotate(triangle.v2 * size, fromEuler(rotationEuler)) + center;

        Vec3 v0v1 = v1 - v0;
        Vec3 v0v2 = v2 - v0;
        Vec3 pvec = ray.dir.cross(v0v2);
        float det = v0v1.dot(pvec);

        if (det < EPS) return hit; // backface culling

        float invDet = 1.0f / det;

        Vec3 tvec = ray.origin - v0;
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
