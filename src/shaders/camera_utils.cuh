#pragma once

#include "../camera.cuh"
#include "../ray.cuh"
#include "../scene.hpp"
#include "../sphere.cuh"
#include "../vec3.cuh"
#include "../vec4.cuh"

__device__ __forceinline__ void computeCameraBasis(const Camera &cam, Vec3 &right, Vec3 &up) {
    right = cam.forward.cross(cam.up).normalize();
    up = right.cross(cam.forward).normalize();
}

__device__ __forceinline__ HitInfo calculateRayCollision(Ray ray, const SceneProperties &sp) {
    HitInfo closestHit{};

    for (int i = 0; i < sp.nSpheres; i++) {
        const Sphere &sphere = sp.spheres[i];
        HitInfo hitInfo = sphere.intersect(ray);

        if (hitInfo.didHit && hitInfo.dst < closestHit.dst || hitInfo.didHit && !closestHit.didHit) {
            closestHit = hitInfo;
        }
    }

    return closestHit;
}

__device__ __forceinline__ Ray generateRay(float u, float v, const Camera &cam) {
    Vec3 right, up;
    computeCameraBasis(cam, right, up);

    float scale = tanf(cam.fov * 0.5f * M_PI / 180.0f);
    Vec3 dir = (cam.forward + right * (u * cam.aspect * scale) + up * (v * scale)).normalize();
    return Ray(cam.position, dir);
}
