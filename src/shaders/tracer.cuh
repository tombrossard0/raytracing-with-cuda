#pragma once

#include "camera_utils.cuh"
#include "lighting.cuh"
#include "sampling.cuh"

__device__ Vec3 __forceinline__ trace(Ray ray, unsigned int &seed, const SceneProperties &sp, float &dst) {
    Vec3 incomingLight = 0; // No color
    Vec3 rayColour = 1;     // White

    for (int i = 0; i < sp.cam->maxBounces; i++) {
        HitInfo hitInfo = calculateRayCollision(ray, sp);

        if (i == 0 && hitInfo.didHit) dst = hitInfo.dst;

        if (!hitInfo.didHit) {
            incomingLight += getEnvironmentLight(ray) * rayColour;
            break;
        }

        ray.origin = hitInfo.hitPoint;
        ray.dir = randomHemisphereDirection(hitInfo.normal, seed);

        RayTracingMaterial material = hitInfo.material;
        incomingLight += (material.emissionColour * material.emissionStrength) * rayColour;

        rayColour *= material.colour; // Absorb light

        if (rayColour < 1e-3f) break;
    }

    return incomingLight;
}
