#pragma once

#include "camera_utils.cuh"
#include "lighting.cuh"
#include "sampling.cuh"

__device__ __forceinline__ Vec3 reflect(Vec3 dirIn, Vec3 normal) {
    return dirIn - 2 * dirIn.dot(normal) * normal;
}

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
        RayTracingMaterial material = hitInfo.material;

        Vec3 diffuseDir = randomHemisphereDirection(hitInfo.normal, seed);
        // Vec3 diffuseDir = (hitInfo.normal + randomDirection(seed)).normalize();
        Vec3 specularDir = reflect(ray.dir, hitInfo.normal);
        bool isSpecularBounce = material.specularProbability >= randomValue(seed);
        ray.dir = lerp(diffuseDir, specularDir, material.smoothness * isSpecularBounce);

        Vec3 emittedLight = material.emissionColour * material.emissionStrength;
        incomingLight += emittedLight * rayColour;
        rayColour *= lerp(material.colour, material.specularColour, isSpecularBounce);

        // if (rayColour < 1e-3f) break;
    }

    return incomingLight;
}
