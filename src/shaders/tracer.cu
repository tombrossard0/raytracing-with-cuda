#include "tracer.cuh"

__device__ Vec3 trace(Ray ray, unsigned int &seed, const SceneProperties &sp, float &dst) {
    Vec3 incomingLight = 0; // No color
    Vec3 rayColour = 1;     // White

    for (int i = 0; i < sp.cam->maxBounces; i++) {
        HitInfo hitInfo = calculateRayCollision(ray, sp);

        if (i == 0 && hitInfo.didHit) { dst = hitInfo.dst; }

        if (hitInfo.didHit) {
            ray.origin = hitInfo.hitPoint;
            ray.dir = randomHemisphereDirection(hitInfo.normal, seed);

            RayTracingMaterial material = hitInfo.material;
            Vec3 emittedLight = material.emissionColour * material.emissionStrength;
            incomingLight += emittedLight * rayColour;
            rayColour *= material.colour; // Absorb light
        } else {
            incomingLight += getEnvironmentLight(ray) * rayColour;
            break;
        }
    }

    return incomingLight;
}
