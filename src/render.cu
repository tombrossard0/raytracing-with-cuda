#include "vec3.cuh"
#include "vec4.cuh"

#include "camera.cuh"
#include "ray.cuh"
#include "scene.hpp"
#include "sphere.cuh"

__device__ void computeCameraBasis(const Camera &cam, Vec3 &right, Vec3 &up) {
    right = cam.forward.cross(cam.up).normalize();
    up = right.cross(cam.forward).normalize();
}

__device__ HitInfo calculateRayCollision(Ray ray, const SceneProperties &sp) {
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

__device__ Ray generateRay(float u, float v, const Camera &cam) {
    Vec3 right, up;
    computeCameraBasis(cam, right, up);

    float scale = tanf(cam.fov * 0.5f * M_PI / 180.0f);
    Vec3 dir = (cam.forward + right * (u * cam.aspect * scale) + up * (v * scale)).normalize();
    return Ray(cam.position, dir);
}

__device__ inline float randomValue(unsigned int &state) {
    state = (state ^ 61u) ^ (state >> 16);
    state *= 9u;
    state = state ^ (state >> 4);
    state *= 0x27d4eb2du;
    state = state ^ (state >> 15);
    // map to [0,1)
    return (state & 0x00FFFFFF) / 16777216.0f;
}

__device__ Vec3 randomDirection(unsigned int &seed) {
    float x = randomValue(seed);
    float y = randomValue(seed);
    float z = randomValue(seed);
    Vec3 pointInCube = Vec3(x, y, z);
    return pointInCube.normalize();
}

__device__ Vec3 randomHemisphereDirection(Vec3 normal, unsigned int &seed) {
    Vec3 dir = randomDirection(seed);
    return dir * sinf(normal.dot(dir));
}

__device__ Vec3 getEnvironmentLight(Ray ray) {
    // return Vec3(0, 0, 0.5f);
    return Vec3(0.5f * (ray.dir.x + 1.0f), 0.5f * (ray.dir.y + 1.0f), 0.5f * (ray.dir.z + 1.0f));
}

__device__ Vec3 trace(Ray ray, unsigned int &seed, const SceneProperties &sp) {
    Vec3 incomingLight = 0; // No color
    Vec3 rayColour = 1;     // White

    for (int i = 0; i < sp.cam->maxBounces; i++) {
        HitInfo hitInfo = calculateRayCollision(ray, sp);
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

__device__ void render_pixel(unsigned int &seed, unsigned int idx, Vec3 coords, const SceneProperties &sp) {
    // Normalize and centerize coordinates between [-0.5, 0.5]
    float u = (coords.x - sp.width / 2.0f) / sp.width;
    float v = (coords.y - sp.height / 2.0f) / sp.height;

    // Generate a ray from the camera through pixel (u,v)
    Ray ray = generateRay(u, v, *sp.cam);

    Vec3 totalIncomingLight = Vec3(0, 0, 0);
    for (int i = 0; i < sp.cam->numberOfRayPerPixel; i++) {
        totalIncomingLight = totalIncomingLight + trace(ray, seed, sp);
    }

    sp.fb[idx] = totalIncomingLight / sp.cam->numberOfRayPerPixel;
}

// __device__ void render_gradient(Vec3 *fb_idx, float u, float v) {
//     *fb_idx = Vec3(u + 0.5, u + 0.5, u + 0.5);
// }

__global__ void render_scene(const SceneProperties sp) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= sp.width || y >= sp.height) return;

    unsigned int seed = 1469598103u ^ x * 16777619u ^ y;
    unsigned int idx = y * sp.width + x;

    render_pixel(seed, idx, Vec3(x, y, 0), sp);
}
