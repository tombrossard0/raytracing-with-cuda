#include "vec3.h"
#include "vec4.h"
#include "mat4x4.h"

#include "ray.h"
#include "sphere.h"
#include "camera.h"

__device__ void computeCameraBasis(const Camera &cam, Vec3 &right, Vec3 &up) {
    right = cam.forward.cross(cam.up).normalize();
    up = right.cross(cam.forward).normalize();
}

__device__ HitInfo calculateRayCollision(Ray ray, int nSphere, const Sphere* spheres) {
    HitInfo closestHit{};

    for (int i = 0; i < nSphere; i++) {
        const Sphere &sphere = spheres[i];
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

    float scale = tanf(cam.fov * 0.5f * M_PI/180.0f);
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
    return Vec3(0, 0, 0.5f);
}

__device__ Vec3 trace(Ray ray, unsigned int &seed, const Sphere *spheres, int nSpheres) {
    Vec3 incomingLight = Vec3(0, 0, 0);
    Vec3 rayColour = Vec3(1, 1, 1);

    for (int i = 0; i < 30; i++) {
        HitInfo hitInfo = calculateRayCollision(ray, nSpheres, spheres);
        if (hitInfo.didHit) {
            ray.origin = hitInfo.hitPoint;
            ray.dir = randomHemisphereDirection(hitInfo.normal, seed);

            RayTracingMaterial material = hitInfo.material;
            Vec3 emittedLight = material.emissionColour * material.emissionStrength;
            incomingLight = incomingLight + emittedLight * rayColour;
            rayColour = rayColour * material.colour;
        } else {
            incomingLight = incomingLight + getEnvironmentLight(ray) * rayColour;
            break;
        }
    }

    return incomingLight;
}

__device__ void render_scene(unsigned int seed, Vec3 *fb_idx, Vec3 coords, Vec3 aspect, const Camera &cam, const Sphere *spheres, int nSpheres) {
    // Normalize and centerize coordinates between [-0.5, 0.5]
    float u = (coords.x - aspect.x / 2.0f) / aspect.x;
    float v = (coords.y - aspect.y / 2.0f) / aspect.y;
    
    // Generate a ray from the camera through pixel (u,v)
    Ray ray = generateRay(u, v, cam);

    Vec3 totalIncomingLight = Vec3(0, 0, 0);
    int numberOfRayPerPixel = 100;

    for (int i = 0; i < numberOfRayPerPixel; i++) {
        totalIncomingLight = totalIncomingLight + trace(ray, seed, spheres, nSpheres);
    }

    *fb_idx = totalIncomingLight / numberOfRayPerPixel;

    // HitInfo closesHit = calculateRayCollision(ray, nSpheres, spheres);
    // if (closesHit.didHit) {
    //     // *fb_idx = randomHemisphereDirection(closesHit.normal, &seed);
    //     *fb_idx = Vec3(closesHit.material.colour.x, closesHit.material.colour.y, closesHit.material.colour.z);
    // } else {
    //     // Map direction [-1,1] to [0,1] for visualization
    //     *fb_idx = Vec3(
    //         0.5f * (ray.dir.x + 1.0f),
    //         0.5f * (ray.dir.y + 1.0f),
    //         0.5f * (ray.dir.z + 1.0f)
    //     );
    // }
}

__device__ void render_gradient(Vec3 *fb_idx, float u, float v) {
    *fb_idx = Vec3(u + 0.5, u + 0.5, u + 0.5);
}

__global__ void render_kernel(Vec3 *fb, int width, int height, Sphere *spheres, int nSpheres, Camera cam) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    unsigned int seed = 1469598103u ^ (unsigned int)x * 16777619u ^ (unsigned int)y;
    int idx = y * width + x;

    render_scene(seed, &fb[idx], Vec3(x, y, 0), Vec3(width, height, 0), cam, spheres, nSpheres);
}

void render(Vec3 *fb, int width, int height, Sphere *spheres, int nSpheres, Camera *cam) {
    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    // Note: the kernel runs on the GPU, which cannot directly access host
    // memory unless we use managed memory or cudaMemcpy
    render_kernel<<<blocks, threads>>>(fb, width, height, spheres, nSpheres, *cam);

    cudaDeviceSynchronize();
}
