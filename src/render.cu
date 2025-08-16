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

__device__ void render_scene(Vec3 *fb_idx, float u, float v, const Camera &cam, const Sphere *spheres, int nSpheres) {
    // Generate a ray from the camera through pixel (u,v)
    Ray ray = generateRay(u, v, cam);

    HitInfo closesHit = calculateRayCollision(ray, nSpheres, spheres);
    if (closesHit.didHit) {
        *fb_idx = closesHit.material.colour;
    } else {
        // Map direction [-1,1] to [0,1] for visualization
        *fb_idx = Vec3(
            0.5f * (ray.dir.x + 1.0f),
            0.5f * (ray.dir.y + 1.0f),
            0.5f * (ray.dir.z + 1.0f)
        );
    }
}

__device__ void render_gradient(Vec3 *fb_idx, float u, float v) {
    *fb_idx = Vec3(u + 0.5, u + 0.5, u + 0.5);
}

__global__ void render_kernel(Vec3 *fb, int width, int height, Sphere *spheres, int nSpheres, Camera cam) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    
    // Normalize and centerize coordinates between [-0.5, 0.5]
    float u = (x - width / 2.0f) / width;
    float v = (y - height / 2.0f) / height;

    render_scene(&fb[idx], u, v, cam, spheres, nSpheres);
    // render_sphere(&fb[idx], u, v, sphere);
}

void render(Vec3 *fb, int width, int height, Sphere *spheres, int nSpheres, Camera *cam) {
    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    // Note: the kernel runs on the GPU, which cannot directly access host
    // memory unless we use managed memory or cudaMemcpy
    render_kernel<<<blocks, threads>>>(fb, width, height, spheres, nSpheres, *cam);

    cudaDeviceSynchronize();
}
