#include "render.cuh"

__device__ void render_pixel(unsigned int &seed, unsigned int idx, Vec3 coords, SceneProperties &sp) {
    // Normalize and centerize coordinates between [-0.5, 0.5]
    float u = (coords.x - sp.width / 2.0f) / sp.width;
    float v = (coords.y - sp.height / 2.0f) / sp.height;

    // Generate a ray from the camera through pixel (u,v)
    Ray ray = generateRay(u, v, *sp.cam);

    Vec3 totalIncomingLight = 0;
    float dst = -1;

    for (int i = 0; i < sp.cam->numberOfRayPerPixel; i++) { totalIncomingLight += trace(ray, seed, sp, dst); }

    totalIncomingLight /= float(sp.cam->numberOfRayPerPixel);

    int frameCount = sp.numRenderedFramesB - sp.numRenderedFramesA + 1;
    if (frameCount <= 1) {
        // First frame after movement
        sp.fb[idx] = totalIncomingLight;
    } else {
        Vec3 oldRender = sp.fb[idx];
        sp.fb[idx] = (oldRender * (frameCount - 1) + totalIncomingLight) / float(frameCount);
    }

    // Generate an infinite grid
    float tGrid;
    if (intersectGrid(ray, tGrid)) {
        Vec3 hitPoint = ray.origin + ray.dir * tGrid;
        Vec3 gridCol = gridColor(hitPoint, 1.0f, 0.02f);

        float maxGridDst = 50.f;
        if (gridCol != 0 && (dst > -1 && dst > tGrid || dst == -1) && tGrid < maxGridDst) {
            float alpha = 1 - tGrid / maxGridDst;
            sp.fb[idx] = sp.fb[idx] * (1.f - alpha) + gridCol * alpha;
        }
    }
}

// __device__ void render_gradient(Vec3 *fb_idx, float u, float v) {
//     *fb_idx = Vec3(u + 0.5, u + 0.5, u + 0.5);
// }

__global__ void render_scene(SceneProperties sp) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= sp.width || y >= sp.height) return;

    unsigned int seed = 1469598103u ^ x * 16777619u ^ y * 14697619u ^ sp.numRenderedFramesB;
    unsigned int idx = y * sp.width + x;

    render_pixel(seed, idx, Vec3(x, y, 0), sp);
}
