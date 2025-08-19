#include "grid.cuh"

__device__ bool intersectGrid(const Ray &ray, float &t) {
    if (ray.dir.y == 0) return false; // parallel to grid plane

    t = -ray.origin.y / ray.dir.y; // intersection with y=0 plane
    return t > 0;
}

__device__ Vec3 gridColor(const Vec3 &point, float gridSize, float lineWidth) {
    // Compute distance to nearest grid line
    float fx = fmodf(fabsf(point.x), gridSize);
    float fz = fmodf(fabsf(point.z), gridSize);

    // If the point is close enough to a grid line, draw it
    if (fx < lineWidth || fx > gridSize - lineWidth || fz < lineWidth || fz > gridSize - lineWidth)
        return 0.8f; // grid line color
    else
        return 0; // background color (no square fill)
}
