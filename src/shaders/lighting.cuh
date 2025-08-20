#pragma once

#include "../ray.cuh"
#include "../vec3.cuh"

float __device__ __forceinline__ clamp(float x, float lowerlimit = 0.0f, float upperlimit = 1.0f) {
    if (x < lowerlimit) return lowerlimit;
    if (x > upperlimit) return upperlimit;
    return x;
}

float __device__ __forceinline__ smoothstep(float edge0, float edge1, float x) {
    // Scale, and clamp x to 0..1 range
    x = clamp((x - edge0) / (edge1 - edge0));

    return x * x * (3.0f - 2.0f * x);
}

template <typename T> __host__ __device__ T lerp(const T &a, const T &b, float t) {
    return a + t * (b - a);
}

__device__ __forceinline__ float fmax2(float a, float b) {
    return a >= b ? a : b;
}

__device__ __forceinline__ Vec3 getEnvironmentLight(Ray ray) {
    Vec3 skyColourHorizon = Vec3(1.f, 1.f, 1.f);
    Vec3 skyColourZenith = Vec3(0.42f, 0.737f, 1.f);
    Vec3 groundColour = Vec3(0.2f, 0.2f, 0.2f);

    int sunFocus = 100; // lower = larger sun
    float sunIntensity = 50.0f;
    Vec3 sunLightDir = Vec3(0, -1, -1).normalize(); // test directly forward

    // sky gradient
    float skyGradientT = pow(smoothstep(0, 0.4f, -ray.dir.y), 0.35f);
    Vec3 skyGradient = lerp(skyColourHorizon, skyColourZenith, skyGradientT);

    // sun disk
    float sunAmount = powf(fmaxf(0.0f, ray.dir.dot(sunLightDir)), sunFocus) * sunIntensity;

    // ground blend (0 = ground, 1 = sky)
    float groundToSkyT = smoothstep(-0.01f, 0.0f, -ray.dir.y);

    // combine
    Vec3 base = lerp(groundColour, skyGradient, groundToSkyT);
    return base + Vec3(sunAmount) * groundToSkyT;
}