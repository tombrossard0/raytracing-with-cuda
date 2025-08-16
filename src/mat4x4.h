#pragma once

struct Mat4x4
{
    float m[4][4];
};

__device__ Mat4x4 identityMatrix()
{
    Mat4x4 mat = {};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            mat.m[i][j] = (i == j) ? 1.0f : 0.0f;
    return mat;
}