/// @file GpuMesher.cu
/// @brief Basic CUDA marching-cubes mesher for FLIP/MPM particle sets.
///
/// Pipeline: splat particles (quadratic B-spline) into a dense scalar grid,
/// classify active cells, then march cubes with edge-hash vertex welding.
/// Marching cubes tables follow the classic Paul Bourke convention
/// (corners v0..v7, edges 0..11, "inside" = field < iso).

#include "flipcore/GpuMesher.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>

#define CUDA_CHECK(call) do {                                    \
    cudaError_t _e = (call);                                     \
    if (_e != cudaSuccess) {                                     \
        fprintf(stderr, "GpuMesher CUDA error %s:%d: %s\n",      \
                __FILE__, __LINE__, cudaGetErrorString(_e));      \
    }                                                            \
} while(0)

// ── Quadratic B-spline (same corrected kernel as MpmSolver.cu) ─────────────

__host__ __device__ inline float bsplineN(float x) {
    float ax = fabsf(x);
    if (ax <= 0.5f) return 0.75f - ax * ax;
    if (ax < 1.5f) {
        float t = 1.5f - ax;
        return 0.5f * t * t;
    }
    return 0.0f;
}

// ── Murmur3 finalizer (edge-hash welding) ──────────────────────────────────

__host__ __device__ inline uint32_t murmur3_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

// Edge key: axis(2b) | i(10b) | j(10b) | k(10b)
__host__ __device__ inline uint32_t edgeKey(int axis, int i, int j, int k) {
    return (uint32_t(axis) << 30u) | (uint32_t(i & 1023) << 20u)
         | (uint32_t(j & 1023) << 10u) | uint32_t(k & 1023);
}

// ── Marching cubes tables (Paul Bourke convention) ─────────────────────────

__constant__ int c_edgeTable[256] = {
0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   };

__constant__ int c_triTable[256][16] = {
{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1,-1},
{3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1,-1},
{3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1,-1},
{3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1,-1},
{9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,1,9,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,4,7,3,0,4,1,2,10,-1,-1,-1,-1,-1,-1,-1},
{9,2,10,9,0,2,8,4,7,-1,-1,-1,-1,-1,-1,-1},
{2,9,7,2,7,10,10,7,3,7,9,4,-1,-1,-1,-1},
{8,4,7,3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{11,4,7,11,2,4,2,0,4,-1,-1,-1,-1,-1,-1,-1},
{9,0,1,8,4,7,2,3,11,-1,-1,-1,-1,-1,-1,-1},
{4,7,11,9,4,11,9,11,2,9,2,1,-1,-1,-1,-1},
{3,10,1,3,11,10,7,8,4,-1,-1,-1,-1,-1,-1,-1},
{1,11,10,1,4,11,1,0,4,7,11,4,-1,-1,-1,-1},
{4,7,8,9,0,11,9,11,10,11,0,3,-1,-1,-1,-1},
{4,7,11,4,11,9,9,11,10,-1,-1,-1,-1,-1,-1,-1},
{9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,5,4,0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,5,4,1,5,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{8,5,4,8,3,5,3,1,5,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,0,8,1,2,10,4,9,5,-1,-1,-1,-1,-1,-1,-1},
{5,2,10,5,4,2,4,0,2,-1,-1,-1,-1,-1,-1,-1},
{2,10,5,3,2,5,3,5,4,3,4,8,-1,-1,-1,-1},
{9,5,4,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,11,2,0,8,11,4,9,5,-1,-1,-1,-1,-1,-1,-1},
{0,5,4,0,1,5,2,3,11,-1,-1,-1,-1,-1,-1,-1},
{2,1,5,2,5,8,2,8,11,4,8,5,-1,-1,-1,-1},
{10,3,11,10,1,3,9,5,4,-1,-1,-1,-1,-1,-1,-1},
{4,9,5,0,8,1,8,10,1,8,11,10,-1,-1,-1,-1},
{5,4,0,5,0,11,5,11,10,11,0,3,-1,-1,-1,-1},
{5,4,8,5,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1},
{9,7,8,5,7,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,3,0,9,5,3,5,7,3,-1,-1,-1,-1,-1,-1,-1},
{0,7,8,0,1,7,1,5,7,-1,-1,-1,-1,-1,-1,-1},
{1,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,7,8,9,5,7,10,1,2,-1,-1,-1,-1,-1,-1,-1},
{10,1,2,9,5,0,5,3,0,5,7,3,-1,-1,-1,-1},
{8,0,2,8,2,5,8,5,7,10,5,2,-1,-1,-1,-1},
{2,10,5,2,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1},
{7,9,5,7,8,9,3,11,2,-1,-1,-1,-1,-1,-1,-1},
{9,5,7,9,7,2,9,2,0,2,7,11,-1,-1,-1,-1},
{2,3,11,0,1,8,1,7,8,1,5,7,-1,-1,-1,-1},
{11,2,1,11,1,7,7,1,5,-1,-1,-1,-1,-1,-1,-1},
{9,5,8,8,5,7,10,1,3,10,3,11,-1,-1,-1,-1},
{5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1},
{11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1},
{11,10,5,7,11,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,0,1,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,8,3,1,9,8,5,10,6,-1,-1,-1,-1,-1,-1,-1},
{1,6,5,2,6,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,6,5,1,2,6,3,0,8,-1,-1,-1,-1,-1,-1,-1},
{9,6,5,9,0,6,0,2,6,-1,-1,-1,-1,-1,-1,-1},
{5,9,8,5,8,2,5,2,6,3,2,8,-1,-1,-1,-1},
{2,3,11,10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{11,0,8,11,2,0,10,6,5,-1,-1,-1,-1,-1,-1,-1},
{0,1,9,2,3,11,5,10,6,-1,-1,-1,-1,-1,-1,-1},
{5,10,6,1,9,2,9,11,2,9,8,11,-1,-1,-1,-1},
{6,3,11,6,5,3,5,1,3,-1,-1,-1,-1,-1,-1,-1},
{0,8,11,0,11,5,0,5,1,5,11,6,-1,-1,-1,-1},
{3,11,6,0,3,6,0,6,5,0,5,9,-1,-1,-1,-1},
{6,5,9,6,9,11,11,9,8,-1,-1,-1,-1,-1,-1,-1},
{5,10,6,4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,3,0,4,7,3,6,5,10,-1,-1,-1,-1,-1,-1,-1},
{1,9,0,5,10,6,8,4,7,-1,-1,-1,-1,-1,-1,-1},
{10,6,5,1,9,7,1,7,3,7,9,4,-1,-1,-1,-1},
{6,1,2,6,5,1,4,7,8,-1,-1,-1,-1,-1,-1,-1},
{1,2,5,5,2,6,3,0,4,3,4,7,-1,-1,-1,-1},
{8,4,7,9,0,5,0,6,5,0,2,6,-1,-1,-1,-1},
{7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1},
{3,11,2,7,8,4,10,6,5,-1,-1,-1,-1,-1,-1,-1},
{5,10,6,4,7,2,4,2,0,2,7,11,-1,-1,-1,-1},
{0,1,9,4,7,8,2,3,11,5,10,6,-1,-1,-1,-1},
{9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1},
{8,4,7,3,11,5,3,5,1,5,11,6,-1,-1,-1,-1},
{5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1},
{0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1},
{6,5,9,6,9,11,4,7,9,7,11,9,-1,-1,-1,-1},
{10,4,9,6,4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,10,6,4,9,10,0,8,3,-1,-1,-1,-1,-1,-1,-1},
{10,0,1,10,6,0,6,4,0,-1,-1,-1,-1,-1,-1,-1},
{8,3,1,8,1,6,8,6,4,6,1,10,-1,-1,-1,-1},
{1,4,9,1,2,4,2,6,4,-1,-1,-1,-1,-1,-1,-1},
{3,0,8,1,2,9,2,4,9,2,6,4,-1,-1,-1,-1},
{0,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{8,3,2,8,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1},
{10,4,9,10,6,4,11,2,3,-1,-1,-1,-1,-1,-1,-1},
{0,8,2,2,8,11,4,9,10,4,10,6,-1,-1,-1,-1},
{3,11,2,0,1,6,0,6,4,6,1,10,-1,-1,-1,-1},
{6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1},
{9,6,4,9,3,6,9,1,3,11,6,3,-1,-1,-1,-1},
{8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1},
{3,11,6,3,6,0,0,6,4,-1,-1,-1,-1,-1,-1,-1},
{6,4,8,11,6,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{7,10,6,7,8,10,8,9,10,-1,-1,-1,-1,-1,-1,-1},
{0,7,3,0,10,7,0,9,10,6,7,10,-1,-1,-1,-1},
{10,6,7,1,10,7,1,7,8,1,8,0,-1,-1,-1,-1},
{10,6,7,10,7,1,1,7,3,-1,-1,-1,-1,-1,-1,-1},
{1,2,6,1,6,8,1,8,9,8,6,7,-1,-1,-1,-1},
{2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1},
{7,8,0,7,0,6,6,0,2,-1,-1,-1,-1,-1,-1,-1},
{7,3,2,6,7,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{2,3,11,10,6,8,10,8,9,8,6,7,-1,-1,-1,-1},
{2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1},
{1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1},
{11,2,1,11,1,7,10,6,1,6,7,1,-1,-1,-1,-1},
{8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1},
{0,9,1,11,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{7,8,0,7,0,6,3,11,0,11,6,0,-1,-1,-1,-1},
{7,11,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,0,8,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,1,9,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{8,1,9,8,3,1,11,7,6,-1,-1,-1,-1,-1,-1,-1},
{10,1,2,6,11,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,3,0,8,6,11,7,-1,-1,-1,-1,-1,-1,-1},
{2,9,0,2,10,9,6,11,7,-1,-1,-1,-1,-1,-1,-1},
{6,11,7,2,10,3,10,8,3,10,9,8,-1,-1,-1,-1},
{7,2,3,6,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{7,0,8,7,6,0,6,2,0,-1,-1,-1,-1,-1,-1,-1},
{2,7,6,2,3,7,0,1,9,-1,-1,-1,-1,-1,-1,-1},
{1,6,2,1,8,6,1,9,8,8,7,6,-1,-1,-1,-1},
{10,7,6,10,1,7,1,3,7,-1,-1,-1,-1,-1,-1,-1},
{10,7,6,1,7,10,1,8,7,1,0,8,-1,-1,-1,-1},
{0,3,7,0,7,10,0,10,9,6,10,7,-1,-1,-1,-1},
{7,6,10,7,10,8,8,10,9,-1,-1,-1,-1,-1,-1,-1},
{6,8,4,11,8,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,6,11,3,0,6,0,4,6,-1,-1,-1,-1,-1,-1,-1},
{8,6,11,8,4,6,9,0,1,-1,-1,-1,-1,-1,-1,-1},
{9,4,6,9,6,3,9,3,1,11,3,6,-1,-1,-1,-1},
{6,8,4,6,11,8,2,10,1,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,3,0,11,0,6,11,0,4,6,-1,-1,-1,-1},
{4,11,8,4,6,11,0,2,9,2,10,9,-1,-1,-1,-1},
{10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1},
{8,2,3,8,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1},
{0,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,9,0,2,3,4,2,4,6,4,3,8,-1,-1,-1,-1},
{1,9,4,1,4,2,2,4,6,-1,-1,-1,-1,-1,-1,-1},
{8,1,3,8,6,1,8,4,6,6,10,1,-1,-1,-1,-1},
{10,1,0,10,0,6,6,0,4,-1,-1,-1,-1,-1,-1,-1},
{4,6,3,4,3,8,6,10,3,0,3,9,10,9,3,-1},
{10,9,4,6,10,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,9,5,7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,4,9,5,11,7,6,-1,-1,-1,-1,-1,-1,-1},
{5,0,1,5,4,0,7,6,11,-1,-1,-1,-1,-1,-1,-1},
{11,7,6,8,3,4,3,5,4,3,1,5,-1,-1,-1,-1},
{9,5,4,10,1,2,7,6,11,-1,-1,-1,-1,-1,-1,-1},
{6,11,7,1,2,10,0,8,3,4,9,5,-1,-1,-1,-1},
{7,6,11,5,4,10,4,2,10,4,0,2,-1,-1,-1,-1},
{3,4,8,3,5,4,3,2,5,10,5,2,11,7,6,-1},
{7,2,3,7,6,2,5,4,9,-1,-1,-1,-1,-1,-1,-1},
{9,5,4,0,8,6,0,6,2,6,8,7,-1,-1,-1,-1},
{3,6,2,3,7,6,1,5,0,5,4,0,-1,-1,-1,-1},
{6,2,8,6,8,7,2,1,8,4,8,5,1,5,8,-1},
{9,5,4,10,1,6,1,7,6,1,3,7,-1,-1,-1,-1},
{1,6,10,1,7,6,1,0,7,8,7,0,9,5,4,-1},
{4,0,10,4,10,5,0,3,10,6,10,7,3,7,10,-1},
{7,6,10,7,10,8,5,4,10,4,8,10,-1,-1,-1,-1},
{6,9,5,6,11,9,11,8,9,-1,-1,-1,-1,-1,-1,-1},
{3,6,11,0,6,3,0,5,6,0,9,5,-1,-1,-1,-1},
{0,11,8,0,5,11,0,1,5,5,6,11,-1,-1,-1,-1},
{6,11,3,6,3,5,5,3,1,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,9,5,11,9,11,8,11,5,6,-1,-1,-1,-1},
{0,11,3,0,6,11,0,9,6,5,6,9,1,2,10,-1},
{11,8,5,11,5,6,8,0,5,10,5,2,0,2,5,-1},
{6,11,3,6,3,5,2,10,3,10,5,3,-1,-1,-1,-1},
{5,8,9,5,2,8,5,6,2,3,8,2,-1,-1,-1,-1},
{9,5,6,9,6,0,0,6,2,-1,-1,-1,-1,-1,-1,-1},
{1,5,8,1,8,0,5,6,8,3,8,2,6,2,8,-1},
{1,5,6,2,1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,3,6,1,6,10,3,8,6,5,6,9,8,9,6,-1},
{10,1,0,10,0,6,9,5,0,5,6,0,-1,-1,-1,-1},
{0,3,8,5,6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{10,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{11,5,10,7,5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{11,5,10,11,7,5,8,3,0,-1,-1,-1,-1,-1,-1,-1},
{5,11,7,5,10,11,1,9,0,-1,-1,-1,-1,-1,-1,-1},
{10,7,5,10,11,7,9,8,1,8,3,1,-1,-1,-1,-1},
{11,1,2,11,7,1,7,5,1,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,1,2,7,1,7,5,7,2,11,-1,-1,-1,-1},
{9,7,5,9,2,7,9,0,2,2,11,7,-1,-1,-1,-1},
{7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1},
{2,5,10,2,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1},
{8,2,0,8,5,2,8,7,5,10,2,5,-1,-1,-1,-1},
{9,0,1,5,10,3,5,3,7,3,10,2,-1,-1,-1,-1},
{9,8,2,9,2,1,8,7,2,10,2,5,7,5,2,-1},
{1,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,8,7,0,7,1,1,7,5,-1,-1,-1,-1,-1,-1,-1},
{9,0,3,9,3,5,5,3,7,-1,-1,-1,-1,-1,-1,-1},
{9,8,7,5,9,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{5,8,4,5,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1},
{5,0,4,5,11,0,5,10,11,11,3,0,-1,-1,-1,-1},
{0,1,9,8,4,10,8,10,11,10,4,5,-1,-1,-1,-1},
{10,11,4,10,4,5,11,3,4,9,4,1,3,1,4,-1},
{2,5,1,2,8,5,2,11,8,4,5,8,-1,-1,-1,-1},
{0,4,11,0,11,3,4,5,11,2,11,1,5,1,11,-1},
{0,2,5,0,5,9,2,11,5,4,5,8,11,8,5,-1},
{9,4,5,2,11,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{2,5,10,3,5,2,3,4,5,3,8,4,-1,-1,-1,-1},
{5,10,2,5,2,4,4,2,0,-1,-1,-1,-1,-1,-1,-1},
{3,10,2,3,5,10,3,8,5,4,5,8,0,1,9,-1},
{5,10,2,5,2,4,1,9,2,9,4,2,-1,-1,-1,-1},
{8,4,5,8,5,3,3,5,1,-1,-1,-1,-1,-1,-1,-1},
{0,4,5,1,0,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{8,4,5,8,5,3,9,0,5,0,3,5,-1,-1,-1,-1},
{9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,11,7,4,9,11,9,10,11,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,4,9,7,9,11,7,9,10,11,-1,-1,-1,-1},
{1,10,11,1,11,4,1,4,0,7,4,11,-1,-1,-1,-1},
{3,1,4,3,4,8,1,10,4,7,4,11,10,11,4,-1},
{4,11,7,9,11,4,9,2,11,9,1,2,-1,-1,-1,-1},
{9,7,4,9,11,7,9,1,11,2,11,1,0,8,3,-1},
{11,7,4,11,4,2,2,4,0,-1,-1,-1,-1,-1,-1,-1},
{11,7,4,11,4,2,8,3,4,3,2,4,-1,-1,-1,-1},
{2,9,10,2,7,9,2,3,7,7,4,9,-1,-1,-1,-1},
{9,10,7,9,7,4,10,2,7,8,7,0,2,0,7,-1},
{3,7,10,3,10,2,7,4,10,1,10,0,4,0,10,-1},
{1,10,2,8,7,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,9,1,4,1,7,7,1,3,-1,-1,-1,-1,-1,-1,-1},
{4,9,1,4,1,7,0,8,1,8,7,1,-1,-1,-1,-1},
{4,0,3,7,4,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,0,9,3,9,11,11,9,10,-1,-1,-1,-1,-1,-1,-1},
{0,1,10,0,10,8,8,10,11,-1,-1,-1,-1,-1,-1,-1},
{3,1,10,11,3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,2,11,1,11,9,9,11,8,-1,-1,-1,-1,-1,-1,-1},
{3,0,9,3,9,11,1,2,9,2,11,9,-1,-1,-1,-1},
{0,2,11,8,0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{2,3,8,2,8,10,10,8,9,-1,-1,-1,-1,-1,-1,-1},
{9,10,2,0,9,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{2,3,8,2,8,10,0,1,8,1,10,8,-1,-1,-1,-1},
{1,10,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,3,8,9,1,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,9,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}};

// Edge → (axis, di, dj, dk): canonical grid-edge index of each MC edge.
// Corners: v0=(0,0,0) v1=(1,0,0) v2=(1,1,0) v3=(0,1,0)
//          v4=(0,0,1) v5=(1,0,1) v6=(1,1,1) v7=(0,1,1)
__constant__ int c_edgeAxis[12] = {0,1,0,1,0,1,0,1,2,2,2,2};
__constant__ int c_edgeDi[12]   = {0,1,0,0,0,1,0,0,0,1,1,0};
__constant__ int c_edgeDj[12]   = {0,0,1,0,0,0,1,0,0,0,1,1};
__constant__ int c_edgeDk[12]   = {0,0,0,0,1,1,1,1,0,0,0,0};

// Corner offsets for a cell (i,j,k): index = i + di, j + dj, k + dk
__constant__ int c_cornerDi[8] = {0,1,1,0,0,1,1,0};
__constant__ int c_cornerDj[8] = {0,0,1,1,0,0,1,1};
__constant__ int c_cornerDk[8] = {0,0,0,0,1,1,1,1};

namespace flipcore {

// ═══════════════════════════════════════════════════════════════════════════
// Kernels
// ═══════════════════════════════════════════════════════════════════════════

__global__ void zeroFieldKernel(float* field, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) field[i] = 0.0f;
}

// Splat particles into the field with a quadratic B-spline (3×3×3 stencil)
__global__ void splatKernel(
    const float* positions, int numParticles,
    float* field, int nx, int ny, int nz,
    float originX, float originY, float originZ, float h)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= numParticles) return;

    float xp = (positions[3*p+0] - originX) / h;
    float yp = (positions[3*p+1] - originY) / h;
    float zp = (positions[3*p+2] - originZ) / h;

    int baseX = (int)floorf(xp - 0.5f);
    int baseY = (int)floorf(yp - 0.5f);
    int baseZ = (int)floorf(zp - 0.5f);

    for (int dx = 0; dx < 3; dx++) {
        int gx = baseX + dx;
        if (gx < 0 || gx >= nx) continue;
        float wx = bsplineN(xp - (float)gx);
        for (int dy = 0; dy < 3; dy++) {
            int gy = baseY + dy;
            if (gy < 0 || gy >= ny) continue;
            float wy = bsplineN(yp - (float)gy);
            for (int dz = 0; dz < 3; dz++) {
                int gz = baseZ + dz;
                if (gz < 0 || gz >= nz) continue;
                float wz = bsplineN(zp - (float)gz);
                float w = wx * wy * wz;
                if (w <= 0.0f) continue;
                int idx = gx + nx * (gy + ny * gz);
                atomicAdd(&field[idx], w);
            }
        }
    }
}

// Classify active cells (edgeTable[cubeIdx] != 0) and compact to a list
__global__ void classifyKernel(
    const float* field, int nx, int ny, int nz,
    float iso, int* activeList, uint32_t* activeCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nxy = nx * ny;
    int cellCount = (nx-1) * (ny-1) * (nz-1);
    if (idx >= cellCount) return;

    int i = idx % (nx-1);
    int j = (idx / (nx-1)) % (ny-1);
    int k = idx / ((nx-1)*(ny-1));

    int cubeIdx = 0;
    #pragma unroll
    for (int c = 0; c < 8; c++) {
        int gi = i + c_cornerDi[c];
        int gj = j + c_cornerDj[c];
        int gk = k + c_cornerDk[c];
        float v = field[gi + nx*(gj + ny*gk)];
        if (v < iso) cubeIdx |= (1 << c);
    }
    if (c_edgeTable[cubeIdx] != 0) {
        uint32_t slot = atomicAdd(activeCount, 1u);
        activeList[slot] = idx;
    }
}

// Marching cubes over active cells, welding shared grid edges via hash table.
// The table stores (edgeKey << 32) | vertexId packed in 64 bits, so probing
// verifies the exact edge and publishes its vertex id atomically.
__global__ void marchingCubesKernel(
    const int* activeList, uint32_t activeCount,
    const float* field, int nx, int ny, int nz,
    float iso, float originX, float originY, float originZ, float h,
    float* outVerts, uint32_t* vertCount,
    uint32_t* outTris, uint32_t* triCount,
    unsigned long long* edgeTable2, uint32_t edgeMask)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (int)activeCount) return;

    int cellIdx = activeList[t];
    int i = cellIdx % (nx-1);
    int j = (cellIdx / (nx-1)) % (ny-1);
    int k = cellIdx / ((nx-1)*(ny-1));

    // Corner values + positions
    float v[8];
    #pragma unroll
    for (int c = 0; c < 8; c++) {
        int gi = i + c_cornerDi[c];
        int gj = j + c_cornerDj[c];
        int gk = k + c_cornerDk[c];
        v[c] = field[gi + nx*(gj + ny*gk)];
    }

    int cubeIdx = 0;
    #pragma unroll
    for (int c = 0; c < 8; c++)
        if (v[c] < iso) cubeIdx |= (1 << c);

    int edgeMaskBits = c_edgeTable[cubeIdx];
    if (edgeMaskBits == 0) return;

    // Local edge → global vertex id
    int localVert[12];
    #pragma unroll
    for (int e = 0; e < 12; e++) localVert[e] = -1;

    // Corner world positions
    float px[8], py[8], pz[8];
    #pragma unroll
    for (int c = 0; c < 8; c++) {
        px[c] = originX + (float)(i + c_cornerDi[c]) * h;
        py[c] = originY + (float)(j + c_cornerDj[c]) * h;
        pz[c] = originZ + (float)(k + c_cornerDk[c]) * h;
    }

    // For each used edge: compute canonical grid-edge key, weld via hash
    for (int e = 0; e < 12; e++) {
        if (!(edgeMaskBits & (1 << e))) continue;

        int axis = c_edgeAxis[e];
        int ei = i + c_edgeDi[e];
        int ej = j + c_edgeDj[e];
        int ek = k + c_edgeDk[e];
        uint32_t key = edgeKey(axis, ei, ej, ek);

        uint32_t slot = murmur3_u32(key) & edgeMask;
        int vertId = -1;
        // Sentinel: high word 0xFFFFFFFF never appears in a valid entry
        // (valid entries pack key < 0xC0000000 in the high word).
        const unsigned long long PENDING = 0xffffffff00000000ull;
        while (true) {
            unsigned long long cur = atomicAdd(&edgeTable2[slot], 0ull);
            if (cur == PENDING) continue;   // winner is mid-claim — retry
            if (cur == 0ull) {
                unsigned long long old = atomicCAS(&edgeTable2[slot], 0ull, PENDING);
                if (old != 0ull) continue;  // lost race — re-read
                uint32_t vid = atomicAdd(vertCount, 1u);
                unsigned long long val =
                    ((unsigned long long)key << 32u) | (unsigned long long)vid;
                atomicExch(&edgeTable2[slot], val);   // publish
                vertId = (int)vid;
                break;
            }
            if ((uint32_t)(cur >> 32u) == key) { vertId = (int)(cur & 0xffffffffull); break; }
            slot = (slot + 1u) & edgeMask;
        }
        localVert[e] = vertId;

        // Compute interpolated vertex position and write (only the claiming
        // thread reaches the write for a given id)
        int c0, c1;
        // Bourke edge endpoints
        const int EP0[12] = {0,1,2,3,4,5,6,7,0,1,2,3};
        const int EP1[12] = {1,2,3,0,5,6,7,4,4,5,6,7};
        c0 = EP0[e]; c1 = EP1[e];
        float denom = v[c1] - v[c0];
        float mu = (fabsf(denom) > 1e-12f) ? (iso - v[c0]) / denom : 0.5f;
        mu = fminf(fmaxf(mu, 0.0f), 1.0f);
        float wx = px[c0] + mu * (px[c1] - px[c0]);
        float wy = py[c0] + mu * (py[c1] - py[c0]);
        float wz = pz[c0] + mu * (pz[c1] - pz[c0]);
        outVerts[vertId*3+0] = wx;
        outVerts[vertId*3+1] = wy;
        outVerts[vertId*3+2] = wz;
    }

    // Emit triangles (reversed winding → normals point to decreasing field,
    // i.e. outward for a density field that is large inside)
    int triCountCell = 0;
    #pragma unroll
    for (int n = 0; n < 16; n += 3) {
        int eA = c_triTable[cubeIdx][n];
        if (eA < 0) break;
        int eB = c_triTable[cubeIdx][n+1];
        int eC = c_triTable[cubeIdx][n+2];
        uint32_t slot = atomicAdd(triCount, 1u);
        outTris[slot*3+0] = (uint32_t)localVert[eA];
        outTris[slot*3+1] = (uint32_t)localVert[eC];  // reversed winding
        outTris[slot*3+2] = (uint32_t)localVert[eB];
        triCountCell++;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Host side
// ═══════════════════════════════════════════════════════════════════════════

static inline size_t nextPow2(size_t v) {
    size_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

bool particlesToMeshGpu(const float* positions, size_t numParticles,
                        float voxelSize, float iso,
                        std::vector<float>& outVertices,
                        std::vector<uint32_t>& outTriangles)
{
    outVertices.clear();
    outTriangles.clear();
    if (numParticles < 4 || voxelSize <= 0.0f) return false;

    // Bounding box
    float mn[3] = {positions[0], positions[1], positions[2]};
    float mx[3] = {mn[0], mn[1], mn[2]};
    for (size_t p = 1; p < numParticles; p++) {
        for (int d = 0; d < 3; d++) {
            float c = positions[3*p + d];
            if (c < mn[d]) mn[d] = c;
            if (c > mx[d]) mx[d] = c;
        }
    }

    const float pad = 3.0f * voxelSize;   // room for the kernel support radius
    int nx, ny, nz;
    {
        nx = (int)ceilf((mx[0] + pad - (mn[0] - pad)) / voxelSize) + 1;
        ny = (int)ceilf((mx[1] + pad - (mn[1] - pad)) / voxelSize) + 1;
        nz = (int)ceilf((mx[2] + pad - (mn[2] - pad)) / voxelSize) + 1;
        if (nx < 9) nx = 9;
        if (ny < 9) ny = 9;
        if (nz < 9) nz = 9;

        const size_t maxCells = 32000000ull;   // 32M floats = 128 MB
        while ((size_t)nx * (size_t)ny * (size_t)nz > maxCells) {
            voxelSize *= 1.25f;  // coarsen and re-derive dimensions
            nx = (int)ceilf((mx[0] + pad - (mn[0] - pad)) / voxelSize) + 1;
            ny = (int)ceilf((mx[1] + pad - (mn[1] - pad)) / voxelSize) + 1;
            nz = (int)ceilf((mx[2] + pad - (mn[2] - pad)) / voxelSize) + 1;
        }
    }

    const float originX = mn[0] - pad;
    const float originY = mn[1] - pad;
    const float originZ = mn[2] - pad;
    const size_t fieldCount = (size_t)nx * (size_t)ny * (size_t)nz;

    // Device field + particle positions
    float* d_field = nullptr;
    float* d_positions = nullptr;
    CUDA_CHECK(cudaMalloc(&d_field, fieldCount * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_field, 0, fieldCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_positions, numParticles * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_positions, positions, numParticles * 3 * sizeof(float),
                          cudaMemcpyHostToDevice));

    int b = 256;
    int pGrid = ((int)numParticles + b - 1) / b;
    splatKernel<<<pGrid, b>>>(
        d_positions, (int)numParticles, d_field, nx, ny, nz,
        originX, originY, originZ, voxelSize);

    int cellCount = (nx-1) * (ny-1) * (nz-1);
    int* d_activeList = nullptr;
    uint32_t* d_activeCount = nullptr;
    CUDA_CHECK(cudaMalloc(&d_activeList, cellCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_activeCount, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_activeCount, 0, sizeof(uint32_t)));

    int cGrid = (cellCount + b - 1) / b;
    classifyKernel<<<cGrid, b>>>(
        d_field, nx, ny, nz, iso, d_activeList, d_activeCount);

    uint32_t activeCount = 0;
    CUDA_CHECK(cudaMemcpy(&activeCount, d_activeCount, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    if (activeCount == 0) {
        CUDA_CHECK(cudaFree(d_field));
        CUDA_CHECK(cudaFree(d_positions));
        CUDA_CHECK(cudaFree(d_activeList));
        CUDA_CHECK(cudaFree(d_activeCount));
        return false;
    }

    // Vertex/triangle buffers (worst-case sizing)
    const uint32_t maxVerts = activeCount * 12u;
    const uint32_t maxTris  = activeCount * 5u;
    float* d_verts = nullptr;
    uint32_t* d_tris = nullptr;
    uint32_t* d_vertCount = nullptr;
    uint32_t* d_triCount = nullptr;
    unsigned long long* d_edgeTable = nullptr;
    size_t edgeTableSize = nextPow2((size_t)maxVerts * 2u);
    CUDA_CHECK(cudaMalloc(&d_verts, (size_t)maxVerts * 3u * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tris, (size_t)maxTris * 3u * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_vertCount, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_triCount, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_edgeTable, edgeTableSize * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_vertCount, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_triCount, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_edgeTable, 0, edgeTableSize * sizeof(unsigned long long)));

    int mGrid = ((int)activeCount + b - 1) / b;
    marchingCubesKernel<<<mGrid, b>>>(
        d_activeList, activeCount, d_field, nx, ny, nz, iso,
        originX, originY, originZ, voxelSize,
        d_verts, d_vertCount, d_tris, d_triCount,
        d_edgeTable, (uint32_t)(edgeTableSize - 1));

    uint32_t vertCount = 0, triCount = 0;
    CUDA_CHECK(cudaMemcpy(&vertCount, d_vertCount, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&triCount, d_triCount, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    if (vertCount > 0 && triCount > 0) {
        outVertices.resize((size_t)vertCount * 3u);
        outTriangles.resize((size_t)triCount * 3u);
        CUDA_CHECK(cudaMemcpy(outVertices.data(), d_verts,
                              (size_t)vertCount * 3u * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(outTriangles.data(), d_tris,
                              (size_t)triCount * 3u * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_edgeTable));
    CUDA_CHECK(cudaFree(d_triCount));
    CUDA_CHECK(cudaFree(d_vertCount));
    CUDA_CHECK(cudaFree(d_tris));
    CUDA_CHECK(cudaFree(d_verts));
    CUDA_CHECK(cudaFree(d_activeCount));
    CUDA_CHECK(cudaFree(d_activeList));
    CUDA_CHECK(cudaFree(d_field));
    CUDA_CHECK(cudaFree(d_positions));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return !outVertices.empty() && !outTriangles.empty();
}

} // namespace flipcore
