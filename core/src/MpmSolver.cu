/// @file MpmSolver.cu
/// @brief CUDA APIC-FLIP MPM solver with a sparse murmur3 hash grid,
///        fixed-corotated elasticity, von Mises plasticity, and an
///        optional sand model (sand_alpha). Architecture follows the
///        squishy_volumes sparse-grid / APIC transfer scheme.

#include "flipcore/MpmSolver.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>

// ── Minimal 3×3 math (column-major layout, float[9]) ───────────────────────

#define IDX(i,j) ((i)+(j)*3)

__host__ __device__ inline void mat3_identity(float* M) {
    for (int i = 0; i < 9; i++) M[i] = 0.0f;
    M[IDX(0,0)] = M[IDX(1,1)] = M[IDX(2,2)] = 1.0f;
}

__host__ __device__ inline void mat3_mul(const float* A, const float* B, float* C) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            C[IDX(i,j)] = A[IDX(i,0)]*B[IDX(0,j)]
                        + A[IDX(i,1)]*B[IDX(1,j)]
                        + A[IDX(i,2)]*B[IDX(2,j)];
        }
    }
}

__host__ __device__ inline void mat3_mul_vec(const float* M, const float* v, float* out) {
    for (int i = 0; i < 3; i++)
        out[i] = M[IDX(i,0)]*v[0] + M[IDX(i,1)]*v[1] + M[IDX(i,2)]*v[2];
}

__host__ __device__ inline float mat3_det(const float* M) {
    return M[IDX(0,0)] * (M[IDX(1,1)]*M[IDX(2,2)] - M[IDX(1,2)]*M[IDX(2,1)])
         - M[IDX(0,1)] * (M[IDX(1,0)]*M[IDX(2,2)] - M[IDX(1,2)]*M[IDX(2,0)])
         + M[IDX(0,2)] * (M[IDX(1,0)]*M[IDX(2,1)] - M[IDX(1,1)]*M[IDX(2,0)]);
}

__host__ __device__ inline float mat3_trace(const float* M) {
    return M[IDX(0,0)] + M[IDX(1,1)] + M[IDX(2,2)];
}

__host__ __device__ inline void mat3_transpose(const float* M, float* out) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            out[IDX(i,j)] = M[IDX(j,i)];
}

__host__ __device__ inline void mat3_add_I_times(const float* M, float s, float* out) {
    for (int i = 0; i < 9; i++) out[i] = M[i];
    out[IDX(0,0)] += s; out[IDX(1,1)] += s; out[IDX(2,2)] += s;
}

__host__ __device__ inline void mat3_scale(float* M, float s) {
    for (int i = 0; i < 9; i++) M[i] *= s;
}

__host__ __device__ inline void mat3_inverse(const float* M, float* out) {
    // Adjugate / determinant (3x3 inverse via cofactors)
    float a = M[IDX(0,0)], b = M[IDX(0,1)], c = M[IDX(0,2)];
    float d = M[IDX(1,0)], e = M[IDX(1,1)], f = M[IDX(1,2)];
    float g = M[IDX(2,0)], h = M[IDX(2,1)], i = M[IDX(2,2)];

    float A =  (e*i - f*h), B = -(d*i - f*g), C =  (d*h - e*g);
    float D = -(b*i - c*h), E =  (a*i - c*g), F = -(a*h - b*g);
    float G =  (b*f - c*e), H = -(a*f - c*d), I =  (a*e - b*d);

    float det = a*A + b*B + c*C;
    float invDet = (fabsf(det) > 1e-12f) ? 1.0f / det : 0.0f;

    out[IDX(0,0)] = A*invDet; out[IDX(0,1)] = D*invDet; out[IDX(0,2)] = G*invDet;
    out[IDX(1,0)] = B*invDet; out[IDX(1,1)] = E*invDet; out[IDX(1,2)] = H*invDet;
    out[IDX(2,0)] = C*invDet; out[IDX(2,1)] = F*invDet; out[IDX(2,2)] = I*invDet;
}

// ── 3×3 SVD via power iteration (sufficient for MPM plasticity) ────────────

__host__ __device__ inline void svd3x3(const float* F,
                                        float* U, float* S, float* V) {
    // Build F^T * F
    float FtF[9];
    float Ft[9]; mat3_transpose(F, Ft);
    mat3_mul(Ft, F, FtF);

    // Power iteration for largest eigenvector of FtF
    float v[3] = {1.0f, 0.0f, 0.0f};
    for (int iter = 0; iter < 5; iter++) {
        float w[3];
        mat3_mul_vec(FtF, v, w);
        float len = sqrtf(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
        if (len > 1e-12f) { v[0] = w[0]/len; v[1] = w[1]/len; v[2] = w[2]/len; }
    }
    float sigma0 = 0.0f;
    { float w[3]; mat3_mul_vec(FtF, v, w); sigma0 = sqrtf(fmaxf(w[0]*v[0]+w[1]*v[1]+w[2]*v[2], 0.0f)); }

    // Deflate: remove v1 component, find next eigenvector
    float P1[9]; mat3_identity(P1);
    P1[IDX(0,0)] -= v[0]*v[0]; P1[IDX(0,1)] -= v[0]*v[1]; P1[IDX(0,2)] -= v[0]*v[2];
    P1[IDX(1,0)] -= v[1]*v[0]; P1[IDX(1,1)] -= v[1]*v[1]; P1[IDX(1,2)] -= v[1]*v[2];
    P1[IDX(2,0)] -= v[2]*v[0]; P1[IDX(2,1)] -= v[2]*v[1]; P1[IDX(2,2)] -= v[2]*v[2];

    float R1[9], tmp[9];
    mat3_mul(P1, FtF, tmp);
    mat3_mul(tmp, P1, R1);

    float v2[3] = {0.0f, 1.0f, 0.0f};
    for (int iter = 0; iter < 5; iter++) {
        float w[3];
        mat3_mul_vec(R1, v2, w);
        float len = sqrtf(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
        if (len > 1e-12f) { v2[0] = w[0]/len; v2[1] = w[1]/len; v2[2] = w[2]/len; }
    }
    float sigma1 = 0.0f;
    { float w[3]; mat3_mul_vec(FtF, v2, w); sigma1 = sqrtf(fmaxf(w[0]*v2[0]+w[1]*v2[1]+w[2]*v2[2], 0.0f)); }

    // Third singular value = sqrt(|det F| / (σ0·σ1))
    float det = fabsf(mat3_det(F));
    float sigma2 = (sigma0 * sigma1 > 1e-12f) ? det / (sigma0 * sigma1) : 0.0f;

    S[0] = sigma0; S[1] = sigma1; S[2] = sigma2;

    // Build U columns: u_i = F·v_i / σ_i
    for (int i = 0; i < 3; i++) {
        float Fi[3] = {F[IDX(0,i)], F[IDX(1,i)], F[IDX(2,i)]};
        // Actually u_i should use V columns. For now, compute V columns as
        // eigenvectors of F^T*F. We already have v (=V[:,0]) and v2.
        // Use cross product for V[:,2], then U = F * V * diag(1/σ)
    }

    // Build V from eigenvectors v, v2, and v×v2
    float v3[3] = {
        v[1]*v2[2] - v[2]*v2[1],
        v[2]*v2[0] - v[0]*v2[2],
        v[0]*v2[1] - v[1]*v2[0]
    };
    // V columns
    for (int r = 0; r < 3; r++) {
        V[IDX(r,0)] = v[r];
        V[IDX(r,1)] = v2[r];
        V[IDX(r,2)] = v3[r];
    }

    // U columns: u_i = F * v_i / σ_i
    float eps = 1e-10f;
    for (int c = 0; c < 3; c++) {
        float Vc[3] = {V[IDX(0,c)], V[IDX(1,c)], V[IDX(2,c)]};
        float Uc[3];
        mat3_mul_vec(F, Vc, Uc);
        float invSig = (c == 0) ? 1.0f/(S[0]+eps) : (c == 1 ? 1.0f/(S[1]+eps) : 1.0f/(S[2]+eps));
        for (int r = 0; r < 3; r++) U[IDX(r,c)] = Uc[r] * invSig;
    }
}

// ── Quadratic B-spline weight ──────────────────────────────────────────────

__host__ __device__ inline float bspline(float x) {
    // Quadratic B-spline: N(x) = 0.75 - x²            for |x| ≤ 0.5,
    //                             0.5·(1.5-|x|)²      for 0.5 < |x| ≤ 1.5,
    //                             0                   otherwise
    float ax = fabsf(x);
    if (ax <= 0.5f) {
        return 0.75f - ax * ax;
    } else if (ax < 1.5f) {
        float t = 1.5f - ax;
        return 0.5f * t * t;
    }
    return 0.0f;
}

__host__ __device__ inline float bsplineGrad(float x) {
    float ax = fabsf(x);
    float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    if (ax <= 0.5f) {
        return -2.0f * x;               // derivative of 0.75 - x²
    } else if (ax < 1.5f) {
        return -sign * (1.5f - ax);     // derivative of 0.5·(1.5-|x|)²
    }
    return 0.0f;
}

// ── Murmur3 hash for sparse grid cells (squishy_volumes scheme) ────────────

__host__ __device__ inline uint32_t murmur3_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

__host__ __device__ inline uint32_t cell_hash(int i, int j, int k) {
    uint32_t h = murmur3_u32((uint32_t)i);
    h = murmur3_u32(h ^ (uint32_t)j);
    h = murmur3_u32(h ^ (uint32_t)k);
    return h;
}

// ── Sparse grid lookup: returns node index for a cell, inserting it via
//    atomic CAS if missing. Linear probing with power-of-two table.
//    The __threadfence before the CAS publishes cell coords first, and the
//    atomic table read synchronizes with it, keeping the probe race-free. ────

__device__ inline int sparseGridInsert(
    uint32_t* hashTable, uint32_t* gridCount,
    int32_t* cellCoords, uint32_t tableMask,
    int ci, int cj, int ck)
{
    uint32_t slot = cell_hash(ci, cj, ck) & tableMask;
    while (true) {
        uint32_t cur = atomicAdd(&hashTable[slot], 0u);   // atomic (acquire-ish) read
        if (cur == 0u) {
            uint32_t nodeIdx = atomicAdd(gridCount, 1u);
            cellCoords[nodeIdx*3 + 0] = ci;
            cellCoords[nodeIdx*3 + 1] = cj;
            cellCoords[nodeIdx*3 + 2] = ck;
            __threadfence();                               // publish coords first
            uint32_t old = atomicCAS(&hashTable[slot], 0u, nodeIdx + 1u);
            if (old == 0u) return (int)nodeIdx;            // claimed this slot
            continue;                                      // lost race — retry
        }
        // Slot occupied — verify it holds our cell, else probe next slot
        int node = (int)(cur - 1u);
        if (cellCoords[node*3+0] == ci && cellCoords[node*3+1] == cj &&
            cellCoords[node*3+2] == ck) {
            return node;
        }
        slot = (slot + 1u) & tableMask;
    }
}

__device__ inline int sparseGridLookup(
    uint32_t* hashTable, const int32_t* cellCoords, uint32_t tableMask,
    int ci, int cj, int ck)
{
    uint32_t slot = cell_hash(ci, cj, ck) & tableMask;
    while (true) {
        uint32_t cur = atomicAdd(&hashTable[slot], 0u);
        if (cur == 0u) return -1;                          // empty slot → not found
        int node = (int)(cur - 1u);
        if (cellCoords[node*3+0] == ci && cellCoords[node*3+1] == cj &&
            cellCoords[node*3+2] == ck) {
            return node;
        }
        slot = (slot + 1u) & tableMask;
    }
}

// ── Accessor macros for flat particle/grid arrays ──────────────────────────

// Particle: struct MpmParticleGPU {
//   pos(3), vel(3), B(9), F(9), mass, vol0, mu0, lam0, H, critC, critS,
//   sandAlpha, dynVisc, bulkVisc, prevPicVel(3) }
//  = 3 + 3 + 9 + 9 + 10 + 3 = 37 floats
#define PARTICLE_FLOATS 37
#define P_IDX(i) ((i) * PARTICLE_FLOATS)
#define P_POS(i)   (P_IDX(i))
#define P_VEL(i)   (P_IDX(i) + 3)
#define P_B(i)     (P_IDX(i) + 6)     // APIC affine matrix C = B·D⁻¹
#define P_F(i)     (P_IDX(i) + 15)    // deformation gradient
#define P_MASS(i)  (P_IDX(i) + 24)
#define P_VOL0(i)  (P_IDX(i) + 25)
#define P_MU0(i)   (P_IDX(i) + 26)
#define P_LAM0(i)  (P_IDX(i) + 27)
#define P_HARD(i)  (P_IDX(i) + 28)
#define P_CRITC(i) (P_IDX(i) + 29)
#define P_CRITS(i) (P_IDX(i) + 30)
#define P_SAND(i)  (P_IDX(i) + 31)    // sand_alpha
#define P_VDYN(i)  (P_IDX(i) + 32)    // dynamic viscosity
#define P_VBULK(i) (P_IDX(i) + 33)    // bulk viscosity
#define P_PIC0(i)  (P_IDX(i) + 34)    // previous step's PIC velocity (FLIP)

// Grid node: struct MpmGridNodeGPU { mass(1), vel(3), mom(3), force(3) } = 10 floats
#define GRID_FLOATS 10
#define G_IDX(i)      ((i) * GRID_FLOATS)
#define G_MASS(i)     (G_IDX(i))
#define G_VEL(i)      (G_IDX(i) + 1)
#define G_MOM(i)      (G_IDX(i) + 4)
#define G_FORCE(i)    (G_IDX(i) + 7)

namespace flipcore {

// ═══════════════════════════════════════════════════════════════════════════
// CUDA Kernels
// ═══════════════════════════════════════════════════════════════════════════

// ── resetGrid: clear hash table + dense node array + active counter ────────

__global__ void resetGridKernel(
    float* grid, int maxNodes,
    uint32_t* hashTable, uint32_t tableSize,
    uint32_t* gridCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Clear the grid node array (all maxNodes slots)
    int gridFloats = maxNodes * GRID_FLOATS;
    if (i < gridFloats) grid[i] = 0.0f;

    // Clear the hash table
    int hIdx = i - gridFloats;
    if (hIdx >= 0 && hIdx < (int)tableSize) hashTable[hIdx] = 0u;

    // Reset the active-node counter (single thread)
    if (threadIdx.x == 0 && blockIdx.x == 0) *gridCount = 0u;
}

// ── P2G (Particle to Grid) — sparse grid insertion + APIC momentum ─────────

__global__ void P2GKernel(
    float* particles, int numParticles,
    float* grid,
    uint32_t* hashTable, uint32_t tableMask,
    uint32_t* gridCount, int32_t* cellCoords,
    float originX, float originY, float originZ,
    float h)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= numParticles) return;

    float px = particles[P_POS(p)];
    float py = particles[P_POS(p)+1];
    float pz = particles[P_POS(p)+2];
    float mass = particles[P_MASS(p)];

    float* F_ptr = &particles[P_F(p)];

    // Fixed-corotated: compute Cauchy stress × volume
    //   σ·vol = 2μ·(F - R)·F^T + λ·(J-1)·J·I
    float U[9], S[3], V[9];
    svd3x3(F_ptr, U, S, V);

    float R[9];
    float Vt[9]; mat3_transpose(V, Vt);
    mat3_mul(U, Vt, R);

    float FmR[9];
    for (int i = 0; i < 9; i++) FmR[i] = F_ptr[i] - R[i];

    float Ft[9]; mat3_transpose(F_ptr, Ft);
    float FmRFt[9];
    mat3_mul(FmR, Ft, FmRFt);

    float J = fmaxf(mat3_det(F_ptr), 1e-8f);
    float mu = particles[P_MU0(p)];
    float lambda = particles[P_LAM0(p)];
    float vol0 = particles[P_VOL0(p)];

    float stressVol[9];
    for (int i = 0; i < 9; i++) stressVol[i] = vol0 * 2.0f * mu * FmRFt[i];
    float lamJ = vol0 * lambda * (J - 1.0f) * J;
    stressVol[IDX(0,0)] += lamJ;
    stressVol[IDX(1,1)] += lamJ;
    stressVol[IDX(2,2)] += lamJ;

    // Sand model (sand_alpha): blend the elastic stress toward the stress of
    // the plastically-clamped deformation gradient F_c = U·S_clamped·Vᵀ.
    // F_c is already in SVD form, so its rotation is U·Vᵀ — no second SVD.
    float sandAlpha = particles[P_SAND(p)];
    if (sandAlpha > 0.0f) {
        float critC = particles[P_CRITC(p)];
        float critS = particles[P_CRITS(p)];
        float Sc[3] = {S[0], S[1], S[2]};
        for (int i = 0; i < 3; i++) {
            if (Sc[i] < 1.0f - critC) Sc[i] = 1.0f - critC;
            if (Sc[i] > 1.0f + critS) Sc[i] = 1.0f + critS;
        }
        // F_clamped = U · diag(Sc) · Vᵀ
        float Fc[9], USc[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                USc[IDX(r,c)] = U[IDX(r,c)] * Sc[c];
        mat3_mul(USc, Vt, Fc);

        float FcR[9];
        for (int i = 0; i < 9; i++) FcR[i] = Fc[i] - R[i];   // R = U·Vᵀ shared
        float Fct[9]; mat3_transpose(Fc, Fct);
        float FcRFct[9];
        mat3_mul(FcR, Fct, FcRFct);
        float Jc = fmaxf(mat3_det(Fc), 1e-8f);

        for (int i = 0; i < 9; i++) {
            float stressClamped = vol0 * 2.0f * mu * FcRFct[i];
            if (i == IDX(0,0) || i == IDX(1,1) || i == IDX(2,2))
                stressClamped += vol0 * lambda * (Jc - 1.0f) * Jc;
            stressVol[i] = (1.0f - sandAlpha) * stressVol[i] + sandAlpha * stressClamped;
        }
    }

    // Internal force = -stressVol (negative sign from variational derivative)
    float forceVol[9];
    for (int i = 0; i < 9; i++) forceVol[i] = -stressVol[i];

    // APIC affine matrix C = B·D⁻¹ (normalized by G2P), stored in P_B
    float C[9];
    for (int i = 0; i < 9; i++) C[i] = particles[P_B(p)+i];

    // Quadratic B-spline stencil: iterate 3x3x3 grid cells around particle.
    // Grid is unbounded — cells are (floor(pos/h) + offset) with hash mapping.
    float xp = (px - originX) / h;
    float yp = (py - originY) / h;
    float zp = (pz - originZ) / h;
    int baseX = (int)floorf(xp - 0.5f);
    int baseY = (int)floorf(yp - 0.5f);
    int baseZ = (int)floorf(zp - 0.5f);

    for (int dx = 0; dx < 3; dx++) {
        for (int dy = 0; dy < 3; dy++) {
            for (int dz = 0; dz < 3; dz++) {
                int gx = baseX + dx;
                int gy = baseY + dy;
                int gz = baseZ + dz;

                float wx = bspline(xp - (float)gx);
                float wy = bspline(yp - (float)gy);
                float wz = bspline(zp - (float)gz);
                float w = wx * wy * wz;
                if (w < 1e-12f) continue;

                int gi = sparseGridInsert(hashTable, gridCount, cellCoords,
                                          tableMask, gx, gy, gz);

                // Node world position — nodes sit at integer cell indices,
                // matching the integer-centered B-spline weights N(xp - gx).
                float nx = originX + (float)gx * h;
                float ny = originY + (float)gy * h;
                float nz = originZ + (float)gz * h;
                float dpos[3] = {nx - px, ny - py, nz - pz};

                // Scatter mass
                atomicAdd(&grid[G_MASS(gi)], mass * w);

                // APIC momentum: w · (m·v + m·C·dpos)
                float Cd[3];
                mat3_mul_vec(C, dpos, Cd);
                atomicAdd(&grid[G_MOM(gi)],   mass * (particles[P_VEL(p)]   + Cd[0]) * w);
                atomicAdd(&grid[G_MOM(gi)+1], mass * (particles[P_VEL(p)+1] + Cd[1]) * w);
                atomicAdd(&grid[G_MOM(gi)+2], mass * (particles[P_VEL(p)+2] + Cd[2]) * w);

                // Scatter elastic force: f_i = Σ_j σ_ij · ∇_j w
                float gradWx = bsplineGrad(xp - (float)gx) / h;
                float gradWy = bsplineGrad(yp - (float)gy) / h;
                float gradWz = bsplineGrad(zp - (float)gz) / h;

                float fx = forceVol[IDX(0,0)]*gradWx*wy*wz + forceVol[IDX(0,1)]*wx*gradWy*wz + forceVol[IDX(0,2)]*wx*wy*gradWz;
                float fy = forceVol[IDX(1,0)]*gradWx*wy*wz + forceVol[IDX(1,1)]*wx*gradWy*wz + forceVol[IDX(1,2)]*wx*wy*gradWz;
                float fz = forceVol[IDX(2,0)]*gradWx*wy*wz + forceVol[IDX(2,1)]*wx*gradWy*wz + forceVol[IDX(2,2)]*wx*wy*gradWz;

                atomicAdd(&grid[G_FORCE(gi)],   fx);
                atomicAdd(&grid[G_FORCE(gi)+1], fy);
                atomicAdd(&grid[G_FORCE(gi)+2], fz);
            }
        }
    }
}

// ── Update Grid (velocity = momentum/mass, apply force + gravity) ──────────
//    Only iterates the active nodes recorded in the sparse grid.

__global__ void updateGridKernel(
    float* grid, uint32_t* gridCount,
    float gx, float gy, float gz,
    float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int)(*gridCount)) return;
    int g = G_IDX(i);

    float mass = grid[G_MASS(i)];
    if (mass < 1e-12f) return;

    float invMass = 1.0f / mass;

    // Velocity = momentum / mass + dt * (force / mass + gravity)
    for (int d = 0; d < 3; d++) {
        float vel = grid[G_MOM(i)+d] * invMass;
        vel += dt * (grid[G_FORCE(i)+d] * invMass + (d == 0 ? gx : (d == 1 ? gy : gz)));
        grid[G_VEL(i)+d] = vel;
    }
}

// ── Grid collision (stick/slip at domain boundaries) ───────────────────────
//    Sparse version: nodes are identified by their cell; the domain walls sit
//    at the first cell (cell index 0) and last cell (res-1) on each axis,
//    matching the fixed-grid behaviour.

__global__ void gridCollisionKernel(
    float* grid, uint32_t* gridCount,
    const int32_t* cellCoords,
    int resX, int resY, int resZ,
    float friction)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int)(*gridCount)) return;

    float mass = grid[G_MASS(idx)];
    if (mass < 1e-12f) return;

    int cx = cellCoords[idx*3 + 0];
    int cy = cellCoords[idx*3 + 1];
    int cz = cellCoords[idx*3 + 2];

    float vx = grid[G_VEL(idx)];
    float vy = grid[G_VEL(idx)+1];
    float vz = grid[G_VEL(idx)+2];

    // Stick at walls: zero normal velocity at boundary cells
    if (cx <= 0 || cx >= resX-1) {
        vx = 0.0f;
        if (friction > 0.0f) { vy *= (1.0f - friction); vz *= (1.0f - friction); }
    }
    if (cy <= 0 || cy >= resY-1) {
        vy = 0.0f;
        if (friction > 0.0f) { vx *= (1.0f - friction); vz *= (1.0f - friction); }
    }
    if (cz <= 0 || cz >= resZ-1) {
        vz = 0.0f;
        if (friction > 0.0f) { vx *= (1.0f - friction); vy *= (1.0f - friction); }
    }

    grid[G_VEL(idx)]   = vx;
    grid[G_VEL(idx)+1] = vy;
    grid[G_VEL(idx)+2] = vz;
}

// ── G2P (Grid to Particle) — APIC affine gather + FLIP blend + plasticity ──

__global__ void G2PKernel(
    float* particles, int numParticles,
    const float* grid,
    uint32_t* hashTable, const int32_t* cellCoords, uint32_t tableMask,
    float originX, float originY, float originZ,
    float h, float flipRatio, float dt)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= numParticles) return;

    float px = particles[P_POS(p)];
    float py = particles[P_POS(p)+1];
    float pz = particles[P_POS(p)+2];

    // Gather PIC velocity, B matrix, and D matrix
    float picVel[3] = {0, 0, 0};
    float Bnew[9] = {0};
    float D[9] = {0};

    float xp = (px - originX) / h;
    float yp = (py - originY) / h;
    float zp = (pz - originZ) / h;
    int baseX = (int)floorf(xp - 0.5f);
    int baseY = (int)floorf(yp - 0.5f);
    int baseZ = (int)floorf(zp - 0.5f);

    for (int dx = 0; dx < 3; dx++) {
        for (int dy = 0; dy < 3; dy++) {
            for (int dz = 0; dz < 3; dz++) {
                int gx = baseX + dx;
                int gy = baseY + dy;
                int gz = baseZ + dz;

                float wx = bspline(xp - (float)gx);
                float wy = bspline(yp - (float)gy);
                float wz = bspline(zp - (float)gz);
                float w = wx * wy * wz;
                if (w < 1e-12f) continue;

                int gi = sparseGridLookup(hashTable, cellCoords, tableMask,
                                          gx, gy, gz);
                if (gi < 0) continue;

                float nodeMass = grid[G_MASS(gi)];
                if (nodeMass < 1e-12f) continue;

                float gvx = grid[G_VEL(gi)];
                float gvy = grid[G_VEL(gi)+1];
                float gvz = grid[G_VEL(gi)+2];

                // Node world position — integer cell indices, consistent
                // with the integer-centered B-spline weights used above.
                float nx = originX + (float)gx * h;
                float ny = originY + (float)gy * h;
                float nz = originZ + (float)gz * h;
                float dpos[3] = {nx - px, ny - py, nz - pz};

                // PIC velocity
                picVel[0] += w * gvx;
                picVel[1] += w * gvy;
                picVel[2] += w * gvz;

                // APIC affine momentum: B = Σ w · gv ⊗ dpos
                Bnew[IDX(0,0)] += w * gvx * dpos[0];
                Bnew[IDX(0,1)] += w * gvx * dpos[1];
                Bnew[IDX(0,2)] += w * gvx * dpos[2];
                Bnew[IDX(1,0)] += w * gvy * dpos[0];
                Bnew[IDX(1,1)] += w * gvy * dpos[1];
                Bnew[IDX(1,2)] += w * gvy * dpos[2];
                Bnew[IDX(2,0)] += w * gvz * dpos[0];
                Bnew[IDX(2,1)] += w * gvz * dpos[1];
                Bnew[IDX(2,2)] += w * gvz * dpos[2];

                // D = Σ w · dpos ⊗ dpos
                D[IDX(0,0)] += w * dpos[0]*dpos[0];
                D[IDX(0,1)] += w * dpos[0]*dpos[1];
                D[IDX(0,2)] += w * dpos[0]*dpos[2];
                D[IDX(1,0)] += w * dpos[1]*dpos[0];
                D[IDX(1,1)] += w * dpos[1]*dpos[1];
                D[IDX(1,2)] += w * dpos[1]*dpos[2];
                D[IDX(2,0)] += w * dpos[2]*dpos[0];
                D[IDX(2,1)] += w * dpos[2]*dpos[1];
                D[IDX(2,2)] += w * dpos[2]*dpos[2];
            }
        }
    }

    // APIC: C = B · D⁻¹  (regularized)
    D[IDX(0,0)] += 1e-8f; D[IDX(1,1)] += 1e-8f; D[IDX(2,2)] += 1e-8f;
    float Dinv[9];
    mat3_inverse(D, Dinv);
    float Cnew[9];
    mat3_mul(Bnew, Dinv, Cnew);

    // Per-particle viscosity: shear damping on C, bulk damping on tr(C)
    float dynVisc  = particles[P_VDYN(p)];
    float bulkVisc = particles[P_VBULK(p)];
    if (dynVisc > 0.0f || bulkVisc > 0.0f) {
        float tr = (Cnew[IDX(0,0)] + Cnew[IDX(1,1)] + Cnew[IDX(2,2)]) / 3.0f;
        float shear = 1.0f - dt * dynVisc;
        for (int i = 0; i < 9; i++) Cnew[i] *= shear;
        float bulk = dt * bulkVisc * tr;
        Cnew[IDX(0,0)] -= bulk;
        Cnew[IDX(1,1)] -= bulk;
        Cnew[IDX(2,2)] -= bulk;
    }

    // FLIP blend with previous PIC velocity (robust for sparse neighborhoods):
    //   v_FLIP = v_old + (v_pic − v_pic_prev)
    float newVel[3];
    for (int d = 0; d < 3; d++) {
        float v0 = particles[P_VEL(p)+d];
        float picPrev = particles[P_PIC0(p)+d];
        float vFlip = v0 + (picVel[d] - picPrev);
        newVel[d] = (1.0f - flipRatio) * picVel[d] + flipRatio * vFlip;
    }

    // Update deformation gradient: F_new = (I + dt·C) · F
    float IPlusDtC[9]; mat3_identity(IPlusDtC);
    for (int i = 0; i < 9; i++) IPlusDtC[i] += dt * Cnew[i];

    float* F_ptr = &particles[P_F(p)];
    float Fnew[9];
    mat3_mul(IPlusDtC, F_ptr, Fnew);

    // von Mises plasticity: clamp singular values
    float U[9], S[3], V[9];
    svd3x3(Fnew, U, S, V);

    float critC = particles[P_CRITC(p)];
    float critS = particles[P_CRITS(p)];
    bool plastic = false;
    for (int i = 0; i < 3; i++) {
        if (S[i] < 1.0f - critC) { S[i] = 1.0f - critC; plastic = true; }
        if (S[i] > 1.0f + critS) { S[i] = 1.0f + critS; plastic = true; }
    }

    if (plastic) {
        // Reconstruct F after plasticity
        float Sdiag[9] = {0};
        Sdiag[IDX(0,0)] = S[0]; Sdiag[IDX(1,1)] = S[1]; Sdiag[IDX(2,2)] = S[2];
        float US[9]; mat3_mul(U, Sdiag, US);
        float Vt[9]; mat3_transpose(V, Vt);
        mat3_mul(US, Vt, Fnew);
    }

    // Store updated F, affine matrix, velocity, and previous PIC velocity
    for (int i = 0; i < 9; i++) particles[P_F(p)+i] = Fnew[i];
    for (int i = 0; i < 9; i++) particles[P_B(p)+i] = Cnew[i];
    for (int d = 0; d < 3; d++) {
        particles[P_VEL(p)+d]  = newVel[d];
        particles[P_PIC0(p)+d] = picVel[d];
    }
}

// ── Advect Particles (with particle-level domain clamp) ────────────────────

__global__ void advectParticlesKernel(
    float* particles, int numParticles, float dt,
    float ox, float oy, float oz,
    float tx, float ty, float tz)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= numParticles) return;

    for (int d = 0; d < 3; d++) {
        particles[P_POS(p)+d] += dt * particles[P_VEL(p)+d];
    }

    // Clamp to the domain box — grid-level collision alone lets particles
    // drift slightly past walls before their stencil weight shifts.
    float lo[3] = {ox, oy, oz};
    float hi[3] = {tx, ty, tz};
    for (int d = 0; d < 3; d++) {
        if (particles[P_POS(p)+d] < lo[d]) {
            particles[P_POS(p)+d] = lo[d];
            particles[P_VEL(p)+d] = 0.0f;
        }
        if (particles[P_POS(p)+d] > hi[d]) {
            particles[P_POS(p)+d] = hi[d];
            particles[P_VEL(p)+d] = 0.0f;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Host-side solver
// ═══════════════════════════════════════════════════════════════════════════

#define CUDA_CHECK(call) do {                                    \
    cudaError_t _e = (call);                                     \
    if (_e != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(_e));      \
    }                                                            \
} while(0)

MpmSolver::MpmSolver()  = default;
MpmSolver::~MpmSolver() { _freeAll(); }

void MpmSolver::_freeAll() {
    if (_d_particles)  { CUDA_CHECK(cudaFree(_d_particles));  _d_particles  = nullptr; }
    if (_d_grid)       { CUDA_CHECK(cudaFree(_d_grid));       _d_grid       = nullptr; }
    if (_d_hashTable)  { CUDA_CHECK(cudaFree(_d_hashTable));  _d_hashTable  = nullptr; }
    if (_d_cellCoords) { CUDA_CHECK(cudaFree(_d_cellCoords)); _d_cellCoords = nullptr; }
    if (_d_gridCount)  { CUDA_CHECK(cudaFree(_d_gridCount));  _d_gridCount  = nullptr; }
    _numParticles = 0;
    _maxNodes = 0;
    _tableSize = 0;
    _initialised = false;
}

// Sparse grid: worst case every particle touches a disjoint 3×3×3 stencil,
// so maxNodes = 27·numParticles bounds the insertion counter. The hash table
// is the next power of two ≥ 2·maxNodes (load factor ≤ 0.5).
static inline size_t nextPow2(size_t v) {
    size_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

void MpmSolver::_allocGrid() {
    _maxNodes  = _numParticles * 27 + 1;
    _tableSize = nextPow2(_maxNodes * 2);

    CUDA_CHECK(cudaMalloc(&_d_grid,       _maxNodes * GRID_FLOATS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&_d_hashTable,  _tableSize * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&_d_cellCoords, _maxNodes * 3 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&_d_gridCount,  sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(_d_gridCount, 0, sizeof(uint32_t)));
}

void MpmSolver::init(const float* positions, size_t numParticles,
                     const MpmSettings& settings) {
    _freeAll();
    _settings = settings;
    _numParticles = numParticles;

    if (numParticles == 0) return;

    // Allocate device particles
    size_t pBytes = numParticles * PARTICLE_FLOATS * sizeof(float);
    CUDA_CHECK(cudaMalloc(&_d_particles, pBytes));

    // Build host particle array
    std::vector<float> hostParticles(numParticles * PARTICLE_FLOATS, 0.0f);
    const auto& mat = settings.material;
    float mu0  = mat.youngsModulus / (2.0f * (1.0f + mat.poissonRatio));
    float lam0 = (mat.youngsModulus * mat.poissonRatio)
                / ((1.0f + mat.poissonRatio) * (1.0f - 2.0f * mat.poissonRatio));
    float h    = settings.gridStride;
    float pVol = h * h * h * 0.125f;  // (h/2)^3 — particle volume for 2ppc
    float pMass = pVol * mat.density;

    for (size_t i = 0; i < numParticles; i++) {
        int pid = (int)i * PARTICLE_FLOATS;
        // Position
        hostParticles[pid + 0] = positions[3*i + 0];
        hostParticles[pid + 1] = positions[3*i + 1];
        hostParticles[pid + 2] = positions[3*i + 2];
        // Velocity = 0, APIC affine B = 0
        // Deformation gradient = Identity
        hostParticles[pid + 15 + IDX(0,0)] = 1.0f;
        hostParticles[pid + 15 + IDX(1,1)] = 1.0f;
        hostParticles[pid + 15 + IDX(2,2)] = 1.0f;
        // Mass & volume
        hostParticles[pid + 24] = pMass;  // P_MASS
        hostParticles[pid + 25] = pVol;   // P_VOL0
        hostParticles[pid + 26] = mu0;    // P_MU0
        hostParticles[pid + 27] = lam0;   // P_LAM0
        hostParticles[pid + 28] = mat.hardening;            // P_HARD
        hostParticles[pid + 29] = mat.criticalCompression;  // P_CRITC
        hostParticles[pid + 30] = mat.criticalStretch;      // P_CRITS
        hostParticles[pid + 31] = mat.sandAlpha;            // P_SAND
        hostParticles[pid + 32] = mat.dynamicViscosity;     // P_VDYN
        hostParticles[pid + 33] = mat.bulkViscosity;        // P_VBULK
    }

    CUDA_CHECK(cudaMemcpy(_d_particles, hostParticles.data(), pBytes,
                          cudaMemcpyHostToDevice));

    _allocGrid();
    _initialised = true;
}

void MpmSolver::step() {
    if (!_initialised) return;

    float dt = _settings.deltaTime;
    float h  = _settings.gridStride;
    int resX = _settings.gridResX;
    int resY = _settings.gridResY;
    int resZ = _settings.gridResZ;
    float ox = _settings.gridOriginX;
    float oy = _settings.gridOriginY;
    float oz = _settings.gridOriginZ;

    int pBlock = 256;
    int pGrid  = ((int)_numParticles + pBlock - 1) / pBlock;

    // 1. Reset sparse grid (nodes + hash table + counter)
    size_t resetTotal = _maxNodes * GRID_FLOATS + _tableSize;
    int resetBlock = 256;
    int resetGrid  = (int)((resetTotal + resetBlock - 1) / resetBlock);
    resetGridKernel<<<resetGrid, resetBlock>>>(
        _d_grid, (int)_maxNodes,
        _d_hashTable, (uint32_t)_tableSize,
        _d_gridCount);

    // 2. P2G — insert cells into the sparse grid and scatter mass/momentum/force
    uint32_t tableMask = (uint32_t)(_tableSize - 1);
    P2GKernel<<<pGrid, pBlock>>>(
        _d_particles, (int)_numParticles, _d_grid,
        _d_hashTable, tableMask,
        _d_gridCount, _d_cellCoords,
        ox, oy, oz, h);

    // 3. Update grid velocities (momentum→velocity, apply gravity + force)
    //    Launch enough threads to cover all possible nodes; the kernel exits
    //    early for indices past the active counter.
    int gBlock = 256;
    int gGrid  = (int)((_maxNodes + gBlock - 1) / gBlock);
    updateGridKernel<<<gGrid, gBlock>>>(
        _d_grid, _d_gridCount,
        _settings.gravityX, _settings.gravityY, _settings.gravityZ,
        dt);

    // 4. Grid collision (stick at domain walls)
    gridCollisionKernel<<<gGrid, gBlock>>>(
        _d_grid, _d_gridCount,
        _d_cellCoords, resX, resY, resZ,
        _settings.boundaryFriction);

    // 5. G2P — APIC gather + FLIP blend + plasticity
    G2PKernel<<<pGrid, pBlock>>>(
        _d_particles, (int)_numParticles, _d_grid,
        _d_hashTable, _d_cellCoords, tableMask,
        ox, oy, oz, h,
        _settings.flipRatio, dt);

    // 6. Advect (with particle-level domain clamp)
    float tx = ox + resX * h;
    float ty = oy + resY * h;
    float tz = oz + resZ * h;
    advectParticlesKernel<<<pGrid, pBlock>>>(
        _d_particles, (int)_numParticles, dt,
        ox, oy, oz, tx, ty, tz);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void MpmSolver::getPositions(float* outPositions, size_t maxCount) const {
    if (!_initialised || maxCount == 0) return;
    size_t n = _numParticles < maxCount ? _numParticles : maxCount;

    // Copy full particle array, extract positions
    std::vector<float> hostBuf(n * PARTICLE_FLOATS);
    CUDA_CHECK(cudaMemcpy(hostBuf.data(), _d_particles,
                          n * PARTICLE_FLOATS * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; i++) {
        outPositions[3*i + 0] = hostBuf[P_POS(i)];
        outPositions[3*i + 1] = hostBuf[P_POS(i)+1];
        outPositions[3*i + 2] = hostBuf[P_POS(i)+2];
    }
}

void MpmSolver::setBoundary(const float origin[3], const float target[3]) {
    _settings.gridOriginX = origin[0];
    _settings.gridOriginY = origin[1];
    _settings.gridOriginZ = origin[2];
    float h = _settings.gridStride;
    _settings.gridResX = (int)((target[0] - origin[0]) / h + 0.5f);
    _settings.gridResY = (int)((target[1] - origin[1]) / h + 0.5f);
    _settings.gridResZ = (int)((target[2] - origin[2]) / h + 0.5f);
}

} // namespace flipcore
