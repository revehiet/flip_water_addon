#include "flipcore/PressureSolverGPU.h"
#include "flipcore/FlipSolver.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cmath>
#include <cstdio>

namespace flipcore {
namespace {

// ── helpers ────────────────────────────────────────────────────────────────

inline void checkCuda(cudaError_t err, const char* tag) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA] %s: %s\n", tag, cudaGetErrorString(err));
    }
}

inline void checkCublas(cublasStatus_t st, const char* tag) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "[cuBLAS] %s: status %d\n", tag, int(st));
    }
}

// ── kernels ────────────────────────────────────────────────────────────────

/// out[ci] = A*x[ci]  (7-point Laplacian stencil, compacted over fluid cells)
__global__ void applyA_kernel(
    const int*       __restrict__ cellOfIndex,
    const int*       __restrict__ indexOfCell,
    const signed char* __restrict__ cellType,
    const float*     __restrict__ x,
    float*           __restrict__ out,
    int nx, int ny, int nz, int n)
{
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= n) return;

    int cellId = cellOfIndex[ci];
    int nxNy   = nx * ny;
    int k = cellId / nxNy;
    int rem = cellId % nxNy;
    int j = rem / nx;
    int i = rem % nx;

    const float kReg = 1e-4f;
    float diag = kReg;
    float sum  = 0.f;

    // +x neighbor
    if (i + 1 < nx) {
        signed char nt = cellType[(i + 1) + nx * (j + ny * k)];
        if (nt != 2) { diag += 1.f; if (nt == 1) sum += x[indexOfCell[(i + 1) + nx * (j + ny * k)]]; }
    }
    // -x
    if (i - 1 >= 0) {
        signed char nt = cellType[(i - 1) + nx * (j + ny * k)];
        if (nt != 2) { diag += 1.f; if (nt == 1) sum += x[indexOfCell[(i - 1) + nx * (j + ny * k)]]; }
    }
    // +y
    if (j + 1 < ny) {
        signed char nt = cellType[i + nx * ((j + 1) + ny * k)];
        if (nt != 2) { diag += 1.f; if (nt == 1) sum += x[indexOfCell[i + nx * ((j + 1) + ny * k)]]; }
    }
    // -y
    if (j - 1 >= 0) {
        signed char nt = cellType[i + nx * ((j - 1) + ny * k)];
        if (nt != 2) { diag += 1.f; if (nt == 1) sum += x[indexOfCell[i + nx * ((j - 1) + ny * k)]]; }
    }
    // +z
    if (k + 1 < nz) {
        signed char nt = cellType[i + nx * (j + ny * (k + 1))];
        if (nt != 2) { diag += 1.f; if (nt == 1) sum += x[indexOfCell[i + nx * (j + ny * (k + 1))]]; }
    }
    // -z
    if (k - 1 >= 0) {
        signed char nt = cellType[i + nx * (j + ny * (k - 1))];
        if (nt != 2) { diag += 1.f; if (nt == 1) sum += x[indexOfCell[i + nx * (j + ny * (k - 1))]]; }
    }

    out[ci] = diag * x[ci] - sum;
}

/// Compute diagonal of A in-place into diagOut
__global__ void computeDiagonal_kernel(
    const int*       __restrict__ cellOfIndex,
    const signed char* __restrict__ cellType,
    float*           __restrict__ diagOut,
    int nx, int ny, int nz, int n)
{
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= n) return;

    int cellId = cellOfIndex[ci];
    int nxNy   = nx * ny;
    int k = cellId / nxNy;
    int rem = cellId % nxNy;
    int j = rem / nx;
    int i = rem % nx;

    const float kReg = 1e-4f;
    float diag = kReg;
    if (i + 1 < nx && cellType[(i + 1) + nx * (j + ny * k)] != 2) diag += 1.f;
    if (i - 1 >= 0 && cellType[(i - 1) + nx * (j + ny * k)] != 2) diag += 1.f;
    if (j + 1 < ny && cellType[i + nx * ((j + 1) + ny * k)] != 2) diag += 1.f;
    if (j - 1 >= 0 && cellType[i + nx * ((j - 1) + ny * k)] != 2) diag += 1.f;
    if (k + 1 < nz && cellType[i + nx * (j + ny * (k + 1))] != 2) diag += 1.f;
    if (k - 1 >= 0 && cellType[i + nx * (j + ny * (k - 1))] != 2) diag += 1.f;
    diagOut[ci] = diag;
}

/// z = r / diag  (Jacobi preconditioner, element-wise)
__global__ void applyJacobi_kernel(
    const float* __restrict__ diag,
    const float* __restrict__ r,
    float*       __restrict__ z,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    z[i] = r[i] / diag[i];
}

} // anonymous namespace

// ── public entry point ─────────────────────────────────────────────────────

extern "C" int solvePressureCUDA(void* pg, float dt, float rho,
                                 int maxIterations, float tolerance,
                                 const float* x0Host)
{
    flipcore::MacGrid& grid = *static_cast<flipcore::MacGrid*>(pg);
    // ── build fluid index (CPU) ────────────────────────────────────────
    int nx = grid.nx(), ny = grid.ny(), nz = grid.nz();
    int nxNy = nx * ny;
    size_t totalCells = size_t(nx) * ny * nz;

    std::vector<int> cellOfIndex;
    std::vector<int> indexOfCell(totalCells, -1);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                if (grid.cellType(i, j, k) != CELL_FLUID) continue;
                size_t cellId = size_t(i) + size_t(nx) * (size_t(j) + size_t(ny) * size_t(k));
                indexOfCell[cellId] = int(cellOfIndex.size());
                cellOfIndex.push_back(int(cellId));
            }

    size_t n = cellOfIndex.size();
    int iter = 0;
    if (n == 0) return 0;

    // ── build RHS on CPU ───────────────────────────────────────────────
    float scale = rho * grid.h() * grid.h() / std::max(dt, 1e-6f);
    auto div = [&](int i, int j, int k) -> float {
        float du = grid.u(i + 1, j, k) - grid.u(i, j, k);
        float dv = grid.v(i, j + 1, k) - grid.v(i, j, k);
        float dw = grid.w(i, j, k + 1) - grid.w(i, j, k);
        return (du + dv + dw) / grid.h();
    };
    std::vector<float> b_host(n);
    for (size_t ci = 0; ci < n; ++ci) {
        int cellId = cellOfIndex[ci];
        int k = cellId / nxNy, rem = cellId % nxNy, j = rem / nx, i = rem % nx;
        b_host[ci] = -scale * div(i, j, k);
    }

    // ── GPU allocations ────────────────────────────────────────────────
    cudaError_t err;
    int *d_cellOfIndex = nullptr, *d_indexOfCell = nullptr;
    signed char *d_cellType = nullptr;
    float *d_b = nullptr, *d_x = nullptr, *d_r = nullptr, *d_p = nullptr,
          *d_z = nullptr, *d_Ap = nullptr, *d_diag = nullptr;

    auto gpuAlloc = [&](auto*& ptr, size_t bytes) -> bool {
        err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) { checkCuda(err, "cudaMalloc"); return false; }
        return true;
    };
    #define ALLOC_OR_FAIL(ptr, bytes) if (!gpuAlloc(ptr, bytes)) goto cleanup

    ALLOC_OR_FAIL(d_cellOfIndex, n * sizeof(int));
    ALLOC_OR_FAIL(d_indexOfCell, totalCells * sizeof(int));
    ALLOC_OR_FAIL(d_cellType,    totalCells * sizeof(signed char));
    ALLOC_OR_FAIL(d_b,           n * sizeof(float));
    ALLOC_OR_FAIL(d_x,           n * sizeof(float));
    ALLOC_OR_FAIL(d_r,           n * sizeof(float));
    ALLOC_OR_FAIL(d_p,           n * sizeof(float));
    ALLOC_OR_FAIL(d_z,           n * sizeof(float));
    ALLOC_OR_FAIL(d_Ap,          n * sizeof(float));
    ALLOC_OR_FAIL(d_diag,        n * sizeof(float));

    // x0 = 0 must hold for the CG recurrence (r0 = b), so zero d_x explicitly.
    cudaMemset(d_x, 0, n * sizeof(float));

    // ── upload to GPU ──────────────────────────────────────────────────
    cudaMemcpy(d_cellOfIndex, cellOfIndex.data(), n * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_indexOfCell, indexOfCell.data(), totalCells * sizeof(int),    cudaMemcpyHostToDevice);
    // cellType is stored as Array3<signed char> with contiguous .data()
    cudaMemcpy(d_cellType,    grid.cellType.data(), totalCells * sizeof(signed char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,           b_host.data(),        n * sizeof(float),              cudaMemcpyHostToDevice);

    // ── cuBLAS handle ──────────────────────────────────────────────────
    cublasHandle_t cublas = nullptr;
    if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS) {
        err = cudaErrorUnknown; // signal CPU fallback to the caller
        goto cleanup;
    }

    // ── precompute diagonal ─────────────────────────────────────────────
    {
        int block = 256, gridDimC = int((n + block - 1) / block);
        computeDiagonal_kernel<<<gridDimC, block>>>(d_cellOfIndex, d_cellType, d_diag, nx, ny, nz, int(n));
        cudaDeviceSynchronize();
    }

    // ── CG loop ────────────────────────────────────────────────────────
    {
        int block = 256, gridDimC = int((n + block - 1) / block);
        float one = 1.f;

        // Initial guess: previous pressure snapshot (warm start) or zero.
        if (x0Host != nullptr) {
            std::vector<float> x0(n, 0.f);
            for (size_t ci = 0; ci < n; ++ci) x0[ci] = x0Host[size_t(cellOfIndex[ci])];
            cudaMemcpy(d_x, x0.data(), n * sizeof(float), cudaMemcpyHostToDevice);
            // r0 = b - A*x0
            applyA_kernel<<<gridDimC, block>>>(d_cellOfIndex, d_indexOfCell, d_cellType,
                                                d_x, d_Ap, nx, ny, nz, int(n));
            cudaDeviceSynchronize();
            cublasScopy(cublas, int(n), d_b, 1, d_r, 1);
            float mone = -1.f;
            cublasSaxpy(cublas, int(n), &mone, d_Ap, 1, d_r, 1);
        } else {
            // x0 = 0, r0 = b
            cublasScopy(cublas, int(n), d_b, 1, d_r, 1);
        }

        // z0 = M^-1 * r0
        applyJacobi_kernel<<<gridDimC, block>>>(d_diag, d_r, d_z, int(n));
        cudaDeviceSynchronize();

        // p0 = z0
        cublasScopy(cublas, int(n), d_z, 1, d_p, 1);

        float rzOld = 0.f, rzNew = 0.f;
        cublasSdot(cublas, int(n), d_r, 1, d_z, 1, &rzOld);

        float rsInit = 0.f, rsNew = 0.f;
        cublasSdot(cublas, int(n), d_r, 1, d_r, 1, &rsInit);

        if (std::sqrt(std::max(rsInit, 0.f)) < 1e-9f) { iter = 0; goto cleanup_cublas; }

        for (; iter < maxIterations; ++iter) {
            // Ap = A * p
            applyA_kernel<<<gridDimC, block>>>(d_cellOfIndex, d_indexOfCell, d_cellType,
                                                d_p, d_Ap, nx, ny, nz, int(n));
            cudaDeviceSynchronize();

            float pAp = 0.f;
            cublasSdot(cublas, int(n), d_p, 1, d_Ap, 1, &pAp);
            if (std::fabs(pAp) < 1e-20f) break;

            float alpha = rzOld / pAp;

            // x += alpha * p
            cublasSaxpy(cublas, int(n), &alpha, d_p, 1, d_x, 1);

            // r -= alpha * Ap
            float malpha = -alpha;
            cublasSaxpy(cublas, int(n), &malpha, d_Ap, 1, d_r, 1);

            cublasSdot(cublas, int(n), d_r, 1, d_r, 1, &rsNew);
            if (std::sqrt(std::max(rsNew, 0.f)) < tolerance) { ++iter; break; }

            // z = M^-1 * r
            applyJacobi_kernel<<<gridDimC, block>>>(d_diag, d_r, d_z, int(n));
            cudaDeviceSynchronize();

            cublasSdot(cublas, int(n), d_r, 1, d_z, 1, &rzNew);

            float beta = rzNew / rzOld;

            // p = z + beta * p
            cublasSscal(cublas, int(n), &beta, d_p, 1);
            cublasSaxpy(cublas, int(n), &one, d_z, 1, d_p, 1);

            rzOld = rzNew;
        }
    }

cleanup_cublas:
    cublasDestroy(cublas);

    // ── download pressure back to CPU ──────────────────────────────────
    {
        std::vector<float> x_host(n, 0.f);
        cudaMemcpy(x_host.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t ci = 0; ci < n; ++ci) {
            int cellId = cellOfIndex[ci];
            int k = cellId / nxNy, rem = cellId % nxNy, j = rem / nx, i = rem % nx;
            grid.pressure(i, j, k) = x_host[ci];
        }
    }

cleanup:
    cudaFree(d_cellOfIndex); cudaFree(d_indexOfCell); cudaFree(d_cellType);
    cudaFree(d_b); cudaFree(d_x); cudaFree(d_r); cudaFree(d_p);
    cudaFree(d_z); cudaFree(d_Ap); cudaFree(d_diag);

    #undef ALLOC_OR_FAIL

    if (err != cudaSuccess) {
        // If any cudaMalloc failed, the CG didn't run - signal fallback
        return -1;
    }

    return iter;
}

// ────────────────────────────────────────────────────────────────────────────
// GPU SDF collision: per-particle penetration detection & push-out
// ────────────────────────────────────────────────────────────────────────────

/// Reads a value from a flat cell-centered 3D field using trilinear
/// interpolation, clamping lookups to grid bounds.
__device__ float sampleSDF(const float* sdf, int nx, int ny, int nz,
                           float gx, float gy, float gz) {
    if (nx <= 0 || ny <= 0 || nz <= 0) return 1e6f;
    gx = fmaxf(0.f, fminf(gx, float(nx - 1)));
    gy = fmaxf(0.f, fminf(gy, float(ny - 1)));
    gz = fmaxf(0.f, fminf(gz, float(nz - 1)));
    int i0 = int(floorf(gx)), i1 = min(i0 + 1, nx - 1);
    int j0 = int(floorf(gy)), j1 = min(j0 + 1, ny - 1);
    int k0 = int(floorf(gz)), k1 = min(k0 + 1, nz - 1);
    float fx = gx - float(i0), fy = gy - float(j0), fz = gz - float(k0);
    auto idx = [nx, ny](int i, int j, int k) { return i + nx * (j + ny * k); };
    float c00 = sdf[idx(i0,j0,k0)] * (1-fx) + sdf[idx(i1,j0,k0)] * fx;
    float c10 = sdf[idx(i0,j1,k0)] * (1-fx) + sdf[idx(i1,j1,k0)] * fx;
    float c01 = sdf[idx(i0,j0,k1)] * (1-fx) + sdf[idx(i1,j0,k1)] * fx;
    float c11 = sdf[idx(i0,j1,k1)] * (1-fx) + sdf[idx(i1,j1,k1)] * fx;
    float c0  = c00 * (1-fy) + c10 * fy;
    float c1  = c01 * (1-fy) + c11 * fy;
    return c0 * (1-fz) + c1 * fz;
}

__global__ void sdfCollisionKernel(
    float* posX, float* posY, float* posZ,
    float* velX, float* velY, float* velZ,
    const float* sdf,
    int nx, int ny, int nz,
    float domainMinX, float domainMinY, float domainMinZ,
    float domainMaxX, float domainMaxY, float domainMaxZ,
    float cellSize, float margin,
    int nParticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nParticles) return;

    float px = posX[i], py = posY[i], pz = posZ[i];

    // World → grid-index space
    float ix = (px - domainMinX) / cellSize;
    float iy = (py - domainMinY) / cellSize;
    float iz = (pz - domainMinZ) / cellSize;

    // Cell-centered SDF sample (index − 0.5)
    float phi = sampleSDF(sdf, nx, ny, nz, ix - 0.5f, iy - 0.5f, iz - 0.5f);
    if (phi >= margin) return;

    // Central-difference gradient in index space
    float eps = 0.5f;
    float gx = sampleSDF(sdf, nx, ny, nz, ix - 0.5f + eps, iy - 0.5f, iz - 0.5f)
             - sampleSDF(sdf, nx, ny, nz, ix - 0.5f - eps, iy - 0.5f, iz - 0.5f);
    float gy = sampleSDF(sdf, nx, ny, nz, ix - 0.5f, iy - 0.5f + eps, iz - 0.5f)
             - sampleSDF(sdf, nx, ny, nz, ix - 0.5f, iy - 0.5f - eps, iz - 0.5f);
    float gz = sampleSDF(sdf, nx, ny, nz, ix - 0.5f, iy - 0.5f, iz - 0.5f + eps)
             - sampleSDF(sdf, nx, ny, nz, ix - 0.5f, iy - 0.5f, iz - 0.5f - eps);

    float glen = sqrtf(gx*gx + gy*gy + gz*gz);
    if (glen < 1e-6f) return;

    float nx_n = gx / glen, ny_n = gy / glen, nz_n = gz / glen;
    float push = margin - phi;  // world units

    // Push particle out along gradient
    px += nx_n * push;
    py += ny_n * push;
    pz += nz_n * push;

    // Clamp to domain
    px = fmaxf(domainMinX, fminf(domainMaxX, px));
    py = fmaxf(domainMinY, fminf(domainMaxY, py));
    pz = fmaxf(domainMinZ, fminf(domainMaxZ, pz));

    posX[i] = px; posY[i] = py; posZ[i] = pz;

    // Kill inward velocity
    float vn = velX[i]*nx_n + velY[i]*ny_n + velZ[i]*nz_n;
    if (vn < 0.f) {
        velX[i] -= nx_n * vn;
        velY[i] -= ny_n * vn;
        velZ[i] -= nz_n * vn;
    }
}

extern "C" int resolveObstacleCollisionsCUDA(void* psolver) {
    flipcore::FlipSolver& s = *static_cast<flipcore::FlipSolver*>(psolver);
    if (!s.hasObstacleSDF()) return 0;

    size_t n = s.positions().size();
    if (n == 0) return 0;

    const flipcore::Array3<float>& sdf = s.obstacleSDF();
    int nx = sdf.nx(), ny = sdf.ny(), nz = sdf.nz();
    if (nx == 0 || ny == 0 || nz == 0) return 0;

    float h = s.grid().h();
    float margin = s.settings().sdfCollisionMargin * h;
    flipcore::Vec3 dmin = s.domainMin(), dmax = s.domainMax();
    size_t sdfBytes = size_t(nx) * ny * nz * sizeof(float);

    // Host-side buffers (declared before any goto for MSVC compliance)
    const auto& pos = s.positions();
    const auto& vel = s.velocities();
    std::vector<float> px(n), py(n), pz(n), vx(n), vy(n), vz(n);
    for (size_t i = 0; i < n; ++i) {
        px[i]=pos[i].x; py[i]=pos[i].y; pz[i]=pos[i].z;
        vx[i]=vel[i].x; vy[i]=vel[i].y; vz[i]=vel[i].z;
    }

    // Alloc GPU
    float *d_posX=nullptr, *d_posY=nullptr, *d_posZ=nullptr;
    float *d_velX=nullptr, *d_velY=nullptr, *d_velZ=nullptr, *d_sdf=nullptr;
    cudaError_t err = cudaSuccess;
    #define CUALLOC(p, sz) if (cudaMalloc(&p, sz) != cudaSuccess) goto col_cleanup
    CUALLOC(d_posX, n * sizeof(float)); CUALLOC(d_posY, n * sizeof(float));
    CUALLOC(d_posZ, n * sizeof(float)); CUALLOC(d_velX, n * sizeof(float));
    CUALLOC(d_velY, n * sizeof(float)); CUALLOC(d_velZ, n * sizeof(float));
    CUALLOC(d_sdf, sdfBytes);

    // Upload: positions, velocities, SDF grid
    cudaMemcpy(d_posX, px.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posY, py.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posZ, pz.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velX, vx.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velY, vy.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velZ, vz.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sdf, sdf.data(), sdfBytes, cudaMemcpyHostToDevice);

    int block = 256, gridDim = int((n + block - 1) / block);
    sdfCollisionKernel<<<gridDim, block>>>(
        d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ,
        d_sdf, nx, ny, nz,
        dmin.x, dmin.y, dmin.z, dmax.x, dmax.y, dmax.z,
        h, margin, int(n));
    cudaDeviceSynchronize();

    // Download results
    cudaMemcpy(px.data(), d_posX, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(py.data(), d_posY, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(pz.data(), d_posZ, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vx.data(), d_velX, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vy.data(), d_velY, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vz.data(), d_velZ, n*sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; ++i) {
        s.positions()[i] = flipcore::Vec3(px[i], py[i], pz[i]);
        s.velocities()[i] = flipcore::Vec3(vx[i], vy[i], vz[i]);
    }

col_cleanup:
    cudaFree(d_posX); cudaFree(d_posY); cudaFree(d_posZ);
    cudaFree(d_velX); cudaFree(d_velY); cudaFree(d_velZ);
    cudaFree(d_sdf);
    #undef CUALLOC

    return (err == cudaSuccess) ? int(n) : -1;
}

} // namespace flipcore
