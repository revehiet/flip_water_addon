#include "flipcore/PressureSolverGPU.h"
#include "flipcore/FlipSolver.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>

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

// ── persistent GPU scratch ─────────────────────────────────────────────────
// Historically every substep performed ~10 cudaMalloc/cudaFree pairs plus a
// full SDF re-upload. cudaMalloc/cudaFree are device-synchronizing driver
// calls, so this dominated the profile at typical grid sizes. These buffers
// live for the process lifetime and grow geometrically, so steady-state
// substeps perform zero allocations; only the data that actually changes
// (cell types, particle positions, ...) crosses PCIe, over pinned staging
// memory (direct DMA instead of pageable-buffer staging copies).

inline bool cok(cudaError_t err, const char* tag) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA] %s: %s\n", tag, cudaGetErrorString(err));
        return false;
    }
    return true;
}

struct GpuBuf {
    void* ptr = nullptr;
    size_t cap = 0;
    void* acquire(size_t bytes) {
        if (bytes > cap) {
            if (ptr) cudaFree(ptr); // ignore errors on grow/teardown paths
            size_t bigger = bytes + bytes / 2; // 1.5x growth headroom
            if (cudaMalloc(&ptr, bigger) != cudaSuccess) { ptr = nullptr; cap = 0; return nullptr; }
            cap = bigger;
        }
        return ptr;
    }
    ~GpuBuf() { if (ptr) cudaFree(ptr); }
};

// Pinned host staging buffer for async H2D/D2H copies (pageable memory would
// force a synchronous staging copy inside the driver).
struct PinnedBuf {
    void* ptr = nullptr;
    size_t cap = 0;
    void* acquire(size_t bytes) {
        if (bytes > cap) {
            if (ptr) cudaFreeHost(ptr);
            size_t bigger = bytes + bytes / 2;
            if (cudaHostAlloc(&ptr, bigger, cudaHostAllocDefault) != cudaSuccess) { ptr = nullptr; cap = 0; return nullptr; }
            cap = bigger;
        }
        return ptr;
    }
    ~PinnedBuf() { if (ptr) cudaFreeHost(ptr); }
};

struct GpuScratch {
    GpuBuf cellType, cellOfIndex, indexOfCell;
    GpuBuf uF, vF, wF, pressureFull;   // staggered MAC fields + solved pressure (device-resident)
    GpuBuf b, x, r, p, z, Ap, diag, counter;
    GpuBuf posX, posY, posZ, velX, velY, velZ, sdf;
    PinnedBuf up, down;
    cublasHandle_t cublas = nullptr;
    bool cublasOk = false;
    // SDF device-copy provenance: re-upload only when the host field differs
    // (host data pointer or per-solver revision changed).
    const void* sdfHostPtr = nullptr;
    size_t sdfBytes = 0;
    uint64_t sdfRevision = 0;
    // Device-resident warm-start pressure (scatterPressure_kernel writes it,
    // gatherX0_kernel consumes it next substep). Invalidated when the grid is
    // (re)created at a different resolution or a different solver instance
    // uses the scratch (scratch is process-global; warm start is per-solver).
    size_t pressureCells = 0;
    bool pressureValid = false;
    const void* owner = nullptr;
    GpuScratch() { cublasOk = (cublasCreate(&cublas) == CUBLAS_STATUS_SUCCESS); }
    ~GpuScratch() { if (cublas) cublasDestroy(cublas); }
};

GpuScratch& scratch() {
    static GpuScratch s;
    return s;
}

/// Marks every FLUID cell (cellType == 1) with a compact index on the GPU:
/// indexOfCell[cellId] = ci, cellOfIndex[ci] = cellId, counter = total fluid
/// cells. Replaces the previous per-substep CPU scan over the whole grid.
__global__ void buildFluidIndex_kernel(
    const signed char* __restrict__ cellType,
    int* __restrict__ indexOfCell,
    int* __restrict__ cellOfIndex,
    int* __restrict__ counter,
    int totalCells)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= totalCells) return;
    if (cellType[c] != CELL_FLUID) return;
    int ci = atomicAdd(counter, 1);
    indexOfCell[c] = ci;
    cellOfIndex[ci] = c;
}

/// Compact divergence RHS: b[ci] = -rho*h²/dt * div(u,v,w) at the cell center
/// (same scale/units as the host solver's RHS build).
__global__ void buildRHS_kernel(
    const int* __restrict__ cellOfIndex,
    const float* __restrict__ u, const float* __restrict__ v,
    const float* __restrict__ w,
    float* __restrict__ bOut,
    int nx, int ny, int nz, float scale, float invH, int n)
{
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= n) return;
    int cellId = cellOfIndex[ci];
    int nxNy = nx * ny;
    int k = cellId / nxNy, rem = cellId % nxNy, j = rem / nx, i = rem % nx;
    // Staggered Array3 layouts: u(nx+1,ny,nz), v(nx,ny+1,nz), w(nx,ny,nz+1)
    float du = u[(i + 1) + (nx + 1) * (j + ny * k)] - u[i + (nx + 1) * (j + ny * k)];
    float dv = v[i + nx * ((j + 1) + (ny + 1) * k)] - v[i + nx * (j + (ny + 1) * k)];
    float dw = w[i + nx * (j + ny * (k + 1))] - w[i + nx * (j + ny * k)];
    bOut[ci] = -scale * invH * (du + dv + dw);
}

/// Warm start: x0[ci] = previous pressure at this step's fluid cell.
__global__ void gatherX0_kernel(int n, const int* __restrict__ cellOfIndex,
                                const float* __restrict__ pressureFull,
                                float* __restrict__ x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = pressureFull[cellOfIndex[i]];
}

/// pressureFull[cellId] = x[ci] — the solved pressure stays on the device as
/// the next substep's warm start AND as input to the projection below.
__global__ void scatterPressure_kernel(int n, const int* __restrict__ cellOfIndex,
                                       const float* __restrict__ x,
                                       float* __restrict__ pressureFull)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    pressureFull[cellOfIndex[i]] = x[i];
}

__device__ __forceinline__ bool isPressureCellD(signed char t) {
    return t == CELL_FLUID || t == CELL_AIR_ACTIVE;
}

// Face projections, replicating projectPressureVelocities()
// (PressureSolver.cpp) exactly — including the air-band density scaling —
// followed by the solid-face zeroing (zeroSolidNormalVelocities parity:
// the domain edge counts as solid on its missing side).

__global__ void projectU_kernel(
    const signed char* __restrict__ cellType,
    const float* __restrict__ pressureFull,
    float* __restrict__ u,
    int nx, int ny, int nz,
    float dtOverH, float invRho, float invRhoAir, float airDensityRatio)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int faces = (nx - 1) * ny * nz;              // interior u-faces, i in [1, nx-1]
    if (idx >= faces) return;
    int k = idx / ((nx - 1) * ny);
    int rem = idx % ((nx - 1) * ny);
    int j = rem / (nx - 1);
    int i = rem % (nx - 1) + 1;

    signed char a = cellType[(i - 1) + nx * (j + ny * k)];
    signed char c = cellType[i + nx * (j + ny * k)];
    if (a == CELL_SOLID || c == CELL_SOLID) return;
    bool pa = isPressureCellD(a), pc = isPressureCellD(c);
    if (!pa && !pc) return;
    float pA = pa ? pressureFull[(i - 1) + nx * (j + ny * k)] : 0.f;
    float pB = pc ? pressureFull[i + nx * (j + ny * k)] : 0.f;
    float rA = (a == CELL_AIR_ACTIVE && airDensityRatio > 0.f) ? invRhoAir : invRho;
    float rC = (c == CELL_AIR_ACTIVE && airDensityRatio > 0.f) ? invRhoAir : invRho;
    float inv = (pa && pc) ? 0.5f * (rA + rC) : (rA + rC - invRho);
    u[i + (nx + 1) * (j + ny * k)] -= dtOverH * inv * (pB - pA);
}

__global__ void projectV_kernel(
    const signed char* __restrict__ cellType,
    const float* __restrict__ pressureFull,
    float* __restrict__ v,
    int nx, int ny, int nz,
    float dtOverH, float invRho, float invRhoAir, float airDensityRatio)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int faces = nx * (ny - 1) * nz;              // interior v-faces, j in [1, ny-1]
    if (idx >= faces) return;
    int k = idx / (nx * (ny - 1));
    int rem = idx % (nx * (ny - 1));
    int j = rem / nx + 1;
    int i = rem % nx;

    signed char a = cellType[i + nx * ((j - 1) + ny * k)];
    signed char c = cellType[i + nx * (j + ny * k)];
    if (a == CELL_SOLID || c == CELL_SOLID) return;
    bool pa = isPressureCellD(a), pc = isPressureCellD(c);
    if (!pa && !pc) return;
    float pA = pa ? pressureFull[i + nx * ((j - 1) + ny * k)] : 0.f;
    float pB = pc ? pressureFull[i + nx * (j + ny * k)] : 0.f;
    float rA = (a == CELL_AIR_ACTIVE && airDensityRatio > 0.f) ? invRhoAir : invRho;
    float rC = (c == CELL_AIR_ACTIVE && airDensityRatio > 0.f) ? invRhoAir : invRho;
    float inv = (pa && pc) ? 0.5f * (rA + rC) : (rA + rC - invRho);
    v[i + nx * (j + (ny + 1) * k)] -= dtOverH * inv * (pB - pA);
}

__global__ void projectW_kernel(
    const signed char* __restrict__ cellType,
    const float* __restrict__ pressureFull,
    float* __restrict__ w,
    int nx, int ny, int nz,
    float dtOverH, float invRho, float invRhoAir, float airDensityRatio)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int faces = nx * ny * (nz - 1);              // interior w-faces, k in [1, nz-1]
    if (idx >= faces) return;
    int k = idx / (nx * ny) + 1;
    int rem = idx % (nx * ny);
    int j = rem / nx;
    int i = rem % nx;

    signed char a = cellType[i + nx * (j + ny * (k - 1))];
    signed char c = cellType[i + nx * (j + ny * k)];
    if (a == CELL_SOLID || c == CELL_SOLID) return;
    bool pa = isPressureCellD(a), pc = isPressureCellD(c);
    if (!pa && !pc) return;
    float pA = pa ? pressureFull[i + nx * (j + ny * (k - 1))] : 0.f;
    float pB = pc ? pressureFull[i + nx * (j + ny * k)] : 0.f;
    float rA = (a == CELL_AIR_ACTIVE && airDensityRatio > 0.f) ? invRhoAir : invRho;
    float rC = (c == CELL_AIR_ACTIVE && airDensityRatio > 0.f) ? invRhoAir : invRho;
    float inv = (pa && pc) ? 0.5f * (rA + rC) : (rA + rC - invRho);
    w[i + nx * (j + ny * k)] -= dtOverH * inv * (pB - pA);
}

__global__ void zeroSolidU_kernel(const signed char* __restrict__ cellType,
                                  float* __restrict__ u, int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int faces = (nx + 1) * ny * nz;
    if (idx >= faces) return;
    int k = idx / ((nx + 1) * ny);
    int rem = idx % ((nx + 1) * ny);
    int j = rem / (nx + 1);
    int i = rem % (nx + 1);
    bool sl = (i > 0) ? (cellType[(i - 1) + nx * (j + ny * k)] == CELL_SOLID) : true;
    bool sr = (i < nx) ? (cellType[i + nx * (j + ny * k)] == CELL_SOLID) : true;
    if (sl || sr) u[idx] = 0.f;
}

__global__ void zeroSolidV_kernel(const signed char* __restrict__ cellType,
                                  float* __restrict__ v, int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int faces = nx * (ny + 1) * nz;
    if (idx >= faces) return;
    int k = idx / (nx * (ny + 1));
    int rem = idx % (nx * (ny + 1));
    int j = rem / nx;
    int i = rem % nx;
    bool sb = (j > 0) ? (cellType[i + nx * ((j - 1) + ny * k)] == CELL_SOLID) : true;
    bool sa = (j < ny) ? (cellType[i + nx * (j + ny * k)] == CELL_SOLID) : true;
    if (sb || sa) v[idx] = 0.f;
}

__global__ void zeroSolidW_kernel(const signed char* __restrict__ cellType,
                                  float* __restrict__ w, int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int faces = nx * ny * (nz + 1);
    if (idx >= faces) return;
    int k = idx / (nx * ny);
    int rem = idx % (nx * ny);
    int j = rem / nx;
    int i = rem % nx;
    bool sbk = (k > 0) ? (cellType[i + nx * (j + ny * (k - 1))] == CELL_SOLID) : true;
    bool sf = (k < nz) ? (cellType[i + nx * (j + ny * k)] == CELL_SOLID) : true;
    if (sbk || sf) w[idx] = 0.f;
}

} // anonymous namespace

// ── public entry point ─────────────────────────────────────────────────────
//
// Tier-A rewrite: persistent scratch buffers (zero per-substep cudaMalloc),
// GPU-built fluid index, pinned-memory transfers, and the velocity projection
// + solid-face zeroing fused into the same device pass so the pressure field
// never round-trips through the host. On success the MAC grid fields
// (u, v, w) come back projected and solid-zeroed; on ANY failure the host
// state is untouched and -1 is returned so the caller falls back to the CPU
// solver. Warm start = device-resident previous pressure (useWarmStart).

extern "C" int solvePressureCUDA(void* pg, float dt, float rho,
                                 int maxIterations, float tolerance,
                                 int useWarmStart, float airDensityRatio)
{
    flipcore::MacGrid& grid = *static_cast<flipcore::MacGrid*>(pg);
    GpuScratch& S = scratch();
    if (!S.cublasOk) return -1;

    const int nx = grid.nx(), ny = grid.ny(), nz = grid.nz();
    const size_t totalCells = size_t(nx) * ny * nz;
    if (totalCells == 0) return 0;
    const size_t nu = size_t(nx + 1) * ny * nz;
    const size_t nv = size_t(nx) * (ny + 1) * nz;
    const size_t nw = size_t(nx) * ny * (nz + 1);
    const int block = 256;
    auto gd = [block](size_t count) { return dim3(unsigned((count + block - 1) / block)); };

    // Warm-start provenance: scratch is process-global, so the resident
    // pressure is only valid for the same solver instance and grid size.
    if (S.owner != pg || S.pressureCells != totalCells) {
        S.owner = pg;
        S.pressureCells = totalCells;
        S.pressureValid = false;
    }

    // ── upload everything the device needs (pinned staging) ────────────
    void* up = S.up.acquire(totalCells + (nu + nv + nw) * sizeof(float));
    signed char* d_cellType = static_cast<signed char*>(S.cellType.acquire(totalCells));
    float* d_uF = static_cast<float*>(S.uF.acquire(nu * sizeof(float)));
    float* d_vF = static_cast<float*>(S.vF.acquire(nv * sizeof(float)));
    float* d_wF = static_cast<float*>(S.wF.acquire(nw * sizeof(float)));
    float* d_pressureFull = static_cast<float*>(S.pressureFull.acquire(totalCells * sizeof(float)));
    int* d_counter = static_cast<int*>(S.counter.acquire(sizeof(int)));
    int* d_indexOfCell = static_cast<int*>(S.indexOfCell.acquire(totalCells * sizeof(int)));
    int* d_cellOfIndex = static_cast<int*>(S.cellOfIndex.acquire(totalCells * sizeof(int)));
    if (!up || !d_cellType || !d_uF || !d_vF || !d_wF || !d_pressureFull ||
        !d_counter || !d_indexOfCell || !d_cellOfIndex)
        return -1;

    signed char* hCell = static_cast<signed char*>(up);
    float* hU = reinterpret_cast<float*>(hCell + totalCells);
    float* hV = hU + nu;
    float* hW = hV + nv;
    std::memcpy(hCell, grid.cellType.data(), totalCells);
    std::memcpy(hU, grid.u.data(), nu * sizeof(float));
    std::memcpy(hV, grid.v.data(), nv * sizeof(float));
    std::memcpy(hW, grid.w.data(), nw * sizeof(float));
    if (!cok(cudaMemcpyAsync(d_cellType, hCell, totalCells, cudaMemcpyHostToDevice), "up cellType")) return -1;
    if (!cok(cudaMemcpyAsync(d_uF, hU, nu * sizeof(float), cudaMemcpyHostToDevice), "up u")) return -1;
    if (!cok(cudaMemcpyAsync(d_vF, hV, nv * sizeof(float), cudaMemcpyHostToDevice), "up v")) return -1;
    if (!cok(cudaMemcpyAsync(d_wF, hW, nw * sizeof(float), cudaMemcpyHostToDevice), "up w")) return -1;

    // ── fluid index built on the GPU (replaces the host's full-grid scan) ─
    if (!cok(cudaMemsetAsync(d_counter, 0, sizeof(int)), "zero counter")) return -1;
    buildFluidIndex_kernel<<<gd(totalCells), block>>>(
        d_cellType, d_indexOfCell, d_cellOfIndex, d_counter, int(totalCells));
    if (!cok(cudaGetLastError(), "buildFluidIndex")) return -1;

    int n = 0;
    if (!cok(cudaMemcpy(&n, d_counter, sizeof(int), cudaMemcpyDeviceToHost), "read counter")) return -1;
    if (n <= 0) return 0; // no fluid cells: nothing to solve, host state untouched
    const size_t nc = size_t(n);

    float* d_b    = static_cast<float*>(S.b.acquire(nc * sizeof(float)));
    float* d_x    = static_cast<float*>(S.x.acquire(nc * sizeof(float)));
    float* d_r    = static_cast<float*>(S.r.acquire(nc * sizeof(float)));
    float* d_p    = static_cast<float*>(S.p.acquire(nc * sizeof(float)));
    float* d_z    = static_cast<float*>(S.z.acquire(nc * sizeof(float)));
    float* d_Ap   = static_cast<float*>(S.Ap.acquire(nc * sizeof(float)));
    float* d_diag = static_cast<float*>(S.diag.acquire(nc * sizeof(float)));
    cublasHandle_t cublas = S.cublas;
    if (!d_b || !d_x || !d_r || !d_p || !d_z || !d_Ap || !d_diag) return -1;

    const float h = grid.h();
    const float scale = rho * h * h / std::max(dt, 1e-6f);
    const float dtOverH = dt / h;
    const float invRho = 1.f / std::max(rho, 1e-6f);
    const float invRhoAir = (airDensityRatio > 0.f) ? 1.f / (rho * airDensityRatio) : invRho;

    buildRHS_kernel<<<gd(nc), block>>>(d_cellOfIndex, d_uF, d_vF, d_wF, d_b,
                                       nx, ny, nz, scale, 1.f / h, n);
    if (!cok(cudaGetLastError(), "buildRHS")) return -1;
    computeDiagonal_kernel<<<gd(nc), block>>>(d_cellOfIndex, d_cellType, d_diag, nx, ny, nz, n);
    if (!cok(cudaGetLastError(), "computeDiagonal")) return -1;
    int iter = 0;

    // ── CG loop ────────────────────────────────────────────────────────
    {
        int block = 256, gridDimC = int((n + block - 1) / block);
        float one = 1.f;

        // Initial guess: device-resident previous pressure (warm start) or
        // zero. The snapshot round-trip through the host is gone entirely.
        if (useWarmStart && S.pressureValid) {
            gatherX0_kernel<<<gridDimC, block>>>(n, d_cellOfIndex, d_pressureFull, d_x);
            cudaDeviceSynchronize();
            // r0 = b - A*x0
            applyA_kernel<<<gridDimC, block>>>(d_cellOfIndex, d_indexOfCell, d_cellType,
                                                d_x, d_Ap, nx, ny, nz, n);
            cudaDeviceSynchronize();
            cublasScopy(cublas, n, d_b, 1, d_r, 1);
            float mone = -1.f;
            cublasSaxpy(cublas, n, &mone, d_Ap, 1, d_r, 1);
        } else {
            // x0 = 0, r0 = b
            cudaMemset(d_x, 0, n * sizeof(float));
            cublasScopy(cublas, n, d_b, 1, d_r, 1);
        }

        // z0 = M^-1 * r0
        applyJacobi_kernel<<<gridDimC, block>>>(d_diag, d_r, d_z, n);
        cudaDeviceSynchronize();

        // p0 = z0
        cublasScopy(cublas, n, d_z, 1, d_p, 1);

        float rzOld = 0.f, rzNew = 0.f;
        cublasSdot(cublas, n, d_r, 1, d_z, 1, &rzOld);

        float rsInit = 0.f, rsNew = 0.f;
        cublasSdot(cublas, n, d_r, 1, d_r, 1, &rsInit);

        if (std::sqrt(std::max(rsInit, 0.f)) >= 1e-9f) {

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
        } // rsInit guard
    }

    // ── device-side finish: scatter pressure, project velocities, zero
    // solid faces. Pressure never touches the host on this path; it stays
    // resident as the next substep's warm start.
    scatterPressure_kernel<<<gd(nc), block>>>(n, d_cellOfIndex, d_x, d_pressureFull);
    if (!cok(cudaGetLastError(), "scatterPressure")) return -1;
    S.pressureValid = true;

    const float airRatio = airDensityRatio;
    projectU_kernel<<<gd(size_t(std::max(nx - 1, 0)) * ny * nz), block>>>(
        d_cellType, d_pressureFull, d_uF, nx, ny, nz, dtOverH, invRho, invRhoAir, airRatio);
    projectV_kernel<<<gd(size_t(nx) * std::max(ny - 1, 0) * nz), block>>>(
        d_cellType, d_pressureFull, d_vF, nx, ny, nz, dtOverH, invRho, invRhoAir, airRatio);
    projectW_kernel<<<gd(size_t(nx) * ny * std::max(nz - 1, 0)), block>>>(
        d_cellType, d_pressureFull, d_wF, nx, ny, nz, dtOverH, invRho, invRhoAir, airRatio);
    zeroSolidU_kernel<<<gd(nu), block>>>(d_cellType, d_uF, nx, ny, nz);
    zeroSolidV_kernel<<<gd(nv), block>>>(d_cellType, d_vF, nx, ny, nz);
    zeroSolidW_kernel<<<gd(nw), block>>>(d_cellType, d_wF, nx, ny, nz);
    if (!cok(cudaGetLastError(), "project/zero")) return -1;

    // ── download the projected staggered fields (pinned staging) ───────
    void* down = S.down.acquire((nu + nv + nw) * sizeof(float));
    if (!down) return -1;
    if (!cok(cudaMemcpyAsync(down, d_uF, nu * sizeof(float), cudaMemcpyDeviceToHost), "down u")) return -1;
    if (!cok(cudaMemcpyAsync(static_cast<char*>(down) + nu * sizeof(float),
                             d_vF, nv * sizeof(float), cudaMemcpyDeviceToHost), "down v")) return -1;
    if (!cok(cudaMemcpyAsync(static_cast<char*>(down) + (nu + nv) * sizeof(float),
                             d_wF, nw * sizeof(float), cudaMemcpyDeviceToHost), "down w")) return -1;
    if (!cok(cudaDeviceSynchronize(), "download sync")) return -1;

    std::memcpy(grid.u.data(), down, nu * sizeof(float));
    std::memcpy(grid.v.data(), static_cast<const char*>(down) + nu * sizeof(float), nv * sizeof(float));
    std::memcpy(grid.w.data(), static_cast<const char*>(down) + (nu + nv) * sizeof(float), nw * sizeof(float));

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

    GpuScratch& S = scratch();
    // Persistent device buffers + pinned staging: zero cudaMalloc/cudaFree on
    // the steady-state path (this used to be 7 mallocs + frees per substep).
    float* d_posX = static_cast<float*>(S.posX.acquire(n * sizeof(float)));
    float* d_posY = static_cast<float*>(S.posY.acquire(n * sizeof(float)));
    float* d_posZ = static_cast<float*>(S.posZ.acquire(n * sizeof(float)));
    float* d_velX = static_cast<float*>(S.velX.acquire(n * sizeof(float)));
    float* d_velY = static_cast<float*>(S.velY.acquire(n * sizeof(float)));
    float* d_velZ = static_cast<float*>(S.velZ.acquire(n * sizeof(float)));
    float* d_sdf  = static_cast<float*>(S.sdf.acquire(sdfBytes));
    void* up = S.up.acquire(6 * n * sizeof(float) + sdfBytes);
    void* down = S.down.acquire(6 * n * sizeof(float));
    if (!d_posX || !d_posY || !d_posZ || !d_velX || !d_velY || !d_velZ ||
        !d_sdf || !up || !down)
        return -1;

    // ── upload positions/velocities/SDF (single pinned window, async) ────
    float* hPos = static_cast<float*>(up);
    float* hVel = hPos + 3 * n;
    const auto& pos = s.positions();
    const auto& vel = s.velocities();
    for (size_t i = 0; i < n; ++i) {
        hPos[3 * i + 0] = pos[i].x; hPos[3 * i + 1] = pos[i].y; hPos[3 * i + 2] = pos[i].z;
        hVel[3 * i + 0] = vel[i].x; hVel[3 * i + 1] = vel[i].y; hVel[3 * i + 2] = vel[i].z;
    }
    float* hSdf = hVel + 3 * n;
    std::memcpy(hSdf, sdf.data(), sdfBytes);

    // The SDF field only changes when setObstacleSDF() is called again
    // (revision bump) or a different host field arrives; skip its PCIe
    // transfer otherwise — this was the whole-SDF re-upload per substep.
    bool sdfDirty = (S.sdfHostPtr != sdf.data()) || (S.sdfBytes != sdfBytes) ||
                    (S.sdfRevision != s.obstacleSdfRevision());
    if (!cok(cudaMemcpyAsync(d_posX, hPos, 3 * n * sizeof(float), cudaMemcpyHostToDevice), "col up pos")) return -1;
    if (!cok(cudaMemcpyAsync(d_velX, hVel, 3 * n * sizeof(float), cudaMemcpyHostToDevice), "col up vel")) return -1;
    if (sdfDirty) {
        if (!cok(cudaMemcpyAsync(d_sdf, hSdf, sdfBytes, cudaMemcpyHostToDevice), "col up sdf")) return -1;
        S.sdfHostPtr = sdf.data();
        S.sdfBytes = sdfBytes;
        S.sdfRevision = s.obstacleSdfRevision();
    }

    int block = 256, gridDim = int((n + block - 1) / block);
    sdfCollisionKernel<<<gridDim, block>>>(
        d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ,
        d_sdf, nx, ny, nz,
        dmin.x, dmin.y, dmin.z, dmax.x, dmax.y, dmax.z,
        h, margin, int(n));
    if (!cok(cudaGetLastError(), "sdfCollisionKernel")) return -1;

    // ── download updated positions/velocities ───────────────────────────
    if (!cok(cudaMemcpyAsync(down, d_posX, 3 * n * sizeof(float), cudaMemcpyDeviceToHost), "col down pos")) return -1;
    if (!cok(cudaMemcpyAsync(static_cast<char*>(down) + 3 * n * sizeof(float),
                             d_velX, 3 * n * sizeof(float), cudaMemcpyDeviceToHost), "col down vel")) return -1;
    if (!cok(cudaDeviceSynchronize(), "col sync")) return -1;

    const float* hPosOut = static_cast<const float*>(down);
    const float* hVelOut = hPosOut + 3 * n;
    for (size_t i = 0; i < n; ++i) {
        s.positions()[i] = flipcore::Vec3(hPosOut[3 * i + 0], hPosOut[3 * i + 1], hPosOut[3 * i + 2]);
        s.velocities()[i] = flipcore::Vec3(hVelOut[3 * i + 0], hVelOut[3 * i + 1], hVelOut[3 * i + 2]);
    }

    return int(n);
}

} // namespace flipcore
