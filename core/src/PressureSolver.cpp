#include "flipcore/PressureSolver.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace flipcore {
namespace {

float computeDivergence(const MacGrid& g, int i, int j, int k) {
    float h = g.h();
    float du = g.u(i + 1, j, k) - g.u(i, j, k);
    float dv = g.v(i, j + 1, k) - g.v(i, j, k);
    float dw = g.w(i, j, k + 1) - g.w(i, j, k);
    return (du + dv + dw) / h;
}

// Maps between "compact" unknown indices [0, numFluid) and the underlying
// (i,j,k) grid cells that are actually FLUID. AIR/SOLID cells are pinned at
// pressure = 0 implicitly and never enter the linear system at all, which is
// what makes the CG solve scale with the number of *wet* cells instead of
// the whole domain (important: most of a domain's cells are usually empty
// air, especially at higher resolutions).
struct FluidIndex {
    std::vector<int> cellOfIndex;  // compact index -> flat cell id (i + nx*(j + ny*k))
    std::vector<int> indexOfCell;  // flat cell id -> compact index, or -1 if not fluid
};

FluidIndex buildFluidIndex(const MacGrid& g) {
    int nx = g.nx(), ny = g.ny(), nz = g.nz();
    FluidIndex fi;
    fi.indexOfCell.assign(size_t(nx) * size_t(ny) * size_t(nz), -1);
    fi.cellOfIndex.reserve(fi.indexOfCell.size() / 4 + 16);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                signed char t = g.cellType(i, j, k);
                if (t != CELL_FLUID && t != CELL_AIR_ACTIVE) continue;
                size_t cellId = g.cellType.idx(i, j, k);
                fi.indexOfCell[cellId] = int(fi.cellOfIndex.size());
                fi.cellOfIndex.push_back(int(cellId));
            }
        }
    }
    return fi;
}

inline bool isPressureCell(signed char t) {
    return t == CELL_FLUID || t == CELL_AIR_ACTIVE;
}

inline void decode(int cellId, int nx, int nxTimesNy, int& i, int& j, int& k) {
    k = cellId / nxTimesNy;
    int rem = cellId % nxTimesNy;
    j = rem / nx;
    i = rem % nx;
}

// out = A * x over the compacted fluid-cell unknown vector. SOLID neighbors
// (including out-of-grid, treated as solid) are dropped (zero-flux); AIR
// neighbors contribute their implicit zero pressure (Dirichlet free surface).
void applyA(const MacGrid& g, const FluidIndex& fi, const std::vector<float>& x, std::vector<float>& out) {
    int nx = g.nx(), ny = g.ny(), nz = g.nz();
    int nxTimesNy = nx * ny;
    const float kReg = 1e-4f; // small regularization for disconnected fluid pockets
    static const int dI[6] = {1, -1, 0, 0, 0, 0};
    static const int dJ[6] = {0, 0, 1, -1, 0, 0};
    static const int dK[6] = {0, 0, 0, 0, 1, -1};

    size_t n = fi.cellOfIndex.size();
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (long long ci64 = 0; ci64 < (long long)n; ++ci64) {
        size_t ci = size_t(ci64);
        int i, j, k;
        decode(fi.cellOfIndex[ci], nx, nxTimesNy, i, j, k);

        float diag = kReg;
        float sum = 0.f;
        for (int d = 0; d < 6; ++d) {
            int ni = i + dI[d], nj = j + dJ[d], nk = k + dK[d];
            if (ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 || nk >= nz) continue; // out of grid = solid
            signed char nt = g.cellType(ni, nj, nk);
            if (nt == CELL_SOLID) continue;
            diag += 1.f;
            if (isPressureCell(nt)) {
                int nci = fi.indexOfCell[g.cellType.idx(ni, nj, nk)];
                sum += x[nci];
            }
        }
        out[ci] = diag * x[ci] - sum;
    }
}

float dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    long long n = (long long)a.size();
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:s) schedule(static)
    #endif
    for (long long i = 0; i < n; ++i) s += double(a[size_t(i)]) * double(b[size_t(i)]);
    return float(s);
}

// The diagonal of A doesn't change during the solve (it only depends on
// cell types, not on the current pressure guess), so we compute it once and
// reuse it every iteration as a Jacobi preconditioner: M^-1 = diag(A)^-1.
// This is a cheap, embarrassingly-parallel preconditioner that typically
// cuts the number of CG iterations needed to reach a given tolerance
// noticeably on Poisson-like systems such as this one.
void computeDiagonal(const MacGrid& g, const FluidIndex& fi, std::vector<float>& diagOut) {
    int nx = g.nx(), ny = g.ny(), nz = g.nz();
    const float kReg = 1e-4f;
    static const int dI[6] = {1, -1, 0, 0, 0, 0};
    static const int dJ[6] = {0, 0, 1, -1, 0, 0};
    static const int dK[6] = {0, 0, 0, 0, 1, -1};

    size_t n = fi.cellOfIndex.size();
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (long long ci64 = 0; ci64 < (long long)n; ++ci64) {
        size_t ci = size_t(ci64);
        int i, j, k;
        decode(fi.cellOfIndex[ci], nx, nx * ny, i, j, k);
        float diag = kReg;
        for (int d = 0; d < 6; ++d) {
            int ni = i + dI[d], nj = j + dJ[d], nk = k + dK[d];
            if (ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 || nk >= nz) continue;
            if (g.cellType(ni, nj, nk) == CELL_SOLID) continue;
            diag += 1.f;
        }
        diagOut[ci] = diag;
    }
}

void applyJacobi(const std::vector<float>& diag, const std::vector<float>& r, std::vector<float>& z) {
    long long n = (long long)diag.size();
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (long long i = 0; i < n; ++i) z[size_t(i)] = r[size_t(i)] / diag[size_t(i)];
}

} // namespace

void projectPressureVelocities(MacGrid& g, float dt, float rho, float airDensityRatio) {
    // Project velocities using the solved pressure field so the fluid region
    // becomes (approximately) divergence-free. Plain AIR cells are pinned at
    // pressure = 0; CELL_AIR_ACTIVE cells (when airDensityRatio > 0) carry
    // pressure with density rho*ratio.
    const Array3<float>& x = g.pressure;
    int nx = g.nx(), ny = g.ny(), nz = g.nz();
    float h = g.h();
    const float invRho = 1.f / rho;
    const float invRhoAir = (airDensityRatio > 0.f) ? 1.f / (rho * airDensityRatio) : invRho;

    auto faceRho = [&](signed char t) -> float {
        return (t == CELL_AIR_ACTIVE && airDensityRatio > 0.f) ? invRhoAir : invRho;
    };
    auto cellPressure = [&](signed char t, int i, int j, int k) -> float {
        return isPressureCell(t) ? x(i, j, k) : 0.f;
    };

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 1; i < nx; ++i) {
                signed char a = g.cellType(i - 1, j, k);
                signed char c = g.cellType(i, j, k);
                if (a == CELL_SOLID || c == CELL_SOLID) continue;
                if (!isPressureCell(a) && !isPressureCell(c)) continue;
                float pA = cellPressure(a, i - 1, j, k);
                float pB = cellPressure(c, i, j, k);
                float rA = faceRho(a), rC = faceRho(c);
                float inv = isPressureCell(a) && isPressureCell(c) ? 0.5f * (rA + rC) : rA + rC - invRho;
                g.u(i, j, k) -= dt * inv * (pB - pA) / h;
            }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int k = 0; k < nz; ++k)
        for (int j = 1; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                signed char a = g.cellType(i, j - 1, k);
                signed char c = g.cellType(i, j, k);
                if (a == CELL_SOLID || c == CELL_SOLID) continue;
                if (!isPressureCell(a) && !isPressureCell(c)) continue;
                float pA = cellPressure(a, i, j - 1, k);
                float pB = cellPressure(c, i, j, k);
                float rA = faceRho(a), rC = faceRho(c);
                float inv = isPressureCell(a) && isPressureCell(c) ? 0.5f * (rA + rC) : rA + rC - invRho;
                g.v(i, j, k) -= dt * inv * (pB - pA) / h;
            }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int k = 1; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                signed char a = g.cellType(i, j, k - 1);
                signed char c = g.cellType(i, j, k);
                if (a == CELL_SOLID || c == CELL_SOLID) continue;
                if (!isPressureCell(a) && !isPressureCell(c)) continue;
                float pA = cellPressure(a, i, j, k - 1);
                float pB = cellPressure(c, i, j, k);
                float rA = faceRho(a), rC = faceRho(c);
                float inv = isPressureCell(a) && isPressureCell(c) ? 0.5f * (rA + rC) : rA + rC - invRho;
                g.w(i, j, k) -= dt * inv * (pB - pA) / h;
            }

    g.zeroSolidNormalVelocities();
}

int solvePressure(MacGrid& g, float dt, float rho, int maxIterations,
                  float tolerance, const float* warmStart, float airDensityRatio) {
    g.pressure.fill(0.f);

    FluidIndex fi = buildFluidIndex(g);
    size_t n = fi.cellOfIndex.size();
    int iter = 0;

    if (n > 0) {
        int nx = g.nx(), ny = g.ny();
        int nxTimesNy = nx * ny;
        float scale = rho * g.h() * g.h() / std::max(dt, 1e-6f);

        std::vector<float> b(n, 0.f), x(n, 0.f), r(n), p(n), z(n), Ap(n, 0.f), diagA(n);
        for (size_t ci = 0; ci < n; ++ci) {
            int i, j, k;
            decode(fi.cellOfIndex[ci], nx, nxTimesNy, i, j, k);
            float rhoScale = (g.cellType(i, j, k) == CELL_AIR_ACTIVE && airDensityRatio > 0.f)
                                 ? airDensityRatio : 1.f;
            b[ci] = -scale * rhoScale * computeDivergence(g, i, j, k);
        }
        computeDiagonal(g, fi, diagA);

        if (warmStart != nullptr) {
            // x0 = previous pressure snapshot; r0 = b - A x0.
            for (size_t ci = 0; ci < n; ++ci) x[ci] = warmStart[size_t(fi.cellOfIndex[ci])];
            applyA(g, fi, x, Ap);
            for (size_t ci = 0; ci < n; ++ci) r[ci] = b[ci] - Ap[ci];
        } else {
            r = b; // x0 = 0 => r0 = b
        }
        applyJacobi(diagA, r, z);
        p = z;

        float rzOld = dotProduct(r, z);
        float rsInit = dotProduct(r, r);
        if (std::sqrt(std::max(rsInit, 0.f)) >= 1e-9f) {
            for (; iter < maxIterations; ++iter) {
                applyA(g, fi, p, Ap);
                float pAp = dotProduct(p, Ap);
                if (std::fabs(pAp) < 1e-20f) break;
                float alpha = rzOld / pAp;
                long long nn = (long long)n;
                #ifdef _OPENMP
                #pragma omp parallel for schedule(static)
                #endif
                for (long long idx64 = 0; idx64 < nn; ++idx64) {
                    size_t idx = size_t(idx64);
                    x[idx] += alpha * p[idx];
                    r[idx] -= alpha * Ap[idx];
                }
                float rsNew = dotProduct(r, r);
                if (std::sqrt(std::max(rsNew, 0.f)) < tolerance) { iter++; break; }
                applyJacobi(diagA, r, z);
                float rzNew = dotProduct(r, z);
                float beta = rzNew / rzOld;
                #ifdef _OPENMP
                #pragma omp parallel for schedule(static)
                #endif
                for (long long idx64 = 0; idx64 < nn; ++idx64) {
                    size_t idx = size_t(idx64);
                    p[idx] = z[idx] + beta * p[idx];
                }
                rzOld = rzNew;
            }
        }

        for (size_t ci = 0; ci < n; ++ci) {
            int i, j, k;
            decode(fi.cellOfIndex[ci], nx, nxTimesNy, i, j, k);
            g.pressure(i, j, k) = x[ci];
        }
    }

    projectPressureVelocities(g, dt, rho, airDensityRatio);
    return iter;
}

} // namespace flipcore
