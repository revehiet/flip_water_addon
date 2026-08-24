#pragma once
#include "Array3.h"
#include "Vec3.h"
#include <vector>

namespace flipcore {

enum CellType : signed char { CELL_AIR = 0, CELL_FLUID = 1, CELL_SOLID = 2, CELL_AIR_ACTIVE = 3 };

// An inclusive integer cell-index bounding box, used to restrict operations
// (like extrapolation) to a narrow band around the fluid instead of the
// whole domain grid - this matters a lot for large domains that are mostly
// empty air.
struct CellBounds {
    int iMin = 0, iMax = -1, jMin = 0, jMax = -1, kMin = 0, kMax = -1;
    bool empty() const { return iMax < iMin || jMax < jMin || kMax < kMin; }
};

// Axis-aligned MAC (marker-and-cell) staggered grid.
// Domain occupies [0, nx*h] x [0, ny*h] x [0, nz*h] in grid-local space.
class MacGrid {
public:
    void resize(int nx, int ny, int nz, float h);

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    float h() const { return h_; }

    // Staggered velocity components
    Array3<float> u, v, w;       // current velocities
    Array3<float> uOld, vOld, wOld; // snapshot before pressure solve (for FLIP delta)
    Array3<float> uWeight, vWeight, wWeight; // accumulation weights during P2G
    Array3<float> pressure;
    Array3<signed char> cellType;

    // Cell-centered phase-field mass accumulator (ST-FLIP, Braun et al. 2026,
    // Sec 3.6): replaces per-particle-position cell marking with a
    // temporal-kernel-weighted deposit, reused directly as the fluid/air
    // classification for pressure projection. Cleared and re-accumulated
    // every grid step alongside the velocity splat.
    Array3<float> cellWeight;

    void clearVelocities();
    void clearWeights();
    void snapshotVelocities();     // uOld = u, etc.
    void addGravity(float dt, const Vec3& g);

    void markSolidBoundary();      // marks the outer shell of the domain as solid

    // Trilinear sample of the full velocity field at an arbitrary grid-local position.
    Vec3 sampleVelocity(const Vec3& posGrid) const;

    // Cell-centered velocity at cell (i,j,k), averaged from the surrounding
    // staggered MAC faces (used for vorticity and Whitewater source fields).
    Vec3 cellCenteredVelocity(int i, int j, int k) const;

    // Splat a single particle's velocity into u,v,w with trilinear weights,
    // scaled by `temporalWeight` (the ST-FLIP W_T kernel evaluated at this
    // particle's jittered sample time; pass 1.0 to recover plain FLIP P2G).
    void splatParticle(const Vec3& posGrid, const Vec3& vel, float temporalWeight = 1.f);
    void normalizeBySplatWeight();

    // Adds `amount` directly to the single cell containing posGrid (nearest,
    // not trilinear-spread) - used for the phase-field mass accumulator.
    void addCellWeight(const Vec3& posGrid, float amount);

    // FLIP delta: for a particle at posGrid, sample (u - uOld) trilinearly.
    Vec3 sampleVelocityDelta(const Vec3& posGrid) const;

    void extrapolateAll(int iterations);
    void extrapolateAll(int iterations, const CellBounds& bounds);

    void zeroSolidNormalVelocities();

private:
    int nx_ = 0, ny_ = 0, nz_ = 0;
    float h_ = 1.f;

    static void trilinearWeights(float fx, float fy, float fz, int& i0, int& j0, int& k0, float w[8]);
};

} // namespace flipcore
