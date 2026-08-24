#include "flipcore/MacGrid.h"
#include <cmath>
#include <algorithm>

namespace flipcore {

namespace {

inline float lerp(float a, float b, float t) { return a + (b - a) * t; }

float sampleField(const Array3<float>& f, float gx, float gy, float gz) {
    int nx = f.nx(), ny = f.ny(), nz = f.nz();
    if (nx == 0 || ny == 0 || nz == 0) return 0.f;
    gx = clampf(gx, 0.f, float(nx - 1));
    gy = clampf(gy, 0.f, float(ny - 1));
    gz = clampf(gz, 0.f, float(nz - 1));

    int i0 = (int)std::floor(gx); int i1 = std::min(i0 + 1, nx - 1);
    int j0 = (int)std::floor(gy); int j1 = std::min(j0 + 1, ny - 1);
    int k0 = (int)std::floor(gz); int k1 = std::min(k0 + 1, nz - 1);
    float fx = gx - i0, fy = gy - j0, fz = gz - k0;

    float c00 = lerp(f(i0, j0, k0), f(i1, j0, k0), fx);
    float c10 = lerp(f(i0, j1, k0), f(i1, j1, k0), fx);
    float c01 = lerp(f(i0, j0, k1), f(i1, j0, k1), fx);
    float c11 = lerp(f(i0, j1, k1), f(i1, j1, k1), fx);
    float c0 = lerp(c00, c10, fy);
    float c1 = lerp(c01, c11, fy);
    return lerp(c0, c1, fz);
}

void splatField(Array3<float>& f, Array3<float>& wgt, float gx, float gy, float gz, float value, float scale) {
    int nx = f.nx(), ny = f.ny(), nz = f.nz();
    if (nx == 0 || ny == 0 || nz == 0) return;
    gx = clampf(gx, 0.f, float(nx - 1));
    gy = clampf(gy, 0.f, float(ny - 1));
    gz = clampf(gz, 0.f, float(nz - 1));

    int i0 = (int)std::floor(gx); int i1 = std::min(i0 + 1, nx - 1);
    int j0 = (int)std::floor(gy); int j1 = std::min(j0 + 1, ny - 1);
    int k0 = (int)std::floor(gz); int k1 = std::min(k0 + 1, nz - 1);
    float fx = gx - i0, fy = gy - j0, fz = gz - k0;

    float w000 = scale * (1 - fx) * (1 - fy) * (1 - fz);
    float w100 = scale * fx * (1 - fy) * (1 - fz);
    float w010 = scale * (1 - fx) * fy * (1 - fz);
    float w110 = scale * fx * fy * (1 - fz);
    float w001 = scale * (1 - fx) * (1 - fy) * fz;
    float w101 = scale * fx * (1 - fy) * fz;
    float w011 = scale * (1 - fx) * fy * fz;
    float w111 = scale * fx * fy * fz;

    f(i0, j0, k0) += w000 * value; wgt(i0, j0, k0) += w000;
    f(i1, j0, k0) += w100 * value; wgt(i1, j0, k0) += w100;
    f(i0, j1, k0) += w010 * value; wgt(i0, j1, k0) += w010;
    f(i1, j1, k0) += w110 * value; wgt(i1, j1, k0) += w110;
    f(i0, j0, k1) += w001 * value; wgt(i0, j0, k1) += w001;
    f(i1, j0, k1) += w101 * value; wgt(i1, j0, k1) += w101;
    f(i0, j1, k1) += w011 * value; wgt(i0, j1, k1) += w011;
    f(i1, j1, k1) += w111 * value; wgt(i1, j1, k1) += w111;
}

void extrapolateField(Array3<float>& f, const Array3<float>& weight, int iterations,
                       int i0, int i1, int j0, int j1, int k0, int k1) {
    int nx = f.nx(), ny = f.ny(), nz = f.nz();
    i0 = std::max(0, i0); i1 = std::min(nx - 1, i1);
    j0 = std::max(0, j0); j1 = std::min(ny - 1, j1);
    k0 = std::max(0, k0); k1 = std::min(nz - 1, k1);
    if (i0 > i1 || j0 > j1 || k0 > k1) return; // empty sub-box, nothing to do

    int sx = i1 - i0 + 1, sy = j1 - j0 + 1, sz = k1 - k0 + 1;
    Array3<signed char> valid(sx, sy, sz, 0);
    for (int k = k0; k <= k1; ++k)
        for (int j = j0; j <= j1; ++j)
            for (int i = i0; i <= i1; ++i)
                valid(i - i0, j - j0, k - k0) = weight(i, j, k) > 1e-6f ? 1 : 0;

    const int dI[6] = {1, -1, 0, 0, 0, 0};
    const int dJ[6] = {0, 0, 1, -1, 0, 0};
    const int dK[6] = {0, 0, 0, 0, 1, -1};

    for (int it = 0; it < iterations; ++it) {
        Array3<signed char> newValid = valid;
        bool anyChange = false;
        for (int k = 0; k < sz; ++k) {
            for (int j = 0; j < sy; ++j) {
                for (int i = 0; i < sx; ++i) {
                    if (valid(i, j, k)) continue;
                    float sum = 0.f; int cnt = 0;
                    for (int n = 0; n < 6; ++n) {
                        int ni = i + dI[n], nj = j + dJ[n], nk = k + dK[n];
                        if (ni < 0 || ni >= sx || nj < 0 || nj >= sy || nk < 0 || nk >= sz) continue;
                        if (valid(ni, nj, nk)) { sum += f(ni + i0, nj + j0, nk + k0); cnt++; }
                    }
                    if (cnt > 0) {
                        f(i + i0, j + j0, k + k0) = sum / cnt;
                        newValid(i, j, k) = 1;
                        anyChange = true;
                    }
                }
            }
        }
        valid = newValid;
        if (!anyChange) break;
    }
}

} // namespace

void MacGrid::resize(int nx, int ny, int nz, float h) {
    nx_ = nx; ny_ = ny; nz_ = nz; h_ = h;
    u.resize(nx + 1, ny, nz, 0.f);
    v.resize(nx, ny + 1, nz, 0.f);
    w.resize(nx, ny, nz + 1, 0.f);
    uOld.resize(nx + 1, ny, nz, 0.f);
    vOld.resize(nx, ny + 1, nz, 0.f);
    wOld.resize(nx, ny, nz + 1, 0.f);
    uWeight.resize(nx + 1, ny, nz, 0.f);
    vWeight.resize(nx, ny + 1, nz, 0.f);
    wWeight.resize(nx, ny, nz + 1, 0.f);
    pressure.resize(nx, ny, nz, 0.f);
    cellType.resize(nx, ny, nz, CELL_AIR);
    cellWeight.resize(nx, ny, nz, 0.f);
}

void MacGrid::clearVelocities() {
    u.fill(0.f); v.fill(0.f); w.fill(0.f);
}

void MacGrid::clearWeights() {
    uWeight.fill(0.f); vWeight.fill(0.f); wWeight.fill(0.f);
}

void MacGrid::snapshotVelocities() {
    uOld = u; vOld = v; wOld = w;
}

void MacGrid::addGravity(float dt, const Vec3& g) {
    // gravity only has y (up) component typically in Blender's Z-up world, but we
    // work in solver-local axes where "y" of MacGrid maps to whichever axis the
    // caller passes as g. Here we simply add g to all three staggered components
    // using the appropriate face field.
    for (size_t i = 0; i < u.size(); ++i) u.data()[i] += g.x * dt;
    for (size_t i = 0; i < v.size(); ++i) v.data()[i] += g.y * dt;
    for (size_t i = 0; i < w.size(); ++i) w.data()[i] += g.z * dt;
}

void MacGrid::markSolidBoundary() {
    for (int k = 0; k < nz_; ++k) {
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                bool boundary = (i == 0 || i == nx_ - 1 || j == 0 || j == ny_ - 1 || k == 0 || k == nz_ - 1);
                if (boundary) cellType(i, j, k) = CELL_SOLID;
            }
        }
    }
}

Vec3 MacGrid::sampleVelocity(const Vec3& p) const {
    float uu = sampleField(u, p.x, p.y - 0.5f, p.z - 0.5f);
    float vv = sampleField(v, p.x - 0.5f, p.y, p.z - 0.5f);
    float ww = sampleField(w, p.x - 0.5f, p.y - 0.5f, p.z);
    return {uu, vv, ww};
}

Vec3 MacGrid::cellCenteredVelocity(int i, int j, int k) const {
    float uu = 0.5f * (u(i, j, k) + u(i + 1, j, k));
    float vv = 0.5f * (v(i, j, k) + v(i, j + 1, k));
    float ww = 0.5f * (w(i, j, k) + w(i, j, k + 1));
    return {uu, vv, ww};
}

Vec3 MacGrid::sampleVelocityDelta(const Vec3& p) const {
    float du = sampleField(u, p.x, p.y - 0.5f, p.z - 0.5f) - sampleField(uOld, p.x, p.y - 0.5f, p.z - 0.5f);
    float dv = sampleField(v, p.x - 0.5f, p.y, p.z - 0.5f) - sampleField(vOld, p.x - 0.5f, p.y, p.z - 0.5f);
    float dw = sampleField(w, p.x - 0.5f, p.y - 0.5f, p.z) - sampleField(wOld, p.x - 0.5f, p.y - 0.5f, p.z);
    return {du, dv, dw};
}

void MacGrid::splatParticle(const Vec3& p, const Vec3& vel, float temporalWeight) {
    splatField(u, uWeight, p.x, p.y - 0.5f, p.z - 0.5f, vel.x, temporalWeight);
    splatField(v, vWeight, p.x - 0.5f, p.y, p.z - 0.5f, vel.y, temporalWeight);
    splatField(w, wWeight, p.x - 0.5f, p.y - 0.5f, p.z, vel.z, temporalWeight);
}

void MacGrid::addCellWeight(const Vec3& p, float amount) {
    int i = std::clamp((int)std::floor(p.x), 0, nx_ - 1);
    int j = std::clamp((int)std::floor(p.y), 0, ny_ - 1);
    int k = std::clamp((int)std::floor(p.z), 0, nz_ - 1);
    cellWeight(i, j, k) += amount;
}

void MacGrid::normalizeBySplatWeight() {
    for (size_t i = 0; i < u.size(); ++i) if (uWeight.data()[i] > 1e-6f) u.data()[i] /= uWeight.data()[i];
    for (size_t i = 0; i < v.size(); ++i) if (vWeight.data()[i] > 1e-6f) v.data()[i] /= vWeight.data()[i];
    for (size_t i = 0; i < w.size(); ++i) if (wWeight.data()[i] > 1e-6f) w.data()[i] /= wWeight.data()[i];
}

void MacGrid::extrapolateAll(int iterations) {
    CellBounds full;
    full.iMin = 0; full.iMax = nx_ - 1;
    full.jMin = 0; full.jMax = ny_ - 1;
    full.kMin = 0; full.kMax = nz_ - 1;
    extrapolateAll(iterations, full);
}

void MacGrid::extrapolateAll(int iterations, const CellBounds& b) {
    if (b.empty()) return;
    // u/v/w each have one extra entry along their own staggered axis; pad
    // the cell-index box by 1 in that axis so the corresponding face range
    // is fully covered (extrapolateField clamps to valid array bounds).
    extrapolateField(u, uWeight, iterations, b.iMin, b.iMax + 1, b.jMin, b.jMax, b.kMin, b.kMax);
    extrapolateField(v, vWeight, iterations, b.iMin, b.iMax, b.jMin, b.jMax + 1, b.kMin, b.kMax);
    extrapolateField(w, wWeight, iterations, b.iMin, b.iMax, b.jMin, b.jMax, b.kMin, b.kMax + 1);
}

void MacGrid::zeroSolidNormalVelocities() {
    // Zero any staggered velocity face that touches a SOLID cell on either
    // side (this covers both the outer domain wall and any internal
    // obstacle-marked cells). Faces at the very edge of the array (i==0 or
    // i==nx_, etc.) have no neighbor on one side; that side is treated as
    // solid too since it represents the boundary of the simulated region.
    for (int k = 0; k < nz_; ++k) {
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i <= nx_; ++i) {
                bool solidLeft = (i > 0) ? (cellType(i - 1, j, k) == CELL_SOLID) : true;
                bool solidRight = (i < nx_) ? (cellType(i, j, k) == CELL_SOLID) : true;
                if (solidLeft || solidRight) u(i, j, k) = 0.f;
            }
        }
    }
    for (int k = 0; k < nz_; ++k) {
        for (int j = 0; j <= ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                bool solidBelow = (j > 0) ? (cellType(i, j - 1, k) == CELL_SOLID) : true;
                bool solidAbove = (j < ny_) ? (cellType(i, j, k) == CELL_SOLID) : true;
                if (solidBelow || solidAbove) v(i, j, k) = 0.f;
            }
        }
    }
    for (int k = 0; k <= nz_; ++k) {
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                bool solidBack = (k > 0) ? (cellType(i, j, k - 1) == CELL_SOLID) : true;
                bool solidFront = (k < nz_) ? (cellType(i, j, k) == CELL_SOLID) : true;
                if (solidBack || solidFront) w(i, j, k) = 0.f;
            }
        }
    }
}

} // namespace flipcore
