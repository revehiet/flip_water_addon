#include "flipcore/FlipSolver.h"
#include "flipcore/PressureSolver.h"
#ifdef FLIP_HAS_CUDA
#include "flipcore/PressureSolverGPU.h"
#endif
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace flipcore {

namespace {

// One-sided temporal transfer kernel W_T (ST-FLIP, Braun et al. 2026, Eq.
// 19): a poly6 bump shifted so it peaks at tau=0.5 and is truncated to zero
// beyond it, i.e. only the "past half" of a symmetric kernel is used, giving
// the most weight to the most recent samples within the current time slab.
// Normalized so that E[W_T(tau)] = 1 for tau ~ Uniform(-0.5, 0.5), verified
// numerically (35/16 is exact, not a rounded approximation).
inline float temporalKernel(float tau) {
    if (tau > 0.5f) return 0.f;
    float r = tau - 0.5f;
    float x = 1.f - r * r;
    if (x <= 0.f) return 0.f;
    return (35.f / 16.f) * (x * x * x);
}

// smoothstep(0,1;x) as defined in the paper's Preliminaries (Sec 3.2).
inline float smoothstep01(float x) {
    float s = clampf(x, 0.f, 1.f);
    return s * s * (3.f - 2.f * s);
}

// Fast deterministic hash -> pseudo-random float in [0,1). Used for the
// per-particle temporal jitter so the (OpenMP-parallel) advection loop needs
// no shared/locked RNG state, while remaining fully reproducible for a given
// step counter (Sec 3.10: "per-thread PRNGs seeded with a hash over the
// thread-id, the global step number, and a global seed").
inline uint32_t hashU32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
inline float hashRandom01(uint32_t particleIndex, uint32_t stepCounter) {
    uint32_t h = hashU32(particleIndex * 0x9e3779b9U + hashU32(stepCounter));
    return float(h & 0x00FFFFFFu) / float(0x01000000u); // 24-bit precision, in [0,1)
}

// Trilinear sample of a cell-centered field (cell i's value represents the
// sample at index i+0.5), so callers pass gx/gy/gz already offset by -0.5
// from a worldToIndex()-space position - same convention as MacGrid's
// staggered-axis sampling.
float sampleCellCenteredField(const Array3<float>& f, float gx, float gy, float gz) {
    int nx = f.nx(), ny = f.ny(), nz = f.nz();
    if (nx == 0 || ny == 0 || nz == 0) return 1e6f;
    gx = clampf(gx, 0.f, float(nx - 1));
    gy = clampf(gy, 0.f, float(ny - 1));
    gz = clampf(gz, 0.f, float(nz - 1));
    int i0 = (int)std::floor(gx); int i1 = std::min(i0 + 1, nx - 1);
    int j0 = (int)std::floor(gy); int j1 = std::min(j0 + 1, ny - 1);
    int k0 = (int)std::floor(gz); int k1 = std::min(k0 + 1, nz - 1);
    float fx = gx - i0, fy = gy - j0, fz = gz - k0;
    float c00 = f(i0, j0, k0) * (1 - fx) + f(i1, j0, k0) * fx;
    float c10 = f(i0, j1, k0) * (1 - fx) + f(i1, j1, k0) * fx;
    float c01 = f(i0, j0, k1) * (1 - fx) + f(i1, j0, k1) * fx;
    float c11 = f(i0, j1, k1) * (1 - fx) + f(i1, j1, k1) * fx;
    float c0 = c00 * (1 - fy) + c10 * fy;
    float c1 = c01 * (1 - fy) + c11 * fy;
    return c0 * (1 - fz) + c1 * fz;
}

} // namespace

void FlipSolver::initDomain(const Vec3& domainMin, const Vec3& domainMax, const SolverSettings& settings) {
    settings_ = settings;
    Vec3 size = domainMax - domainMin;
    float longest = std::max({size.x, size.y, size.z});
    if (longest <= 0.f) longest = 1.f;
    float h = longest / std::max(1, settings_.resolution);

    int nx = std::max(1, (int)std::round(size.x / h));
    int ny = std::max(1, (int)std::round(size.y / h));
    int nz = std::max(1, (int)std::round(size.z / h));

    grid_.resize(nx, ny, nz, h);
    grid_.markSolidBoundary();

    domainMin_ = domainMin;
    domainMax_ = domainMin + Vec3(nx * h, ny * h, nz * h);

    positions_.clear();
    velocities_.clear();
    particleDt_.clear();
    lastGridDt_ = 0.f;
    stepCounter_ = 0;
}

Vec3 FlipSolver::worldToIndex(const Vec3& p) const {
    float h = grid_.h();
    return {(p.x - domainMin_.x) / h, (p.y - domainMin_.y) / h, (p.z - domainMin_.z) / h};
}

Vec3 FlipSolver::indexToWorld(const Vec3& p) const {
    float h = grid_.h();
    return {domainMin_.x + p.x * h, domainMin_.y + p.y * h, domainMin_.z + p.z * h};
}

void FlipSolver::clampParticleToDomain(Vec3& idx) const {
    // Safety net keeping particles inside the grid. Only a QUARTER cell is
    // needed: the trilinear P2G/G2P stencils sample at most 0.5 cells away
    // (and clamp into range), while the pressure projection and
    // zeroSolidNormalVelocities() enforce the actual wall collision. A
    // larger margin used to leave a visible ~1-voxel gap between resting
    // fluid and every domain wall (resolution-independent, since this is
    // in grid-index units).
    const float margin = 0.25f;
    idx.x = clampf(idx.x, margin, grid_.nx() - margin);
    idx.y = clampf(idx.y, margin, grid_.ny() - margin);
    idx.z = clampf(idx.z, margin, grid_.nz() - margin);
}

size_t FlipSolver::addParticles(const float* positions, const float* velocities, size_t count) {
    size_t room = (positions_.size() >= settings_.maxParticles) ? 0 : settings_.maxParticles - positions_.size();
    size_t toAdd = std::min(count, room);
    positions_.reserve(positions_.size() + toAdd);
    velocities_.reserve(velocities_.size() + toAdd);
    particleDt_.reserve(particleDt_.size() + toAdd);
    for (size_t i = 0; i < toAdd; ++i) {
        positions_.emplace_back(positions[3 * i + 0], positions[3 * i + 1], positions[3 * i + 2]);
        if (velocities) {
            velocities_.emplace_back(velocities[3 * i + 0], velocities[3 * i + 1], velocities[3 * i + 2]);
        } else {
            velocities_.emplace_back(0.f, 0.f, 0.f);
        }
        particleDt_.push_back(0.f); // brand-new particles start exactly on-time (tau=0)
    }
    return toAdd;
}

size_t FlipSolver::addParticlesBox(const Vec3& boxMin, const Vec3& boxMax, int perCell,
                                    const Vec3& initialVelocity, uint32_t seed) {
    perCell = std::max(1, perCell);
    float h = grid_.h();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> jitter(-0.4f, 0.4f);

    std::vector<Vec3> newPos;
    for (float z = boxMin.z + h * 0.5f; z < boxMax.z; z += h / perCell) {
        for (float y = boxMin.y + h * 0.5f; y < boxMax.y; y += h / perCell) {
            for (float x = boxMin.x + h * 0.5f; x < boxMax.x; x += h / perCell) {
                Vec3 jitterVec(jitter(rng) * h / perCell, jitter(rng) * h / perCell, jitter(rng) * h / perCell);
                newPos.push_back(Vec3(x, y, z) + jitterVec);
            }
        }
    }
    std::vector<float> flatPos(newPos.size() * 3);
    std::vector<float> flatVel(newPos.size() * 3);
    for (size_t i = 0; i < newPos.size(); ++i) {
        flatPos[3 * i + 0] = newPos[i].x; flatPos[3 * i + 1] = newPos[i].y; flatPos[3 * i + 2] = newPos[i].z;
        flatVel[3 * i + 0] = initialVelocity.x; flatVel[3 * i + 1] = initialVelocity.y; flatVel[3 * i + 2] = initialVelocity.z;
    }
    return addParticles(flatPos.data(), flatVel.data(), newPos.size());
}

void FlipSolver::clearParticles() {
    positions_.clear();
    velocities_.clear();
    particleDt_.clear();
}

void FlipSolver::setObstacleMask(const uint8_t* mask, size_t count) {
    int nx = grid_.nx(), ny = grid_.ny(), nz = grid_.nz();
    if (count == 0) { obstacleMask_.clear(); return; }
    if (count != size_t(nx) * ny * nz) return; // silently ignore mismatched size
    obstacleMask_.assign(count, 0);
    for (size_t i = 0; i < count; ++i) obstacleMask_[i] = mask[i] ? 1 : 0;
}

void FlipSolver::setObstacleSDF(const float* sdf, size_t count) {
    int nx = grid_.nx(), ny = grid_.ny(), nz = grid_.nz();
    if (count == 0) { hasObstacleSDF_ = false; return; }
    if (count != size_t(nx) * ny * nz) return; // silently ignore mismatched size
    obstacleSDF_.resize(nx, ny, nz, 1e6f);
    std::memcpy(obstacleSDF_.data(), sdf, count * sizeof(float));
    hasObstacleSDF_ = true;
    ++obstacleSdfRevision_; // invalidates the GPU collision path's device copy
}

float FlipSolver::sampleObstacleSDF(const Vec3& idxPos) const {
    if (!hasObstacleSDF_) return 1e6f;
    return sampleCellCenteredField(obstacleSDF_, idxPos.x - 0.5f, idxPos.y - 0.5f, idxPos.z - 0.5f);
}

Vec3 FlipSolver::obstacleSDFGradientIndex(const Vec3& idxPos) const {
    if (!hasObstacleSDF_) return Vec3(0.f, 0.f, 0.f);
    const float eps = 0.5f;
    float dx = sampleObstacleSDF(idxPos + Vec3(eps, 0.f, 0.f)) - sampleObstacleSDF(idxPos - Vec3(eps, 0.f, 0.f));
    float dy = sampleObstacleSDF(idxPos + Vec3(0.f, eps, 0.f)) - sampleObstacleSDF(idxPos - Vec3(0.f, eps, 0.f));
    float dz = sampleObstacleSDF(idxPos + Vec3(0.f, 0.f, eps)) - sampleObstacleSDF(idxPos - Vec3(0.f, 0.f, eps));
    return Vec3(dx, dy, dz); // direction only, caller normalizes
}

// SDF collision response: pushes particles that penetrate (or come within a
// small margin of) solid geometry back out along the SDF gradient, and
// cancels any velocity component pointing further into the solid. This is
// the main practical difference vs. voxel-mask collision, which only blocks
// grid cells and can still let particles tunnel through thin obstacles
// between steps.
void FlipSolver::resolveObstacleCollisions() {
    if (!settings_.collisionUseSDF || !hasObstacleSDF_) return;

#ifdef FLIP_HAS_CUDA
    if (settings_.solverBackend == SolverBackend::CUDA) {
        resolveObstacleCollisionsCUDA(this);
        return;
    }
#endif

    float h = grid_.h();
    float margin = settings_.sdfCollisionMargin * h;
    long long n = (long long)positions_.size();
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (long long i64 = 0; i64 < n; ++i64) {
        size_t i = size_t(i64);
        Vec3 idx = worldToIndex(positions_[i]);
        float phi = sampleObstacleSDF(idx);
        if (phi >= margin) continue;
        Vec3 grad = obstacleSDFGradientIndex(idx);
        float glen = grad.length();
        if (glen < 1e-6f) continue;
        Vec3 nrm = grad * (1.f / glen);
        float pushWorld = margin - phi;
        idx = idx + nrm * (pushWorld / h);
        clampParticleToDomain(idx);
        positions_[i] = indexToWorld(idx);
        float vn = velocities_[i].dot(nrm);
        if (vn < 0.f) velocities_[i] -= nrm * vn;
    }
}

// Classifies cells as SOLID (boundary/obstacle) / FLUID / AIR from the
// phase-field mass accumulator already deposited into grid_.cellWeight
// during P2G, instead of the old "does any particle's instantaneous
// position fall in this cell" test. This is ST-FLIP's key trick for
// avoiding large-CFL aliasing: because the P2G deposit already integrates
// each particle's contribution over its jittered spatiotemporal sample
// (Sec 3.6), the resulting phase field is a smooth, temporally-averaged
// estimate of occupancy rather than a hard instantaneous snapshot.
void FlipSolver::classifyCells() {
    int nx = grid_.nx(), ny = grid_.ny(), nz = grid_.nz();
    bool hasMask = obstacleMask_.size() == size_t(nx) * ny * nz;

    // Reference mass m0 for a "fully sampled" cell: since E[W_T(tau)] = 1
    // exactly for our jitter distribution (verified numerically), the
    // expected total deposited weight for a cell containing the nominal
    // particles-per-cell count is simply that count - no separate Monte
    // Carlo calibration pass needed (paper's Sec 3.6 does this empirically
    // for their trilinear-spread deposit; our simplified single-cell,
    // non-spread accumulator makes it exact and analytic instead).
    float nppc = float(settings_.particlesPerCellPerAxis) *
                 float(settings_.particlesPerCellPerAxis) *
                 float(settings_.particlesPerCellPerAxis);
    float refMass = std::max(settings_.phaseFieldEta * nppc, 1e-6f);

    bool useSDF = settings_.collisionUseSDF && hasObstacleSDF_ &&
                  obstacleSDF_.nx() == nx && obstacleSDF_.ny() == ny && obstacleSDF_.nz() == nz;

    int iMin = nx, iMax = -1, jMin = ny, jMax = -1, kMin = nz, kMax = -1;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                bool boundary = (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1);
                bool obstacle = useSDF
                    ? (obstacleSDF_(i, j, k) <= 0.f)
                    : (hasMask && obstacleMask_[size_t(i) + size_t(nx) * (size_t(j) + size_t(ny) * size_t(k))]);
                if (boundary || obstacle) {
                    grid_.cellType(i, j, k) = CELL_SOLID;
                    continue;
                }
                float mass = grid_.cellWeight(i, j, k);
                float phi = std::min(std::sqrt(std::max(mass, 0.f) / refMass), 1.f);
                if (phi >= 0.5f) {
                    grid_.cellType(i, j, k) = CELL_FLUID;
                    iMin = std::min(iMin, i); iMax = std::max(iMax, i);
                    jMin = std::min(jMin, j); jMax = std::max(jMax, j);
                    kMin = std::min(kMin, k); kMax = std::max(kMax, k);
                } else {
                    grid_.cellType(i, j, k) = CELL_AIR;
                }
            }
        }
    }

    // Air-incompressibility band (two-phase FLIP approximation): promote AIR
    // cells within `airBandCells` of the fluid region to AIR_ACTIVE so they
    // join the pressure solve as a low-density second phase. The active band
    // is pinned at p=0 on its outer face, so far-away air stays cheap.
    if (settings_.airBandCells > 0 && iMax >= iMin) {
        int band = settings_.airBandCells;
        int i0 = std::max(0, iMin - band), i1 = std::min(nx - 1, iMax + band);
        int j0 = std::max(0, jMin - band), j1 = std::min(ny - 1, jMax + band);
        int k0 = std::max(0, kMin - band), k1 = std::min(nz - 1, kMax + band);
        for (int k = k0; k <= k1; ++k)
            for (int j = j0; j <= j1; ++j)
                for (int i = i0; i <= i1; ++i)
                    if (grid_.cellType(i, j, k) == CELL_AIR)
                        grid_.cellType(i, j, k) = CELL_AIR_ACTIVE;
    }

    // Pad the fluid's bounding box so extrapolation (which fills a halo of
    // air cells around the fluid so particle sampling near the surface
    // stays stable) doesn't need to scan the whole domain grid. At large
    // target CFL numbers particles travel further per grid step, and
    // extrapolateField only propagates 1 cell per iteration, so both the
    // margin AND the iteration count must scale with the target CFL for
    // the halo to actually reach far enough - see the effectiveExtrapIters
    // computation in substep().
    if (iMax < iMin) {
        fluidBoundsPadded_ = CellBounds{}; // no fluid cells this step -> empty()
    } else {
        int margin = std::max(4, (int)std::ceil(settings_.cflNumber) + settings_.extrapolateIterations + 2);
        fluidBoundsPadded_.iMin = std::max(0, iMin - margin);
        fluidBoundsPadded_.iMax = std::min(nx - 1, iMax + margin);
        fluidBoundsPadded_.jMin = std::max(0, jMin - margin);
        fluidBoundsPadded_.jMax = std::min(ny - 1, jMax + margin);
        fluidBoundsPadded_.kMin = std::max(0, kMin - margin);
        fluidBoundsPadded_.kMax = std::min(nz - 1, kMax + margin);
    }
}

void FlipSolver::advectParticleLocalSubstepped(size_t i, float dtAct) {
    // Locally sub-stepped advection at CFLlocal ~= 1 (paper Sec 3.3): even
    // though a particle's own active time dtAct can correspond to a very
    // large CFL number (that's the whole point - the grid step itself is
    // large), we still integrate its trajectory in small hops bounded to
    // roughly one cell each. This is cheap (just repeated sampling of the
    // *already-computed, static-for-this-grid-step* velocity field, no new
    // P2G/pressure solve), and is what prevents fast particles from
    // tunnelling through thin obstacles at large target CFL.
    if (dtAct <= 0.f) return;
    float h = grid_.h();
    float remaining = dtAct;
    int guard = 0;
    const int maxInnerSteps = 256;
    while (remaining > 1e-8f && guard < maxInnerSteps) {
        Vec3 idx0 = worldToIndex(positions_[i]);
        Vec3 v0 = grid_.sampleVelocity(idx0);
        float speed = v0.length();
        float subDt = (speed > 1e-6f) ? (h / speed) : remaining;
        subDt = std::min(subDt, remaining);
        // Avoid degenerating into an excessive number of tiny steps for
        // pathological/near-zero velocities close to `remaining`.
        subDt = std::max(subDt, remaining / float(maxInnerSteps - guard));

        Vec3 idxMid = idx0 + (v0 * (0.5f * subDt / h));
        clampParticleToDomain(idxMid);
        Vec3 vMid = grid_.sampleVelocity(idxMid);
        Vec3 idx1 = idx0 + (vMid * (subDt / h));
        // Wall response: clamp the position only. The pressure projection and
        // zeroSolidNormalVelocities() already enforce no-penetration/no-slip
        // at walls; killing the particle's own velocity here instead removes
        // momentum flux before the next P2G and makes splashes die on walls.
        clampParticleToDomain(idx1);
        positions_[i] = indexToWorld(idx1);

        remaining -= subDt;
        guard++;
    }
}

void FlipSolver::substep(float dtGrid) {
    if (positions_.empty()) { lastGridDt_ = dtGrid; return; }

    // Function-scope so the G2P/advection stages below can use them (the
    // stage braces only exist for the perf counters).
    bool stFlip = settings_.stFlipEnabled;
    size_t n = positions_.size();

    { ScopedStage _stageP2G(*this, SG_P2G); // splat + phase-field deposit
    grid_.clearVelocities();
    grid_.clearWeights();
    grid_.cellWeight.fill(0.f);

    float divisor = (lastGridDt_ > 1e-8f) ? lastGridDt_ : dtGrid;

    // --- Spatiotemporal P2G: deposit velocity AND phase-field mass, both
    // weighted by the same temporal kernel evaluated at each particle's
    // jittered sample time (Sec 3.4). With stFlipEnabled=false this
    // collapses to temporalWeight=1 for everyone, i.e. plain instantaneous
    // FLIP P2G - useful for direct comparison / as a safe fallback.
    for (size_t i = 0; i < n; ++i) {
        float tau = stFlip ? clampf(-particleDt_[i] / divisor, -0.5f, 0.5f) : 0.f;
        float wT = stFlip ? temporalKernel(tau) : 1.f;
        Vec3 idx = worldToIndex(positions_[i]);
        grid_.splatParticle(idx, velocities_[i], wT);
        grid_.addCellWeight(idx, wT);
    }
    grid_.normalizeBySplatWeight();
    }

    { ScopedStage _stageCls(*this, SG_CLASSIFY);
    classifyCells();
    }

    int effectiveExtrapIters = std::max(settings_.extrapolateIterations,
                                         (int)std::ceil(settings_.cflNumber) + 2);

    { ScopedStage _stageEx1(*this, SG_EXTRAP);
    grid_.extrapolateAll(effectiveExtrapIters, fluidBoundsPadded_);
    grid_.snapshotVelocities();
    }

    { ScopedStage _stageGrav(*this, SG_GRAVITY);
    grid_.addGravity(dtGrid, settings_.gravity);
    grid_.zeroSolidNormalVelocities();
    }

    int pressureIters = 0;
    bool solved = false;

    // Adaptive CG iteration cap: grow when the solve hits the ceiling, relax
    // toward the minimum when it converges early. (Function scope: the
    // adaptive update after the solve reads maxIters.)
    int maxIters = settings_.pressureIterations;
    if (settings_.adaptivePressureIterations) {
        if (adaptiveMaxIters_ < settings_.pressureMinIterations)
            adaptiveMaxIters_ = settings_.pressureIterations;
        maxIters = adaptiveMaxIters_;
    }

    { ScopedStage _stagePress(*this, SG_PRESSURE); // CG solve incl. transfers

    const float* warm = nullptr;
    if (settings_.pressureWarmStart) {
        size_t pn = size_t(grid_.nx()) * grid_.ny() * grid_.nz();
        if (pressureGuess_.size() == pn) warm = pressureGuess_.data();
    }

#ifdef FLIP_HAS_CUDA
    if (settings_.solverBackend == SolverBackend::CUDA && settings_.airBandCells <= 0) {
        pressureIters = solvePressureCUDA(&grid_, dtGrid, settings_.density,
                                          maxIters, settings_.pressureTolerance,
                                          settings_.pressureWarmStart ? 1 : 0,
                                          settings_.airDensityRatio);
        if (pressureIters >= 0) {
            // The device path also performs the velocity projection and the
            // solid-face zeroing; pressure/velocities never round-trip the
            // host here (pressure stays resident as the next warm start).
            solved = true;
        }
        // pressureIters < 0 means CUDA failed at runtime -> CPU fallback below
    }
#endif
    if (!solved) {
        // The CPU solver also handles the air-incompressibility band (the
        // CUDA CG kernel only knows FLUID/SOLID/AIR cell roles).
        pressureIters = solvePressure(grid_, dtGrid, settings_.density, maxIters,
                                      settings_.pressureTolerance, warm,
                                      settings_.airBandCells > 0 ? settings_.airDensityRatio : 0.f);
    }
    lastPressureIters_ = pressureIters;

    } // SG_PRESSURE

    if (settings_.adaptivePressureIterations) {
        if (pressureIters >= maxIters - 1) {
            adaptiveMaxIters_ = std::min(2 * settings_.pressureIterations, adaptiveMaxIters_ + 8);
        } else {
            adaptiveMaxIters_ = std::max(settings_.pressureMinIterations, adaptiveMaxIters_ - 1);
        }
    }
    if (!solved && settings_.pressureWarmStart) {
        // The CUDA path keeps its warm-start pressure device-resident; this
        // host snapshot is only meaningful for the CPU solver.
        size_t pn = size_t(grid_.nx()) * grid_.ny() * grid_.nz();
        if (pressureGuess_.size() != pn) pressureGuess_.resize(grid_.nx(), grid_.ny(), grid_.nz(), 0.f);
        std::memcpy(pressureGuess_.data(), grid_.pressure.data(), pn * sizeof(float));
    }

    { ScopedStage _stageEx2(*this, SG_EXTRAP);
    grid_.extrapolateAll(effectiveExtrapIters, fluidBoundsPadded_);
    }

    float flip = clampf(settings_.flipRatio, 0.f, 1.f);
    float h = grid_.h();
    long long numParticles = (long long)n;

    { ScopedStage _stageG2P(*this, SG_G2P); // FLIP/PIC velocity blend
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (long long i64 = 0; i64 < numParticles; ++i64) {
        size_t i = size_t(i64);
        Vec3 idx = worldToIndex(positions_[i]);
        Vec3 picVel = grid_.sampleVelocity(idx);
        Vec3 flipVel = velocities_[i] + grid_.sampleVelocityDelta(idx);
        velocities_[i] = flip * flipVel + (1.f - flip) * picVel;
    }
    }

    // --- Temporal jitter, residual carryover, and locally sub-stepped
    // advection (Sec 3.5, Algorithm 1 lines 23-29). This is what lets the
    // NEXT grid step be large without aliasing: each particle ends this
    // step at its own randomized sample time within the slab, not exactly
    // at dtGrid.
    { ScopedStage _stageAdv(*this, SG_ADVECT); // jitter + sub-stepped advection
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
    #endif
    for (long long i64 = 0; i64 < numParticles; ++i64) {
        size_t i = size_t(i64);
        float dtAct;
        if (stFlip) {
            float speed = velocities_[i].length();
            float localCfl = speed * dtGrid / h;
            float gamma = clampf(settings_.jitterStrength, 0.f, 1.f) * smoothstep01(localCfl);
            float xi = hashRandom01(uint32_t(i), stepCounter_) - 0.5f; // ~ Uniform(-0.5, 0.5)
            float jitterAmt = gamma * xi * dtGrid;
            dtAct = clampf(dtGrid + particleDt_[i] + jitterAmt, 0.f, 2.f * dtGrid);
            particleDt_[i] = dtGrid + particleDt_[i] - dtAct;
        } else {
            dtAct = dtGrid;
        }
        advectParticleLocalSubstepped(i, dtAct);
    }
    }

    { ScopedStage _stageCol(*this, SG_COLLIDE);
    resolveObstacleCollisions();
    }

    // Houdini-parity post-passes (each disabled by default -> exact no-ops).
    { ScopedStage _stagePost(*this, SG_POST);
    if (settings_.viscosityStrength > 0.f) applyViscosity(dtGrid);
    if (settings_.surfaceTensionStrength > 0.f) applySurfaceTension(dtGrid);
    if (settings_.vorticityConfinement > 0.f) applyVorticityConfinement(dtGrid);
    if (settings_.reseedEnabled) reseedParticles();
    }

    lastGridDt_ = dtGrid;
    stepCounter_++;
}

void FlipSolver::step(float dt) {
    if (dt <= 0.f) return;

    float maxSpeed = 0.1f;
    for (const Vec3& v : velocities_) maxSpeed = std::max(maxSpeed, v.length());
    float h = grid_.h();

    float remaining = dt;
    int steps = 0;
    while (remaining > 1e-6f && steps < settings_.maxSubsteps) {
        float rawDt = settings_.cflNumber * h / std::max(maxSpeed, 1e-4f);
        rawDt = clampf(rawDt, dt / settings_.maxSubsteps, remaining);
        // Quantize so the remaining frame time divides evenly into an
        // integer number of equal-sized grid steps (paper Algorithm 1,
        // lines 6-7) - avoids tiny leftover "catch-up" steps and keeps
        // step-size changes gradual, which matters for the temporal
        // jitter's clamping behavior (Sec 3.5).
        int nSteps = std::max(1, (int)std::ceil(remaining / rawDt));
        float dtGrid = remaining / float(nSteps);

        substep(dtGrid);
        remaining -= dtGrid;
        steps++;

        maxSpeed = 0.1f;
        for (const Vec3& v : velocities_) maxSpeed = std::max(maxSpeed, v.length());
    }
}

std::vector<float> FlipSolver::positionsFlat() const {
    std::vector<float> out(positions_.size() * 3);
    for (size_t i = 0; i < positions_.size(); ++i) {
        out[3 * i + 0] = positions_[i].x;
        out[3 * i + 1] = positions_[i].y;
        out[3 * i + 2] = positions_[i].z;
    }
    return out;
}

std::vector<float> FlipSolver::renderPositionsFlat() const {
    // "Un-jitters" each particle by advecting it by its own small residual
    // time offset so it lands exactly at the current (nominal) time, using
    // the current velocity field - matching Algorithm 1 lines 31-34's
    // render-time re-synchronization. delta_t_p = t_n - t_p^n (Sec 3.5), so
    // advecting forward by +particleDt_[i] brings the sample time to t_n.
    // A single small step suffices since |particleDt_[i]| is bounded by at
    // most half the largest recent grid step (Appendix A).
    std::vector<float> out(positions_.size() * 3);
    float h = grid_.h();
    long long n = (long long)positions_.size();
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (long long i64 = 0; i64 < n; ++i64) {
        size_t i = size_t(i64);
        Vec3 pos = positions_[i];
        float dt = particleDt_.empty() ? 0.f : particleDt_[i];
        if (settings_.stFlipEnabled && std::fabs(dt) > 1e-8f) {
            Vec3 idx0 = worldToIndex(pos);
            Vec3 v0 = grid_.sampleVelocity(idx0);
            Vec3 idx1 = idx0 + (v0 * (dt / h));
            Vec3 clamped;
            clamped.x = clampf(idx1.x, 0.f, float(grid_.nx()));
            clamped.y = clampf(idx1.y, 0.f, float(grid_.ny()));
            clamped.z = clampf(idx1.z, 0.f, float(grid_.nz()));
            pos = indexToWorld(clamped);
        }
        out[3 * i + 0] = pos.x;
        out[3 * i + 1] = pos.y;
        out[3 * i + 2] = pos.z;
    }
    return out;
}

std::vector<float> FlipSolver::velocitiesFlat() const {
    std::vector<float> out(velocities_.size() * 3);
    for (size_t i = 0; i < velocities_.size(); ++i) {
        out[3 * i + 0] = velocities_[i].x;
        out[3 * i + 1] = velocities_[i].y;
        out[3 * i + 2] = velocities_[i].z;
    }
    return out;
}

// ── Houdini-parity passes ──────────────────────────────────────────────────

void FlipSolver::buildParticleCellIndex(std::vector<int>& cellHead,
                                        std::vector<int>& particleOrder) const {
    int nx = grid_.nx(), ny = grid_.ny(), nz = grid_.nz();
    size_t ncells = size_t(nx) * ny * nz;
    size_t n = positions_.size();
    cellHead.assign(ncells + 1, 0);
    particleOrder.resize(n);

    std::vector<int> cellOf(n);
    for (size_t i = 0; i < n; ++i) {
        Vec3 idx = worldToIndex(positions_[i]);
        int ci = clampf((int)std::floor(idx.x), 0, nx - 1);
        int cj = clampf((int)std::floor(idx.y), 0, ny - 1);
        int ck = clampf((int)std::floor(idx.z), 0, nz - 1);
        cellOf[i] = ci + nx * (cj + ny * ck);
    }
    for (size_t i = 0; i < n; ++i) cellHead[size_t(cellOf[size_t(i)]) + 1]++;
    for (size_t c = 0; c < ncells; ++c) cellHead[c + 1] += cellHead[c];
    std::vector<int> cursor = cellHead; // write pointers
    for (size_t i = 0; i < n; ++i) {
        int cell = cellOf[size_t(i)];
        particleOrder[size_t(cursor[size_t(cell)]++)] = int(i);
    }
}

void FlipSolver::applyViscosity(float dtGrid) {
    // XSPH-style velocity diffusion: blend each particle's velocity toward
    // the average of its neighbours within one cell (27-cell neighborhood).
    std::vector<int> cellHead, order;
    buildParticleCellIndex(cellHead, order);

    int nx = grid_.nx(), ny = grid_.ny(), nz = grid_.nz();
    float h = grid_.h();
    float r2 = (1.5f * h) * (1.5f * h);
    float eps = clampf(settings_.viscosityStrength * dtGrid * 30.f, 0.f, 0.5f);
    if (eps <= 1e-6f) return;

    std::vector<Vec3> newVel(velocities_.size());
    long long n = (long long)velocities_.size();
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 128)
    #endif
    for (long long i64 = 0; i64 < n; ++i64) {
        size_t i = size_t(i64);
        Vec3 pos = positions_[i];
        Vec3 idx = worldToIndex(pos);
        int ci = (int)std::floor(idx.x), cj = (int)std::floor(idx.y), ck = (int)std::floor(idx.z);
        Vec3 avg = velocities_[i];
        int cnt = 1;
        for (int dk = -1; dk <= 1; ++dk) {
            for (int dj = -1; dj <= 1; ++dj) {
                for (int di = -1; di <= 1; ++di) {
                    int ni = ci + di, nj = cj + dj, nk = ck + dk;
                    if (ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 || nk >= nz) continue;
                    int cell = ni + nx * (nj + ny * nk);
                    for (int p = cellHead[size_t(cell)]; p < cellHead[size_t(cell) + 1]; ++p) {
                        size_t j = size_t(order[size_t(p)]);
                        Vec3 d = positions_[j] - pos;
                        if (d.lengthSq() > r2) continue;
                        avg += velocities_[j];
                        cnt++;
                    }
                }
            }
        }
        newVel[i] = velocities_[i] + (avg * (1.f / float(cnt)) - velocities_[i]) * eps;
    }
    velocities_ = std::move(newVel);
}

void FlipSolver::applySurfaceTension(float dtGrid) {
    // Cohesion pass: under-sampled (surface) particles feel a pull toward the
    // centroid of their neighbours, approximating curvature-driven surface
    // tension. Interior particles (full neighbourhood) feel nothing, so the
    // net effect is a rounding/tightening of the free surface.
    std::vector<int> cellHead, order;
    buildParticleCellIndex(cellHead, order);

    int nx = grid_.nx(), ny = grid_.ny(), nz = grid_.nz();
    float h = grid_.h();
    float r2 = h * h;
    // Expected neighbours in a fully sampled ball of radius h.
    float nppc = float(settings_.particlesPerCellPerAxis);
    float fullCount = 4.19f * nppc * nppc * nppc;

    long long n = (long long)velocities_.size();
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 128)
    #endif
    for (long long i64 = 0; i64 < n; ++i64) {
        size_t i = size_t(i64);
        Vec3 pos = positions_[i];
        Vec3 idx = worldToIndex(pos);
        int ci = (int)std::floor(idx.x), cj = (int)std::floor(idx.y), ck = (int)std::floor(idx.z);
        Vec3 dir(0.f, 0.f, 0.f);
        int cnt = 0;
        for (int dk = -1; dk <= 1; ++dk) {
            for (int dj = -1; dj <= 1; ++dj) {
                for (int di = -1; di <= 1; ++di) {
                    int ni = ci + di, nj = cj + dj, nk = ck + dk;
                    if (ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 || nk >= nz) continue;
                    int cell = ni + nx * (nj + ny * nk);
                    for (int p = cellHead[size_t(cell)]; p < cellHead[size_t(cell) + 1]; ++p) {
                        size_t j = size_t(order[size_t(p)]);
                        if (j == i) continue;
                        Vec3 d = positions_[j] - pos;
                        if (d.lengthSq() > r2) continue;
                        dir += d; // vector toward neighbours (away from free space)
                        cnt++;
                    }
                }
            }
        }
        if (cnt == 0) continue;
        // Pull toward the neighbour centroid, scaled by how under-sampled
        // this particle is. Interior particles cancel out (cnt ~= fullCount).
        float deficit = clampf(1.f - float(cnt) / std::max(fullCount, 1.f), 0.f, 1.f);
        float len = dir.length();
        if (len < 1e-8f) continue;
        Vec3 nrm = dir * (1.f / len);
        velocities_[i] += nrm * (settings_.surfaceTensionStrength * deficit * dtGrid);
    }
}

void FlipSolver::applyVorticityConfinement(float dtGrid) {
    // Fedkiw-style vorticity confinement: F = eps * (N x omega) with
    // N = grad|omega| / |grad|omega||. Sample trilinearly per particle.
    int nx = grid_.nx(), ny = grid_.ny(), nz = grid_.nz();
    if (nx < 3 || ny < 3 || nz < 3) return;
    float h = grid_.h();

    std::vector<float> wx(size_t(nx) * ny * nz, 0.f);
    std::vector<float> wy(size_t(nx) * ny * nz, 0.f);
    std::vector<float> wz(size_t(nx) * ny * nz, 0.f);

    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                Vec3 vxm = grid_.cellCenteredVelocity(i - 1, j, k);
                Vec3 vxp = grid_.cellCenteredVelocity(i + 1, j, k);
                Vec3 vym = grid_.cellCenteredVelocity(i, j - 1, k);
                Vec3 vyp = grid_.cellCenteredVelocity(i, j + 1, k);
                Vec3 vzm = grid_.cellCenteredVelocity(i, j, k - 1);
                Vec3 vzp = grid_.cellCenteredVelocity(i, j, k + 1);
                float inv2h = 1.f / (2.f * h);
                size_t c = size_t(i) + size_t(nx) * (size_t(j) + size_t(ny) * size_t(k));
                wx[c] = (vyp.z - vym.z) * inv2h - (vzp.y - vzm.y) * inv2h;
                wy[c] = (vzp.x - vzm.x) * inv2h - (vxp.z - vxm.z) * inv2h;
                wz[c] = (vxp.y - vxm.y) * inv2h - (vyp.x - vym.x) * inv2h;
            }
        }
    }

    // grad|omega| via central differences of the magnitude.
    std::vector<float> mag(size_t(nx) * ny * nz, 0.f);
    for (int k = 1; k < nz - 1; ++k)
        for (int j = 1; j < ny - 1; ++j)
            for (int i = 1; i < nx - 1; ++i) {
                size_t c = size_t(i) + size_t(nx) * (size_t(j) + size_t(ny) * size_t(k));
                mag[c] = std::sqrt(wx[c] * wx[c] + wy[c] * wy[c] + wz[c] * wz[c]);
            }

    auto fieldAt = [&](const std::vector<float>& f, float gx, float gy, float gz) -> float {
        gx = clampf(gx - 0.5f, 0.f, float(nx - 1));
        gy = clampf(gy - 0.5f, 0.f, float(ny - 1));
        gz = clampf(gz - 0.5f, 0.f, float(nz - 1));
        int i0 = (int)std::floor(gx), i1 = std::min(i0 + 1, nx - 1);
        int j0 = (int)std::floor(gy), j1 = std::min(j0 + 1, ny - 1);
        int k0 = (int)std::floor(gz), k1 = std::min(k0 + 1, nz - 1);
        float fx = gx - i0, fy = gy - j0, fz = gz - k0;
        float c00 = f[size_t(i0) + size_t(nx) * (size_t(j0) + size_t(ny) * size_t(k0))] * (1 - fx)
                  + f[size_t(i1) + size_t(nx) * (size_t(j0) + size_t(ny) * size_t(k0))] * fx;
        float c10 = f[size_t(i0) + size_t(nx) * (size_t(j1) + size_t(ny) * size_t(k0))] * (1 - fx)
                  + f[size_t(i1) + size_t(nx) * (size_t(j1) + size_t(ny) * size_t(k0))] * fx;
        float c01 = f[size_t(i0) + size_t(nx) * (size_t(j0) + size_t(ny) * size_t(k1))] * (1 - fx)
                  + f[size_t(i1) + size_t(nx) * (size_t(j0) + size_t(ny) * size_t(k1))] * fx;
        float c11 = f[size_t(i0) + size_t(nx) * (size_t(j1) + size_t(ny) * size_t(k1))] * (1 - fx)
                  + f[size_t(i1) + size_t(nx) * (size_t(j1) + size_t(ny) * size_t(k1))] * fx;
        float c0 = c00 * (1 - fy) + c10 * fy;
        float c1 = c01 * (1 - fy) + c11 * fy;
        return c0 * (1 - fz) + c1 * fz;
    };

    float eps = settings_.vorticityConfinement;
    long long n = (long long)velocities_.size();
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (long long i64 = 0; i64 < n; ++i64) {
        size_t i = size_t(i64);
        Vec3 idx = worldToIndex(positions_[i]);
        float ox = fieldAt(wx, idx.x, idx.y, idx.z);
        float oy = fieldAt(wy, idx.x, idx.y, idx.z);
        float oz = fieldAt(wz, idx.x, idx.y, idx.z);

        float gx = (fieldAt(mag, idx.x + 0.5f, idx.y, idx.z) - fieldAt(mag, idx.x - 0.5f, idx.y, idx.z)) / h;
        float gy = (fieldAt(mag, idx.x, idx.y + 0.5f, idx.z) - fieldAt(mag, idx.x, idx.y - 0.5f, idx.z)) / h;
        float gz = (fieldAt(mag, idx.x, idx.y, idx.z + 0.5f) - fieldAt(mag, idx.x, idx.y, idx.z - 0.5f)) / h;
        float glen = std::sqrt(gx * gx + gy * gy + gz * gz) + 1e-8f;
        float nxv = gx / glen, nyv = gy / glen, nzv = gz / glen;

        // N x omega
        Vec3 force(nyv * oz - nzv * oy, nzv * ox - nxv * oz, nxv * oy - nyv * ox);
        velocities_[i] += force * (eps * h * dtGrid);
    }
}

void FlipSolver::reseedParticles() {
    // Keep per-cell particle density inside [minRatio, maxRatio] x nominal.
    int nx = grid_.nx(), ny = grid_.ny(), nz = grid_.nz();
    float h = grid_.h();
    int ppc = std::max(1, settings_.particlesPerCellPerAxis);
    int target = ppc * ppc * ppc;
    int minCount = std::max(0, int(std::floor(settings_.reseedMinRatio * target)));
    int maxCount = std::max(target, int(std::ceil(settings_.reseedMaxRatio * target)));

    std::vector<int> cellHead, order;
    buildParticleCellIndex(cellHead, order);

    std::vector<int> counts(size_t(nx) * ny * nz, 0);
    std::vector<int> budget(size_t(nx) * ny * nz, -1); // -1 = untouched
    for (size_t oi = 0; oi < order.size(); ++oi) {
        Vec3 idx = worldToIndex(positions_[size_t(order[oi])]);
        int ci = clampf((int)std::floor(idx.x), 0, nx - 1);
        int cj = clampf((int)std::floor(idx.y), 0, ny - 1);
        int ck = clampf((int)std::floor(idx.z), 0, nz - 1);
        counts[size_t(ci) + size_t(nx) * (size_t(cj) + size_t(ny) * size_t(ck))]++;
    }

    // Gather under-filled FLUID cells for seeding.
    struct Seed { Vec3 pos; Vec3 vel; };
    std::vector<Seed> seeds;
    uint32_t rngSeed = stepCounter_ * 2654435761u + 12345u;
    auto jitter01 = [&](uint32_t s) -> float {
        uint32_t x = hashU32(s ^ rngSeed);
        return float(x & 0x00FFFFFFu) / float(0x01000000u);
    };

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (grid_.cellType(i, j, k) != CELL_FLUID) continue;
                size_t cell = size_t(i) + size_t(nx) * (size_t(j) + size_t(ny) * size_t(k));
                int cnt = counts[cell];
                if (cnt >= maxCount) budget[cell] = target; // over-sampled: keep `target`
                else if (cnt < minCount) {
                    budget[cell] = cnt;
                    int toAdd = target - cnt;
                    for (int a = 0; a < toAdd; ++a) {
                        Vec3 center(float(i) + 0.5f + (jitter01(uint32_t(cell * 3u + a)) - 0.5f) * 0.8f,
                                    float(j) + 0.5f + (jitter01(uint32_t(cell * 5u + a)) - 0.5f) * 0.8f,
                                    float(k) + 0.5f + (jitter01(uint32_t(cell * 7u + a)) - 0.5f) * 0.8f);
                        seeds.push_back({indexToWorld(center), grid_.sampleVelocity(center)});
                    }
                }
            }
        }
    }

    // Remove excess particles in over-sampled cells (swap-erase from the end).
    size_t n = positions_.size();
    std::vector<int> keepFlag(n, 1);
    size_t kept = n;
    for (size_t oi = 0; oi < order.size(); ++oi) {
        int pi = order[oi];
        Vec3 idx = worldToIndex(positions_[size_t(pi)]);
        int ci = clampf((int)std::floor(idx.x), 0, nx - 1);
        int cj = clampf((int)std::floor(idx.y), 0, ny - 1);
        int ck = clampf((int)std::floor(idx.z), 0, nz - 1);
        size_t cell = size_t(ci) + size_t(nx) * (size_t(cj) + size_t(ny) * size_t(ck));
        if (budget[cell] < 0) continue; // not over-sampled
        if (budget[cell] > 0) { budget[cell]--; continue; }
        keepFlag[size_t(pi)] = 0;
        kept--;
    }

    if (kept < n) {
        size_t write = 0;
        for (size_t i = 0; i < n; ++i) {
            if (!keepFlag[i]) continue;
            if (write != i) {
                positions_[write] = positions_[i];
                velocities_[write] = velocities_[i];
                particleDt_[write] = particleDt_[i];
            }
            write++;
        }
        positions_.resize(kept);
        velocities_.resize(kept);
        particleDt_.resize(kept);
    }

    for (const Seed& s : seeds) {
        if (positions_.size() >= settings_.maxParticles) break;
        positions_.push_back(s.pos);
        velocities_.push_back(s.vel);
        particleDt_.push_back(0.f);
    }
}

std::vector<float> FlipSolver::cellVelocityField() const {
    int nx = grid_.nx(), ny = grid_.ny(), nz = grid_.nz();
    std::vector<float> out(size_t(nx) * ny * nz * 3, 0.f);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                Vec3 v = grid_.cellCenteredVelocity(i, j, k);
                size_t c = size_t(i) + size_t(nx) * (size_t(j) + size_t(ny) * size_t(k));
                out[3 * c + 0] = v.x; out[3 * c + 1] = v.y; out[3 * c + 2] = v.z;
            }
        }
    }
    return out;
}

std::vector<float> FlipSolver::vorticityField() const {
    int nx = grid_.nx(), ny = grid_.ny(), nz = grid_.nz();
    std::vector<float> out(size_t(nx) * ny * nz * 3, 0.f);
    if (nx < 3 || ny < 3 || nz < 3) return out;
    float inv2h = 1.f / (2.f * grid_.h());
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                Vec3 vxm = grid_.cellCenteredVelocity(i - 1, j, k);
                Vec3 vxp = grid_.cellCenteredVelocity(i + 1, j, k);
                Vec3 vym = grid_.cellCenteredVelocity(i, j - 1, k);
                Vec3 vyp = grid_.cellCenteredVelocity(i, j + 1, k);
                Vec3 vzm = grid_.cellCenteredVelocity(i, j, k - 1);
                Vec3 vzp = grid_.cellCenteredVelocity(i, j, k + 1);
                float ox = (vyp.z - vym.z) * inv2h - (vzp.y - vzm.y) * inv2h;
                float oy = (vzp.x - vzm.x) * inv2h - (vxp.z - vxm.z) * inv2h;
                float oz = (vxp.y - vxm.y) * inv2h - (vyp.x - vym.x) * inv2h;
                size_t c = size_t(i) + size_t(nx) * (size_t(j) + size_t(ny) * size_t(k));
                out[3 * c + 0] = ox; out[3 * c + 1] = oy; out[3 * c + 2] = oz;
            }
        }
    }
    return out;
}

} // namespace flipcore
