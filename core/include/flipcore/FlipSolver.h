#pragma once
#include "MacGrid.h"
#include "Vec3.h"
#include <vector>
#include <cstdint>

namespace flipcore {

enum class SolverBackend : int { CPU = 0, CUDA = 1 };

struct SolverSettings {
    int resolution = 48;          // cells along the domain's longest axis
    float flipRatio = 0.95f;      // 0 = pure PIC (stable/viscous), 1 = pure FLIP (energetic/noisy)
    float density = 1000.0f;      // kg/m^3 (water)
    Vec3 gravity{0.f, 0.f, -9.81f};

    // Target CFL number for the physics grid step (P2G / pressure solve /
    // G2P), following Braun, Winchenbach, Bender & Thuerey, "Spatiotemporal
    // FLIP for Fast Free-Surface and Two-Phase Simulation With Very Large
    // Time Steps" (ACM TOG 45(4), SIGGRAPH 2026). Unlike plain FLIP, this can
    // be pushed well beyond 1-3 (the paper reports good results up to
    // ~10-16, and usable results beyond that) because particles are treated
    // as 4D space-time samples (see stFlipEnabled) rather than instantaneous
    // 3D ones, which is what normally makes large steps alias/blow up.
    float cflNumber = 8.0f;
    int maxSubsteps = 48;         // safety cap per frame
    int pressureIterations = 150;
    float pressureTolerance = 1e-4f;
    int extrapolateIterations = 6; // minimum; actual count auto-scales with cflNumber

    // --- ST-FLIP spatiotemporal sampling ---
    bool stFlipEnabled = true;
    float jitterStrength = 1.0f;  // gamma in the paper; 0 disables temporal jitter (falls back to instantaneous P2G)
    float phaseFieldEta = 0.5f;   // eta_phi: phase-transition steepness for fluid/air classification
    int particlesPerCellPerAxis = 2; // must match actual emission density for correct phase-field calibration

    // Collision A/B toggle: false = legacy per-cell binary voxel mask
    // (setObstacleMask); true = signed distance field (setObstacleSDF), used
    // both for cell classification and a per-particle penetration push-out.
    bool collisionUseSDF = false;

    // SDF collision margin as fraction of cell size (0.001-0.5).
    float sdfCollisionMargin = 0.01f;

    // --- Houdini FLIP Solver parity extensions ---

    // Reseeding: keep particle density inside [min, max] fractions of the
    // nominal particles-per-cell count in FLUID cells.
    bool reseedEnabled = false;
    float reseedMinRatio = 0.5f;   // under-sampled cells get topped up
    float reseedMaxRatio = 2.5f;   // over-sampled cells have excess removed

    // Viscosity (XSPH velocity diffusion) and surface tension (cohesion of
    // under-sampled surface particles toward the local density centroid).
    float viscosityStrength = 0.0f;      // 0..1 blend per unit time
    float surfaceTensionStrength = 0.0f; // acceleration, world units/s^2

    // Vorticity confinement (Fedkiw-style epsilon; 0 disables).
    float vorticityConfinement = 0.0f;

    // Pressure solve: reuse the previous step's pressure as the CG initial
    // guess, and adaptively grow/shrink the iteration cap.
    bool pressureWarmStart = false;
    bool adaptivePressureIterations = false;
    int pressureMinIterations = 10;

    // Air incompressibility: include a band of air cells around the liquid
    // in the pressure solve as a low-density second phase (two-phase FLIP
    // approximation). airBandCells = 0 disables.
    int airBandCells = 0;
    float airDensityRatio = 0.01f; // rho_air / rho_water

    SolverBackend solverBackend = SolverBackend::CPU;

    uint64_t maxParticles = 4000000ULL;
};

class FlipSolver {
public:
    // Defines the simulation domain as an axis-aligned box in world units and
    // builds the underlying MAC grid so that `resolution` cells span the
    // longest domain axis.
    void initDomain(const Vec3& domainMin, const Vec3& domainMax, const SolverSettings& settings);

    // Bulk-add particles (e.g. sampled from an emitter mesh on the Python side).
    // positions/velocities are flat arrays of length 3*count, in world units.
    // Returns the number of particles actually added (capped by maxParticles).
    size_t addParticles(const float* positions, const float* velocities, size_t count);

    // Convenience: jittered regular seeding inside a world-space AABB, useful
    // for simple box/sphere-ish emitters without mesh sampling.
    size_t addParticlesBox(const Vec3& boxMin, const Vec3& boxMax, int particlesPerCellPerAxis,
                            const Vec3& initialVelocity, uint32_t seed);

    void clearParticles();

    // Marks static solid (obstacle) cells from a dense nx*ny*nz mask (1 = solid,
    // 0 = open), indexed as i + nx*(j + ny*k) matching the domain's grid
    // resolution. Typically built once from an obstacle mesh via voxelization
    // on the Python side before baking. Pass count=0 to clear.
    void setObstacleMask(const uint8_t* mask, size_t count);

    // Same indexing as setObstacleMask, but a float signed distance (world
    // units, negative = inside solid) instead of a binary flag. Pass count=0
    // to clear. Only used when settings().collisionUseSDF is true.
    void setObstacleSDF(const float* sdf, size_t count);

    // Advances the simulation by one frame worth of time `dt` (seconds),
    // internally subdividing into adaptive, CFL-target-limited grid steps.
    void step(float dt);

    size_t particleCount() const { return positions_.size(); }
    const std::vector<Vec3>& positions() const { return positions_; }
    const std::vector<Vec3>& velocities() const { return velocities_; }

    // Flat float getters, convenient for pybind11 <-> numpy.
    std::vector<float> positionsFlat() const;
    std::vector<float> velocitiesFlat() const;

    // Cell-centered velocity field (from the staggered MAC faces), flat
    // (ncells,3) floats in i + nx*(j + ny*k) order - used by the Whitewater
    // solver as its source velocity volume.
    std::vector<float> cellVelocityField() const;

    // Curl of the cell-centered velocity field, same layout. The Whitewater
    // solver derives its vorticity/churn emission from this.
    std::vector<float> vorticityField() const;

    // Like positionsFlat(), but "un-jitters" each particle by its own small
    // residual time offset so positions are synchronized to the exact
    // current time - matching Algorithm 1's render-time re-synchronization
    // in the ST-FLIP paper. Use this (not positionsFlat()) for anything
    // that will be cached/meshed/rendered, to avoid motion-blur-like
    // artifacts from the spatiotemporal jitter.
    std::vector<float> renderPositionsFlat() const;

    const MacGrid& grid() const { return grid_; }
    MacGrid& grid() { return grid_; }

    Vec3 domainMin() const { return domainMin_; }
    Vec3 domainMax() const { return domainMax_; }
    float cellSize() const { return grid_.h(); }
    const SolverSettings& settings() const { return settings_; }
    void setSettings(const SolverSettings& s) { settings_ = s; }

    // Mutable access for in-place GPU operations
    std::vector<Vec3>& positions() { return positions_; }
    std::vector<Vec3>& velocities() { return velocities_; }

    // SDF obstacle data access for GPU collision
    bool hasObstacleSDF() const { return hasObstacleSDF_; }
    const Array3<float>& obstacleSDF() const { return obstacleSDF_; }

    // GPU collision dispatch (called from CUDA code)
    void resolveObstacleCollisions();

private:
    void substep(float dtGrid);
    void classifyCells();
    void advectParticleLocalSubstepped(size_t i, float dtAct);
    Vec3 worldToIndex(const Vec3& worldPos) const;
    Vec3 indexToWorld(const Vec3& idxPos) const;
    void clampParticleToDomain(Vec3& idxPos) const;

    // Houdini-parity per-substep passes (each is a no-op when disabled).
    void buildParticleCellIndex(std::vector<int>& cellHead, std::vector<int>& particleOrder) const;
    void applyViscosity(float dtGrid);
    void applySurfaceTension(float dtGrid);
    void applyVorticityConfinement(float dtGrid);
    void reseedParticles();

    // SDF collision mode: samples the obstacle SDF/gradient at a grid-index
    // position, and pushes penetrating particles back out along the normal.
    float sampleObstacleSDF(const Vec3& idxPos) const;
    Vec3 obstacleSDFGradientIndex(const Vec3& idxPos) const;

    MacGrid grid_;
    SolverSettings settings_;
    Vec3 domainMin_{0.f, 0.f, 0.f};
    Vec3 domainMax_{1.f, 1.f, 1.f};

    std::vector<Vec3> positions_;    // world-space
    std::vector<Vec3> velocities_;   // world-space
    std::vector<float> particleDt_;  // ST-FLIP: per-particle residual time offset (seconds)
    std::vector<signed char> obstacleMask_; // nx*ny*nz, 1 = static solid obstacle
    Array3<float> obstacleSDF_;      // nx*ny*nz cell-centered SDF, world-units, negative = inside solid
    bool hasObstacleSDF_ = false;
    CellBounds fluidBoundsPadded_;   // narrow band around current fluid cells, for extrapolation

    float lastGridDt_ = 0.f;  // previous grid step's Delta t, for temporal-kernel normalization
    uint32_t stepCounter_ = 0; // drives the deterministic per-particle jitter RNG

    Array3<float> pressureGuess_;  // warm-start pressure snapshot (grid-sized)
    int adaptiveMaxIters_ = 0;     // current adaptive CG iteration cap (0 = uninitialized)
};

} // namespace flipcore
