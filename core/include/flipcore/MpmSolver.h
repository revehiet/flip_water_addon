#pragma once
/// @file MpmSolver.h
/// @brief PIC-FLIP Material Point Method solver with presets for sand, snow,
///        jello, water, and honey.
///
/// Based on the fixed-corotated hyperelastic model with von Mises plasticity
/// (Stomakhin et al. 2013 / Hu et al. 2018 MLS-MPM) and the PIC-FLIP blending
/// scheme from H-YWu/mpm.

#include <vector>
#include <cstdint>

namespace flipcore {

// ── Material parameters ────────────────────────────────────────────────────

struct MpmMaterial {
    float youngsModulus;        // E  (Pa)  – stiffness
    float poissonRatio;         // ν         – volume preservation (0.49 ≈ water)
    float hardening;            // H         – plasticity hardening (0 = elastic)
    float criticalCompression;  // λ_c⁻      – plastic yield (compression)
    float criticalStretch;      // λ_c⁺      – plastic yield (stretch)
    float dynamicViscosity;     // μ_dyn     – shear velocity damping (honey >> water)
    float bulkViscosity;        // μ_bulk    – volumetric (dilation) damping
    float sandAlpha;            // α         – 0 = elastic, 1 = full sand model
                                //             (blends stress toward plastic-clamped)
    float density;              // ρ  (kg/m³)
};

// ── Named presets (matching paper parameters) ──────────────────────────────

enum class MpmPreset : int {
    Sand  = 0,
    Snow  = 1,
    Jello = 2,
    Water = 3,
    Honey = 4,
};

inline MpmMaterial presetMaterial(MpmPreset p) {
    switch (p) {
    case MpmPreset::Sand:
        // Loose granular: stiff with plastic failure, full sand model
        return {3.5e5f, 0.30f, 10.0f, 0.005f, 0.010f, 0.0f, 0.0f, 1.0f, 1600.0f};
    case MpmPreset::Snow:
        // Compressible, plastically deforms under pressure; half sand-like
        return {1.4e5f, 0.20f, 10.0f, 0.015f, 0.010f, 0.0f, 0.0f, 0.5f, 400.0f};
    case MpmPreset::Jello:
        // Elastic, nearly incompressible
        return {3.0e4f, 0.45f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 1000.0f};
    case MpmPreset::Water:
        // Inviscid, nearly incompressible — no plasticity
        return {1.0e5f, 0.49f, 0.0f, 0.0f, 0.0f, 0.01f, 0.0f, 0.0f, 1000.0f};
    case MpmPreset::Honey:
        // Highly viscous, slow flow
        return {5.0e4f, 0.45f, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 1400.0f};
    }
    return {};
}

// ── Solver settings ────────────────────────────────────────────────────────

struct MpmSettings {
    // Domain box — the simulation is unbounded (sparse hash grid), but
    // grid nodes outside origin..origin+res*stride are treated as walls.
    float gridOriginX = 0.0f, gridOriginY = 0.0f, gridOriginZ = 0.0f;
    int   gridResX = 32, gridResY = 32, gridResZ = 32;
    float gridStride      = 0.05f;     // cell size in metres

    // Time
    float deltaTime       = 0.0002f;   // seconds per sub-step
    int   substepsPerFrame = 25;        // sub-steps × dt = frame dt

    // PIC-FLIP blending  (0 = pure PIC, 1 = pure FLIP)
    float flipRatio       = 0.95f;

    // Gravity
    float gravityX = 0.0f, gravityY = 0.0f, gravityZ = -9.81f;

    // Boundary friction
    float boundaryFriction = 0.0f;

    // Material
    MpmMaterial material  = presetMaterial(MpmPreset::Sand);
};

// ── GPU-side particle (POD, no Eigen – compatible with plain CUDA) ─────────

struct MpmParticleGPU {
    float position[3];
    float velocity[3];
    // APIC affine velocity matrix B (column-major, 9 floats) — stored
    // normalized as C = B·D⁻¹ so P2G can scatter m·C·(x_g − x_p).
    float B[9];
    // Deformation gradient F (column-major, 9 floats)
    float F[9];
    float mass;
    float volume0;
    // Lame parameters
    float mu0, lambda0;
    float hardening;
    float critCompression, critStretch;
    // Constitutive extras
    float sandAlpha;          // 0 = elastic, 1 = full sand model
    float dynamicViscosity;   // shear damping
    float bulkViscosity;      // volumetric (dilation) damping
    // Previous step's PIC velocity (FLIP blend reference)
    float prevPicVelocity[3];
};

// ── GPU-side grid node (sparse hash grid) ──────────────────────────────────

struct MpmGridNodeGPU {
    float mass;
    float velocity[3];      // momentum / mass  (PIC velocity)
    float momentum[3];      // raw momentum accumulator
    float force[3];         // elastic force accumulator
};

// ── Main solver ────────────────────────────────────────────────────────────

class MpmSolver {
public:
    MpmSolver();
    ~MpmSolver();

    /// Initialise or re-initialise with a set of particle positions and settings.
    /// @param positions  flat float array of (N,3) particle positions in metres
    /// @param numParticles
    /// @param settings   grid, time, material, etc.
    void init(const float* positions, size_t numParticles,
              const MpmSettings& settings);

    /// Advance the simulation by one sub-step (call substepsPerFrame times per frame).
    void step();

    /// Copy current particle positions back to host (caller provides (N,3) buffer).
    void getPositions(float* outPositions, size_t maxCount) const;

    /// Number of particles.
    size_t particleCount() const { return _numParticles; }

    /// Change boundary box dynamically (e.g. moving domain).
    void setBoundary(const float origin[3], const float target[3]);

private:
    // Device buffers
    float*        _d_particles  = nullptr;   // array of MpmParticleGPU
    float*        _d_grid       = nullptr;   // dense array of MpmGridNodeGPU (sparse-indexed)
    uint32_t*     _d_hashTable  = nullptr;   // cell -> (node index + 1), 0 = empty
    int32_t*      _d_cellCoords = nullptr;   // (i,j,k) cell of each active grid node
    uint32_t*     _d_gridCount  = nullptr;   // atomic counter of active grid nodes
    MpmSettings   _settings;
    size_t        _numParticles = 0;
    size_t        _maxNodes     = 0;         // upper bound: 27 × particles
    size_t        _tableSize    = 0;         // pow2, ~2× maxNodes
    bool          _initialised  = false;

    void _allocGrid();
    void _freeAll();
};

} // namespace flipcore
