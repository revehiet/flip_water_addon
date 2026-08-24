#pragma once
#include "MacGrid.h"

namespace flipcore {

// Solves the variable-coefficient Poisson pressure equation for the grid's
// FLUID cells (AIR cells are Dirichlet p=0, SOLID cells are zero-flux/Neumann)
// using an unpreconditioned Conjugate Gradient method, then projects the
// velocity field to be (approximately) divergence-free.
//
// `warmStart` (optional) is a grid-sized pressure field used as the CG
// initial guess (previous step's solution). `airDensityRatio` > 0 promotes
// CELL_AIR_ACTIVE cells to unknowns with density rho*ratio (two-phase FLIP
// approximation; 0 treats them as plain p=0 air).
// Returns the number of CG iterations actually performed.
int solvePressure(MacGrid& grid, float dt, float rho, int maxIterations = 150,
                  float tolerance = 1e-4f, const float* warmStart = nullptr,
                  float airDensityRatio = 0.f);

// Applies the solved pressure field back onto the staggered velocities:
// u -= (dt/rho) * grad(p), with solid-wall faces left untouched, and finally
// re-zeroes any normal component adjacent to solid cells. Shared by the CPU
// and CUDA solve paths (the CUDA kernel only solves for pressure; this helper
// runs on the host afterwards). When airDensityRatio > 0, CELL_AIR_ACTIVE
// cells use rho*ratio for their gradient scaling.
void projectPressureVelocities(MacGrid& grid, float dt, float rho,
                               float airDensityRatio = 0.f);

} // namespace flipcore
