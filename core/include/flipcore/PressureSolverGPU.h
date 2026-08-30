#pragma once

// Exported as C symbol (no C++ name mangling) so the MinGW/GCC linker can
// resolve it from the nvcc+MSVC-compiled object file.
#ifdef __cplusplus
extern "C" {
#endif

// Pointer to avoid C++ references in extern "C" interface.
// Caller passes &grid from C++ code; CUDA implementation dereferences.
// `useWarmStart` != 0 seeds CG from the device-resident previous pressure
// (managed internally by the CUDA implementation); `airDensityRatio` scales
// CELL_AIR_ACTIVE gradient terms (pass 0 for the free-surface-only path).
int solvePressureCUDA(void* grid, float dt, float rho,
                      int maxIterations, float tolerance,
                      int useWarmStart, float airDensityRatio);

// GPU SDF collision pass.  Pushes penetrating particles out along the SDF
// gradient and zeroes inward velocity.  Returns number of pushed particles,
// or -1 on error.
int resolveObstacleCollisionsCUDA(void* solver);

// GPU velocity extrapolation (parallel dilation port of
// MacGrid::extrapolateAll).  i0..k1 is the CELL box (fluidBoundsPadded_).
// Returns 0 on success (host u/v/w updated in place), -1 on error (host
// untouched; caller falls back to the CPU path).
int extrapolateAllCUDA(void* grid, int iterations,
                       int i0, int i1, int j0, int j1, int k0, int k1);

#ifdef __cplusplus
}
#endif
