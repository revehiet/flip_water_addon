#pragma once

// Exported as C symbol (no C++ name mangling) so the MinGW/GCC linker can
// resolve it from the nvcc+MSVC-compiled object file.
#ifdef __cplusplus
extern "C" {
#endif

// Pointer to avoid C++ references in extern "C" interface.
// Caller passes &grid from C++ code; CUDA implementation dereferences.
// `x0Host` (optional) is a grid-sized host pressure array used as the CG
// initial guess (pressure warm start); pass nullptr to start from zero.
int solvePressureCUDA(void* grid, float dt, float rho,
                      int maxIterations, float tolerance,
                      const float* x0Host);

// GPU SDF collision pass.  Pushes penetrating particles out along the SDF
// gradient and zeroes inward velocity.  Returns number of pushed particles,
// or -1 on error.
int resolveObstacleCollisionsCUDA(void* solver);

#ifdef __cplusplus
}
#endif
