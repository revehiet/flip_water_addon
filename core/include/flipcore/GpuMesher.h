#pragma once
/// @file GpuMesher.h
/// @brief Basic CUDA marching-cubes surface mesher for particle sets.
///
/// Splats particles into a dense scalar field (quadratic B-spline kernel)
/// and extracts an isosurface with classic marching cubes, welding shared
/// grid-edge vertices through a hash table. Output is (N,3) float vertices
/// and (M,3) uint32 triangle indices.

#include <vector>
#include <cstdint>
#include <cstddef>

namespace flipcore {

/// Mesh a particle set on the GPU.
/// @param positions   flat float array of (N,3) particle positions (metres)
/// @param numParticles
/// @param voxelSize   grid cell size (metres)
/// @param iso         isosurface threshold of the splatted density field
///                    (lower = surface further outside the particles)
/// @param outVertices (N,3) float vertices
/// @param outTriangles (M,3) uint32 indices
/// @return true on success
bool particlesToMeshGpu(const float* positions, size_t numParticles,
                        float voxelSize, float iso,
                        std::vector<float>& outVertices,
                        std::vector<uint32_t>& outTriangles);

} // namespace flipcore
