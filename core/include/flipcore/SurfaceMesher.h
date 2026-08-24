#pragma once
#include <vector>
#include <cstdint>

namespace flipcore {

/// Converts raw FLIP particle positions into a triangle mesh using OpenVDB's
/// ParticlesToLevelSet + volumeToMesh pipeline.  Much faster than Python-side
/// SPH reconstruction (pysplashsurf) and produces higher-quality surfaces.
///
/// Returns true on success.  outVertices is (N,3) float32 interleaved,
/// outTriangles is (M,3) uint32 interleaved.
bool particlesToMesh(const float* positions, size_t numParticles,
                     float voxelSize, float halfWidth, float iso, float adaptivity,
                     bool preserveBubbles,
                     std::vector<float>& outVertices,
                     std::vector<uint32_t>& outTriangles);

/// Same as particlesToMesh, but CSG-subtracts a collider triangle mesh from
/// the water level set first (water ∧ ¬solid). The water surface is cut at
/// the solid boundary and the solid's surface appears inside the water
/// region. Obstacle inputs are (N,3) float32 verts and (M,3) uint32 tris.
bool particlesToMeshWithObstacles(
    const float* positions, size_t numParticles,
    float voxelSize, float halfWidth, float iso,
    const float* obstacleVerts, size_t numObstacleVerts,
    const uint32_t* obstacleTris, size_t numObstacleTris,
    float adaptivity, bool preserveBubbles,
    std::vector<float>& outVertices,
    std::vector<uint32_t>& outTriangles);

} // namespace flipcore
