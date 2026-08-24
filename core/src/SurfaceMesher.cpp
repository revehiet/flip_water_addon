#include "flipcore/SurfaceMesher.h"

#include <openvdb/openvdb.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/LevelSetUtil.h>  // signedFloodFill (bubble extraction)
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/math/Transform.h>

namespace flipcore {

// Lightweight particle list adaptor required by OpenVDB's particlesToSdf
struct ParticleList {
    using PosType = openvdb::Vec3R;

    const float* pos;
    size_t       count;
    float        radius; // world-space radius

    size_t size() const { return count; }
    void getPos(size_t n, openvdb::Vec3R& xyz) const {
        xyz[0] = double(pos[3 * n + 0]);
        xyz[1] = double(pos[3 * n + 1]);
        xyz[2] = double(pos[3 * n + 2]);
    }
    void getPosRad(size_t n, openvdb::Vec3R& xyz, openvdb::Real& r) const {
        xyz[0] = double(pos[3 * n + 0]);
        xyz[1] = double(pos[3 * n + 1]);
        xyz[2] = double(pos[3 * n + 2]);
        r = double(radius);
    }
};

// ── Shared helpers ─────────────────────────────────────────────────────────

namespace {

// Build the water narrow-band level set from particles (world-space transform).
openvdb::FloatGrid::Ptr buildWaterGrid(const float* positions, size_t numParticles,
                                       float voxelSize, float halfWidth)
{
    float particleRadius = voxelSize * 1.5f;
    ParticleList particles{positions, numParticles, particleRadius};
    auto grid = openvdb::FloatGrid::create(halfWidth * voxelSize);
    // IMPORTANT: the grid must carry a world-space transform matching the
    // voxel size, otherwise the particle radius (world units) is interpreted
    // in voxel units and everything falls below rasterizeSpheres' Rmin cutoff.
    grid->setTransform(openvdb::math::Transform::createLinearTransform(voxelSize));
    grid->setGridClass(openvdb::GRID_LEVEL_SET);
    openvdb::tools::particlesToSdf(particles, *grid, double(particleRadius));
    if (grid->empty()) return nullptr;
    return grid;
}

// Extract a triangle mesh from a level set (world space) at the given iso value.
// `adaptivity` (0..1) is the polygonization tolerance: 0 keeps the mesh at full
// voxel resolution, higher values decimate more aggressively (fewer, larger
// polygons) - mirrors Houdini's Particle Fluid Surface Adaptivity parameter.
bool extractMesh(openvdb::FloatGrid& grid, float iso, float adaptivity,
                 std::vector<float>& outVertices,
                 std::vector<uint32_t>& outTriangles)
{
    std::vector<openvdb::Vec3s> meshPoints;
    std::vector<openvdb::Vec3I> meshTris;
    std::vector<openvdb::Vec4I> meshQuads;
    openvdb::tools::volumeToMesh(grid, meshPoints, meshTris, meshQuads,
                                 double(iso), double(adaptivity));

    outVertices.clear();
    outVertices.reserve(meshPoints.size() * 3);
    for (const auto& p : meshPoints) {
        outVertices.push_back(p.x());
        outVertices.push_back(p.y());
        outVertices.push_back(p.z());
    }

    outTriangles.clear();
    outTriangles.reserve(meshTris.size() * 3 + meshQuads.size() * 6);
    for (const auto& t : meshTris) {
        outTriangles.push_back(uint32_t(t.x()));
        outTriangles.push_back(uint32_t(t.y()));
        outTriangles.push_back(uint32_t(t.z()));
    }
    for (const auto& q : meshQuads) {
        outTriangles.push_back(uint32_t(q.x())); outTriangles.push_back(uint32_t(q.y())); outTriangles.push_back(uint32_t(q.z()));
        outTriangles.push_back(uint32_t(q.x())); outTriangles.push_back(uint32_t(q.z())); outTriangles.push_back(uint32_t(q.w()));
    }
    return !outVertices.empty();
}

// Extract meshes for enclosed air pockets (bubbles) inside the water.
//
// signedFloodFill re-signs the tree so that regions connected to the
// background become negative and enclosed regions become positive. For a
// water SDF (negative inside): exterior air -> negative, water -> positive,
// and air pockets trapped INSIDE the water keep their original positive
// sign. So cells that are positive in BOTH the original and the flooded
// tree are exactly the enclosed air pockets. We keep those (with negated
// values so the pocket interior is negative) and mesh them at iso 0.
void appendBubbleMeshes(const openvdb::FloatGrid& waterGrid,
                        std::vector<float>& outVertices,
                        std::vector<uint32_t>& outTriangles) {
    auto cav = waterGrid.deepCopy();
    openvdb::tools::signedFloodFill(cav->tree());
    for (openvdb::FloatGrid::ValueOnIter it = cav->beginValueOn(); it; ++it) {
        float orig = waterGrid.tree().getValue(it.getCoord());
        if (orig > 0.f && it.getValue() > 0.f) {
            it.setValue(-orig); // pocket interior becomes negative
        } else {
            it.setValueOff();
        }
    }

    std::vector<openvdb::Vec3s> pts;
    std::vector<openvdb::Vec3I> tris;
    std::vector<openvdb::Vec4I> quads;
    openvdb::tools::volumeToMesh(*cav, pts, tris, quads, 0.0, 0.0);
    if (pts.empty()) return;

    uint32_t off = uint32_t(outVertices.size() / 3);
    for (const auto& p : pts) {
        outVertices.push_back(p.x());
        outVertices.push_back(p.y());
        outVertices.push_back(p.z());
    }
    for (const auto& t : tris) {
        outTriangles.push_back(off + uint32_t(t.x()));
        outTriangles.push_back(off + uint32_t(t.y()));
        outTriangles.push_back(off + uint32_t(t.z()));
    }
    for (const auto& q : quads) {
        outTriangles.push_back(off + uint32_t(q.x())); outTriangles.push_back(off + uint32_t(q.y())); outTriangles.push_back(off + uint32_t(q.z()));
        outTriangles.push_back(off + uint32_t(q.x())); outTriangles.push_back(off + uint32_t(q.z())); outTriangles.push_back(off + uint32_t(q.w()));
    }
}

} // namespace

bool particlesToMesh(const float* positions, size_t numParticles,
                     float voxelSize, float halfWidth, float iso, float adaptivity,
                     bool preserveBubbles,
                     std::vector<float>& outVertices,
                     std::vector<uint32_t>& outTriangles)
{
    if (numParticles < 4) return false;

    openvdb::initialize();
    auto grid = buildWaterGrid(positions, numParticles, voxelSize, halfWidth);
    if (!grid) return false;

    // Light smoothing of the SDF for nicer surface
    openvdb::tools::LevelSetFilter<openvdb::FloatGrid, openvdb::FloatGrid> filter(*grid);
    filter.gaussian(int(voxelSize * 0.5f));

    bool ok = extractMesh(*grid, iso, adaptivity, outVertices, outTriangles);
    if (ok && preserveBubbles) appendBubbleMeshes(*grid, outVertices, outTriangles);
    return ok;
}

bool particlesToMeshWithObstacles(const float* positions, size_t numParticles,
                                  float voxelSize, float halfWidth, float iso,
                                  const float* obstacleVerts, size_t numObstacleVerts,
                                  const uint32_t* obstacleTris, size_t numObstacleTris,
                                  float adaptivity, bool preserveBubbles,
                                  std::vector<float>& outVertices,
                                  std::vector<uint32_t>& outTriangles)
{
    if (numParticles < 4) return false;

    openvdb::initialize();
    auto grid = buildWaterGrid(positions, numParticles, voxelSize, halfWidth);
    if (!grid) return false;

    // CSG-difference with the collider level set: water ∧ ¬solid.
    // The water surface is cut at the solid boundary and the solid's surface
    // is included wherever it lies inside the water region.
    if (numObstacleVerts >= 3 && numObstacleTris >= 1) {
        std::vector<openvdb::Vec3s> opoints;
        opoints.reserve(numObstacleVerts);
        for (size_t i = 0; i < numObstacleVerts; i++) {
            opoints.emplace_back(obstacleVerts[3*i + 0],
                                 obstacleVerts[3*i + 1],
                                 obstacleVerts[3*i + 2]);
        }
        std::vector<openvdb::Vec3I> otris;
        otris.reserve(numObstacleTris);
        for (size_t i = 0; i < numObstacleTris; i++) {
            otris.emplace_back(obstacleTris[3*i + 0],
                               obstacleTris[3*i + 1],
                               obstacleTris[3*i + 2]);
        }
        try {
            auto solidGrid = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(
                grid->transform(), opoints, otris, float(halfWidth * voxelSize));
            if (solidGrid && !solidGrid->empty()) {
                openvdb::tools::csgDifference(*grid, *solidGrid);
            }
        } catch (...) {
            // Non-watertight collider meshes can fail conversion — degrade
            // gracefully to the un-obstructed surface.
        }
    }

    // Light smoothing of the SDF for nicer surface
    openvdb::tools::LevelSetFilter<openvdb::FloatGrid, openvdb::FloatGrid> filter(*grid);
    filter.gaussian(int(voxelSize * 0.5f));

    bool ok = extractMesh(*grid, iso, adaptivity, outVertices, outTriangles);
    if (ok && preserveBubbles) appendBubbleMeshes(*grid, outVertices, outTriangles);
    return ok;
}

} // namespace flipcore
