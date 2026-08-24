#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "flipcore/FlipSolver.h"
#ifdef FLIP_HAS_OPENVDB
#include "flipcore/SurfaceMesher.h"
#endif
#ifdef FLIP_HAS_MPM
#include "flipcore/MpmSolver.h"
#endif
#ifdef FLIP_HAS_MESHER_GPU
#include "flipcore/GpuMesher.h"
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using namespace flipcore;

namespace {

py::array_t<float> toNumpy(std::vector<float>&& flat) {
    size_t n = flat.size() / 3;
    auto result = py::array_t<float>({n, size_t(3)});
    std::memcpy(result.mutable_data(), flat.data(), flat.size() * sizeof(float));
    return result;
}

} // namespace

PYBIND11_MODULE(flip_solver_core, m) {
    m.doc() = "C++ FLIP fluid solver core for the Blender water simulation addon";

#ifdef _OPENMP
    m.attr("openmp_enabled") = true;
    m.attr("openmp_max_threads") = omp_get_max_threads();
#else
    m.attr("openmp_enabled") = false;
    m.attr("openmp_max_threads") = 1;
#endif

#ifdef FLIP_HAS_CUDA
    m.attr("cuda_enabled") = true;
#else
    m.attr("cuda_enabled") = false;
#endif

#ifdef FLIP_HAS_OPENVDB
    m.attr("openvdb_enabled") = true;
    m.def("particles_to_mesh",
          [](py::array_t<float> positions, float voxelSize, float halfWidth, float iso,
             float adaptivity, bool preserveBubbles) -> py::tuple {
              auto buf = positions.request();
              if (buf.ndim != 2 || buf.shape[1] != 3)
                  throw std::runtime_error("positions must be (N,3) float32");
              size_t n = buf.shape[0];
              std::vector<float> verts;
              std::vector<uint32_t> tris;
              bool ok = flipcore::particlesToMesh(
                  static_cast<const float*>(buf.ptr), n, voxelSize, halfWidth, iso,
                  adaptivity, preserveBubbles, verts, tris);
              if (!ok || verts.empty()) return py::make_tuple(py::none(), py::none());
              auto v = py::array_t<float>({verts.size() / 3, size_t(3)});
              std::memcpy(v.mutable_data(), verts.data(), verts.size() * sizeof(float));
              auto t = py::array_t<uint32_t>({tris.size() / 3, size_t(3)});
              std::memcpy(t.mutable_data(), tris.data(), tris.size() * sizeof(uint32_t));
              return py::make_tuple(v, t);
          },
          py::arg("positions"), py::arg("voxel_size") = 0.05f, py::arg("half_width") = 3.0f,
          py::arg("iso") = 0.0f, py::arg("adaptivity") = 0.0f,
          py::arg("preserve_bubbles") = false,
          "Convert particle positions to a triangle mesh using OpenVDB");

    m.def("particles_to_mesh_with_obstacles",
          [](py::array_t<float> positions, float voxelSize, float halfWidth,
             py::array_t<float> obstacleVerts, py::array_t<uint32_t> obstacleTris,
             float iso, float adaptivity, bool preserveBubbles) -> py::tuple {
              auto buf = positions.request();
              if (buf.ndim != 2 || buf.shape[1] != 3)
                  throw std::runtime_error("positions must be (N,3) float32");
              size_t n = buf.shape[0];

              std::vector<float> verts;
              std::vector<uint32_t> tris;
              const float* ov = nullptr;
              size_t nov = 0;
              const uint32_t* ot = nullptr;
              size_t not_ = 0;
              if (!obstacleVerts.is_none()) {
                  auto ob = obstacleVerts.request();
                  if (ob.ndim != 2 || ob.shape[1] != 3)
                      throw std::runtime_error("obstacle_verts must be (N,3) float32");
                  ov = static_cast<const float*>(ob.ptr);
                  nov = ob.shape[0];
              }
              if (!obstacleTris.is_none()) {
                  auto tb = obstacleTris.request();
                  if (tb.ndim != 2 || tb.shape[1] != 3)
                      throw std::runtime_error("obstacle_tris must be (M,3) uint32");
                  ot = static_cast<const uint32_t*>(tb.ptr);
                  not_ = tb.shape[0];
              }
              bool ok = flipcore::particlesToMeshWithObstacles(
                  static_cast<const float*>(buf.ptr), n, voxelSize, halfWidth, iso,
                  ov, nov, ot, not_, adaptivity, preserveBubbles, verts, tris);
              if (!ok || verts.empty()) return py::make_tuple(py::none(), py::none());
              auto v = py::array_t<float>({verts.size() / 3, size_t(3)});
              std::memcpy(v.mutable_data(), verts.data(), verts.size() * sizeof(float));
              auto t = py::array_t<uint32_t>({tris.size() / 3, size_t(3)});
              std::memcpy(t.mutable_data(), tris.data(), tris.size() * sizeof(uint32_t));
              return py::make_tuple(v, t);
          },
          py::arg("positions"), py::arg("voxel_size") = 0.05f, py::arg("half_width") = 3.0f,
          py::arg("obstacle_verts") = py::none(), py::arg("obstacle_tris") = py::none(),
          py::arg("iso") = 0.0f, py::arg("adaptivity") = 0.0f,
          py::arg("preserve_bubbles") = false,
          "Convert particle positions to a triangle mesh, CSG-cut by collider meshes");
#else
    m.attr("openvdb_enabled") = false;
#endif

#ifdef FLIP_HAS_MESHER_GPU
    m.attr("mesher_gpu_enabled") = true;
    m.def("particles_to_mesh_gpu",
          [](py::array_t<float> positions, float voxelSize, float iso) -> py::tuple {
              auto buf = positions.request();
              if (buf.ndim != 2 || buf.shape[1] != 3)
                  throw std::runtime_error("positions must be (N,3) float32");
              size_t n = buf.shape[0];
              std::vector<float> verts;
              std::vector<uint32_t> tris;
              bool ok = flipcore::particlesToMeshGpu(
                  static_cast<const float*>(buf.ptr), n, voxelSize, iso, verts, tris);
              if (!ok || verts.empty()) return py::make_tuple(py::none(), py::none());
              auto v = py::array_t<float>({verts.size() / 3, size_t(3)});
              std::memcpy(v.mutable_data(), verts.data(), verts.size() * sizeof(float));
              auto t = py::array_t<uint32_t>({tris.size() / 3, size_t(3)});
              std::memcpy(t.mutable_data(), tris.data(), tris.size() * sizeof(uint32_t));
              return py::make_tuple(v, t);
          },
          py::arg("positions"), py::arg("voxel_size") = 0.05f, py::arg("iso") = 0.25f,
          "Convert particle positions to a triangle mesh using the GPU marching-cubes mesher");
#else
    m.attr("mesher_gpu_enabled") = false;
#endif

#ifdef FLIP_HAS_MPM
    m.attr("mpm_enabled") = true;

    py::class_<MpmMaterial>(m, "MpmMaterial")
        .def(py::init<>())
        .def_readwrite("youngs_modulus",       &MpmMaterial::youngsModulus)
        .def_readwrite("poisson_ratio",        &MpmMaterial::poissonRatio)
        .def_readwrite("hardening",            &MpmMaterial::hardening)
        .def_readwrite("critical_compression", &MpmMaterial::criticalCompression)
        .def_readwrite("critical_stretch",     &MpmMaterial::criticalStretch)
        .def_readwrite("dynamic_viscosity",    &MpmMaterial::dynamicViscosity)
        .def_readwrite("bulk_viscosity",       &MpmMaterial::bulkViscosity)
        .def_readwrite("sand_alpha",           &MpmMaterial::sandAlpha)
        .def_readwrite("density",              &MpmMaterial::density);

    py::enum_<MpmPreset>(m, "MpmPreset")
        .value("Sand",  MpmPreset::Sand)
        .value("Snow",  MpmPreset::Snow)
        .value("Jello", MpmPreset::Jello)
        .value("Water", MpmPreset::Water)
        .value("Honey", MpmPreset::Honey)
        .export_values();

    m.def("mpm_preset_material", &presetMaterial,
          "Return MpmMaterial for a given preset");

    py::class_<MpmSettings>(m, "MpmSettings")
        .def(py::init<>())
        .def_readwrite("grid_origin_x",     &MpmSettings::gridOriginX)
        .def_readwrite("grid_origin_y",     &MpmSettings::gridOriginY)
        .def_readwrite("grid_origin_z",     &MpmSettings::gridOriginZ)
        .def_readwrite("grid_res_x",        &MpmSettings::gridResX)
        .def_readwrite("grid_res_y",        &MpmSettings::gridResY)
        .def_readwrite("grid_res_z",        &MpmSettings::gridResZ)
        .def_readwrite("grid_stride",       &MpmSettings::gridStride)
        .def_readwrite("delta_time",        &MpmSettings::deltaTime)
        .def_readwrite("substeps_per_frame",&MpmSettings::substepsPerFrame)
        .def_readwrite("flip_ratio",        &MpmSettings::flipRatio)
        .def_readwrite("gravity_x",         &MpmSettings::gravityX)
        .def_readwrite("gravity_y",         &MpmSettings::gravityY)
        .def_readwrite("gravity_z",         &MpmSettings::gravityZ)
        .def_readwrite("boundary_friction", &MpmSettings::boundaryFriction)
        .def_readwrite("material",          &MpmSettings::material);

    py::class_<MpmSolver>(m, "MpmSolver")
        .def(py::init<>())
        .def("init", [](MpmSolver& s, py::array_t<float> positions, const MpmSettings& settings) {
            auto buf = positions.request();
            if (buf.ndim != 2 || buf.shape[1] != 3)
                throw std::runtime_error("positions must be (N,3) float32");
            s.init(static_cast<const float*>(buf.ptr), buf.shape[0], settings);
        })
        .def("step", &MpmSolver::step)
        .def("get_positions", [](const MpmSolver& s) -> py::array_t<float> {
            size_t n = s.particleCount();
            if (n == 0) return py::array_t<float>({size_t(0), size_t(3)});
            auto result = py::array_t<float>({n, size_t(3)});
            s.getPositions(result.mutable_data(), n);
            return result;
        })
        .def("particle_count", &MpmSolver::particleCount)
        .def("set_boundary", [](MpmSolver& s, float ox, float oy, float oz,
                                  float tx, float ty, float tz) {
            float o[3] = {ox, oy, oz};
            float t[3] = {tx, ty, tz};
            s.setBoundary(o, t);
        });
#else
    m.attr("mpm_enabled") = false;
#endif

    py::enum_<SolverBackend>(m, "SolverBackend")
        .value("CPU", SolverBackend::CPU)
        .value("CUDA", SolverBackend::CUDA)
        .export_values();

    py::class_<Vec3>(m, "Vec3")
        .def(py::init<>())
        .def(py::init<float, float, float>())
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z);

    py::class_<SolverSettings>(m, "SolverSettings")
        .def(py::init<>())
        .def_readwrite("resolution", &SolverSettings::resolution)
        .def_readwrite("flip_ratio", &SolverSettings::flipRatio)
        .def_readwrite("density", &SolverSettings::density)
        .def_readwrite("gravity", &SolverSettings::gravity)
        .def_readwrite("cfl_number", &SolverSettings::cflNumber)
        .def_readwrite("max_substeps", &SolverSettings::maxSubsteps)
        .def_readwrite("pressure_iterations", &SolverSettings::pressureIterations)
        .def_readwrite("pressure_tolerance", &SolverSettings::pressureTolerance)
        .def_readwrite("extrapolate_iterations", &SolverSettings::extrapolateIterations)
        .def_readwrite("st_flip_enabled", &SolverSettings::stFlipEnabled)
        .def_readwrite("jitter_strength", &SolverSettings::jitterStrength)
        .def_readwrite("phase_field_eta", &SolverSettings::phaseFieldEta)
        .def_readwrite("particles_per_cell_per_axis", &SolverSettings::particlesPerCellPerAxis)
        .def_readwrite("collision_use_sdf", &SolverSettings::collisionUseSDF)
        .def_readwrite("sdf_collision_margin", &SolverSettings::sdfCollisionMargin)
        .def_readwrite("reseed_enabled", &SolverSettings::reseedEnabled)
        .def_readwrite("reseed_min_ratio", &SolverSettings::reseedMinRatio)
        .def_readwrite("reseed_max_ratio", &SolverSettings::reseedMaxRatio)
        .def_readwrite("viscosity_strength", &SolverSettings::viscosityStrength)
        .def_readwrite("surface_tension_strength", &SolverSettings::surfaceTensionStrength)
        .def_readwrite("vorticity_confinement", &SolverSettings::vorticityConfinement)
        .def_readwrite("pressure_warm_start", &SolverSettings::pressureWarmStart)
        .def_readwrite("adaptive_pressure_iterations", &SolverSettings::adaptivePressureIterations)
        .def_readwrite("pressure_min_iterations", &SolverSettings::pressureMinIterations)
        .def_readwrite("air_band_cells", &SolverSettings::airBandCells)
        .def_readwrite("air_density_ratio", &SolverSettings::airDensityRatio)
        .def_readwrite("solver_backend", &SolverSettings::solverBackend)
        .def_readwrite("max_particles", &SolverSettings::maxParticles);

    py::class_<FlipSolver>(m, "FlipSolver")
        .def(py::init<>())
        .def("init_domain",
             [](FlipSolver& s, py::array_t<float> domainMin, py::array_t<float> domainMax,
                const SolverSettings& settings) {
                 auto mn = domainMin.unchecked<1>();
                 auto mx = domainMax.unchecked<1>();
                 s.initDomain(Vec3(mn(0), mn(1), mn(2)), Vec3(mx(0), mx(1), mx(2)), settings);
             },
             py::arg("domain_min"), py::arg("domain_max"), py::arg("settings"))
        .def("add_particles",
             [](FlipSolver& s, py::array_t<float> positions, py::object velocities) {
                 auto posBuf = positions.request();
                 if (posBuf.ndim != 2 || posBuf.shape[1] != 3)
                     throw std::runtime_error("positions must be an (N,3) float32 array");
                 size_t n = posBuf.shape[0];
                 const float* posPtr = static_cast<const float*>(posBuf.ptr);
                 const float* velPtr = nullptr;
                 py::array_t<float> velArr;
                 if (!velocities.is_none()) {
                     velArr = velocities.cast<py::array_t<float>>();
                     auto velBuf = velArr.request();
                     if (velBuf.ndim != 2 || size_t(velBuf.shape[0]) != n || velBuf.shape[1] != 3)
                         throw std::runtime_error("velocities must be an (N,3) float32 array matching positions");
                     velPtr = static_cast<const float*>(velBuf.ptr);
                 }
                 return s.addParticles(posPtr, velPtr, n);
             },
             py::arg("positions"), py::arg("velocities") = py::none())
        .def("add_particles_box",
             [](FlipSolver& s, py::array_t<float> boxMin, py::array_t<float> boxMax, int perCell,
                py::object initialVelocity, uint32_t seed) {
                 auto mn = boxMin.unchecked<1>();
                 auto mx = boxMax.unchecked<1>();
                 Vec3 vel(0.f, 0.f, 0.f);
                 if (!initialVelocity.is_none()) {
                     auto velArr = initialVelocity.cast<py::array_t<float>>();
                     auto v = velArr.unchecked<1>();
                     vel = Vec3(v(0), v(1), v(2));
                 }
                 return s.addParticlesBox(Vec3(mn(0), mn(1), mn(2)), Vec3(mx(0), mx(1), mx(2)), perCell,
                                           vel, seed);
             },
             py::arg("box_min"), py::arg("box_max"), py::arg("particles_per_cell_per_axis") = 2,
             py::arg("initial_velocity") = py::none(), py::arg("seed") = 1u)
        .def("clear_particles", &FlipSolver::clearParticles)
        .def("set_obstacle_mask",
             [](FlipSolver& s, py::array_t<uint8_t> mask) {
                 auto buf = mask.request();
                 s.setObstacleMask(static_cast<const uint8_t*>(buf.ptr), size_t(buf.size));
             },
             py::arg("mask"))
        .def("set_obstacle_sdf",
             [](FlipSolver& s, py::array_t<float> sdf) {
                 auto buf = sdf.request();
                 s.setObstacleSDF(static_cast<const float*>(buf.ptr), size_t(buf.size));
             },
             py::arg("sdf"))
        .def("step", &FlipSolver::step, py::arg("dt"))
        .def("particle_count", &FlipSolver::particleCount)
        .def("get_positions", [](const FlipSolver& s) { return toNumpy(s.positionsFlat()); })
        .def("get_render_positions", [](const FlipSolver& s) { return toNumpy(s.renderPositionsFlat()); })
        .def("get_velocities", [](const FlipSolver& s) { return toNumpy(s.velocitiesFlat()); })
        .def("velocity_field", [](const FlipSolver& s) {
            std::vector<float> flat = s.cellVelocityField();
            auto a = py::array_t<float>({py::ssize_t(flat.size() / 3), py::ssize_t(3)});
            if (!flat.empty()) std::memcpy(a.mutable_data(), flat.data(), flat.size() * sizeof(float));
            return a;
        })
        .def("vorticity_field", [](const FlipSolver& s) {
            std::vector<float> flat = s.vorticityField();
            auto a = py::array_t<float>({py::ssize_t(flat.size() / 3), py::ssize_t(3)});
            if (!flat.empty()) std::memcpy(a.mutable_data(), flat.data(), flat.size() * sizeof(float));
            return a;
        })
        .def("cell_size", &FlipSolver::cellSize)
        .def("grid_dims", [](const FlipSolver& s) {
            return py::make_tuple(s.grid().nx(), s.grid().ny(), s.grid().nz());
        })
        .def("domain_min", [](const FlipSolver& s) {
            Vec3 v = s.domainMin(); return py::make_tuple(v.x, v.y, v.z);
        })
        .def("domain_max", [](const FlipSolver& s) {
            Vec3 v = s.domainMax(); return py::make_tuple(v.x, v.y, v.z);
        })
        // DEBUG-ONLY: dump the internal MAC grid state (cell types, staggered
        // velocities, pressure, phase-field weights) for headless diagnosis.
        .def("debug_grid", [](const FlipSolver& s) -> py::dict {
            const MacGrid& g = s.grid();
            py::dict out;
            out["dims"] = py::make_tuple(g.nx(), g.ny(), g.nz());
            out["h"] = g.h();
            auto copyArr = [](const float* p, size_t n) -> py::array_t<float> {
                auto a = py::array_t<float>({py::ssize_t(n)});
                if (n) std::memcpy(a.mutable_data(), p, n * sizeof(float));
                return a;
            };
            out["u"] = copyArr(g.u.data(), g.u.size());
            out["v"] = copyArr(g.v.data(), g.v.size());
            out["w"] = copyArr(g.w.data(), g.w.size());
            out["u_old"] = copyArr(g.uOld.data(), g.uOld.size());
            out["v_old"] = copyArr(g.vOld.data(), g.vOld.size());
            out["w_old"] = copyArr(g.wOld.data(), g.wOld.size());
            out["pressure"] = copyArr(g.pressure.data(), g.pressure.size());
            out["cell_weight"] = copyArr(g.cellWeight.data(), g.cellWeight.size());
            auto ct = py::array_t<int8_t>({py::ssize_t(g.cellType.size())});
            if (g.cellType.size()) {
                std::memcpy(ct.mutable_data(), g.cellType.data(), g.cellType.size());
            }
            out["cell_type"] = ct;
            return out;
        });
}
