cmake_minimum_required(VERSION 3.15)
project(flip_solver_core LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/O2 /W3)
else()
  add_compile_options(-O3 -Wall -Wextra -Wno-unused-parameter)
endif()

# --- CUDA (optional GPU-accelerated pressure solve) ------------------------
# If the CUDA toolkit is installed, the CUDA backend will be compiled in and
# can be toggled at runtime via SolverSettings.solverBackend.
# NOTE: CUDA 13.x + CMake 4.4's FindCUDAToolkit has a known incompatibility.
# We locate nvcc and the libraries manually as a workaround.
option(FLIP_ENABLE_CUDA "Build pressure-solver CUDA backend" OFF)
set(FLIP_HAS_CUDA FALSE)

if(FLIP_ENABLE_CUDA)
  find_program(CUDA_NVCC_EXECUTABLE nvcc
    PATHS "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA"
    PATH_SUFFIXES v12.8/bin v12.6/bin v13.3/bin bin
    DOC "nvcc compiler from the CUDA Toolkit"
  )
  if(CUDA_NVCC_EXECUTABLE)
    get_filename_component(CUDA_TOOLKIT_ROOT "${CUDA_NVCC_EXECUTABLE}" DIRECTORY)
    get_filename_component(CUDA_TOOLKIT_ROOT "${CUDA_TOOLKIT_ROOT}" DIRECTORY)
    message(STATUS "nvcc found: ${CUDA_NVCC_EXECUTABLE}")
    message(STATUS "CUDA toolkit root: ${CUDA_TOOLKIT_ROOT}")

    # Manually locate cudart and cublas libraries
    find_library(CUDA_CUDART_LIBRARY cudart
      PATHS "${CUDA_TOOLKIT_ROOT}"
      PATH_SUFFIXES lib/x64 lib
      NO_DEFAULT_PATH
    )
    find_library(CUDA_CUBLAS_LIBRARY cublas
      PATHS "${CUDA_TOOLKIT_ROOT}"
      PATH_SUFFIXES lib/x64 lib
      NO_DEFAULT_PATH
    )
    if(CUDA_CUDART_LIBRARY AND CUDA_CUBLAS_LIBRARY)
      # Compile .cu manually with nvcc (uses MSVC host compiler from PATH).
      # We bypass CMake's enable_language(CUDA) because it triggers linker
      # tests that fail when MinGW ld.exe is the system linker on Windows.
      # The .cu code is C-only at the API surface (extern "C" in the header)
      # so the resulting .obj has no C++ name mangling from either toolchain.
      set(NVCC_FLAGS
        -O2 -std=c++17 --use-local-env
        -Xcompiler /MD
        -Wno-deprecated-gpu-targets
        -diag-suppress 177,546
        -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
      )
      foreach(_arch 61 75 86)
        list(APPEND NVCC_FLAGS -gencode "arch=compute_${_arch},code=sm_${_arch}")
      endforeach()
      if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND NVCC_FLAGS -G)
      endif()

      set(CUDA_OBJ "${CMAKE_CURRENT_BINARY_DIR}/PressureSolverGPU.cu.obj")
      add_custom_command(
        OUTPUT "${CUDA_OBJ}"
        COMMAND "${CUDA_NVCC_EXECUTABLE}" ${NVCC_FLAGS}
                -c "${CMAKE_CURRENT_SOURCE_DIR}/src/PressureSolverGPU.cu"
                -o "${CUDA_OBJ}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/PressureSolverGPU.cu"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/flipcore/PressureSolverGPU.h"
        COMMENT "Compiling CUDA pressure solver (nvcc + MSVC)"
        VERBATIM
      )
      add_library(flip_solver_cuda STATIC "${CUDA_OBJ}")
      set_target_properties(flip_solver_cuda PROPERTIES LINKER_LANGUAGE C)

      set(FLIP_HAS_CUDA TRUE)
      set(CUDAToolkit_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT}/include")
      set(CUDAToolkit_LIBRARY_DIR "${CUDA_TOOLKIT_ROOT}/lib/x64")

      message(STATUS "CUDA GPU pressure solver enabled (manual nvcc)")
      message(STATUS "  CUDA architectures: 61, 75, 86")
      message(STATUS "  cudart: ${CUDA_CUDART_LIBRARY}")

      # --- MPM solver (CUDA) ---
      option(FLIP_ENABLE_MPM "Build MPM CUDA solver" ON)
      if(FLIP_ENABLE_MPM)
        set(MPM_OBJ "${CMAKE_CURRENT_BINARY_DIR}/MpmSolver.cu.obj")
        add_custom_command(
          OUTPUT "${MPM_OBJ}"
          COMMAND "${CUDA_NVCC_EXECUTABLE}" ${NVCC_FLAGS}
                  -c "${CMAKE_CURRENT_SOURCE_DIR}/src/MpmSolver.cu"
                  -o "${MPM_OBJ}"
          DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/MpmSolver.cu"
                  "${CMAKE_CURRENT_SOURCE_DIR}/include/flipcore/MpmSolver.h"
          COMMENT "Compiling CUDA MPM solver (nvcc + MSVC)"
          VERBATIM
        )
        add_library(flip_solver_mpm STATIC "${MPM_OBJ}")
        set_target_properties(flip_solver_mpm PROPERTIES LINKER_LANGUAGE C)
        set(FLIP_HAS_MPM TRUE)
        message(STATUS "MPM CUDA solver enabled")
      endif()

      # --- GPU marching-cubes mesher (CUDA) ---
      option(FLIP_ENABLE_MESHER_GPU "Build CUDA surface mesher" ON)
      if(FLIP_ENABLE_MESHER_GPU)
        set(MESHER_OBJ "${CMAKE_CURRENT_BINARY_DIR}/GpuMesher.cu.obj")
        add_custom_command(
          OUTPUT "${MESHER_OBJ}"
          COMMAND "${CUDA_NVCC_EXECUTABLE}" ${NVCC_FLAGS}
                  -c "${CMAKE_CURRENT_SOURCE_DIR}/src/GpuMesher.cu"
                  -o "${MESHER_OBJ}"
          DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/GpuMesher.cu"
                  "${CMAKE_CURRENT_SOURCE_DIR}/include/flipcore/GpuMesher.h"
          COMMENT "Compiling CUDA surface mesher (nvcc + MSVC)"
          VERBATIM
        )
        add_library(flip_solver_mesher STATIC "${MESHER_OBJ}")
        set_target_properties(flip_solver_mesher PROPERTIES LINKER_LANGUAGE C)
        set(FLIP_HAS_MESHER_GPU TRUE)
        message(STATUS "GPU surface mesher enabled")
      endif()
    else()
      message(STATUS "cudart.lib not found in ${CUDA_TOOLKIT_ROOT}/lib/x64 - CPU only")
    endif()
  else()
    message(STATUS "nvcc not found - building CPU-only pressure solver")
  endif()
endif()

# --- Locate the exact Python interpreter to build against -------------------
# IMPORTANT: this must match the Python version embedded in the target
# Blender build (Blender 4.x -> Python 3.11, Blender 5.0/5.1 -> Python 3.13,
# etc). Pass -DPython3_EXECUTABLE=/path/to/python to CMake to pin it
# explicitly instead of relying on auto-detection. See ../README.md.
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# --- pybind11 -----------------------------------------------------------
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  # Fall back to asking the interpreter where its pip-installed copy lives.
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_LOOKUP_RESULT
  )
  if(NOT PYBIND11_LOOKUP_RESULT EQUAL 0)
    message(FATAL_ERROR
      "pybind11 was not found. Install it for the Python you are building "
      "against, e.g.:\n  \"${Python3_EXECUTABLE}\" -m pip install pybind11")
  endif()
  list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
  find_package(pybind11 CONFIG REQUIRED)
endif()

# --- OpenVDB (optional native surface mesher) --------------------------------
option(FLIP_ENABLE_OPENVDB "Build native OpenVDB surface mesher" ON)
if(FLIP_ENABLE_OPENVDB)
  find_package(OpenVDB CONFIG QUIET)
  if(OpenVDB_FOUND)
    message(STATUS "OpenVDB native surface mesher enabled (vcpkg)")
  else()
    message(STATUS "OpenVDB not found — falling back to pysplashsurf. Install with: vcpkg install openvdb:x64-windows")
  endif()
endif()

file(GLOB CORE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
# SurfaceMesher.cpp requires OpenVDB — exclude it if not found
if(NOT OpenVDB_FOUND)
  list(REMOVE_ITEM CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/SurfaceMesher.cpp")
endif()

pybind11_add_module(flip_solver_core MODULE ${CORE_SOURCES} bindings/pybind_module.cpp)
target_include_directories(flip_solver_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(FLIP_HAS_CUDA)
  target_link_libraries(flip_solver_core PRIVATE flip_solver_cuda "${CUDA_CUDART_LIBRARY}" "${CUDA_CUBLAS_LIBRARY}")
  target_compile_definitions(flip_solver_core PRIVATE FLIP_HAS_CUDA)
  message(STATUS "GPU pressure solver linked into flip_solver_core")
endif()

if(FLIP_HAS_MPM)
  target_link_libraries(flip_solver_core PRIVATE flip_solver_mpm)
  target_compile_definitions(flip_solver_core PRIVATE FLIP_HAS_MPM)
  message(STATUS "MPM solver linked into flip_solver_core")
endif()

if(FLIP_HAS_MESHER_GPU)
  target_link_libraries(flip_solver_core PRIVATE flip_solver_mesher)
  target_compile_definitions(flip_solver_core PRIVATE FLIP_HAS_MESHER_GPU)
  message(STATUS "GPU surface mesher linked into flip_solver_core")
endif()

if(OpenVDB_FOUND)
  target_link_libraries(flip_solver_core PRIVATE OpenVDB::openvdb)
  target_compile_definitions(flip_solver_core PRIVATE FLIP_HAS_OPENVDB)
endif()

# Optional multi-core speedup for the pressure solve / particle loops.
# Not a hard requirement - falls back to single-threaded silently if unavailable
# (e.g. stock AppleClang on macOS without `brew install libomp`).
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(flip_solver_core PRIVATE OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP found - building with multi-core support")
else()
  message(STATUS "OpenMP NOT found - building single-threaded (still works, just slower). "
                  "On macOS: `brew install libomp`. On Windows/Linux this is normally bundled with the compiler.")
endif()

# Land the built extension directly in addon/bin/<platform>-py<major><minor>/
# so the addon can locate it deterministically at runtime.
set(FLIP_PLATFORM_TAG "${CMAKE_SYSTEM_NAME}-py${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}")
string(TOLOWER "${FLIP_PLATFORM_TAG}" FLIP_PLATFORM_TAG)
set(FLIP_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../bin/${FLIP_PLATFORM_TAG}")
set_target_properties(flip_solver_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${FLIP_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${FLIP_OUTPUT_DIR}"
)

message(STATUS "flip_solver_core will be built for Python ${Python3_VERSION} at ${Python3_EXECUTABLE}")
message(STATUS "Output directory: ${FLIP_OUTPUT_DIR}")
