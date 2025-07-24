cmake_minimum_required(VERSION 3.25.2)
project(NBodySimulation LANGUAGES CXX CUDA)

# Set C++ and CUDA standards to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA settings
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr -Wno-deprecated-gpu-targets")

# OpenMP support
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Build options - build everything by default
# We'll keep these for backward compatibility but don't use them to conditionally compile
set(ENABLE_VISUALIZATION ON)
set(ENABLE_BENCHMARK ON)
option(ENABLE_PROFILE "Enable profiling mode" OFF)

# Add compile flags for CPU optimization
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -ffast-math -mavx2 -fopenmp")

# Define source files
set(COMMON_SOURCES
    simulator/core/main.cpp
    simulator/core/System.cpp
    simulator/core/Initializer.cpp
    simulator/core/Forces.cpp
    simulator/output/OutputUtils.cpp
)

# Define CUDA sources - include all functionality
set(ALL_CUDA_SOURCES
    simulator/cuda/CudaUtils.cu
    simulator/cuda/ForceKernels.cu
    simulator/cuda/IntegrationKernels.cu
    simulator/cuda/SimulationKernels.cu
    simulator/cuda/BenchmarkKernels.cu
    simulator/cuda/BHKernels2.cu
    #simulator/cuda/MinimalBHKernels.cu
    simulator/cuda/Constants.cu
)

# Define visualization sources
set(VISUALIZATION_SOURCES
    simulator/visualization/ParticleRenderer.cpp
    simulator/visualization/ShaderManager.cpp
    simulator/visualization/CameraController.cpp
    simulator/visualization/cuda/VisualizationKernels.cu
    simulator/visualization/VisualizationUtils.cpp
)


# Define include directory
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Find all subdirectories and add them to include path
file(GLOB_RECURSE ALL_SUBDIRS LIST_DIRECTORIES true "${CMAKE_CURRENT_SOURCE_DIR}/simulator/*")
foreach(DIR ${ALL_SUBDIRS})
    if(IS_DIRECTORY ${DIR})
        include_directories(${DIR})
    endif()
endforeach()

# Define CUDA_ENABLED flag for conditional compilation
add_definitions(-DCUDA_ENABLED)

# Always enable CUDA
add_definitions(-DUSE_CUDA=1)

# Enable all features
add_definitions(-DBENCHMARK_MODE=1)
add_definitions(-DVISUALIZATION_ENABLED=1)

# Find required packages for visualization
find_package(OpenGL REQUIRED)
find_package(glfw3 QUIET)
if(NOT glfw3_FOUND)
    find_package(GLFW QUIET)
    if(NOT GLFW_FOUND)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(GLFW3 REQUIRED glfw3)
        set(GLFW_LIBRARIES ${GLFW3_LIBRARIES})
    endif()
endif()
find_package(GLEW REQUIRED)

# Add nlohmann/json library (using the recommended approach)
include(FetchContent)
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Create a single executable with all features
add_executable(n-body-simulation
    ${COMMON_SOURCES}
    ${ALL_CUDA_SOURCES}
    ${VISUALIZATION_SOURCES}
)

# Always link with CUDA and OpenMP
target_link_libraries(n-body-simulation
    CUDA::cudart
    ${OpenMP_CXX_LIBRARIES}
    ${OPENGL_LIBRARIES}
    glfw
    GLEW::GLEW
    nlohmann_json::nlohmann_json
)

# Print configuration summary
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "OpenMP support: ${OpenMP_CXX_FOUND}")
message(STATUS "OpenGL support: ${OPENGL_FOUND}")
message(STATUS "GLFW support: ${glfw3_FOUND}")
message(STATUS "GLEW support: ${GLEW_FOUND}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
