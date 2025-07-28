# N-Body Simulation Project

This project is a high-performance, modular N-body simulation framework designed for modern computing physics research and experimentation. It supports large-scale astrophysical simulations with GPU acceleration, interactive visualization, flexible initialization, and robust output options.
It's a codebase for future improvement and extensions. Work and presentation for the project of Modern Computing for Physics for MSc Physics of Data, UniPD.
You no not need presentation folder to run the project. In fact there is a lot of extra material here.

---

## Features

- **High Performance:**  
  - GPU acceleration via CUDA for millions of particles.
  - Optimized CPU execution with OpenMP parallelism.
  - Efficient memory management (pinned memory, device-host transfers).
  - Very low memory usage (~310MiB for 1M particle simulation, ~1.1GiB for 1B particles)

- **Flexible Physics Models:**  
  - Multiple force calculation methods: Pairwise, Adaptive Mutual Softening, Barnes-Hut.
  - Integration schemes: Euler, Velocity Verlet.
  - Configurable physical constants and softening parameters.

- **Rich Initialization Options:**  
  - Random particle distributions.
  - Realistic galaxy and stellar system generators.
  - Spiral galaxy models with configurable arms and perturbations.
  - Load initial conditions from file.

- **Modular Output:**  
  - Benchmark mode for performance testing.
  - CSV file output for analysis and plotting.
  - Real-time interactive visualization with OpenGL and CUDA interop.

- **Interactive Visualization:**  
  - GPU-accelerated rendering of particles.
  - Camera controls (rotation, pan, zoom).
  - Multiple color and size modes.
  - Keyboard and mouse interaction.

- **Extensibility & Maintainability:**  
  - Modular codebase (core, cuda, output, visualization).
  - Easy to add new force models, integration schemes, or output formats, so it's very easy to use this codebase for another physical model.
  - Only does gravitational many body but could be extended to other forms of long range interaction systems with the same kind of performances.
  - Can be extended to short range based models practically at the cost of writing the core function for Force interactions only.
  - Clear separation between simulation, output, and visualization logic.

---

## Requirements

- **Operating System:** Linux
- **Compiler:** GCC (C++17), NVCC (CUDA 11+)
- **Libraries:**  
  - CUDA Toolkit  
  - OpenMP  
  - OpenGL  
  - GLFW  
  - GLEW  
  - nlohmann/json (automatically fetched via CMake)
- **Hardware:**  
  - NVIDIA GPU with CUDA support (for GPU acceleration and visualization)

---

## Build Instructions

1. **Build the project:**
   - Standard build:
     ```bash
     ./build.sh
     ```
   - Clean previous build folder:
     ```bash
     ./build.sh clean
     ```

   The executable `n-body-simulation` will be placed in the project root.

2. **Manual build (alternative):**
  i expect you know how to read a cmake list if you are reading this, then good luck :)

---

## Usage Instructions

1. **Configure the simulation:**
   - Edit `config.json` to set initial conditions, simulation parameters, output options, and execution mode.
   - Example configuration:
     ```json
     {
       "init": { ... }, // see config.json for details
       "threads": 4, // set CPU threads to use
       "years": 10.0,
       "dtYears": 0.3,
       "output": {
         "dir": "data_out/",
         "file": "simulation_data_file_spiral_gpu.csv"
       },
       "integrationMethod": "VELOCITY_VERLET" //or EULER
       "forceMethod": "PAIRWISE", //or ADAPTIVE_MUTUAL, PAIRWISE_AVX2_FP32, BARNES_HUT (only CPU)
       "executionMode": "GPU", // or CPU
       "outputMode": "BENCHMARK" // or VISUALIZATION, FILE_CSV
     }
     ```

2. **Run the simulation:**
   - Standard execution:
     ```bash
     ./n-body-simulation
     ```
   - For direct GPU rendering (on hybrid graphics systems):
     ```bash
     __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./n-body-simulation
     ```

3. **Output:**
   - Results are saved in the specified output directory and file (CSV format) if the corresponding option is specified.
   - Visualization mode opens an interactive window for real-time exploration.
   - Benchmark mode only outputs relevant benchmark metrics.
   - Optionally you can create a 2d animation of a CSV output plus some plots to check if the energy/momentum is conserved, or you can trust me because I already did it.

---

## Project Strengths

- **Scalable:** Handles very large particle counts efficiently on modern GPUs.
- **Modular:** Easy to extend and maintain; clear separation of concerns.
- **Flexible:** Supports a wide range of physical models and initialization scenarios.
- **Interactive:** Real-time visualization and user controls for exploration.
- **Robust:** Error handling, progress feedback, and efficient output management.
- **Very little memory needed:** The overall base memory needed it's only a few MiBs higher than the theoretical one would need to compute the integration of the system, when scaling with larger number of particles you can do even ~billions of particles if your systems allow for it and you are fine with the time constraint of your specific hardware. Basically you are not memory limited in any case more than you are performance limited.
- **User friendly:** with almost all technical variables taken care of automatically besides thread count, this is my first project where one could use it without knowing anything about the code. It is not perfect (not a real GUI) and it lacks documentation but it's a nice starting point for a backend of a *real* usable scientific framework.

## Project Weaknesses

- **Platform Dependency:** Primarily tested on Linux; Windows/Mac support may require additional setup.
- **GPU Requirement:** Full performance and visualization require a CUDA-capable NVIDIA GPU.
- **Documentation:** While code is modular, some advanced features may require further documentation for new users.
- **Still slow with respect to-state-of-the-art libraries** While memory usage is minimal we are still limited to ~25k particles in real-time (~30fps with modern CPUs) so everything is slower than my initial goal (billion particles in real time). I believe that finishing up Barnes-Hut in CUDA with morton sorting and trasversal tree could achieve that, but I did not manage to do it. Also the visualization feature was my first ever try so it's very slow compared to the raw compute, in some cases even 10 times slower with modern hardware. I believe there is still a lot of vectorization and optimization to do in this department.
- **Not a real documentation:** Please note that all the other READMEs are all generated and while they were inspected by me they are readable at best.
- **Code fragmentation:** Finally the code is not organised as much as I'd hoped it would be, you have to blame my lack of vision for this project as well as my inexperience in project design. It's my first time doing a project of this size, without real research on best practices in anything other than pure algorithm optimization. Also some code, particularly the last benchmark blob in the main, is generated, so it's unrefined in contrast to the rest (here i was at the last straws of patience after a month going back and forth with BH-Kernels).

---

## Example Configuration

See `config.json` for a template and parameter descriptions.

---

## Further Information

For detailed architecture and code organization, see the README files in each subdirectory:
- [`simulator/core/README-core.md`](simulator/core/README-core.md)
- [`simulator/cuda/README-cuda.md`](simulator/cuda/README-cuda.md)
- [`simulator/output/README-output.md`](simulator/output/README-output.md)
- [`simulator/visualization/README-visualization.md`](simulator/visualization/README-visualization.md)

---

## License

GPLv3
