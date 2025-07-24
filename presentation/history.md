# N-Body Simulation: Design, Optimizations, and Analysis

## 1. Introduction and Motivation

This project started as a simple single-file C++ code to compute the trajectory of Voyager II together with the rest of the solar system. The initial goal was to create a functional and straightforward simulation, without focusing on modularity or extensibility. Over time, the project evolved into a highly modular and configurable N-body simulator, capable of running efficiently on both CPU and GPU, and supporting advanced features such as interactive visualization and benchmarking.

The main objectives of the project are:
- To provide a flexible and extensible framework for N-body simulations.
- To enable fair and reproducible performance comparisons between CPU and GPU implementations.
- To explore and implement various algorithmic and memory optimizations.
- To offer an interactive and high-performance visualization stack for large-scale simulations.

---

## 2. Project Evolution

The development followed an iterative and exploratory approach, with features and optimizations added progressively as new challenges and opportunities emerged.

**Key milestones:**
- **Initial version:**  
  Everything was implemented in a single file (`system.cpp`), including main logic, initialization, force calculation, and integration. A simple Python script was used for 2D animation of the simulation results.
- **First modularization:**  
  Output logic was separated into a dedicated module. An initial attempt at implementing Barnes-Hut on CPU was made to improve scalability.
- **Parallelization and data layout:**  
  OpenMP was introduced for multithreading. The data layout was switched from array-of-structures to structure-of-arrays (`double4`), and raw pointers were adopted for memory management.
- **CUDA integration:**  
  The first CUDA version was developed, with a clear separation between core and CUDA-specific code. Pinned memory was introduced for efficient host-device transfers.
- **Refactoring and configuration:**  
  Enum classes and dynamic method selection were added for force and integration algorithms. The build system was migrated to CMake, and configuration was moved from hardcoded variables to an external JSON file.
- **Visualization and benchmarking:**  
  An interactive visualization stack was implemented, supporting both CPU and GPU rendering. Benchmarking and CSV output were added, along with scripts for performance analysis and code diagrams.
- **Advanced optimizations:**  
  AVX2-optimized pairwise force calculation was introduced on CPU. Multiple attempts were made to implement Barnes-Hut on GPU, but the working version remains CPU-only.

---

## 3. General Architecture

The project is organized into several modular components, each responsible for a specific aspect of the simulation:

- **Core:**  
  Contains the main simulation logic, data structures (`Particles`), force and integration methods, and memory management.
- **CUDA:**  
  Implements GPU-accelerated kernels for force calculation, integration, and benchmarking. Includes utilities for automatic block size selection and efficient memory transfers.
- **Visualization:**  
  Provides an interactive visualization stack based on OpenGL and CUDA/OpenGL interop, supporting real-time rendering of large-scale simulations.
- **Benchmarks:**  
  Includes scripts and utilities for automated benchmarking, performance analysis, and result visualization.
- **Configuration:**  
  All simulation parameters are specified in a single JSON file, enabling easy switching between CPU and GPU modes and reproducible experiments.

A high-level diagram or Mermaid chart can be included here to illustrate the relationships between modules.

---

## 4. User Interface and Configuration

All simulation parameters are managed through a single JSON configuration file. This approach ensures a uniform interface for both CPU and GPU modes, hiding technical details such as CUDA block size or memory allocation strategies from the user.

**Example configuration:**
```json
{
  "num_particles": 10000,
  "execution_mode": "GPU",
  "force_method": "PAIRWISE",
  "integration_method": "VELOCITY_VERLET",
  "time_step": 0.001,
  "num_steps": 1000,
  "output_interval": 100,
  "visualization": true,
  "benchmark": false
}
```

This design philosophy prioritizes usability and reproducibility, allowing users to focus on scientific questions rather than low-level implementation details.

---

## 5. Memory Management and Optimizations

Several memory and algorithmic optimizations have been implemented to maximize performance and scalability:

- **Unified data structure (`double4`):**  
  All particle data (position, mass, velocity) is stored in arrays of `double4`, both on CPU and GPU. This ensures optimal memory alignment and layout for GPU coalescence, and improves cache efficiency on CPU.
- **Pinned memory for GPU:**  
  For large particle counts in GPU mode, pinned host memory (`cudaMallocHost`) is used to accelerate host-device transfers. Standard memory allocation is used for small allocations or CPU mode.
- **Automatic memory management:**  
  The `Particles` class transparently handles allocation, copying, and deallocation of memory, supporting both pinned and standard memory, and reducing the risk of memory leaks.
- **Automatic block size selection:**  
  The CUDA utilities include an automatic block size selection mechanism (Occupancy API), which is benchmarked against heuristic choices. Performance graphs and heatmaps are provided to compare these strategies.
- **Minimal data transfers:**  
  Particle data is copied only once at the beginning of the simulation and, if necessary, once at the end, minimizing transfer overhead.

---

Certo! Proseguo con le prossime sezioni, mantenendo lo stile e la struttura del documento.  
Qui trovi le sezioni su: **Force and Integration Methods**, **Visualization Stack**, **Benchmarking and Performance Analysis**, **Design Choices and Tradeoffs**, **Limitations and Lessons Learned**, **Conclusions**, e una bozza di **Appendix**.

---

## 6. Force and Integration Methods

The simulator supports multiple force calculation and integration methods, each optimized for different scenarios and hardware:

- **Pairwise (Direct Summation):**  
  Computes all pairwise gravitational interactions. Optimized on CPU with tiling (cache blocking), OpenMP parallelization, and an AVX2 SIMD version for significant speedup. On GPU, CUDA kernels leverage shared memory and memory coalescence for high throughput.

- **Adaptive Mutual Softening:**  
  Implements adaptive softening based on particle mass and distance, improving numerical stability in systems with large mass ratios.

- **Barnes-Hut (CPU only):**  
  A hierarchical tree-based algorithm for O(N log N) scaling. Multiple attempts were made to port this to GPU, but the working implementation is currently CPU-only due to complexity and debugging challenges.

- **Integration Methods:**  
  Supports both Euler and Velocity-Verlet integrators, selectable via configuration. The code structure allows easy extension to other integrators.

- **Dynamic Kernel Selection:**  
  On GPU, the appropriate kernel is selected at runtime based on the chosen force and integration method, with block size determined either heuristically or via the CUDA Occupancy API.

---

## 7. Visualization Stack

A modular and high-performance visualization stack enables interactive exploration of large-scale simulations:

- **OpenGL + CUDA Interop:**  
  On supported systems, CUDA kernels update OpenGL vertex buffers directly, allowing real-time rendering of millions of particles without unnecessary host-device transfers.

- **CPU Fallback:**  
  If GPU visualization is unavailable, a CPU-based renderer is used, ensuring portability.

- **User Controls:**  
  Camera movement, zoom, and simulation controls are available via keyboard and mouse. The visualization stack is modular, with separate components for rendering, camera control, and shader management.

- **Design Philosophy:**  
  The visualization system is designed to be decoupled from the simulation core, making it easy to extend or replace.

---

## 8. Benchmarking: Measuring and Comparing Performance

After implementing several versions of the N-body simulator—from the simplest single-threaded brute-force approach to more advanced parallel and GPU-based solutions—the next crucial step was to **systematically measure the performance** of each approach. This benchmarking phase is essential to truly understand the strengths and limitations of each solution, and to make informed decisions about which strategy to use for a given problem size or hardware setup.

### How I Set Up the Benchmarking

- **Multiple Configurations:**  
  For each test, I varied key parameters such as:
  - Number of particles (`N`)
  - Force calculation method (pairwise, AVX2, Barnes-Hut, etc.)
  - Integration method (Euler, Velocity Verlet, etc.)
  - Number of CPU threads
  - Execution mode (CPU/GPU)
- **Repeated Runs:**  
  Each configuration was executed multiple times (typically 10–15) to obtain reliable statistics and minimize the impact of random fluctuations due to OS scheduling or background processes.
- **Data Collection:**  
  For each run, I recorded:
  - Execution time
  - Steps per second (`StepsPerSecond`)
  - Particle-steps per second (`ParticleStepsPerSecond`)
  - Standard deviation and coefficient of variation (CV) to assess stability

### Analysis of Results

- **Aggregation and Statistics:**  
  All raw data are collected in CSV files and analyzed using Python (Pandas, Matplotlib, Seaborn). For each configuration, I compute:
  - Mean, standard deviation, min/max, and CV of the performance metrics
- **Method Comparison:**  
  The resulting plots (log-log, error bars, speedup, boxplots, etc.) allow direct comparison between:
  - CPU single-thread vs multi-thread
  - Standard CPU vs AVX2-optimized
  - CPU vs GPU
  - Barnes-Hut vs brute-force
- **Speedup and Scaling:**  
  I compute the speedup factor between configurations (e.g., GPU vs CPU single-thread) and analyze how performance scales with increasing `N` or thread count.
- **Parallel Efficiency:**  
  I evaluate how much actual speedup is achieved compared to the ideal (speedup/num_threads), which is crucial for understanding the effectiveness of parallelization.

### Example Plots and Insights

- **Log-log plots:**  
  Clearly show how algorithmic complexity (e.g., O(N²) vs O(N log N)) translates into real-world performance as `N` increases.
- **Error bar plots:**  
  Visualize the variability in performance and help assess the stability of each configuration.
- **Speedup plots:**  
  Make it easy to see at a glance when (and if) the GPU outperforms the CPU, or when AVX2 optimizations provide real benefits.
- **Boxplots and heatmaps:**  
  Useful for getting a global overview of performance across many configurations.

### What I Learned from Benchmarking

- The GPU only outperforms the single-threaded CPU above a certain particle count threshold.
- AVX2 optimizations provide significant speedups on CPU, but only for certain ranges of `N` and thread counts.
- The Barnes-Hut method, despite its O(N log N) complexity, is not always faster than brute-force GPU, especially for moderate `N`.
- The impact of CPU thread count on GPU performance is negligible (variation <5%), so I always use 1 CPU thread for GPU benchmarks to ensure consistency.
- Performance stability (low CV) is essential for reliable comparisons.

---

**In summary:**  
Benchmarking is not just about collecting numbers—it’s an experimental research phase that allows you to deeply understand the simulator’s behavior and choose the best solution for real-world problems.  
All analysis notebooks and CSVs are in the benchmarks folder and are designed to be easily reusable and updatable with new data or code versions.

## 9. Design Choices and Tradeoffs

Several important design decisions were made during development:

- **Unified data layout (`double4`):**  
  Using CUDA's `double4` type for both CPU and GPU ensures a fair comparison and simplifies memory transfers, but introduces a dependency on CUDA headers even for CPU-only builds.

- **Pinned memory and minimal transfers:**  
  Pinned memory is used only when beneficial, and data transfers are minimized to reduce overhead.

- **Automatic configuration:**  
  Most technical parameters (e.g., block size, memory allocation) are handled automatically, prioritizing usability and reproducibility.

- **Modularity vs portability:**  
  The codebase is highly modular, but some choices (e.g., CUDA types everywhere) reduce portability to non-CUDA systems.

- **Visualization decoupling:**  
  The visualization stack is designed to be independent from the simulation core, allowing for future extensions or replacements.

---

## 10. Limitations and Lessons Learned

- **CUDA dependency:**  
  The use of CUDA types and headers throughout the codebase means the project cannot be built or run on systems without CUDA.

- **Barnes-Hut on GPU:**  
  Despite multiple attempts, a stable and efficient GPU implementation of Barnes-Hut was not achieved.

- **Benchmark logic:**  
  Some benchmarking and output logic remains in the main function and could be further modularized.

- **Future improvements:**  
  - Refactor to better separate CPU and GPU code.
  - Replace CUDA-specific types with portable alternatives where possible.
  - Further modularize benchmarking and output routines.
  - Explore new algorithms and integrators.

---

## 11. Conclusions

The project evolved from a simple script into a complete, modular, and high-performance N-body simulation framework.  
It supports advanced features such as GPU acceleration, AVX2 optimizations, interactive visualization, and automated benchmarking.  
While some limitations remain, the codebase provides a solid foundation for further research and development in computational physics and high-performance simulation.

---

## 12. Appendix

- **Example configuration files**
- **Sample benchmark commands**
- **Keyboard controls for visualization**
- **References and further reading**
- **Code snippets for key optimizations**

---


