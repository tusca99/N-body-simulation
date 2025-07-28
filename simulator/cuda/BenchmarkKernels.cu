#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>  // Add this for timing
#include "CudaUtils.cuh"
#include "ForceKernels.cuh"
#include "IntegrationKernels.cuh"
#include "Particles.hpp"
#include "System.hpp"
#include "OutputUtils.hpp"  // Include for printProgressBar
#include "BenchmarkKernels.h"  // Include the header
#include "Constants.cuh"

// Run benchmark simulation with minimal overhead (no energy calculation, no data transfers)
extern "C" void runBenchmarkOnGPU(Particles& particles, 
                                IntegrationMethod method,
                                ForceMethod forceMethod,
                                double dt, int steps, int BLOCK_SIZE = 256) {
#ifdef BENCHMARK_MODE
    int n = particles.n;
    //std::cout << "Starting GPU benchmark with " << n << " particles for " << steps << " steps..." << std::endl;
    
    // Initialize CUDA for simulation
    double4 *d_posMass = nullptr, *d_vel = nullptr, *d_accel = nullptr;
    cudaStream_t computeStream;
    //bool useOccupancyAPI = true;  // Use occupancy API for block size determination
    // Create CUDA stream
    gpuErrchk(cudaStreamCreate(&computeStream));
    // Allocate memory on the device
    gpuErrchk(cudaMalloc(&d_posMass, n * sizeof(double4)));
    gpuErrchk(cudaMalloc(&d_vel, n * sizeof(double4)));
    gpuErrchk(cudaMalloc(&d_accel, n * sizeof(double4)));
    
    double4* d_accelOld = nullptr;
    if (method == IntegrationMethod::VELOCITY_VERLET) {
        gpuErrchk(cudaMalloc(&d_accelOld, n * sizeof(double4)));
    }
    
    // Copy data to device
    gpuErrchk(cudaMemcpy(d_posMass, particles.posMass, n * sizeof(double4), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_vel, particles.vel, n * sizeof(double4), cudaMemcpyHostToDevice));
    
    // Calculate initial forces

    //int blockSize = determineOptimalBlockSize(n, sizeof(double4) * 32);
    // If you ever want to try Occupancy API again, you can call it with the useOccupancyAPI flag:
    if (BLOCK_SIZE <= 0) {
        //std::cerr << "Invalid BLOCK_SIZE specified. Using auto block size function" << std::endl;
        BLOCK_SIZE = determineOptimalBlockSize(n, sizeof(double4) * 32, (KernelFunction)getForceKernelPointer(forceMethod), false);
    }
    // Use the provided BLOCK_SIZE or determine it dynamically
    int blockSize = BLOCK_SIZE;
    dim3 blocks = calculateGrid(n, blockSize);
    size_t sharedMemSize = blockSize * sizeof(double4);

    initializeConstants();  // Initialize device constants
    
    // Calculate average mass for adaptive methods
    double avgMass = 0.0;
    if (forceMethod == ForceMethod::ADAPTIVE_MUTUAL || forceMethod == ForceMethod::BARNES_HUT) {
        for (int i = 0; i < n; i++) {
            avgMass += particles.posMass[i].w;
        }
        avgMass /= n;
    }
    
    // Launch force kernel
    launchForceKernel(forceMethod, blocks, blockSize, sharedMemSize, computeStream, 
                    d_posMass, d_accel, avgMass, n);
    gpuErrchk(cudaStreamSynchronize(computeStream));
    
    // Main simulation loop - timed for benchmark
    //auto startTime = std::chrono::high_resolution_clock::now();
    
    for (int currentStep = 0; currentStep < steps; currentStep++) {
        // Integration step
        performGPUIntegrationStep(
            d_posMass, d_vel, d_accel, d_accelOld,
            method, forceMethod,
            dt, avgMass, n,
            blocks, blockSize, sharedMemSize,
            computeStream);
        // Use printProgressBar for consistent progress display
        if (currentStep % 10 == 0) { // Update less frequently to reduce overhead
            cudaStreamSynchronize(computeStream); // Ensure GPU is in sync when displaying progress
            //printProgressBar(currentStep, steps);
        }
    }
    
    // Wait for all operations to complete
    gpuErrchk(cudaStreamSynchronize(computeStream));
    //printProgressBar(steps, steps); // Ensure 100% progress is shown
    //std::cout << std::endl;
    
    //auto endTime = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> totalTime = endTime - startTime;
    //double stepsPerSecond = steps / totalTime.count();
    //double particleStepsPerSecond = stepsPerSecond * n;

    // Copy back final positions and velocities to host
    gpuErrchk(cudaMemcpy(particles.posMass, d_posMass, n * sizeof(double4), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(particles.vel, d_vel, n * sizeof(double4), cudaMemcpyDeviceToHost));
    
    // Free device memory
    gpuErrchk(cudaFree(d_posMass));
    gpuErrchk(cudaFree(d_vel));
    gpuErrchk(cudaFree(d_accel));
    if (method == IntegrationMethod::VELOCITY_VERLET && d_accelOld) {
        gpuErrchk(cudaFree(d_accelOld));
    }
    
    // Destroy stream
    gpuErrchk(cudaStreamDestroy(computeStream));
    
    //std::cout << "GPU-side Benchmark completed: " << steps << " steps in " 
    //          << totalTime.count() << " seconds" << std::endl;
    //std::cout << "Performance: " << stepsPerSecond << " steps/sec" << std::endl;
    //std::cout << "             " << particleStepsPerSecond / 1e6 << " million particle-steps/second" << std::endl;
#else
    // Stub implementation when benchmark mode is disabled
    std::cout << "GPU benchmark mode is not enabled. Please rebuild with ENABLE_BENCHMARK=ON." << std::endl;
    
    // Just run one step with regular simulation to avoid link errors
    runSimulationOnGPU(particles, method, forceMethod, dt, 1, 1, OutputData(particles.n, 1, ExecutionMode::GPU));
#endif
}
