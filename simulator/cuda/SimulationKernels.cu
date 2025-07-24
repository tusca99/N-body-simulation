#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "CudaUtils.cuh"
#include "ForceKernels.cuh"
#include "IntegrationKernels.cuh"
#include "SimulationKernels.h"
#include "Constants.cuh"

// Keep device constant values - good for performance
__constant__ double G_CUDA;
__constant__ double SOFTENING_SQUARED;
__constant__ double PI_CUDA;

// Simplify helper function declarations
std::string getDevicePropertiesAsync(cudaStream_t stream);
void releaseGPUResources(cudaStream_t computeStream, cudaStream_t dataStream, cudaStream_t setupStream,
                        cudaEvent_t computeDone, cudaEvent_t dataReady, cudaEvent_t energyCalculated,
                        void* d_posMass, void* d_vel, void* d_accel, void* d_totalEnergy, void* d_accelOld);

// Main entry point for GPU simulation
extern "C" void runSimulationOnGPU(Particles& particles, 
                                  IntegrationMethod method,
                                  ForceMethod forceMethod,
                                  double dt, int steps, int stepFreq,
                                  OutputData& outputData) {
    
    int n = particles.n;
    std::cout << "Starting GPU simulation with " << n << " particles..." << std::endl;
    
    // Check if CUDA is available
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        std::cerr << "CUDA error: " << (error != cudaSuccess ? 
            cudaGetErrorString(error) : "No CUDA-capable devices found") << std::endl;
        throw std::runtime_error("CUDA initialization failed");
    }
    
    // Get current device information
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    if (props.major < 2) {
        std::cerr << "Error: This simulation requires at least compute capability 2.0" << std::endl;
        throw std::runtime_error("GPU too old for simulation");
    }
    
    std::cout << "Using CUDA device: " << props.name << " (CC " << props.major << "." << props.minor << ")" << std::endl;
    
    // Create CUDA streams for overlapping operations
    cudaStream_t computeStream = nullptr, dataStream = nullptr, setupStream = nullptr;
    cudaEvent_t computeDone = nullptr, dataReady = nullptr, energyCalculated = nullptr;
    
    // Simplify stream and event creation with a try-catch block
    try {
        gpuErrchk(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
        gpuErrchk(cudaStreamCreateWithFlags(&dataStream, cudaStreamNonBlocking));
        gpuErrchk(cudaStreamCreateWithFlags(&setupStream, cudaStreamNonBlocking));
        gpuErrchk(cudaEventCreate(&computeDone));
        gpuErrchk(cudaEventCreate(&dataReady));
        gpuErrchk(cudaEventCreate(&energyCalculated));
    } catch (const std::exception& e) {
        releaseGPUResources(computeStream, dataStream, setupStream, 
                          computeDone, dataReady, energyCalculated, 
                          nullptr, nullptr, nullptr, nullptr, nullptr);
        throw;
    }
    
    // Start device property query in separate stream
    std::string deviceInfo = getDevicePropertiesAsync(setupStream);
    
    // Allocate device memory - simplified approach
    double4 *d_posMass = nullptr, *d_vel = nullptr, *d_accel = nullptr, *d_accelOld = nullptr;
    double *d_totalEnergy = nullptr;
    bool usingAsyncAlloc = false;
    
    try {
        #if CUDART_VERSION >= 11020
        // Check if device supports async allocations
        int driverVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        
        if (driverVersion >= 11020 && props.major >= 7) {
            usingAsyncAlloc = true;
            std::cout << "Using asynchronous memory operations" << std::endl;
            
            // Try asynchronous allocations
            cudaError_t err = cudaMallocAsync(&d_posMass, n * sizeof(double4), computeStream);
            if (err == cudaSuccess) {
                gpuErrchk(cudaMallocAsync(&d_vel, n * sizeof(double4), computeStream));
                gpuErrchk(cudaMallocAsync(&d_accel, n * sizeof(double4), computeStream));
                gpuErrchk(cudaMallocAsync(&d_totalEnergy, sizeof(double), computeStream));
                
                if (method == IntegrationMethod::VELOCITY_VERLET) {
                    gpuErrchk(cudaMallocAsync(&d_accelOld, n * sizeof(double4), computeStream));
                }
            } else {
                usingAsyncAlloc = false;
            }
        }
        #endif
        
        // Fall back to standard allocation if needed
        if (!usingAsyncAlloc) {
            gpuErrchk(cudaMalloc(&d_posMass, n * sizeof(double4)));
            gpuErrchk(cudaMalloc(&d_vel, n * sizeof(double4)));
            gpuErrchk(cudaMalloc(&d_accel, n * sizeof(double4)));
            gpuErrchk(cudaMalloc(&d_totalEnergy, sizeof(double)));
            
            if (method == IntegrationMethod::VELOCITY_VERLET) {
                gpuErrchk(cudaMalloc(&d_accelOld, n * sizeof(double4)));
            }
        }
    } catch (const std::exception& e) {
        releaseGPUResources(computeStream, dataStream, setupStream, 
                          computeDone, dataReady, energyCalculated, 
                          d_posMass, d_vel, d_accel, d_totalEnergy, d_accelOld);
        throw;
    }
    
    // Calculate average mass for adaptive methods
    double avgMass = 0.0;
    if (forceMethod == ForceMethod::ADAPTIVE_MUTUAL) {
        for (int i = 0; i < n; ++i) {
            avgMass += particles.posMass[i].w;
        }
        avgMass /= n;
    }
    
    // Copy initial particle data to device
    gpuErrchk(cudaMemcpyAsync(d_posMass, particles.posMass, 
                          n * sizeof(double4), cudaMemcpyHostToDevice, computeStream));
    gpuErrchk(cudaMemcpyAsync(d_vel, particles.vel, 
                          n * sizeof(double4), cudaMemcpyHostToDevice, computeStream));
    
    // Record when data is ready
    gpuErrchk(cudaEventRecord(dataReady, computeStream));
    
    // Wait for setup to complete and display info
    gpuErrchk(cudaStreamSynchronize(setupStream));
    std::cout << deviceInfo << std::endl;
    
    // Determine optimal block size
    size_t sharedMemSize = sizeof(double4) * 32;
    int blockSize = determineOptimalBlockSize(n, sharedMemSize);
    dim3 blocks = calculateGrid(n, blockSize);
    
    std::cout << "Using block size: " << blockSize << ", grid size: " << blocks.x << std::endl;
    
    // Initialize constants
    initializeConstants();
    
    // Store static data
    storeStaticDataKernel<<<blocks, blockSize, 0, dataStream>>>(
        d_posMass, outputData.getDeviceStaticDataPtr(), n);
    gpuErrchk(cudaGetLastError());
    
    // Wait for data to be on device
    gpuErrchk(cudaStreamWaitEvent(computeStream, dataReady));
    
    // Calculate initial forces
    sharedMemSize = blockSize * sizeof(double4);
    launchForceKernel(forceMethod, blocks, blockSize, sharedMemSize, computeStream, 
                     d_posMass, d_accel, avgMass, n);
    gpuErrchk(cudaGetLastError());
    
    // Calculate initial energy
    gpuErrchk(cudaMemsetAsync(d_totalEnergy, 0, sizeof(double), computeStream));
    size_t energySharedMem = blockSize * sizeof(double) * 5;
    computeTotalEnergyKernel<<<blocks, blockSize, energySharedMem, computeStream>>>(
        d_posMass, d_vel, d_totalEnergy, n);
    gpuErrchk(cudaGetLastError());
    
    // Record energy calculation completion
    gpuErrchk(cudaEventRecord(energyCalculated, computeStream));
    gpuErrchk(cudaStreamWaitEvent(dataStream, energyCalculated));
    
    // Store initial state
    storeParticleStateKernel<<<blocks, blockSize, 0, dataStream>>>(
        d_posMass, d_vel, 
        outputData.d_particleData, outputData.d_systemData,
        d_totalEnergy,
        n, 0, 0.0);
    
    // Main simulation loop
    int printCounter = 0;
    size_t timeIdx = 1;
    
    for (int step = 1; step <= steps; ++step) {
        double time = step * dt;
        
        performGPUIntegrationStep(
            d_posMass, d_vel, d_accel, d_accelOld,
            method, forceMethod,
            dt, avgMass, n,
            blocks, blockSize, sharedMemSize,
            computeStream);
        
        // Store state periodically
        if (++printCounter >= stepFreq) {
            gpuErrchk(cudaMemsetAsync(d_totalEnergy, 0, sizeof(double), computeStream));
            computeTotalEnergyKernel<<<blocks, blockSize, energySharedMem, computeStream>>>(
                d_posMass, d_vel, d_totalEnergy, n);
            gpuErrchk(cudaGetLastError());
            
            gpuErrchk(cudaEventRecord(energyCalculated, computeStream));
            gpuErrchk(cudaStreamWaitEvent(dataStream, energyCalculated));
            
            storeParticleStateKernel<<<blocks, blockSize, 0, dataStream>>>(
                d_posMass, d_vel,
                outputData.d_particleData, outputData.d_systemData,
                d_totalEnergy,
                n, timeIdx, time);
            
            int progressPercent = (step * 100) / steps;
            printf("\rProgress: %d%%", progressPercent);
            fflush(stdout);
            
            timeIdx++;
            printCounter = 0;
        }
    }
    
    // Final synchronization
    gpuErrchk(cudaEventRecord(computeDone, computeStream));
    gpuErrchk(cudaEventSynchronize(computeDone));
    printf("\rProgress: 100%%\n");
    
    // Transfer data
    gpuErrchk(cudaMemcpyAsync(outputData.h_particleData, outputData.d_particleData, 
                           outputData.numParticles * outputData.numTimeSteps * outputData.valuesPerParticle * sizeof(double), 
                           cudaMemcpyDeviceToHost, dataStream));
    
    gpuErrchk(cudaMemcpyAsync(outputData.h_systemData, outputData.d_systemData, 
                           outputData.numTimeSteps * outputData.valuesPerSystem * sizeof(double), 
                           cudaMemcpyDeviceToHost, dataStream));
    
    gpuErrchk(cudaMemcpyAsync(outputData.h_staticData, outputData.d_staticData, 
                           outputData.numParticles * outputData.valuesPerStatic * sizeof(double), 
                           cudaMemcpyDeviceToHost, dataStream));
    
    gpuErrchk(cudaMemcpyAsync(particles.posMass, d_posMass, n * sizeof(double4), 
                           cudaMemcpyDeviceToHost, dataStream));
    gpuErrchk(cudaMemcpyAsync(particles.vel, d_vel, n * sizeof(double4), 
                           cudaMemcpyDeviceToHost, dataStream));
    
    // Wait for all transfers to complete
    gpuErrchk(cudaStreamSynchronize(dataStream));
    
    // Free resources using the appropriate method
    #if CUDART_VERSION >= 11020
    if (usingAsyncAlloc) {
        if (d_posMass) cudaFreeAsync(d_posMass, computeStream);
        if (d_vel) cudaFreeAsync(d_vel, computeStream);
        if (d_accel) cudaFreeAsync(d_accel, computeStream);
        if (d_totalEnergy) cudaFreeAsync(d_totalEnergy, computeStream);
        if (method == IntegrationMethod::VELOCITY_VERLET && d_accelOld)
            cudaFreeAsync(d_accelOld, computeStream);
        cudaStreamSynchronize(computeStream);
    } else {
    #endif
        if (d_posMass) cudaFree(d_posMass);
        if (d_vel) cudaFree(d_vel);
        if (d_accel) cudaFree(d_accel);
        if (d_totalEnergy) cudaFree(d_totalEnergy);
        if (method == IntegrationMethod::VELOCITY_VERLET && d_accelOld)
            cudaFree(d_accelOld);
    #if CUDART_VERSION >= 11020
    }
    #endif
    
    // Cleanup events and streams
    if (computeDone) cudaEventDestroy(computeDone);
    if (dataReady) cudaEventDestroy(dataReady);
    if (energyCalculated) cudaEventDestroy(energyCalculated);
    
    if (computeStream) cudaStreamDestroy(computeStream);
    if (dataStream) cudaStreamDestroy(dataStream);
    if (setupStream) cudaStreamDestroy(setupStream);
    
    std::cout << "GPU simulation completed successfully." << std::endl;
}

// Simplified helper function implementations
std::string getDevicePropertiesAsync(cudaStream_t stream) {
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    std::stringstream ss;
    ss << "Compute capability: " << prop.major << "." << prop.minor << "\n"
       << "Global memory: " << prop.totalGlobalMem / (1024*1024) << " MB\n"
       << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB";
       
    return ss.str();
}

void releaseGPUResources(cudaStream_t computeStream, cudaStream_t dataStream, cudaStream_t setupStream,
                        cudaEvent_t computeDone, cudaEvent_t dataReady, cudaEvent_t energyCalculated,
                        void* d_posMass, void* d_vel, void* d_accel, void* d_totalEnergy, void* d_accelOld) {
    
    // Free device memory
    if (d_posMass) cudaFree(d_posMass);
    if (d_vel) cudaFree(d_vel);
    if (d_accel) cudaFree(d_accel);
    if (d_totalEnergy) cudaFree(d_totalEnergy);
    if (d_accelOld) cudaFree(d_accelOld);
    
    // Destroy events
    if (computeDone) cudaEventDestroy(computeDone);
    if (dataReady) cudaEventDestroy(dataReady);
    if (energyCalculated) cudaEventDestroy(energyCalculated);
    
    // Destroy streams
    if (computeStream) cudaStreamDestroy(computeStream);
    if (dataStream) cudaStreamDestroy(dataStream);
    if (setupStream) cudaStreamDestroy(setupStream);
}