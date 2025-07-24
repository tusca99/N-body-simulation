#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "System.hpp"  // For ForceMethod enum
#include "CudaUtils.cuh"  // Include for KernelFunction typedef
#include "Constants.cuh"
//#include "BHKernels2.cuh"  // Include for FMMNode structure

//#include "SimpleBHKernels.cuh"

// Optimized CUDA kernel for force calculation without softening
__global__ void calculatePairwiseForceKernel(double4* posMass, double4* accel, int n);

// Optimized CUDA kernel for adaptive force calculation with Hill radius consideration
__global__ void calculateAdaptiveForcesKernel(double4* posMass, double4* accel, double avgMass, int n);


// Force kernel selector - a helper function to choose the right kernel based on ForceMethod
void launchForceKernel(ForceMethod method, dim3 blocks, int blockSize, size_t sharedMemSize,
                     cudaStream_t stream, double4* posMass, double4* accel, double avgMass, int n);



// Helper to get the appropriate kernel function pointer for occupancy calculation
// Change the function name to avoid overloading conflicts
void* getForceKernelPointer(ForceMethod method);