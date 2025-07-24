#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "System.hpp"  // For ForceMethod enum
#include "ForceKernels.cuh" 

// Velocity Verlet position update (first half)
__global__ void velocityVerletPositionKernel(double4* posMass, double4* vel, double4* accel, int n, double dt);

// Velocity Verlet velocity update (second half)
__global__ void velocityVerletVelocityKernel(double4* vel, double4* accelOld, double4* accelNew, int n, double dt);

// Euler integration kernel
__global__ void eulerIntegrationKernel(double4* posMass, double4* vel, double4* accel, int n, double dt);

// Optimized kernel to compute total energy (both kinetic and potential) in a single kernel
__global__ void computeTotalEnergyKernel(const double4* __restrict__ posMass, const double4* __restrict__ vel, double* totalEnergy, int n);

// Kernel to store particle and system state into output structure
__global__ void storeParticleStateKernel(double4* posMass, double4* vel, 
                                       double* particleData, double* systemData,
                                       double* totalEnergy, // Device pointer to total energy
                                       int particleCount, int timeStep, double time);

// Kernel to store static particle data (ID and mass)
__global__ void storeStaticDataKernel(double4* posMass, double* staticData, int particleCount);

extern "C" void performGPUIntegrationStep(
    double4* d_posMass, double4* d_vel, double4* d_accel, double4* d_accelOld,
    IntegrationMethod method, ForceMethod forceMethod,
    double dt, double avgMass, int n,
    dim3 blocks, int blockSize, size_t sharedMemSize,
    cudaStream_t stream);
