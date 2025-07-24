#pragma once

#include "Particles.hpp"
#include "System.hpp"
#include "OutputData.hpp"

/**
 * @brief Main CUDA simulation entry point
 * 
 * This function runs the entire N-body simulation on the GPU, with minimal
 * host-device data transfers for maximum performance.
 * 
 * @param particles Particle system data (positions, masses, velocities)
 * @param method Integration method to use (Euler or Velocity Verlet)
 * @param forceMethod Force calculation method (Pairwise, Adaptive, etc)
 * @param dt Time step size
 * @param steps Total number of integration steps to perform
 * @param stepFreq Frequency at which to save the system state
 * @param outputData Output data structure for saving simulation results
 */
extern "C" void runSimulationOnGPU(
    Particles& particles, 
    IntegrationMethod method,
    ForceMethod forceMethod,
    double dt, 
    int steps, 
    int stepFreq,
    OutputData& outputData
);

// Helper function to clean up GPU resources
void releaseGPUResources(cudaStream_t computeStream, cudaStream_t dataStream, cudaStream_t setupStream,
                        cudaEvent_t computeDone, cudaEvent_t dataReady, cudaEvent_t energyCalculated,
                        void* d_posMass, void* d_vel, void* d_accel, void* d_totalEnergy, void* d_accelOld);
