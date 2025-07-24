#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstring>
#include <sstream>
#include "ExecutionModes.hpp" // Include the execution modes directly

// Structure to hold simulation output data in a GPU-friendly format
struct OutputData {
    // Particle data arrays - one value per particle per timestep
    double* h_particleData = nullptr;  // Host particle data array (x,y,z)
    double* d_particleData = nullptr;  // Device particle data array
    static constexpr size_t valuesPerParticle = 6;  // particle_id, x, y, z, vx, vy
    
    // System data arrays - one value per timestep
    double* h_systemData = nullptr;   // Host system data array (time, energy)
    double* d_systemData = nullptr;   // Device system data array
    static constexpr size_t valuesPerSystem = 2;   // time, energy
    
    // Static data that doesn't change during simulation
    double* h_staticData = nullptr;   // Host static data (particle_id, mass)
    double* d_staticData = nullptr;   // Device static data
    static constexpr size_t valuesPerStatic = 2;  // particle_id, mass
    
    bool isPinned = false;     // Flag to track if we're using pinned memory
    size_t numParticles = 0;
    size_t numTimeSteps = 0;
    ExecutionMode mode = ExecutionMode::CPU;  // Store the execution mode
    
    OutputData() = default;

    // Initialize with expected size and allocate memory based on execution mode
    OutputData(size_t particles, size_t timeSteps, ExecutionMode execMode = ExecutionMode::CPU) 
        : numParticles(particles), numTimeSteps(timeSteps), mode(execMode) {
        
        size_t particleDataSize = particles * timeSteps * valuesPerParticle * sizeof(double);
        size_t systemDataSize = timeSteps * valuesPerSystem * sizeof(double);
        size_t staticDataSize = particles * valuesPerStatic * sizeof(double);
        
        if (mode == ExecutionMode::GPU) {
            // Try to use pinned memory for GPU mode
            cudaError_t err1 = cudaMallocHost(&h_particleData, particleDataSize);
            cudaError_t err2 = cudaMallocHost(&h_systemData, systemDataSize);
            cudaError_t err3 = cudaMallocHost(&h_staticData, staticDataSize);
            
            if (err1 == cudaSuccess && err2 == cudaSuccess && err3 == cudaSuccess) {
                isPinned = true;
                // Only zero the first timestep
                memset(h_particleData, 0, particles * valuesPerParticle * sizeof(double));
                memset(h_systemData, 0, valuesPerSystem * sizeof(double));
                memset(h_staticData, 0, staticDataSize);
            } else {
                // Fall back to regular memory
                if (err1 == cudaSuccess) cudaFreeHost(h_particleData);
                if (err2 == cudaSuccess) cudaFreeHost(h_systemData);
                if (err3 == cudaSuccess) cudaFreeHost(h_staticData);
                
                h_particleData = new double[particles * timeSteps * valuesPerParticle]();
                h_systemData = new double[timeSteps * valuesPerSystem]();
                h_staticData = new double[particles * valuesPerStatic]();
                isPinned = false;
                
                // Only zero the first timestep
                memset(h_particleData, 0, particles * valuesPerParticle * sizeof(double));
                memset(h_systemData, 0, valuesPerSystem * sizeof(double));
                memset(h_staticData, 0, staticDataSize);
            }
            
            // For GPU mode, allocate device memory using asynchronous allocation if possible
            cudaStream_t initStream;
            cudaStreamCreate(&initStream);
            
            cudaError_t err4 = cudaMalloc(&d_particleData, particleDataSize);
            if (err4 == cudaSuccess) {
                cudaMemsetAsync(d_particleData, 0, particles * valuesPerParticle * sizeof(double), initStream);
            }
            
            cudaError_t err5 = cudaMalloc(&d_systemData, systemDataSize);
            if (err5 == cudaSuccess) {
                cudaMemsetAsync(d_systemData, 0, valuesPerSystem * sizeof(double), initStream);
            }
            
            cudaError_t err6 = cudaMalloc(&d_staticData, staticDataSize);
            if (err6 == cudaSuccess) {
                cudaMemsetAsync(d_staticData, 0, staticDataSize, initStream);
            }
            
            // Check if any allocations failed
            if (err4 != cudaSuccess || err5 != cudaSuccess || err6 != cudaSuccess) {
                std::cerr << "CUDA memory allocation error" << std::endl;
                // Clean up any successful allocations
                if (err4 == cudaSuccess) cudaFree(d_particleData);
                if (err5 == cudaSuccess) cudaFree(d_systemData);
                if (err6 == cudaSuccess) cudaFree(d_staticData);
                
                d_particleData = nullptr;
                d_systemData = nullptr;
                d_staticData = nullptr;
            }
            
            // Clean up the stream
            cudaStreamDestroy(initStream);
        } else {
            // For CPU mode
            h_particleData = new double[particles * timeSteps * valuesPerParticle]();
            // Only zero out the first timestep
            std::fill_n(h_particleData, particles * valuesPerParticle, 0.0);
            
            h_systemData = new double[timeSteps * valuesPerSystem]();
            std::fill_n(h_systemData, valuesPerSystem, 0.0);
            
            h_staticData = new double[particles * valuesPerStatic]();
            std::fill_n(h_staticData, particles * valuesPerStatic, 0.0);
            
            isPinned = false;
            
            d_particleData = nullptr;
            d_systemData = nullptr;
            d_staticData = nullptr;
        }
    }

    // Destructor - free both host and device memory
    ~OutputData() {
        if (h_particleData) {
            if (isPinned) cudaFreeHost(h_particleData);
            else delete[] h_particleData;
        }
        
        if (h_systemData) {
            if (isPinned) cudaFreeHost(h_systemData);
            else delete[] h_systemData;
        }
        
        if (h_staticData) {
            if (isPinned) cudaFreeHost(h_staticData);
            else delete[] h_staticData;
        }
        
        // Free device memory if allocated (GPU mode)
        if (d_particleData) cudaFree(d_particleData);
        if (d_systemData) cudaFree(d_systemData);
        if (d_staticData) cudaFree(d_staticData);
    }

    // No copy operations
    OutputData(const OutputData&) = delete;
    OutputData& operator=(const OutputData&) = delete;

    // Move operations
    OutputData(OutputData&& other) noexcept 
        : h_particleData(other.h_particleData), d_particleData(other.d_particleData),
          h_systemData(other.h_systemData), d_systemData(other.d_systemData),
          h_staticData(other.h_staticData), d_staticData(other.d_staticData),
          isPinned(other.isPinned),
          numParticles(other.numParticles), numTimeSteps(other.numTimeSteps),
          mode(other.mode) {
        
        other.h_particleData = nullptr;
        other.d_particleData = nullptr;
        other.h_systemData = nullptr;
        other.d_systemData = nullptr;
        other.h_staticData = nullptr;
        other.d_staticData = nullptr;
        other.isPinned = false;
        other.numParticles = 0;
        other.numTimeSteps = 0;
    }

    OutputData& operator=(OutputData&& other) noexcept {
        if (this != &other) {
            // Free existing resources
            if (h_particleData) {
                if (isPinned) cudaFreeHost(h_particleData);
                else delete[] h_particleData;
            }
            if (h_systemData) {
                if (isPinned) cudaFreeHost(h_systemData);
                else delete[] h_systemData;
            }
            if (h_staticData) {
                if (isPinned) cudaFreeHost(h_staticData);
                else delete[] h_staticData;
            }
            
            if (d_particleData) cudaFree(d_particleData);
            if (d_systemData) cudaFree(d_systemData);
            if (d_staticData) cudaFree(d_staticData);
            
            // Move resources from other
            h_particleData = other.h_particleData;
            d_particleData = other.d_particleData;
            h_systemData = other.h_systemData;
            d_systemData = other.d_systemData;
            h_staticData = other.h_staticData;
            d_staticData = other.d_staticData;
            isPinned = other.isPinned;
            numParticles = other.numParticles;
            numTimeSteps = other.numTimeSteps;
            mode = other.mode;
            
            // Clear other
            other.h_particleData = nullptr;
            other.d_particleData = nullptr;
            other.h_systemData = nullptr;
            other.d_systemData = nullptr;
            other.h_staticData = nullptr;
            other.d_staticData = nullptr;
            other.isPinned = false;
            other.numParticles = 0;
            other.numTimeSteps = 0;
        }
        return *this;
    }

    // Set the execution mode and potentially reallocate memory
    void setExecutionMode(ExecutionMode newMode) {
        if (mode == newMode) return; // No change needed
        
        // If switching from CPU to GPU, we need to allocate device memory
        if (newMode == ExecutionMode::GPU && mode == ExecutionMode::CPU) {
            size_t particleDataSize = numParticles * numTimeSteps * valuesPerParticle * sizeof(double);
            size_t systemDataSize = numTimeSteps * valuesPerSystem * sizeof(double);
            size_t staticDataSize = numParticles * valuesPerStatic * sizeof(double);
            
            // Save current data
            double* tempParticleData = new double[numParticles * numTimeSteps * valuesPerParticle];
            double* tempSystemData = new double[numTimeSteps * valuesPerSystem];
            double* tempStaticData = new double[numParticles * valuesPerStatic];
            
            memcpy(tempParticleData, h_particleData, particleDataSize);
            memcpy(tempSystemData, h_systemData, systemDataSize);
            memcpy(tempStaticData, h_staticData, staticDataSize);
            
            // Free existing host memory
            delete[] h_particleData;
            delete[] h_systemData;
            delete[] h_staticData;
            
            // Try to allocate pinned memory
            cudaError_t err1 = cudaMallocHost(&h_particleData, particleDataSize);
            cudaError_t err2 = cudaMallocHost(&h_systemData, systemDataSize);
            cudaError_t err3 = cudaMallocHost(&h_staticData, staticDataSize);
            
            if (err1 == cudaSuccess && err2 == cudaSuccess && err3 == cudaSuccess) {
                isPinned = true;
                
                // Copy data back
                memcpy(h_particleData, tempParticleData, particleDataSize);
                memcpy(h_systemData, tempSystemData, systemDataSize);
                memcpy(h_staticData, tempStaticData, staticDataSize);
            } else {
                // Fall back to regular memory
                if (err1 == cudaSuccess) cudaFreeHost(h_particleData);
                if (err2 == cudaSuccess) cudaFreeHost(h_systemData);
                if (err3 == cudaSuccess) cudaFreeHost(h_staticData);
                
                h_particleData = tempParticleData;
                h_systemData = tempSystemData;
                h_staticData = tempStaticData;
                
                // Don't delete the temp arrays since they're now the main arrays
                tempParticleData = nullptr;
                tempSystemData = nullptr;
                tempStaticData = nullptr;
                isPinned = false;
            }
            
            // Allocate device memory
            cudaError_t err4 = cudaMalloc(&d_particleData, particleDataSize);
            cudaError_t err5 = cudaMalloc(&d_systemData, systemDataSize);
            cudaError_t err6 = cudaMalloc(&d_staticData, staticDataSize);
            
            if (err4 != cudaSuccess || err5 != cudaSuccess || err6 != cudaSuccess) {
                std::cerr << "CUDA memory allocation error during mode change" << std::endl;
                // Clean up any successful device allocations
                if (err4 == cudaSuccess) cudaFree(d_particleData);
                if (err5 == cudaSuccess) cudaFree(d_systemData);
                if (err6 == cudaSuccess) cudaFree(d_staticData);
                
                d_particleData = nullptr;
                d_systemData = nullptr;
                d_staticData = nullptr;
            } else {
                // Initialize device memory to zeros
                cudaMemset(d_particleData, 0, particleDataSize);
                cudaMemset(d_systemData, 0, systemDataSize);
                cudaMemset(d_staticData, 0, staticDataSize);
                
                // Copy data to device
                cudaMemcpy(d_particleData, h_particleData, particleDataSize, cudaMemcpyHostToDevice);
                cudaMemcpy(d_systemData, h_systemData, systemDataSize, cudaMemcpyHostToDevice);
                cudaMemcpy(d_staticData, h_staticData, staticDataSize, cudaMemcpyHostToDevice);
            }
            
            // Clean up temporary arrays if they weren't used
            if (tempParticleData) delete[] tempParticleData;
            if (tempSystemData) delete[] tempSystemData;
            if (tempStaticData) delete[] tempStaticData;
        } 
        // If switching from GPU to CPU, we need to free device memory
        else if (newMode == ExecutionMode::CPU && mode == ExecutionMode::GPU) {
            // Free device memory
            if (d_particleData) {
                cudaFree(d_particleData);
                d_particleData = nullptr;
            }
            
            if (d_systemData) {
                cudaFree(d_systemData);
                d_systemData = nullptr;
            }
            
            if (d_staticData) {
                cudaFree(d_staticData);
                d_staticData = nullptr;
            }
            
            // If using pinned host memory, convert to regular memory
            if (isPinned) {
                size_t particleDataSize = numParticles * numTimeSteps * valuesPerParticle * sizeof(double);
                size_t systemDataSize = numTimeSteps * valuesPerSystem * sizeof(double);
                size_t staticDataSize = numParticles * valuesPerStatic * sizeof(double);
                
                // Save current data
                double* tempParticleData = new double[numParticles * numTimeSteps * valuesPerParticle];
                double* tempSystemData = new double[numTimeSteps * valuesPerSystem];
                double* tempStaticData = new double[numParticles * valuesPerStatic];
                
                memcpy(tempParticleData, h_particleData, particleDataSize);
                memcpy(tempSystemData, h_systemData, systemDataSize);
                memcpy(tempStaticData, h_staticData, staticDataSize);
                
                // Free pinned memory
                cudaFreeHost(h_particleData);
                cudaFreeHost(h_systemData);
                cudaFreeHost(h_staticData);
                
                // Use regular memory
                h_particleData = tempParticleData;
                h_systemData = tempSystemData;
                h_staticData = tempStaticData;
                isPinned = false;
            }
        }
        
        // Update mode
        mode = newMode;
    }

    // Set static data (particle ID and mass) - only called once
    void setStaticData(size_t particleIdx, double id, double mass) {
        if (particleIdx >= numParticles) {
            std::cerr << "OutputData::setStaticData - Index out of bounds\n";
            return;
        }
        
        size_t baseIdx = particleIdx * valuesPerStatic;
        h_staticData[baseIdx + 0] = id;
        h_staticData[baseIdx + 1] = mass;
    }
    
    // Set system data (time and energy) - once per timestep
    void setSystemData(size_t timeIdx, double time, double energy) {
        if (timeIdx >= numTimeSteps) {
            std::cerr << "OutputData::setSystemData - Index out of bounds\n";
            return;
        }
        
        size_t baseIdx = timeIdx * valuesPerSystem;
        h_systemData[baseIdx + 0] = time;
        h_systemData[baseIdx + 1] = energy;
    }
    
    // Set particle data (position and velocity) - for each particle at each timestep
    void setParticleData(size_t particleIdx, size_t timeIdx, 
                         double x, double y, double z, 
                         double vx, double vy, double vz) {
        if (particleIdx >= numParticles || timeIdx >= numTimeSteps) {
            std::cerr << "OutputData::setParticleData - Index out of bounds\n";
            return;
        }
        
        size_t baseIdx = (timeIdx * numParticles + particleIdx) * valuesPerParticle;
        h_particleData[baseIdx + 0] = x;
        h_particleData[baseIdx + 1] = y;
        h_particleData[baseIdx + 2] = z;
        h_particleData[baseIdx + 3] = vx;
        h_particleData[baseIdx + 4] = vy;
        h_particleData[baseIdx + 5] = vz;
    }
    
    // Set particle data directly from double4 structures
    void setParticleDataFromPacked(size_t particleIdx, size_t timeIdx, 
                                   double4 posMass, double4 vel) {
        if (particleIdx >= numParticles || timeIdx >= numTimeSteps) {
            std::cerr << "OutputData::setParticleDataFromPacked - Index out of bounds\n";
            return;
        }
        
        size_t baseIdx = (timeIdx * numParticles + particleIdx) * valuesPerParticle;
        h_particleData[baseIdx + 0] = posMass.x;
        h_particleData[baseIdx + 1] = posMass.y;
        h_particleData[baseIdx + 2] = posMass.z;
        h_particleData[baseIdx + 3] = vel.x;
        h_particleData[baseIdx + 4] = vel.y;
        h_particleData[baseIdx + 5] = vel.z;
        
        // Store mass in static data if it's the first timestep
        if (timeIdx == 0) {
            setStaticData(particleIdx, particleIdx, posMass.w);
        }
    }

    // Copy all device data to host asynchronously
    bool copyDeviceToHostAsync(cudaStream_t stream = 0) {
        if (mode == ExecutionMode::CPU || 
            !d_particleData || !d_systemData || !d_staticData) {
            return true; // No-op for CPU mode
        }
        
        size_t particleDataSize = numParticles * numTimeSteps * valuesPerParticle * sizeof(double);
        size_t systemDataSize = numTimeSteps * valuesPerSystem * sizeof(double);
        size_t staticDataSize = numParticles * valuesPerStatic * sizeof(double);
        
        cudaError_t err1, err2, err3;
        
        if (isPinned) {
            err1 = cudaMemcpyAsync(h_particleData, d_particleData, particleDataSize, cudaMemcpyDeviceToHost, stream);
            err2 = cudaMemcpyAsync(h_systemData, d_systemData, systemDataSize, cudaMemcpyDeviceToHost, stream);
            err3 = cudaMemcpyAsync(h_staticData, d_staticData, staticDataSize, cudaMemcpyDeviceToHost, stream);
        } else {
            err1 = cudaMemcpy(h_particleData, d_particleData, particleDataSize, cudaMemcpyDeviceToHost);
            err2 = cudaMemcpy(h_systemData, d_systemData, systemDataSize, cudaMemcpyDeviceToHost);
            err3 = cudaMemcpy(h_staticData, d_staticData, staticDataSize, cudaMemcpyDeviceToHost);
        }
        
        return (err1 == cudaSuccess && err2 == cudaSuccess && err3 == cudaSuccess);
    }
    
    // Get device pointers to specific data sections
    double* getDeviceParticleDataPtr(size_t timeIdx) const {
        if (mode == ExecutionMode::CPU || !d_particleData || timeIdx >= numTimeSteps) {
            return nullptr;
        }
        return d_particleData + (timeIdx * numParticles * valuesPerParticle);
    }
    
    double* getDeviceSystemDataPtr(size_t timeIdx = 0) const {
        if (mode == ExecutionMode::CPU || !d_systemData || timeIdx >= numTimeSteps) {
            return nullptr;
        }
        return d_systemData + (timeIdx * valuesPerSystem);
    }
    
    double* getDeviceStaticDataPtr() const {
        if (mode == ExecutionMode::CPU || !d_staticData) {
            return nullptr;
        }
        return d_staticData;
    }
};
