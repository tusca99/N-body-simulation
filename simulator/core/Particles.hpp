#pragma once 
#include <cstddef>
#include <stdexcept>
#include <cuda_runtime.h>
#include <iostream>
#include "ExecutionModes.hpp" // Include the execution modes directly

// CUDA-friendly particle structure using double4 vectors with pinned memory
struct Particles {
    size_t n;

    // Using double4 with posMass.w storing mass
    double4* posMass;   // (x, y, z, mass)
    double4* vel;       // (vx, vy, vz, 0)
    bool isPinned;      // Flag to track if memory is pinned
    ExecutionMode mode; // Store the execution mode
    
    // Default constructor:
    Particles()
        : n(0), posMass(nullptr), vel(nullptr), isPinned(false), mode(ExecutionMode::CPU)
    {}

    // Constructor with size that uses pinned memory:
    Particles(size_t nParticles, ExecutionMode execMode = ExecutionMode::CPU)
        : n(nParticles), posMass(nullptr), vel(nullptr), isPinned(false), mode(execMode)
    {
        // Only use pinned memory for GPU mode with significant particle counts
        // Small allocations don't benefit much from pinning and waste memory
        if (mode == ExecutionMode::GPU && n > 1000) {
            cudaError_t err1 = cudaMallocHost(&posMass, n * sizeof(double4));
            cudaError_t err2 = cudaMallocHost(&vel, n * sizeof(double4));
            
            if (err1 == cudaSuccess && err2 == cudaSuccess) {
                isPinned = true;
            } else {
                if (err1 == cudaSuccess) cudaFreeHost(posMass);
                if (err2 == cudaSuccess) cudaFreeHost(vel);
                
                posMass = new double4[n];
                vel = new double4[n];
                isPinned = false;
            }
        } else {
            // For CPU mode or small allocations, use regular memory
            posMass = new double4[n];
            vel = new double4[n];
            isPinned = false;
        }

        // Initialize to zero
        for (size_t i = 0; i < n; ++i) {
            posMass[i] = make_double4(0.0, 0.0, 0.0, 0.0);
            vel[i] = make_double4(0.0, 0.0, 0.0, 0.0);
        }
    }

    // Copy constructor:
    Particles(const Particles& other)
        : n(other.n), posMass(nullptr), vel(nullptr), isPinned(false), mode(other.mode)
    {
        if (mode == ExecutionMode::GPU) {
            // For GPU mode, try to allocate pinned memory
            cudaError_t err1 = cudaMallocHost(&posMass, n * sizeof(double4));
            cudaError_t err2 = cudaMallocHost(&vel, n * sizeof(double4));
            
            if (err1 == cudaSuccess && err2 == cudaSuccess) {
                // Successfully allocated pinned memory
                isPinned = true;
            } else {
                // Fall back to regular memory
                if (err1 == cudaSuccess) cudaFreeHost(posMass);
                if (err2 == cudaSuccess) cudaFreeHost(vel);
                
                posMass = new double4[n];
                vel = new double4[n];
                isPinned = false;
            }
        } else {
            // For CPU mode, use regular memory
            posMass = new double4[n];
            vel = new double4[n];
            isPinned = false;
        }
        
        // Copy data
        for (size_t i = 0; i < n; ++i) {
            posMass[i] = other.posMass[i];
            vel[i] = other.vel[i];
        }
    }

    // Move constructor:
    Particles(Particles&& other) noexcept
        : n(other.n),
          posMass(other.posMass),
          vel(other.vel),
          isPinned(other.isPinned),
          mode(other.mode)
    {
        other.n = 0;
        other.posMass = nullptr;
        other.vel = nullptr;
        other.isPinned = false;
    }

    // Copy assignment operator:
    Particles& operator=(const Particles& other) {
        if (this != &other) {
            // Free existing memory
            freeMemory();

            n = other.n;
            mode = other.mode;
            
            if (mode == ExecutionMode::GPU) {
                // For GPU mode, allocate pinned memory if possible
                cudaError_t err1 = cudaMallocHost(&posMass, n * sizeof(double4));
                cudaError_t err2 = cudaMallocHost(&vel, n * sizeof(double4));
                
                if (err1 == cudaSuccess && err2 == cudaSuccess) {
                    // Successfully allocated pinned memory
                    isPinned = true;
                } else {
                    // Fall back to regular memory
                    if (err1 == cudaSuccess) cudaFreeHost(posMass);
                    if (err2 == cudaSuccess) cudaFreeHost(vel);
                    
                    posMass = new double4[n];
                    vel = new double4[n];
                    isPinned = false;
                }
            } else {
                // For CPU mode, use regular memory
                posMass = new double4[n];
                vel = new double4[n];
                isPinned = false;
            }
            
            // Copy data
            for (size_t i = 0; i < n; ++i) {
                posMass[i] = other.posMass[i];
                vel[i] = other.vel[i];
            }
        }
        return *this;
    }

    // Move assignment operator:
    Particles& operator=(Particles&& other) noexcept {
        if (this != &other) {
            // Free existing memory
            freeMemory();

            n = other.n;
            posMass = other.posMass;
            vel = other.vel;
            isPinned = other.isPinned;
            mode = other.mode;

            other.n = 0;
            other.posMass = nullptr;
            other.vel = nullptr;
            other.isPinned = false;
        }
        return *this;
    }

    // Updated helper to free memory correctly using the isPinned flag
    void freeMemory() {
        if (posMass) {
            if (isPinned) {
                cudaFreeHost(posMass);
            } else {
                delete[] posMass;
            }
            posMass = nullptr;
        }
        
        if (vel) {
            if (isPinned) {
                cudaFreeHost(vel);
            } else {
                delete[] vel;
            }
            vel = nullptr;
        }
        
        isPinned = false;
    }

    // Destructor: 
    ~Particles() {
        freeMemory();
    }

    // Set particle data at index i
    void setParticle(size_t i, double mass, double px, double py, double pz,
                     double vxVal, double vyVal, double vzVal) {
        if (i >= n) throw std::out_of_range("Invalid particle index");
        posMass[i] = make_double4(px, py, pz, mass);
        vel[i] = make_double4(vxVal, vyVal, vzVal, 0.0);
    }

    // Set execution mode (may need to reallocate memory)
    void setExecutionMode(ExecutionMode newMode) {
        if (mode == newMode) return;  // No change needed
        
        // Save current data
        double4* tempPosMass = nullptr;
        double4* tempVel = nullptr;
        bool needToRestore = (n > 0 && posMass != nullptr && vel != nullptr);
        
        if (needToRestore) {
            tempPosMass = new double4[n];
            tempVel = new double4[n];
            for (size_t i = 0; i < n; ++i) {
                tempPosMass[i] = posMass[i];
                tempVel[i] = vel[i];
            }
        }
        
        // Free existing memory
        freeMemory();
        
        // Set new mode
        mode = newMode;
        
        // Reallocate with new mode
        if (n > 0) {
            if (mode == ExecutionMode::GPU) {
                cudaError_t err1 = cudaMallocHost(&posMass, n * sizeof(double4));
                cudaError_t err2 = cudaMallocHost(&vel, n * sizeof(double4));
                
                if (err1 == cudaSuccess && err2 == cudaSuccess) {
                    isPinned = true;
                } else {
                    if (err1 == cudaSuccess) cudaFreeHost(posMass);
                    if (err2 == cudaSuccess) cudaFreeHost(vel);
                    
                    posMass = new double4[n];
                    vel = new double4[n];
                    isPinned = false;
                }
            } else {
                posMass = new double4[n];
                vel = new double4[n];
                isPinned = false;
            }
            
            // Restore data if we had it
            if (needToRestore) {
                for (size_t i = 0; i < n; ++i) {
                    posMass[i] = tempPosMass[i];
                    vel[i] = tempVel[i];
                }
                delete[] tempPosMass;
                delete[] tempVel;
            }
        }
    }

    // Efficient accessor methods
    // These don't allocate memory and are CUDA-friendly
    
    // Get mass of particle i
    __host__ __device__ double getMass(size_t i) const {
        #ifdef __CUDA_ARCH__
        // Device code path - no exceptions
        return (i < n) ? posMass[i].w : 0.0;
        #else
        // Host code path - can use exceptions
        if (i >= n) throw std::out_of_range("Invalid particle index");
        return posMass[i].w;
        #endif
    }
    
    // Get x-coordinate of particle i
    __host__ __device__ double getX(size_t i) const {
        #ifdef __CUDA_ARCH__
        return (i < n) ? posMass[i].x : 0.0;
        #else
        if (i >= n) throw std::out_of_range("Invalid particle index");
        return posMass[i].x;
        #endif
    }
    
    // Get y-coordinate of particle i
    __host__ __device__ double getY(size_t i) const {
        #ifdef __CUDA_ARCH__
        return (i < n) ? posMass[i].y : 0.0;
        #else
        if (i >= n) throw std::out_of_range("Invalid particle index");
        return posMass[i].y;
        #endif
    }
    
    // Get z-coordinate of particle i
    __host__ __device__ double getZ(size_t i) const {
        #ifdef __CUDA_ARCH__
        return (i < n) ? posMass[i].z : 0.0;
        #else
        if (i >= n) throw std::out_of_range("Invalid particle index");
        return posMass[i].z;
        #endif
    }
    
    // Get x-velocity of particle i
    __host__ __device__ double getVx(size_t i) const {
        #ifdef __CUDA_ARCH__
        return (i < n) ? vel[i].x : 0.0;
        #else
        if (i >= n) throw std::out_of_range("Invalid particle index");
        return vel[i].x;
        #endif
    }
    
    // Get y-velocity of particle i
    __host__ __device__ double getVy(size_t i) const {
        #ifdef __CUDA_ARCH__
        return (i < n) ? vel[i].y : 0.0;
        #else
        if (i >= n) throw std::out_of_range("Invalid particle index");
        return vel[i].y;
        #endif
    }
    
    // Get z-velocity of particle i
    __host__ __device__ double getVz(size_t i) const {
        #ifdef __CUDA_ARCH__
        return (i < n) ? vel[i].z : 0.0;
        #else
        if (i >= n) throw std::out_of_range("Invalid particle index");
        return vel[i].z;
        #endif
    }
    
    // Backward compatibility methods
    // These allocate new memory - use with caution and remember to free
    // Ideally, code should be updated to use the element-wise accessors above
    
    double* m() const { 
        double* mass = new double[n];
        for (size_t i = 0; i < n; ++i) {
            mass[i] = posMass[i].w;
        }
        return mass;
    }
    
    double* x() const { 
        double* x_coords = new double[n];
        for (size_t i = 0; i < n; ++i) {
            x_coords[i] = posMass[i].x;
        }
        return x_coords;
    }
    
    double* y() const { 
        double* y_coords = new double[n];
        for (size_t i = 0; i < n; ++i) {
            y_coords[i] = posMass[i].y;
        }
        return y_coords;
    }
    
    double* z() const { 
        double* z_coords = new double[n];
        for (size_t i = 0; i < n; ++i) {
            z_coords[i] = posMass[i].z;
        }
        return z_coords;
    }
    
    double* vx() const { 
        double* vx_vals = new double[n];
        for (size_t i = 0; i < n; ++i) {
            vx_vals[i] = vel[i].x;
        }
        return vx_vals;
    }
    
    double* vy() const { 
        double* vy_vals = new double[n];
        for (size_t i = 0; i < n; ++i) {
            vy_vals[i] = vel[i].y;
        }
        return vy_vals;
    }
    
    double* vz() const { 
        double* vz_vals = new double[n];
        for (size_t i = 0; i < n; ++i) {
            vz_vals[i] = vel[i].z;
        }
        return vz_vals;
    }
};