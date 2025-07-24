#include "ForceKernels.cuh"
#include "CudaUtils.cuh"
#include "BHKernels2.cuh"
//#include "BHKernels.cuh"
//#include "MinimalBHKernels.cuh"
#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cmath>
//#include <cfloat>

// Optimized CUDA kernel for force calculation without softening
__global__ void calculatePairwiseForceKernel(double4* posMass, double4* accel, int n) {
    // Each thread block loads one tile of particles into shared memory
    extern __shared__ double4 sharedPos[];
    
    // Thread index within the grid
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Register cache for my particle's properties
    double4 myPos;
    double acc_x = 0.0f;
    double acc_y = 0.0f;
    double acc_z = 0.0f;
    
    // Cache my particle's position if within bounds
    if (idx < n) {
        myPos = posMass[idx];
    }
    
    // Calculate number of tiles needed
    int numTiles = (n + blockDim.x - 1) / blockDim.x;
    
    // Process all tiles
    for (int tile = 0; tile < numTiles; tile++) {
        // Collaboratively load particle data into shared memory
        int tileStart = tile * blockDim.x;
        int sharedIdx = tileStart + threadIdx.x;
        
        // Make sure we don't access beyond array bounds
        if (sharedIdx < n) {
            sharedPos[threadIdx.x] = posMass[sharedIdx];
        }
        __syncthreads();
        
        // Compute interactions with particles in this tile
        if (idx < n) {
            // Use register-based loop index for better performance
            int limit = min(blockDim.x, n - tileStart);
            
            // Removed #pragma unroll - let compiler decide optimal unrolling
            for (int j = 0; j < limit; j++) {
                // Get jth particle from shared memory
                double4 jPos = sharedPos[j];
                
                // Compute displacement
                double dx = jPos.x - myPos.x;
                double dy = jPos.y - myPos.y;
                double dz = jPos.z - myPos.z;
                
                // Compute distance squared
                double distSqr = dx*dx + dy*dy + dz*dz;
                
                // Skip self-interaction and very close encounters
                bool nonZeroDistMask = (distSqr > 1e-10);
                bool notSelfMask = (tileStart + j != idx);
                bool interactMask = nonZeroDistMask && notSelfMask;
                
                if (interactMask) {
                    // Compute interaction strength (1/r^3) - NO SOFTENING
                    double invDist = rsqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;
                    
                    // Compute acceleration contribution
                    #ifdef __CUDA_ARCH__
                    double fac = G_AU * jPos.w * invDist3;
                    #else
                    double fac = G_AU * jPos.w * invDist3;
                    #endif
                    
                    // Accumulate acceleration
                    acc_x += fac * dx;
                    acc_y += fac * dy;
                    acc_z += fac * dz;
                }
            }
        }
        __syncthreads(); // Wait for all threads before loading next tile
    }
    
    // Write acceleration to global memory
    if (idx < n) {
        accel[idx] = make_double4(acc_x, acc_y, acc_z, 0.0);
    }
}

// Optimized CUDA kernel for adaptive force calculation with Hill radius consideration
__global__ void calculateAdaptiveForcesKernel(double4* posMass, double4* accel, double avgMass, int n) {
    extern __shared__ double4 sharedPos[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double acc_x = 0.0;
    double acc_y = 0.0;
    double acc_z = 0.0;
    
    double4 myPos;
    if (idx < n) {
        myPos = posMass[idx];
    }
    
    int numTiles = (n + blockDim.x - 1) / blockDim.x;
    
    for (int tile = 0; tile < numTiles; tile++) {
        // Collaboratively load particle data
        int tileStart = tile * blockDim.x;
        int sharedIdx = tileStart + threadIdx.x;
        
        if (sharedIdx < n) {
            sharedPos[threadIdx.x] = posMass[sharedIdx];
        }
        __syncthreads();
        
        if (idx < n) {
            int limit = min(blockDim.x, n - tileStart);
            
            // Removed #pragma unroll - consistent with pairwise kernel
            for (int j = 0; j < limit; j++) {
                double4 jPos = sharedPos[j];
                int jIdx = tileStart + j;
                
                if (idx != jIdx) {  // Avoid self-interaction
                    double dx = jPos.x - myPos.x;
                    double dy = jPos.y - myPos.y;
                    double dz = jPos.z - myPos.z;
                    double distSqr = dx*dx + dy*dy + dz*dz;
                    
                    if (distSqr > 1e-10) {
                        double dist = sqrt(distSqr);
                        
                        // Adaptive softening based on local mass and distance
                        double combinedMass = myPos.w + jPos.w;
                        double epsilon = ETA * dist * cbrt(combinedMass / (3.0 * avgMass));
                        epsilon = max(epsilon, EPSILON_MIN);
                        
                        double softDistSqr = distSqr + epsilon*epsilon;
                        double invDist = rsqrt(softDistSqr);
                        double invDist3 = invDist * invDist * invDist;
                        
                        #ifdef __CUDA_ARCH__
                        double fac = G_AU * jPos.w * invDist3;
                        #else
                        double fac = G_AU * jPos.w * invDist3;
                        #endif
                        
                        // Acceleration accumulation
                        acc_x += fac * dx;
                        acc_y += fac * dy;
                        acc_z += fac * dz;
                    }
                }
            }
        }
        __syncthreads();
    }
    
    if (idx < n) {
        accel[idx] = make_double4(acc_x, acc_y, acc_z, 0.0);
    }
}


__global__ void calculatePairwiseAccelOpt(const double4* __restrict__ posMass, double4* __restrict__ accel, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    double4 pi = posMass[tid];
    double ax = 0.0, ay = 0.0, az = 0.0;

    #pragma unroll 4
    for (int j = 0; j < n; ++j) {
        double4 pj = posMass[j];
        double dx = pj.x - pi.x;
        double dy = pj.y - pi.y;
        double dz = pj.z - pi.z;

        double r2 = dx*dx + dy*dy + dz*dz + 1e-10;

        if (r2 > 0.0) {
            float rinv_f = rsqrtf((float)r2);
            double rinv = (double)rinv_f;
            rinv = rinv * (1.5 - 0.5 * r2 * rinv * rinv); // Newton step

            double rinv3 = rinv * rinv * rinv;
            double f = G_AU * pj.w * rinv3;

            ax += f * dx;
            ay += f * dy;
            az += f * dz;
        }
    }

    accel[tid] = make_double4(ax, ay, az, 0.0);
}


// Force kernel selector function implementation
void launchForceKernel(ForceMethod method, dim3 blocks, int blockSize, size_t sharedMemSize,
                     cudaStream_t stream, double4* posMass, double4* accel, double avgMass, int n) {
    switch (method) {
        case ForceMethod::PAIRWISE:
            calculatePairwiseForceKernel<<<blocks, blockSize, sharedMemSize, stream>>>(
                posMass, accel, n);
            break;
        case ForceMethod::PAIRWISE_AVX2_FP32:
            calculatePairwiseAccelOpt<<<blocks, blockSize, sharedMemSize, stream>>>(
                posMass, accel, n);
            break;
        case ForceMethod::ADAPTIVE_MUTUAL:
            calculateAdaptiveForcesKernel<<<blocks, blockSize, sharedMemSize, stream>>>(
                posMass, accel, avgMass, n);
            break;
        case ForceMethod::BARNES_HUT:
            calculateBarnesHutForces(posMass, accel, n, 0.1f, stream);
            break;

        default:
            std::cout << "Warning: Using default pairwise force method on GPU" << std::endl;
            calculatePairwiseForceKernel<<<blocks, blockSize, sharedMemSize, stream>>>(
                posMass, accel, n);
    }
}

// Helper to get appropriate kernel function pointer for occupancy calculation
// Changed return type and name to match the declaration
void* getForceKernelPointer(ForceMethod method) {
    switch (method) {
        case ForceMethod::PAIRWISE:
            return (void*)calculatePairwiseForceKernel;
        case ForceMethod::PAIRWISE_AVX2_FP32:
            return (void*)calculatePairwiseAccelOpt;
        case ForceMethod::ADAPTIVE_MUTUAL:
            return (void*)calculateAdaptiveForcesKernel;
        case ForceMethod::BARNES_HUT:
            return (void*)calculateBarnesHutForces;
        default:
            return (void*)calculatePairwiseForceKernel;
    }
}