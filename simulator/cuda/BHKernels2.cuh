#pragma once

#include <cuda_runtime.h>
#include "Constants.cuh"
#include <cfloat>

// Constants for Barnes-Hut algorithm
#define MAX_DEPTH 16
#define MAX_NODES (1 << 22)  // About 4 million nodes
#define DEFAULT_THETA 0.5f   // Opening angle threshold

__device__ __constant__ double3 d_minBoundInit = {DBL_MAX, DBL_MAX, DBL_MAX};
__device__ __constant__ double3 d_maxBoundInit = {-DBL_MAX, -DBL_MAX, -DBL_MAX};

// Optimized node structure combining the best of both approaches
struct BHNode2 {
    float3 center;      // Center position of node
    float halfWidth;    // Half width of node cube
    int childOffset;    // Offset to child nodes (multiply by 8 for actual index)
    double mass;        // Total mass of node
    double3 com;        // Center of mass
    int flags;          // Bit flags (bit 0 = isLeaf)
};

// Function declarations
bool initializeBH2Memory(int n);
void cleanupBH2Memory();

// BH tree functions
__global__ void initIndices2Kernel(int* indices, int n);
__global__ void computeBoundingBox2Kernel(double4* posMass, int n, double3* minBound, double3* maxBound);

__global__ void calculateNodeProperties2Kernel(BHNode2* nodes, double4* posMass, 
                                            int* sortedParticleIndices, int nodeCount, int n);
__global__ void calculateForces2Kernel(BHNode2* nodes, int* childOffsets, double4* posMass, 
                                     double4* accel, int* indices, int n, float theta, 
                                     double softeningSquared);
__global__ void verifyTree2Kernel(BHNode2* nodes, int* childOffsets, int nodeCount, int n, int* errorCount);
__global__ void diagnosticTree2Kernel(BHNode2* nodes, int* childOffsets, int nodeCount, int* stats, int n);

__global__ void initRootNode2Kernel(BHNode2* nodes, int* nodeCount, double3 center, double halfWidth);



__global__ void computeSimpleBoundingBox2Kernel(double4* posMass, int n, double3* minBound, double3* maxBound);

__global__ void fullResetBarnesHutKernel(BHNode2* nodes, int* nodeCount, int* childOffsets,
                                        int* particleCount, double3* minBound, double3* maxBound);


__global__ void computeTotalMomentumKernel(double4* posMass, double4* accel, int n, 
                                        double* d_momentum, double* d_totalMass);
__global__ void checkMomentumKernel(double4* posMass, double4* accel, int n, double* totalMomentum);
__global__ void applyMomentumCorrectionKernel(double4* accel, int n, 
                                           double* d_momentum, double* d_totalMass);

                                        



__device__ double atomicMaxDouble(double* address, double val);
__device__ double atomicMinDouble(double* address, double val);
__device__ double atomicAddDouble(double* address, double val);


// Memory accessors
BHNode2* getNodes2Pointer();
int* getNodeCount2Pointer();
int* getChildOffsets2Pointer();
int* getSortedParticleIndices2Pointer();
int* getParticleCount2Pointer();
double3* getMinBound2Pointer();
double3* getMaxBound2Pointer();
int* getIndices2Pointer();



extern "C" void calculateBarnesHutForces(double4* posMass, double4* accel, 
                                        int n, float theta, 
                                        cudaStream_t stream);