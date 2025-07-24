#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Constants.cuh"
#include "BHKernels2.cuh"
#include <cfloat>
#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include "CudaUtils.cuh"

// Global persistent device memory for Barnes-Hut algorithm
namespace BH2Memory {
    // Tree structure
    BHNode2* d_nodes = nullptr;
    int* d_nodeCount = nullptr;
    int* d_childOffsets = nullptr;
    
    // Particle data
    int* d_sortedParticleIndices = nullptr;
    int* d_particleCount = nullptr;
    int* d_offsetCount = nullptr;
    
    // Bounding box
    double3* d_minBound = nullptr;
    double3* d_maxBound = nullptr;
    
    // Sorting
    int* d_indices = nullptr;
    
    // Memory management
    bool memory_initialized = false;
    int max_particles = 0;

    // Momentum
    double3* d_blockMomentum = nullptr;
    double* d_blockMass = nullptr;
    double3* d_totalMomentum = nullptr;
    double* d_totalMass = nullptr;
    
}

// Error checking helper
void checkCudaError2(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << ": " << cudaGetErrorString(error) << std::endl;
    }
}

// Memory cleanup
void cleanupBH2Memory() {
    if (!BH2Memory::memory_initialized) return;
    
    // Print debug info before cleanup
    int nodeCount = 0;
    cudaMemcpy(&nodeCount, BH2Memory::d_nodeCount, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Cleaning up Barnes-Hut memory, used " << nodeCount << " nodes" << std::endl;
    
    cudaFree(BH2Memory::d_nodes);
    cudaFree(BH2Memory::d_nodeCount);
    cudaFree(BH2Memory::d_childOffsets);
    cudaFree(BH2Memory::d_sortedParticleIndices);
    cudaFree(BH2Memory::d_particleCount);
    cudaFree(BH2Memory::d_minBound);
    cudaFree(BH2Memory::d_maxBound);
    cudaFree(BH2Memory::d_indices);
    
    // Reset pointers to nullptr
    BH2Memory::d_nodes = nullptr;
    BH2Memory::d_nodeCount = nullptr;
    BH2Memory::d_childOffsets = nullptr;
    BH2Memory::d_sortedParticleIndices = nullptr;
    BH2Memory::d_particleCount = nullptr;
    BH2Memory::d_minBound = nullptr;
    BH2Memory::d_maxBound = nullptr;
    BH2Memory::d_indices = nullptr;
    
    BH2Memory::memory_initialized = false;
}

// Initialize memory
bool initializeBH2Memory(int n) {

    // Cleanup previous allocations
    if (BH2Memory::memory_initialized) {
        cleanupBH2Memory();
    }
    
    if (BH2Memory::memory_initialized && n <= BH2Memory::max_particles) {
        // Reset node counter
        int zero = 0;
        cudaMemcpy(BH2Memory::d_nodeCount, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(BH2Memory::d_particleCount, &zero, sizeof(int), cudaMemcpyHostToDevice);
        
        // Re-initialize child offsets to -1 (invalid)
        cudaMemset(BH2Memory::d_childOffsets, 0xFF, 8 * MAX_NODES * sizeof(int));
        
        // THIS IS IMPORTANT: Zero out the first few nodes to avoid stale data
        cudaMemset(BH2Memory::d_nodes, 0, 256 * sizeof(BHNode2)); // Clear first 256 nodes
        
        // Reset bounds and counters
        fullResetBarnesHutKernel<<<1, 1>>>(
            BH2Memory::d_nodes, 
            BH2Memory::d_nodeCount, 
            BH2Memory::d_childOffsets,
            BH2Memory::d_particleCount, 
            BH2Memory::d_minBound, 
            BH2Memory::d_maxBound
        ); 
        cudaDeviceSynchronize(); // Make sure reset completes
        return true;
    }

    

    
    // Calculate max nodes (using a conservative estimate)
    int maxNodes = std::min(MAX_NODES, n * 2);

    std::cout << "Allocating Barnes-Hut memory for " << n << " particles, " 
            << maxNodes << " nodes" << std::endl;
    
    cudaError_t error;
    
    // Allocate with error checking
    error = cudaMalloc(&BH2Memory::d_nodes, maxNodes * sizeof(BHNode2));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate d_nodes: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMalloc(&BH2Memory::d_nodeCount, sizeof(int));
    if (error != cudaSuccess) {
        cudaFree(BH2Memory::d_nodes);
        std::cerr << "Failed to allocate d_nodeCount: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMalloc(&BH2Memory::d_childOffsets, 8 * maxNodes * sizeof(int));
    if (error != cudaSuccess) {
        cudaFree(BH2Memory::d_nodes);
        cudaFree(BH2Memory::d_nodeCount);
        std::cerr << "Failed to allocate d_childOffsets: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMalloc(&BH2Memory::d_sortedParticleIndices, n * sizeof(int));
    if (error != cudaSuccess) {
        cudaFree(BH2Memory::d_nodes);
        cudaFree(BH2Memory::d_nodeCount);
        cudaFree(BH2Memory::d_childOffsets);
        std::cerr << "Failed to allocate d_sortedParticleIndices: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMalloc(&BH2Memory::d_particleCount, sizeof(int));
    if (error != cudaSuccess) {
        cudaFree(BH2Memory::d_nodes);
        cudaFree(BH2Memory::d_nodeCount);
        cudaFree(BH2Memory::d_childOffsets);
        cudaFree(BH2Memory::d_sortedParticleIndices);
        std::cerr << "Failed to allocate d_particleCount: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMalloc(&BH2Memory::d_minBound, sizeof(double3));
    if (error != cudaSuccess) {
        cudaFree(BH2Memory::d_nodes);
        cudaFree(BH2Memory::d_nodeCount);
        cudaFree(BH2Memory::d_childOffsets);
        cudaFree(BH2Memory::d_sortedParticleIndices);
        cudaFree(BH2Memory::d_particleCount);
        std::cerr << "Failed to allocate d_minBound: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMalloc(&BH2Memory::d_maxBound, sizeof(double3));
    if (error != cudaSuccess) {
        cudaFree(BH2Memory::d_nodes);
        cudaFree(BH2Memory::d_nodeCount);
        cudaFree(BH2Memory::d_childOffsets);
        cudaFree(BH2Memory::d_sortedParticleIndices);
        cudaFree(BH2Memory::d_particleCount);
        cudaFree(BH2Memory::d_minBound);
        std::cerr << "Failed to allocate d_maxBound: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMalloc(&BH2Memory::d_indices, n * sizeof(int));
    if (error != cudaSuccess) {
        cudaFree(BH2Memory::d_nodes);
        cudaFree(BH2Memory::d_nodeCount);
        cudaFree(BH2Memory::d_childOffsets);
        cudaFree(BH2Memory::d_sortedParticleIndices);
        cudaFree(BH2Memory::d_particleCount);
        cudaFree(BH2Memory::d_minBound);
        cudaFree(BH2Memory::d_maxBound);
        std::cerr << "Failed to allocate d_indices: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

        error = cudaMalloc(&BH2Memory::d_offsetCount, sizeof(int));
    if (error != cudaSuccess) {
        cudaFree(BH2Memory::d_nodes);
        cudaFree(BH2Memory::d_nodeCount);
        cudaFree(BH2Memory::d_childOffsets);
        cudaFree(BH2Memory::d_sortedParticleIndices);
        cudaFree(BH2Memory::d_particleCount);
        cudaFree(BH2Memory::d_minBound);
        cudaFree(BH2Memory::d_maxBound);
        cudaFree(BH2Memory::d_indices);
        std::cerr << "Failed to allocate d_offsetCount: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Initialize counters
    int initialNodeCount = 0;
    cudaMemcpy(BH2Memory::d_nodeCount, &initialNodeCount, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(BH2Memory::d_particleCount, &initialNodeCount, sizeof(int), cudaMemcpyHostToDevice);
    
    // Initialize child offsets to -1 (invalid)
    cudaMemset(BH2Memory::d_childOffsets, 0xFF, 8 * maxNodes * sizeof(int));
    
    // Initialize indices
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    initIndices2Kernel<<<gridSize, blockSize>>>(BH2Memory::d_indices, n);
    
    BH2Memory::memory_initialized = true;
    BH2Memory::max_particles = n;
    return true;
}

// Accessor functions
BHNode2* getNodes2Pointer() { return BH2Memory::d_nodes; }
int* getNodeCount2Pointer() { return BH2Memory::d_nodeCount; }
int* getChildOffsets2Pointer() { return BH2Memory::d_childOffsets; }
int* getSortedParticleIndices2Pointer() { return BH2Memory::d_sortedParticleIndices; }
int* getParticleCount2Pointer() { return BH2Memory::d_particleCount; }
double3* getMinBound2Pointer() { return BH2Memory::d_minBound; }
double3* getMaxBound2Pointer() { return BH2Memory::d_maxBound; }
int* getIndices2Pointer() { return BH2Memory::d_indices; }

// Initialize particle indices
__global__ void initIndices2Kernel(int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        indices[idx] = idx;
    }
}

// Compute bounding box for all particles
__global__ void computeBoundingBox2Kernel(double4* posMass, int n, double3* minBound, double3* maxBound) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for reduction
    extern __shared__ double s_data[];
    double* s_min_x = s_data;
    double* s_max_x = s_min_x + blockDim.x;
    double* s_min_y = s_max_x + blockDim.x;
    double* s_max_y = s_min_y + blockDim.x;
    double* s_min_z = s_max_y + blockDim.x;
    double* s_max_z = s_min_z + blockDim.x;
    
    // Initialize shared memory
    s_min_x[threadIdx.x] = DBL_MAX;
    s_max_x[threadIdx.x] = -DBL_MAX;
    s_min_y[threadIdx.x] = DBL_MAX;
    s_max_y[threadIdx.x] = -DBL_MAX;
    s_min_z[threadIdx.x] = DBL_MAX;
    s_max_z[threadIdx.x] = -DBL_MAX;
    
    // Process multiple particles per thread if needed
    while (idx < n) {
        double4 p = posMass[idx];
        
        s_min_x[threadIdx.x] = min(s_min_x[threadIdx.x], p.x);
        s_max_x[threadIdx.x] = max(s_max_x[threadIdx.x], p.x);
        s_min_y[threadIdx.x] = min(s_min_y[threadIdx.x], p.y);
        s_max_y[threadIdx.x] = max(s_max_y[threadIdx.x], p.y);
        s_min_z[threadIdx.x] = min(s_min_z[threadIdx.x], p.z);
        s_max_z[threadIdx.x] = max(s_max_z[threadIdx.x], p.z);
        
        idx += blockDim.x * gridDim.x;
    }
    
    // Synchronize threads in this block
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_min_x[threadIdx.x] = min(s_min_x[threadIdx.x], s_min_x[threadIdx.x + s]);
            s_max_x[threadIdx.x] = max(s_max_x[threadIdx.x], s_max_x[threadIdx.x + s]);
            s_min_y[threadIdx.x] = min(s_min_y[threadIdx.x], s_min_y[threadIdx.x + s]);
            s_max_y[threadIdx.x] = max(s_max_y[threadIdx.x], s_max_y[threadIdx.x + s]);
            s_min_z[threadIdx.x] = min(s_min_z[threadIdx.x], s_min_z[threadIdx.x + s]);
            s_max_z[threadIdx.x] = max(s_max_z[threadIdx.x], s_max_z[threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    // Write result to global memory
    if (threadIdx.x == 0) {
        // Use atomics to update global min/max
        atomicMinDouble(&minBound->x, s_min_x[0]);
        atomicMaxDouble(&maxBound->x, s_max_x[0]);
        atomicMinDouble(&minBound->y, s_min_y[0]);
        atomicMaxDouble(&maxBound->y, s_max_y[0]);
        atomicMinDouble(&minBound->z, s_min_z[0]);
        atomicMaxDouble(&maxBound->z, s_max_z[0]);
    }
}
// Initialize root node
__global__ void initRootNode2Kernel(BHNode2* nodes, int* nodeCount, double3 center, double halfWidth) {
    // Only first thread initializes the root
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Set node count to 1 (root only)
        *nodeCount = 1;
        
        // Initialize root node (index 0)
        nodes[0].center = make_float3(center.x, center.y, center.z);
        nodes[0].halfWidth = halfWidth;
        nodes[0].childOffset = -1;  // No children yet
        nodes[0].mass = 0.0;
        nodes[0].com = make_double3(center.x, center.y, center.z);
        nodes[0].flags = 0;  // Not a leaf yet
        
        // Debug print
        printf("Initialized root node: center=(%f,%f,%f), halfWidth=%f\n", 
               center.x, center.y, center.z, halfWidth);
    }
}

// Fixed version of buildTree2Kernel
__global__ void buildBarnesHutTreeKernel(BHNode2* nodes, int* nodeCount, int* childOffsets,
                                        int* offsetCount, double4* posMass, int n, double rootHalfWidth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Each thread inserts one particle
    double4 myPos = posMass[idx];
    
    // Initial values
    int nodeIdx = 0;
    float nodeHW = rootHalfWidth;
    float3 nodeCenter = nodes[0].center;
    
    int depth = 0;
    const int MAX_TREE_DEPTH = 20;
    
    // Traverse until we find where to insert this particle
    while (depth < MAX_TREE_DEPTH) {
        depth++;
        
        // Determine which octant this particle belongs to
        int octant = 0;
        if (myPos.x >= nodeCenter.x) octant |= 1;
        if (myPos.y >= nodeCenter.y) octant |= 2;
        if (myPos.z >= nodeCenter.z) octant |= 4;
        
        // Calculate child center
        float3 childCenter;
        childCenter.x = nodeCenter.x + ((octant & 1) ? nodeHW*0.5f : -nodeHW*0.5f);
        childCenter.y = nodeCenter.y + ((octant & 2) ? nodeHW*0.5f : -nodeHW*0.5f);
        childCenter.z = nodeCenter.z + ((octant & 4) ? nodeHW*0.5f : -nodeHW*0.5f);
        
        // Get childOffset - carefully check if valid
        int childOffset = nodes[nodeIdx].childOffset;
        
        // Create children if needed
        if (childOffset == -1) {
            // CRITICAL FIX: Allocate offsets separately from nodes
            int newOffsetIdx = atomicAdd(offsetCount, 1);
            int newNodeIdx = atomicAdd(nodeCount, 1);
            
            // Check array bounds
            if (newOffsetIdx >= MAX_NODES || newNodeIdx >= MAX_NODES) {
                return; // Out of space
            }
            
            // Try to set the childOffset atomically
            int oldOffset = atomicCAS(&nodes[nodeIdx].childOffset, -1, newOffsetIdx);
            
            if (oldOffset != -1) {
                // Another thread set it first - use their value
                childOffset = oldOffset;
            } else {
                // We set it - initialize the child slots
                childOffset = newOffsetIdx;
                for (int i = 0; i < 8; i++) {
                    childOffsets[childOffset * 8 + i] = -1;
                }
            }
        }
        
        // Calculate child array index
        int offsetIdx = childOffset * 8 + octant;
        
        // Try to insert our particle
        int currentValue = atomicCAS(&childOffsets[offsetIdx], -1, idx);
        
        if (currentValue == -1) {
            // Success! We inserted our particle
            return;
        } else if (currentValue >= 0 && currentValue < n) {
            // Create a new node to handle the collision
            int newNodeIdx = atomicAdd(nodeCount, 1);
            
            if (newNodeIdx >= MAX_NODES) {
                return; // Out of nodes
            }
            
            // Initialize the new node
            nodes[newNodeIdx].center = childCenter;
            nodes[newNodeIdx].halfWidth = nodeHW * 0.5f;
            nodes[newNodeIdx].childOffset = -1; // No children yet
            nodes[newNodeIdx].mass = 0.0;
            nodes[newNodeIdx].com = make_double3(0.0, 0.0, 0.0);
            nodes[newNodeIdx].flags = 0;
            
            // Try to replace the particle with our new node
            int result = atomicCAS(&childOffsets[offsetIdx], currentValue, newNodeIdx);
            
            if (result != currentValue) {
                // Another thread won the race - retry this octant
                continue;
            }
            
            // Continue traversal with our new node
            nodeIdx = newNodeIdx;
            nodeCenter = childCenter;
            nodeHW *= 0.5f;
            
            // Now we need to re-insert the displaced particle
            idx = currentValue;
            myPos = posMass[idx];
        } else {
            // This is an internal node - continue traversal
            nodeIdx = currentValue;
            nodeCenter = nodes[nodeIdx].center;
            nodeHW = nodes[nodeIdx].halfWidth;
        }
    }
}


__global__ void calculateNodeProperties2Kernel(BHNode2* nodes, double4* posMass, int* childOffsets, int nodeCount, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nodeCount) return;
    
    // Skip leaf nodes that contain particles directly
    if (nodes[idx].flags & 1) return;
    
    double nodeMass = 0.0;
    double comX = 0.0, comY = 0.0, comZ = 0.0;
    int childOffset = nodes[idx].childOffset;
    
    if (childOffset == -1) return;
    
    for (int octant = 0; octant < 8; octant++) {
        int childIdx = childOffsets[childOffset*8 + octant];
        
        if (childIdx >= 0 && childIdx < n) {
            // This is a particle
            double4 p = posMass[childIdx];
            nodeMass += p.w;
            comX += p.w * p.x;
            comY += p.w * p.y;
            comZ += p.w * p.z;
        } else if (childIdx >= n && childIdx < nodeCount) {
            // This is a node
            double childMass = nodes[childIdx].mass;
            if (childMass > 0.0) {
                nodeMass += childMass;
                comX += childMass * nodes[childIdx].com.x;
                comY += childMass * nodes[childIdx].com.y;
                comZ += childMass * nodes[childIdx].com.z;
            }
        }
    }
    
    if (nodeMass > 0.0) {
        comX /= nodeMass;
        comY /= nodeMass;
        comZ /= nodeMass;
        
        nodes[idx].mass = nodeMass;
        nodes[idx].com = make_double3(comX, comY, comZ);
    }
}


// Fixed version of force calculation kernel
__global__ void calculateForces2Kernel(BHNode2* nodes, int* childOffsets, double4* posMass, 
                                     double4* accel, int* indices, int n, float theta, 
                                     double softeningSquared) {
    // Thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Get particle data
    int particleIdx = indices[idx];
    double4 myPos = posMass[particleIdx];
    
    // Accumulate acceleration
    double ax = 0.0, ay = 0.0, az = 0.0;
    
    // Stack for tree traversal
    const int MAX_STACK = 64;
    int stack[MAX_STACK];
    int stackSize = 0;
    
    // Start with root node
    stack[stackSize++] = 0;
    
    // Traverse the tree
    while (stackSize > 0) {
        // Pop node from stack
        int nodeIdx = stack[--stackSize];
        if (nodeIdx < 0 || nodeIdx >= n) continue; // Safety check
        
        BHNode2 node = nodes[nodeIdx];
        
        // Calculate distance to node center of mass
        double dx = node.com.x - myPos.x;
        double dy = node.com.y - myPos.y;
        double dz = node.com.z - myPos.z;
        
        // Compute squared distance
        double distSqr = dx*dx + dy*dy + dz*dz;
        
        // Skip self-interaction
        if (distSqr < 1e-10) continue;
        
        // Node size for opening angle test
        double nodeSize = 2.0 * node.halfWidth;
        double theta_squared = theta * theta;
        
        // Check if node has valid mass and COM
        if (isnan(node.mass) || isnan(node.com.x) || isnan(node.com.y) || isnan(node.com.z)) {
            continue;  // Skip invalid nodes
        }
        
        // Determine whether to use this node as a whole or traverse its children
        // Check if this is a leaf with a particle or if we can use multipole approximation
        bool isLeaf = (node.childOffset == -1);
        bool canUseApproximation = (nodeSize*nodeSize < distSqr * theta_squared);

        if (isLeaf || canUseApproximation) {
            // Use node as a whole - calculate force contribution
            if (node.mass > 0) {
                double invDist = 1.0 / sqrt(distSqr + softeningSquared);
                double invDistCube = invDist * invDist * invDist;
                
                // Use G_AU from the Constants.cuh file
                double factorG = G_AU * node.mass * invDistCube;
                
                // Accumulate acceleration
                ax += factorG * dx;
                ay += factorG * dy;
                az += factorG * dz;
            }
        } else if (node.childOffset >= 0) {
            // Traverse children - fixed indexing
            for (int child = 0; child < 8; child++) {
                int childIdx = childOffsets[node.childOffset + child];
                if (childIdx >= 0 && childIdx < n && nodes[childIdx].mass > 0) {
                    if (stackSize < MAX_STACK) {
                        stack[stackSize++] = childIdx;
                    }
                }
            }
        }
    }
    
    // Write acceleration to global memory
    accel[particleIdx] = make_double4(ax, ay, az, 0.0);
}

// Verify tree integrity
__global__ void verifyTree2Kernel(BHNode2* nodes, int* childOffsets, int nodeCount, int n, int* errorCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nodeCount) return;
    
    BHNode2 node = nodes[idx];
    
    // Check for valid node properties
    if (isnan(node.mass) || isnan(node.com.x) || isnan(node.com.y) || isnan(node.com.z)) {
        atomicAdd(errorCount, 1);
        return;
    }
    
    // Skip nodes with invalid childOffset
    if (node.childOffset < 0 || node.childOffset * 8 >= nodeCount * 8) {
        atomicAdd(errorCount, 1);
        return;
    }
    
    // Check for valid child indices
    bool isLeaf = (node.flags & 1);
    
    for (int i = 0; i < 8; i++) {
        int childIdx = childOffsets[node.childOffset * 8 + i];
        if (childIdx == -1) continue; // Empty slot
        
        if (isLeaf) {
            // Leaf nodes should only have particles
            if (childIdx >= n || childIdx < 0) {
                atomicAdd(errorCount, 1);
            }
        } else {
            // Internal nodes should only have other nodes
            if (childIdx < 0 || childIdx >= nodeCount) {
                atomicAdd(errorCount, 1);
            }
        }
    }
}

// Diagnostic kernel to gather tree statistics
__global__ void diagnosticTree2Kernel(BHNode2* nodes, int* childOffsets, int nodeCount, int* stats, int n) {
    // Reset stats
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        stats[0] = 0; // Internal nodes
        stats[1] = 0; // Leaf nodes
        stats[2] = 0; // Particles
        stats[3] = 0; // Max depth
        stats[4] = 0; // Empty leaves
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nodeCount) return;
    
    BHNode2 node = nodes[idx];
    bool isLeaf = (node.flags & 1);
    
    if (isLeaf) {
        atomicAdd(&stats[1], 1); // Leaf node count
        
        int particleCount = 0;
        for (int i = 0; i < 8; i++) {
            int childIdx = childOffsets[node.childOffset * 8 + i];
            if (childIdx >= 0 && childIdx < n) {
                particleCount++;
            }
        }
        
        atomicAdd(&stats[2], particleCount); // Particle count
        
        if (particleCount == 0) {
            atomicAdd(&stats[4], 1); // Empty leaf count
        }
    } else {
        atomicAdd(&stats[0], 1); // Internal node count
    }
    
    // Track max depth - this is more complex with direct-insert trees
    // Since we don't have a level field anymore, we need another approach
}

// Ensure we have a valid CUDA device
bool ensureValidCudaDevice2() {
    int device = -1;
    cudaError_t error = cudaGetDevice(&device);
    
    if (error == cudaSuccess && device >= 0) {
        void* testPtr = nullptr;
        error = cudaMalloc(&testPtr, 16);
        if (error == cudaSuccess) {
            cudaFree(testPtr);
            return true;
        }
    }
    
    int deviceCount = 0;
    error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA devices available: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    for (int i = 0; i < deviceCount; i++) {
        error = cudaSetDevice(i);
        if (error != cudaSuccess) continue;
        
        void* testPtr = nullptr;
        error = cudaMalloc(&testPtr, 16);
        if (error == cudaSuccess) {
            cudaFree(testPtr);
            std::cout << "Using CUDA device " << i << std::endl;
            return true;
        }
    }
    
    std::cerr << "Failed to initialize any CUDA device" << std::endl;
    return false;
}


// Atomic operations for double precision
__device__ double atomicMinDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                      __double_as_longlong(min(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                      __double_as_longlong(max(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
} 

// Add this helper function for double precision atomic add
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void computeSimpleBoundingBox2Kernel(double4* posMass, int n, double3* minBound, double3* maxBound) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Local min/max for this thread
    double minX = DBL_MAX, minY = DBL_MAX, minZ = DBL_MAX;
    double maxX = -DBL_MAX, maxY = -DBL_MAX, maxZ = -DBL_MAX;
    
    // Process multiple elements per thread
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        double4 p = posMass[i];
        
        // Update local min/max
        minX = min(minX, p.x);
        maxX = max(maxX, p.x);
        minY = min(minY, p.y);
        maxY = max(maxY, p.y);
        minZ = min(minZ, p.z);
        maxZ = max(maxZ, p.z);
    }
    
    // Only update global min/max if we processed at least one particle
    if (minX != DBL_MAX) {
        atomicMinDouble(&minBound->x, minX);
        atomicMaxDouble(&maxBound->x, maxX);
        atomicMinDouble(&minBound->y, minY);
        atomicMaxDouble(&maxBound->y, maxY);
        atomicMinDouble(&minBound->z, minZ);
        atomicMaxDouble(&maxBound->z, maxZ);
    }
}

__global__ void fullResetBarnesHutKernel(BHNode2* nodes, int* nodeCount, int* childOffsets,
                                        int* particleCount, double3* minBound, double3* maxBound) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *nodeCount = 0;
        *particleCount = 0;
        *minBound = make_double3(DBL_MAX, DBL_MAX, DBL_MAX);
        *maxBound = make_double3(-DBL_MAX, -DBL_MAX, -DBL_MAX);
    }
}


// First kernel computes total momentum
__global__ void computeTotalMomentumKernel(double4* posMass, double4* accel, int n, 
                                        double* d_momentum, double* d_totalMass) {
    // Use shared memory for reduction
    extern __shared__ double sharedData[];
    double* s_momentum_x = sharedData;
    double* s_momentum_y = s_momentum_x + blockDim.x;
    double* s_momentum_z = s_momentum_y + blockDim.x;
    double* s_mass = s_momentum_z + blockDim.x;
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Initialize shared memory
    s_momentum_x[tid] = 0.0;
    s_momentum_y[tid] = 0.0;
    s_momentum_z[tid] = 0.0;
    s_mass[tid] = 0.0;
    
    // Accumulate momentum and mass
    if (idx < n) {
        double4 pos = posMass[idx];
        double4 acc = accel[idx];
        
        double mass = pos.w;
        s_momentum_x[tid] = mass * acc.x;
        s_momentum_y[tid] = mass * acc.y;
        s_momentum_z[tid] = mass * acc.z;
        s_mass[tid] = mass;
    }
    
    __syncthreads();
    
    // Perform reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_momentum_x[tid] += s_momentum_x[tid + stride];
            s_momentum_y[tid] += s_momentum_y[tid + stride];
            s_momentum_z[tid] += s_momentum_z[tid + stride];
            s_mass[tid] += s_mass[tid + stride];
        }
        __syncthreads();
    }
    
    // Write block results to global memory
    if (tid == 0) {
        atomicAddDouble(&d_momentum[0], s_momentum_x[0]);
        atomicAddDouble(&d_momentum[1], s_momentum_y[0]);
        atomicAddDouble(&d_momentum[2], s_momentum_z[0]);
        atomicAddDouble(d_totalMass, s_mass[0]);
    }
}

// Second kernel applies the correction
__global__ void applyMomentumCorrectionKernel(double4* accel, int n, 
                                           double* d_momentum, double* d_totalMass) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Load values once from global memory
    double totalMass = *d_totalMass;
    double px = d_momentum[0];
    double py = d_momentum[1];
    double pz = d_momentum[2];
    
    if (totalMass > 0.0) {
        // Calculate correction factors
        double correction_x = px / totalMass;
        double correction_y = py / totalMass;
        double correction_z = pz / totalMass;
        
        // Apply to acceleration
        accel[idx].x -= correction_x;
        accel[idx].y -= correction_y;
        accel[idx].z -= correction_z;
    }
}

__global__ void checkMomentumKernel(double4* posMass, double4* accel, int n, double* totalMomentum) {
    __shared__ double3 s_momentum[256];
    int tid = threadIdx.x;
    s_momentum[tid] = make_double3(0, 0, 0);
    
    // Each thread sums a portion of particles
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        s_momentum[tid].x += posMass[i].w * accel[i].x;
        s_momentum[tid].y += posMass[i].w * accel[i].y;
        s_momentum[tid].z += posMass[i].w * accel[i].z;
    }
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            s_momentum[tid].x += s_momentum[tid + s].x;
            s_momentum[tid].y += s_momentum[tid + s].y;
            s_momentum[tid].z += s_momentum[tid + s].z;
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        totalMomentum[0] = s_momentum[0].x;
        totalMomentum[1] = s_momentum[0].y;
        totalMomentum[2] = s_momentum[0].z;
    }
}



extern "C" void calculateBarnesHutForces(double4* posMass, double4* accel, 
                                        int n, float theta, 
                                        cudaStream_t stream) {
    
    // Start overall timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    // Calculate launch parameters
    int blockSize = determineOptimalBlockSize(n);
    dim3 gridSize = calculateGrid(n, blockSize);
    
    std::cout << "[BH2] Starting Barnes-Hut calculation for " << n << " particles..." << std::endl;
    
    // Ensure we have a valid CUDA device
    if (!ensureValidCudaDevice2()) {
        std::cerr << "Failed to initialize CUDA device for Barnes-Hut" << std::endl;
        return;
    }
    
    // Ensure memory is allocated
    if (!initializeBH2Memory(n)) {
        std::cerr << "Failed to initialize Barnes-Hut memory" << std::endl;
        return;
    }
    
    std::cout << "[BH2] Memory initialized successfully" << std::endl;
    
    BHNode2* d_nodes = getNodes2Pointer();
    int* d_nodeCount = getNodeCount2Pointer();
    int* d_childOffsets = getChildOffsets2Pointer();
    int* d_particleCount = getParticleCount2Pointer();
    double3* d_minBound = getMinBound2Pointer();
    double3* d_maxBound = getMaxBound2Pointer();
    int* d_indices = getIndices2Pointer();
    int* d_sortedParticleIndices = getSortedParticleIndices2Pointer();

    // 1. Compute bounding box
    std::cout << "[BH2] Computing bounding box..." << std::endl;

    // Launch the simpler kernel
    computeSimpleBoundingBox2Kernel<<<gridSize, blockSize, 0, stream>>>(
        posMass, n, d_minBound, d_maxBound);

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Compute bounding box failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    
    // Copy bounds back to host
    double3 minBound, maxBound;
    cudaMemcpy(&minBound, d_minBound, sizeof(double3), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxBound, d_maxBound, sizeof(double3), cudaMemcpyDeviceToHost);
    
    std::cout << "[BH2] Bounding box computed: " 
              << "min=(" << minBound.x << "," << minBound.y << "," << minBound.z << ") "
              << "max=(" << maxBound.x << "," << maxBound.y << "," << maxBound.z << ")" << std::endl;
    
    // Step 2: Create root node with expanded bounds
    double maxRange = std::max(std::max(maxBound.x - minBound.x, maxBound.y - minBound.y), 
                             maxBound.z - minBound.z) * 1.01; // Add 1% margin
    double3 rootCenter = make_double3(
        (minBound.x + maxBound.x) * 0.5,
        (minBound.y + maxBound.y) * 0.5,
        (minBound.z + maxBound.z) * 0.5
    );
    double rootHalfWidth = maxRange * 0.5;
    
    // Step 3: Build the octree using the one-pass approach
    std::cout << "[BH2] Building octree..." << std::endl;

    /// Reset node count to 1 (root only)
    int rootOnly = 1;
    cudaMemcpy(d_nodeCount, &rootOnly, sizeof(int), cudaMemcpyHostToDevice);
    
    // Initialize all child offsets to -1
    cudaMemset(d_childOffsets, 0xFF, 8 * MAX_NODES * sizeof(int));
    
    // Create initial root node
    rootCenter = make_double3(
        (maxBound.x + minBound.x) * 0.5,
        (maxBound.y + minBound.y) * 0.5,
        (maxBound.z + minBound.z) * 0.5
    );
    
    rootHalfWidth = std::max(
        std::max((maxBound.x - minBound.x) * 0.5, (maxBound.y - minBound.y) * 0.5),
        (maxBound.z - minBound.z) * 0.5
    );
    
    // Add some padding to the bounding box
    rootHalfWidth *= 1.01;
    
    // Initialize root node
    initRootNode2Kernel<<<1, 1, 0, stream>>>(d_nodes, d_nodeCount, rootCenter, rootHalfWidth);
    
    // Build the tree with extensive debugging
    rootOnly = 1;
    int offsetZero = 0;
    cudaMemcpy(d_nodeCount, &rootOnly, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(BH2Memory::d_offsetCount, &offsetZero, sizeof(int), cudaMemcpyHostToDevice);

    // Pass the offset counter to the kernel
    buildBarnesHutTreeKernel<<<gridSize, blockSize, 0, stream>>>(
        d_nodes, d_nodeCount, d_childOffsets, BH2Memory::d_offsetCount, 
        posMass, n, rootHalfWidth);
    // Check for errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Build tree failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    // Get final node count
    int finalNodeCount;
    cudaMemcpy(&finalNodeCount, d_nodeCount, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "[BH2] Tree built with " << finalNodeCount << " nodes" << std::endl;
    
    // Step 6: Calculate node properties (mass and center of mass)
    std::cout << "[BH2] Calculating node properties..." << std::endl;
    
    int nodeGridSize = (finalNodeCount + blockSize - 1) / blockSize;
    
    // Calculate properties for leaf nodes first
    calculateNodeProperties2Kernel<<<nodeGridSize, blockSize, 0, stream>>>(
        d_nodes, posMass, d_childOffsets, finalNodeCount, n);
    
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Calculate node properties failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    
    // Replace the propagation loop with:
    // Calculate properties for all nodes in multiple passes
    // This ensures propagation from bottom to top
    const int NUM_PASSES = 5; // Usually log(MAX_DEPTH) is sufficient
    for (int pass = 0; pass < NUM_PASSES; pass++) {
        calculateNodeProperties2Kernel<<<nodeGridSize, blockSize, 0, stream>>>(
            d_nodes, posMass, d_childOffsets, finalNodeCount, n);
        
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "Node properties calculation pass " << pass 
                    << " failed: " << cudaGetErrorString(error) << std::endl;
            return;
        }
    }
    
    // Step 7: Verify tree integrity
    std::cout << "[BH2] Verifying tree integrity..." << std::endl;
    
    int* d_errorCount;
    cudaMalloc(&d_errorCount, sizeof(int));
    cudaMemset(d_errorCount, 0, sizeof(int));
    
    verifyTree2Kernel<<<nodeGridSize, blockSize, 0, stream>>>(
        d_nodes, d_childOffsets, finalNodeCount, n, d_errorCount);
    
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Tree verification failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    
    int errorCount;
    cudaMemcpy(&errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_errorCount);
    
    if (errorCount > 0) {
        std::cerr << "WARNING: Tree verification found " << errorCount << " issues!" << std::endl;
    }
    
    // Tree diagnostics
    int* d_stats;
    cudaMalloc(&d_stats, 5 * sizeof(int));
    cudaMemset(d_stats, 0, 5 * sizeof(int));
    
    diagnosticTree2Kernel<<<1, 1, 0, stream>>>(d_nodes, d_childOffsets, finalNodeCount, d_stats, n);
    
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Tree diagnostics failed: " << cudaGetErrorString(error) << std::endl;
    } else {
        int stats[5];
        cudaMemcpy(stats, d_stats, 5 * sizeof(int), cudaMemcpyDeviceToHost);
        
        std::cout << "[BH2] Tree stats: "
                  << stats[0] << " internal nodes, "
                  << stats[1] << " leaf nodes, "
                  << stats[2] << "/" << n << " particles in leaves, "
                  << "max depth " << stats[3] << ", "
                  << stats[4] << " empty leaves" << std::endl;
    }
    cudaFree(d_stats);
    
    // Step 8: Calculate forces
    std::cout << "[BH2] Calculating forces..." << std::endl;
    
    double softeningSquared = 1e-8; // Small softening to avoid singularities
    
    calculateForces2Kernel<<<gridSize, blockSize, 0, stream>>>(
        d_nodes, d_childOffsets, posMass, accel, d_indices, n, theta, softeningSquared);
    
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Force calculation failed: " << cudaGetErrorString(error) << std::endl;
        
        // Fall back to direct calculation in case of error
        std::cout << "[BH2] Falling back to direct calculation..." << std::endl;
        
        // Call your direct calculation kernel here
        return;
    }
    
    // Record total time
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float total_time = 0;
    cudaEventElapsedTime(&total_time, start, stop);
    
    std::cout << "[BH2] Barnes-Hut calculation complete in " << total_time << " ms" << std::endl;

    // Apply momentum conservation
    std::cout << "[BH2] Applying momentum conservation..." << std::endl;

    // Allocate memory for momentum data
    double* d_momentum;
    double* d_totalMass;
    cudaMalloc(&d_momentum, 3 * sizeof(double));
    cudaMalloc(&d_totalMass, sizeof(double));
    cudaMemset(d_momentum, 0, 3 * sizeof(double));
    cudaMemset(d_totalMass, 0, sizeof(double));

    // Calculate total momentum with the first kernel
    int momentumBlockSize = 256;
    int momentumGridSize = (n + momentumBlockSize - 1) / momentumBlockSize;
    size_t sharedMemSize = momentumBlockSize * 4 * sizeof(double);

    computeTotalMomentumKernel<<<momentumGridSize, momentumBlockSize, sharedMemSize, stream>>>(
        posMass, accel, n, d_momentum, d_totalMass);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Momentum calculation failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    // Apply the correction with the second kernel
    applyMomentumCorrectionKernel<<<momentumGridSize, momentumBlockSize, 0, stream>>>(
        accel, n, d_momentum, d_totalMass);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Momentum correction failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    // Free temporary memory
    cudaFree(d_momentum);
    cudaFree(d_totalMass);
    
    // Clean up timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}