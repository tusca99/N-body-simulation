#include "CudaUtils.cuh"
#include <sstream>
#include <iostream>

void gpuAssert(cudaError_t code, const char *file, int line, bool abort) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

std::string getDeviceProperties() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        return "No CUDA devices found";
    }
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    std::stringstream ss;
    ss << "Device " << device << ": " << props.name << "\n"
       << "  Compute capability: " << props.major << "." << props.minor << "\n"
       << "  Multiprocessors: " << props.multiProcessorCount << "\n"
       << "  Max threads per block: " << props.maxThreadsPerBlock << "\n"
       << "  Shared memory per block: " << props.sharedMemPerBlock / 1024 << " KB\n"
       << "  Registers per block: " << props.regsPerBlock << "\n"
       << "  Warp size: " << props.warpSize;
    
    return ss.str();
}

int determineOptimalBlockSize(int n, size_t sharedMemPerBlock, KernelFunction kernelFunc, bool useOccupancyAPI) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    // Initialize blockSize with a default value to prevent uninitialized use warning
    int blockSize = 1024; 
    
    // Only use occupancy API if explicitly requested AND kernel function is provided
    if (useOccupancyAPI && kernelFunc != nullptr) {
        int minGridSize = 1;
        
        // Use CUDA Occupancy API to get optimal block size
        cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, 
            &blockSize, 
            kernelFunc, 
            sharedMemPerBlock, 
            0  // No dynamic shared memory
        );
        
        // Calculate theoretical occupancy
        int maxActiveBlocks = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks, 
            kernelFunc, 
            blockSize, 
            sharedMemPerBlock
        );
        
        float occupancy = (maxActiveBlocks * blockSize / props.warpSize) / 
                        (float)(props.maxThreadsPerMultiProcessor / props.warpSize);
        
        //printf("CUDA Occupancy API suggests block size: %d. Theoretical occupancy: %f\n", 
        //    blockSize, occupancy);
    }
    else {
        // Use the more sophisticated heuristic approach that worked better

        // Starting value - default to 1024 threads per block (common for many GPUs)
        blockSize = 1024;
        
        // For very small problem sizes
        if (n <= 32) return 32;
        
        // Check if shared memory might be a limitation
        // For N-body we typically need at least sizeof(double4) * blockSize bytes
        size_t sharedMemPerParticle = sizeof(double4);
        size_t requiredSharedMem = sharedMemPerParticle * blockSize;
        
        // If we would exceed shared memory, reduce block size
        while (requiredSharedMem > props.sharedMemPerBlock && blockSize > props.warpSize) {
            blockSize /= 2;
            requiredSharedMem = sharedMemPerParticle * blockSize;
        }
        
        // Ensure block size is multiple of warp size for best performance
        blockSize = (blockSize / props.warpSize) * props.warpSize;
        
        // Cap at maxThreadsPerBlock
        blockSize = std::min(blockSize, props.maxThreadsPerBlock);
        
        // For large simulations, make sure we have enough blocks to saturate SMs
        int minBlocksForSaturation = 2 * props.multiProcessorCount;
        int blocksNeeded = (n + blockSize - 1) / blockSize;
        
        // If we don't have enough blocks to saturate, reduce block size
        while (blocksNeeded < minBlocksForSaturation && blockSize > props.warpSize) {
            blockSize /= 2;
            blocksNeeded = (n + blockSize - 1) / blockSize;
        }
        
        // Ensure it's still a multiple of the warp size
        blockSize = (blockSize / props.warpSize) * props.warpSize;
        
        //printf("Using heuristic block size: %d\n", blockSize);
    }
    
    // Minimum sensible block size for modern GPUs
    if (blockSize < props.warpSize) blockSize = props.warpSize;
    
    return blockSize;
}

dim3 calculateGrid(int n, int blockSize) {
    // Simple 1D grid calculation
    return dim3((n + blockSize - 1) / blockSize);
}
