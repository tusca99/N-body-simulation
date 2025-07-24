#pragma once

#include <cuda_runtime.h>
#include <string>

// Error checking helper
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

// Macro for error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// Function to query device properties and return them as a readable string
std::string getDeviceProperties();

// Generic function type for kernels - make it a plain function pointer
typedef void (*KernelFunction)(void);

// Updated function signature with additional parameter to control whether to use the occupancy API
int determineOptimalBlockSize(int n, size_t sharedMemPerBlock = 0, KernelFunction kernelFunc = nullptr, bool useOccupancyAPI = false);

/*
The key insight here is that theoretical occupancy doesn't always translate to optimal runtime performance, especially for N-body simulations. 
Your empirical finding that 32 threads per block performs much better than 1024 is actually quite common for N-body simulations for several reasons:

    Memory access patterns - N-body simulations involve complex, non-coalesced memory access patterns that can benefit from smaller warps and more blocks
    Shared memory bank conflicts - Larger block sizes can cause more bank conflicts in shared memory operations
    Register pressure - N-body kernels often use many registers per thread, which limits occupancy regardless of block size
    Instruction-level parallelism - Smaller blocks allow more blocks to run concurrently, often better utilizing the GPU

By using this more conservative approach biased toward smaller block sizes (especially favoring 32, which matches your empirical finding), 
you should see much better performance while still benefiting from the occupancy API's hardware-specific information.
*/


// Function to calculate grid dimensions based on block size and problem size
dim3 calculateGrid(int n, int blockSize);
