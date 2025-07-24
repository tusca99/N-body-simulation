#include "IntegrationKernels.cuh"
#include "CudaUtils.cuh"
#include <cooperative_groups.h>
#include "Constants.cuh"

namespace {
    //constexpr double G_AU = 39.47841760435743;  // G in astronomical units
}

// Custom implementation of atomicAdd for double for older GPU architectures
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// Optimized CUDA kernel for velocity-verlet position update (first half)
__global__ void velocityVerletPositionKernel(double4* posMass, double4* vel, double4* accel, int n, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Load particle data into registers
    double4 pos = posMass[i];
    double4 velocity = vel[i];
    double4 acceleration = accel[i];
    
    // Update position using velocity verlet (first half)
    pos.x += velocity.x * dt + 0.5 * acceleration.x * dt * dt;
    pos.y += velocity.y * dt + 0.5 * acceleration.y * dt * dt;
    pos.z += velocity.z * dt + 0.5 * acceleration.z * dt * dt;
    
    // Write updated position back
    posMass[i] = pos;
}

// Optimized CUDA kernel for velocity-verlet velocity update (second half)
__global__ void velocityVerletVelocityKernel(double4* vel, double4* accelOld, double4* accelNew, int n, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Load particle data into registers
    double4 velocity = vel[i];
    double4 oldAccel = accelOld[i];
    double4 newAccel = accelNew[i];
    
    // Update velocity using average acceleration
    velocity.x += 0.5 * (oldAccel.x + newAccel.x) * dt;
    velocity.y += 0.5 * (oldAccel.y + newAccel.y) * dt;
    velocity.z += 0.5 * (oldAccel.z + newAccel.z) * dt;
    
    // Write updated velocity back
    vel[i] = velocity;
}

// Optimized CUDA kernel for Euler integration
__global__ void eulerIntegrationKernel(double4* posMass, double4* vel, double4* accel, int n, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Load particle data into registers
    double4 pos = posMass[i];
    double4 velocity = vel[i];
    double4 acceleration = accel[i];
    
    // Update position using current velocity
    pos.x += velocity.x * dt;
    pos.y += velocity.y * dt;
    pos.z += velocity.z * dt;
    
    // Update velocity using current acceleration
    velocity.x += acceleration.x * dt;
    velocity.y += acceleration.y * dt;
    velocity.z += acceleration.z * dt;
    
    // Write updated values back
    posMass[i] = pos;
    vel[i] = velocity;
}

// Optimized kernel to compute total energy with improved memory access pattern and warp-level operations
__global__ void computeTotalEnergyKernel(
    const double4* __restrict__ posMass,
    const double4* __restrict__ vel,
    double* totalEnergy,
    int n
) {
    extern __shared__ double sharedEnergy[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    double energy = 0.0;

    // Kinetic energy
    if (i < n) {
        double4 myPosMass = posMass[i];
        double4 myVel = vel[i];
        double v2 = myVel.x * myVel.x + myVel.y * myVel.y + myVel.z * myVel.z;
        energy += 0.5 * myPosMass.w * v2;
    }

    // Potential energy (avoid double counting)
    for (int j = 0; j < n; j += blockDim.x) {
        __syncthreads();
        // Load tile into shared memory (SoA)
        if (j + tid < n) {
            double4 other = posMass[j + tid];
            sharedEnergy[tid] = 0.0; // Not used, just for alignment
            sharedEnergy[tid + blockDim.x] = other.x;
            sharedEnergy[tid + 2 * blockDim.x] = other.y;
            sharedEnergy[tid + 3 * blockDim.x] = other.z;
            sharedEnergy[tid + 4 * blockDim.x] = other.w;
        }
        __syncthreads();

        if (i < n) {
            int limit = min(blockDim.x, n - j);
            double4 myPosMass = posMass[i];
            for (int k = 0; k < limit; ++k) {
                int idx = j + k;
                if (i < idx) { // Only compute for i < j to avoid double counting and self-interaction
                    double dx = sharedEnergy[k + blockDim.x] - myPosMass.x;
                    double dy = sharedEnergy[k + 2 * blockDim.x] - myPosMass.y;
                    double dz = sharedEnergy[k + 3 * blockDim.x] - myPosMass.z;
                    double distSqr = dx * dx + dy * dy + dz * dz;
                    if (distSqr > 1e-10) {
                        double dist = sqrt(distSqr);
                        double m1 = myPosMass.w;
                        double m2 = sharedEnergy[k + 4 * blockDim.x];
                        energy -= G_AU * m1 * m2 / dist;
                    }
                }
            }
        }
    }

    // Store thread's energy contribution in shared memory
    sharedEnergy[tid] = energy;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedEnergy[tid] += sharedEnergy[tid + stride];
        }
        __syncthreads();
    }

    // Write block's result to global memory with atomic operation
    if (tid == 0) {
        atomicAdd(totalEnergy, sharedEnergy[0]);
    }
}

// Store particle and system state efficiently (modified to use single energy pointer)
__global__ void storeParticleStateKernel(double4* posMass, double4* vel, 
                                      double* particleData, double* systemData,
                                      double* totalEnergy, // Now a device pointer
                                      int particleCount, int timeStep, double time) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // First thread writes system data (time and energy) - only once per timestep
    if (i == 0) {
        systemData[timeStep * 2] = time;                 // time
        systemData[timeStep * 2 + 1] = *totalEnergy;     // read energy from device pointer
    }
    
    // All applicable threads write particle data
    if (i < particleCount) {
        const int valuesPerParticle = 6;  // x, y, z, vx, vy, vz
        size_t baseIndex = (timeStep * particleCount + i) * valuesPerParticle;
        
        // Coalesced memory access - threads in a warp access consecutive memory
        particleData[baseIndex + 0] = posMass[i].x;  // x
        particleData[baseIndex + 1] = posMass[i].y;  // y
        particleData[baseIndex + 2] = posMass[i].z;  // z
        particleData[baseIndex + 3] = vel[i].x;      // vx
        particleData[baseIndex + 4] = vel[i].y;      // vy
        particleData[baseIndex + 5] = vel[i].z;      // vz
    }
}

// Optimized kernel to store static data with coalesced memory access
__global__ void storeStaticDataKernel(double4* posMass, double* staticData, int particleCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < particleCount) {
        staticData[i * 2] = i;                // particle_id
        staticData[i * 2 + 1] = posMass[i].w; // mass
    }
}

// Unified integration step function for GPU
extern "C" void performGPUIntegrationStep(
    double4* d_posMass, double4* d_vel, double4* d_accel, double4* d_accelOld,
    IntegrationMethod method, ForceMethod forceMethod,
    double dt, double avgMass, int n,
    dim3 blocks, int blockSize, size_t sharedMemSize,
    cudaStream_t stream) {

    /*
    // Debug counters
    static int callCount = 0;
    callCount++;

    // Add debugging info for initial frames
    if (callCount <= 20 || callCount % 100 == 0) {
        printf("Integration call #%d: n=%d, method=%d, force=%d\n", 
            callCount, n, static_cast<int>(method), static_cast<int>(forceMethod));
    }

    // Validate parameters
    if (n <= 0 || blockSize <= 0 || blocks.x <= 0) {
        printf("ERROR: Invalid parameters in integration step: n=%d, blocks=%d, blockSize=%d\n",
            n, blocks.x, blockSize);
        return;
    }

    // Validate pointers
    if (!d_posMass || !d_vel || !d_accel || 
        (method == IntegrationMethod::VELOCITY_VERLET && !d_accelOld)) {
        printf("ERROR: Null pointer in integration step (call #%d)\n", callCount);
        return;
    }
    */
        
    // Step 1: Based on integration method
    if (method == IntegrationMethod::VELOCITY_VERLET) {
        // Save current accelerations before updating positions
        cudaMemcpyAsync(d_accelOld, d_accel, n * sizeof(double4), 
                       cudaMemcpyDeviceToDevice, stream);
        
        // Update positions based on current velocity and acceleration
        velocityVerletPositionKernel<<<blocks, blockSize, 0, stream>>>(
            d_posMass, d_vel, d_accel, n, dt);
        gpuErrchk(cudaGetLastError());
        
        // Recalculate forces/accelerations with new positions
        launchForceKernel(forceMethod, blocks, blockSize, sharedMemSize, stream, 
                         d_posMass, d_accel, avgMass, n);
        gpuErrchk(cudaGetLastError());
        
        // Update velocities using both old and new accelerations
        velocityVerletVelocityKernel<<<blocks, blockSize, 0, stream>>>(
            d_vel, d_accelOld, d_accel, n, dt);
        gpuErrchk(cudaGetLastError());
    }
    else { // Euler integration
        // Update positions and velocities
        gpuErrchk(cudaGetLastError());
        eulerIntegrationKernel<<<blocks, blockSize, 0, stream>>>(
            d_posMass, d_vel, d_accel, n, dt);
        gpuErrchk(cudaGetLastError());
        
        // Calculate new accelerations
        launchForceKernel(forceMethod, blocks, blockSize, sharedMemSize, stream, 
                         d_posMass, d_accel, avgMass, n);
        gpuErrchk(cudaGetLastError());
    }
}