#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <sstream>
#include "CudaUtils.cuh"
#include "ForceKernels.cuh"
#include "IntegrationKernels.cuh"
#include "Particles.hpp"
#include "System.hpp"
#include "OutputUtils.hpp"
#include "VisualizationKernels.h"
#include "OutputData.hpp"
#include "VisualizationUtils.h"
#include "SimulationKernels.h"
#include "Constants.cuh"

#ifdef VISUALIZATION_ENABLED
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#endif

// Kernel for updating OpenGL buffers directly from CUDA
__global__ void updateParticleVBOsKernel(float4* positions, float4* velocities, 
                                        const double4* simPositions, 
                                        const double4* simVelocities,
                                        float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Convert from double to float and apply scale
        positions[i] = make_float4(
            simPositions[i].x * scale,
            simPositions[i].y * scale,
            simPositions[i].z * scale,
            simPositions[i].w  // Mass
        );
        
        velocities[i] = make_float4(
            simVelocities[i].x,
            simVelocities[i].y,
            simVelocities[i].z,
            0.0f
        );
    }
}

// Main visualization function
extern "C" void runVisualizationOnGPU(Particles& particles, 
                                      IntegrationMethod method,
                                      ForceMethod forceMethod,
                                      double dt, int steps, double stepFreq,
                                      int physicsStepsPerFrame) {
#ifdef VISUALIZATION_ENABLED
    int n = particles.n;
    std::cout << "Starting visualization with " << n << " particles..." << std::endl;
    gpuErrchk(cudaGetLastError());
    // Memory allocation for simulation
    double4 *d_posMass = nullptr;
    double4 *d_vel = nullptr;
    double4 *d_accel = nullptr;
    double4 *d_accelOld = nullptr;
    
    // Initialize CUDA (will allocate memory regardless of visualization mode)
    gpuErrchk(cudaMalloc(&d_posMass, n * sizeof(double4)));
    gpuErrchk(cudaMalloc(&d_vel, n * sizeof(double4)));
    gpuErrchk(cudaMalloc(&d_accel, n * sizeof(double4)));
    
    if (method == IntegrationMethod::VELOCITY_VERLET) {
        gpuErrchk(cudaMalloc(&d_accelOld, n * sizeof(double4)));
    }
    
    // Copy initial data to device
    gpuErrchk(cudaMemcpy(d_posMass, particles.posMass, n * sizeof(double4), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_vel, particles.vel, n * sizeof(double4), cudaMemcpyHostToDevice));
    gpuErrchk(cudaGetLastError());
    // Create CUDA streams before OpenGL initialization
    cudaStream_t computeStream;
    gpuErrchk(cudaStreamCreate(&computeStream));


    gpuErrchk(cudaGetLastError());
    bool useCuda = false;

    // Initialize visualization
    if (!VisualizationUtils::initVisualization(n, "N-body Simulation", useCuda)) {
        std::cerr << "Failed to initialize visualization. Aborting." << std::endl;
        cudaFree(d_posMass);
        cudaFree(d_vel);
        cudaFree(d_accel);
        if (d_accelOld) cudaFree(d_accelOld);
        return;
    }
    gpuErrchk(cudaGetLastError());
    VisualizationUtils::setupKeyboardCallback();
    VisualizationUtils::printKeyboardControls();
    gpuErrchk(cudaGetLastError());
    // Kernel launch parameters
    int blockSize = determineOptimalBlockSize(n, sizeof(double4) * 4);
    dim3 blocks = calculateGrid(n, blockSize);
    size_t sharedMemSize = blockSize * sizeof(double4);
    
    // Calculate average mass for needed methods
    double avgMass = 0.0;
    if (forceMethod == ForceMethod::ADAPTIVE_MUTUAL) {
        for (int i = 0; i < n; i++) {
            avgMass += particles.posMass[i].w;
        }
        avgMass /= n;
    }
    gpuErrchk(cudaGetLastError());
    // Initialize physics simulation
    initializeConstants();
    
    // 6. Initial force calculation with error handling
    try {
        launchForceKernel(forceMethod, blocks, blockSize, sharedMemSize, computeStream, 
                        d_posMass, d_accel, avgMass, n);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Initial force calculation failed: ") + 
                                cudaGetErrorString(err));
        }
        gpuErrchk(cudaGetLastError());
    } 
    catch (const std::exception& e) {
        std::cerr << "ERROR in initial setup: " << e.what() << std::endl;
        std::cerr << "Continuing with zero accelerations..." << std::endl;
        cudaMemset(d_accel, 0, n * sizeof(double4));
    }
    gpuErrchk(cudaGetLastError());
    
    // Check for CUDA-OpenGL interoperability once
    bool cudaGLInteropCapable = false;
    cudaStream_t renderStream;
    
    if (VisualizationUtils::renderer) {
        // Only try to register resources if renderer exists
        cudaGLInteropCapable = VisualizationUtils::renderer->registerCudaResources();
        
        if (cudaGLInteropCapable) {
            // Create a separate stream for rendering
            cudaStreamCreate(&renderStream);
            std::cout << "✓ CUDA-OpenGL interop enabled: Using direct GPU-to-GPU rendering" << std::endl;
        } else {
            std::cout << "✓ CUDA-OpenGL interop not available: Using CPU transfer mode" << std::endl;
        }
    }
    gpuErrchk(cudaGetLastError());
    // Set up timing
    auto simStartTime = std::chrono::high_resolution_clock::now();
    int currentStep = 0;
    int framesRendered = 0;
    
    // Calculate physics steps per frame
    int stepsPerFrame = (physicsStepsPerFrame > 0) ? 
                       physicsStepsPerFrame : 1;
    
    // Main simulation loop
    while (currentStep < steps && !glfwWindowShouldClose(VisualizationUtils::window)) {
        // Handle window events
        glfwPollEvents();
        
            // Process physics steps
            for (int i = 0; i < stepsPerFrame && currentStep < steps; i++) {
                // Call unified integration step with detailed error checking
                try {
                    // Check all pointers before calling
                    if (!d_posMass || !d_vel || !d_accel || 
                        (method == IntegrationMethod::VELOCITY_VERLET && !d_accelOld)) {
                        throw std::runtime_error("Null pointer in integration arguments");
                    }
                                        
                    // Synchronize stream before integration to ensure clean state
                    cudaStreamSynchronize(computeStream);

                    gpuErrchk(cudaGetLastError());

                    // Call our unified integration function
                    performGPUIntegrationStep(
                        d_posMass, d_vel, d_accel, d_accelOld,
                        method, forceMethod,
                        dt, avgMass, n,
                        blocks, blockSize, sharedMemSize,
                        computeStream);
                        
                    // Check for errors and synchronize
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        std::stringstream ss;
                        ss << "CUDA error after integration step " << currentStep 
                        << ": " << cudaGetErrorString(err) 
                        << " (code " << err << ")";
                        throw std::runtime_error(ss.str());
                    }
                    
                    currentStep++;
                }
                catch (const std::exception& e) {
                    std::cerr << "Error in physics simulation: " << e.what() << std::endl;
                    // Breaking out of the physics loop but continuing rendering
                    // This helps diagnose issues without crashing
                    break;
                }
            }

        // Make sure compute is done before rendering
        cudaStreamSynchronize(computeStream);
        
        // Simple choice: Direct GPU path or CPU fallback
        if (cudaGLInteropCapable && VisualizationUtils::renderer->mapCudaResources()) {
            // Direct GPU path
            float4* d_positions = VisualizationUtils::renderer->getDevicePositionPtr();
            float4* d_velocities = VisualizationUtils::renderer->getDeviceVelocityPtr();
            
            // Update VBOs directly
            updateParticleVBOsKernel<<<blocks, blockSize, 0, renderStream>>>(
                d_positions, d_velocities, d_posMass, d_vel, VisualizationUtils::scale, n);
            
            // Wait for render operations to complete
            cudaStreamSynchronize(renderStream);
            
            // Unmap resources
            VisualizationUtils::renderer->unmapCudaResources();
        } 
        else {
            // CPU fallback path - simple and direct
            cudaMemcpy(particles.posMass, d_posMass, n * sizeof(double4), cudaMemcpyDeviceToHost);
            cudaMemcpy(particles.vel, d_vel, n * sizeof(double4), cudaMemcpyDeviceToHost);
            
            if (VisualizationUtils::renderer) {
                VisualizationUtils::renderer->updatePositions(
                    particles.posMass, particles.vel, n, 
                    VisualizationUtils::scale,
                    VisualizationUtils::renderer->getTranslateX(),
                    VisualizationUtils::renderer->getTranslateY(),
                    VisualizationUtils::renderer->getZoom());
            }
        }
        
        // Render and swap buffers
        if (VisualizationUtils::renderer) {
            VisualizationUtils::renderer->render();
        }
        glfwSwapBuffers(VisualizationUtils::window);
        framesRendered++;
        
        // Update window title with performance stats occasionally
        if (framesRendered % 30 == 0) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = currentTime - simStartTime;
            double fps = framesRendered / elapsed.count();
            
            std::stringstream title;
            title << "N-body Simulation - Step " << currentStep << "/" << steps
                  << " (" << fps << " FPS)";
            glfwSetWindowTitle(VisualizationUtils::window, title.str().c_str());
        }
    }
    
    // Wait for completion and copy final state back to host
    cudaStreamSynchronize(computeStream);
    cudaMemcpy(particles.posMass, d_posMass, n * sizeof(double4), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.vel, d_vel, n * sizeof(double4), cudaMemcpyDeviceToHost);
    
    // Clean up CUDA resources
    if (VisualizationUtils::renderer) {
        VisualizationUtils::renderer->unregisterCudaResources();
    }
    
    cudaStreamDestroy(computeStream);
    if (cudaGLInteropCapable) {
        cudaStreamDestroy(renderStream);
    }
    
    cudaFree(d_posMass);
    cudaFree(d_vel);
    cudaFree(d_accel);
    if (d_accelOld) cudaFree(d_accelOld);
    
    // Clean up visualization
    VisualizationUtils::cleanupVisualization();
    
    // Print performance summary
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> totalTime = endTime - simStartTime;
    std::cout << "\nSimulation completed: " << currentStep << " steps in "
              << totalTime.count() << " seconds ("
              << (currentStep / totalTime.count()) << " steps/sec, "
              << (framesRendered / totalTime.count()) << " FPS)" << std::endl;
#else
    // Stub implementation when visualization is disabled
    std::cout << "GPU visualization mode is not enabled. Please rebuild with ENABLE_VISUALIZATION=ON." << std::endl;
    
    // Run just a few steps with the regular simulation code
    OutputData dummyOutput(particles.n, 1, ExecutionMode::GPU);
    runSimulationOnGPU(particles, method, forceMethod, dt, 
                      std::min(10, steps), 1, dummyOutput);
#endif
}