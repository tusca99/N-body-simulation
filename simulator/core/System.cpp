#include "System.hpp"
#include "OutputUtils.hpp"
#include "OutputData.hpp"
#include <iostream>
#include <fstream>
#include <cassert>
#include <omp.h>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <array>
#include <chrono>  // Add this for timing functionality
#include "SimulationKernels.h"
#include "VisualizationKernels.h"
#include "BenchmarkKernels.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "VisualizationUtils.h"


// Forward declarations for CUDA functions
// Only declare when needed based on build mode
#ifdef BENCHMARK_MODE
extern "C" void runBenchmarkOnGPU(Particles& particles, 
                                 IntegrationMethod method,
                                 ForceMethod forceMethod,
                                 double dt, int steps);
#endif

#ifdef VISUALIZATION_ENABLED
extern "C" void runVisualizationOnGPU(Particles& particles, 
                                     IntegrationMethod method,
                                     ForceMethod forceMethod,
                                     double dt, int steps, double stepFreq,
                                     int physicsStepsPerFrame);
#endif

namespace
{
    constexpr double G_AU = 39.47841760435743;

    // Helper to subtract center-of-mass velocity
    auto centerOfMassVelocity(const Particles &p) {
        double M = 0.0, vxCM = 0.0, vyCM = 0.0, vzCM = 0.0;
        for (size_t i = 0; i < p.n; ++i) {
            M += p.posMass[i].w;
        }
        for (size_t i = 0; i < p.n; ++i) {
            vxCM += p.posMass[i].w * p.vel[i].x;
            vyCM += p.posMass[i].w * p.vel[i].y;
            vzCM += p.posMass[i].w * p.vel[i].z;
        }
        vxCM /= M; vyCM /= M; vzCM /= M;
        return std::array<double, 3>{vxCM, vyCM, vzCM};
    }
}

double System::computeTotalEnergy() const
{
    double totalKinetic = 0.0;
    double totalPotential = 0.0;

    // Kinetic
    for (size_t i = 0; i < particles.n; ++i) {
        double v2 = particles.vel[i].x * particles.vel[i].x 
                  + particles.vel[i].y * particles.vel[i].y
                  + particles.vel[i].z * particles.vel[i].z;
        totalKinetic += 0.5 * particles.posMass[i].w * v2;
    }

    // Potential
    for (size_t i = 0; i < particles.n; ++i) {
        for (size_t j = i+1; j < particles.n; ++j) {
            double dx = particles.posMass[j].x - particles.posMass[i].x;
            double dy = particles.posMass[j].y - particles.posMass[i].y;
            double dz = particles.posMass[j].z - particles.posMass[i].z;
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (dist > 0)
                totalPotential -= G_AU * particles.posMass[i].w * particles.posMass[j].w / dist;
        }
    }
    return totalKinetic + totalPotential;
}

void System::runSimulation(double dt, int steps, double stepFreq,
                           const std::string &outputFilename,
                           const std::string &metadata)
{
    // Dispatch to appropriate implementation based on output mode
    switch (outputMode) {
        case OutputMode::BENCHMARK:
            runBenchmark(dt, steps);
            return;
            
        case OutputMode::VISUALIZATION:
            runVisualization(dt, steps, stepFreq);
            return;
            
        case OutputMode::FILE_CSV:
            // Continue with existing CSV output logic
            break;
    }
    
    // Original CSV output logic
    OutputData outputData(particles.n, steps / stepFreq + 1, executionMode);
    
    // Handle different execution modes
    if (executionMode == ExecutionMode::GPU) {
        // Run the entire simulation on GPU - use the runSimulationOnGPU function from cuda directory
        runGPUSimulation(dt, steps, stepFreq, outputData);
    }
    else {
        // CPU mode (default)
        
        // Store initial state - always on CPU for first state
        double initialEnergy = computeTotalEnergy();
        outputData.setSystemData(0, 0.0, initialEnergy);
        
        // Store initial particle positions and velocities
        for (size_t p = 0; p < particles.n; ++p) {
            outputData.setParticleDataFromPacked(p, 0, particles.posMass[p], particles.vel[p]);
        }
        
        // Main simulation loop
        int printCounter = 0;
        size_t timeIdx = 1; // Start from 1 since 0 is the initial state
        for (int i = 0; i < steps; ++i) {
            // Perform a single integration step
            performIntegrationStep(dt);
            
            // Store state if it's time
            if (++printCounter >= static_cast<int>(stepFreq)) {
                double currentTime = (i+1) * dt;
                double currentEnergy = computeTotalEnergy();
                
                // Store system data (time and energy)
                outputData.setSystemData(timeIdx, currentTime, currentEnergy);
                
                // Store particle data (position and velocity)
                for (size_t p = 0; p < particles.n; ++p) {
                    outputData.setParticleDataFromPacked(p, timeIdx, particles.posMass[p], particles.vel[p]);
                }
                
                timeIdx++;
                printCounter = 0;
            }
            
            // Update progress bar using existing utility
            printProgressBar(i, steps);
        }
    }
    
    // Write to file at the end, regardless of execution mode
    flushCSVOutput(outputData, outputFilename, metadata);
}

// GPU simulation method
void System::runGPUSimulation(double dt, int steps, double stepFreq, OutputData& outputData) {
    std::cout << "Running simulation on GPU..." << std::endl;
    
    // Call the CUDA implementation from cuda directory
    runSimulationOnGPU(particles, method, forceMethod, dt, steps, stepFreq, outputData);
}

// Benchmark method - no output, no energy calculation, just pure performance
void System::runBenchmark(double dt, int steps) {
    //std::cout << "Running benchmark with " << steps << " steps..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if (executionMode == ExecutionMode::GPU) {
        runGPUBenchmark(dt, steps);
    } else {
        // CPU benchmark implementation - no output, no energy calculation
        for (int i = 0; i < steps; ++i) {
            // Perform a single integration step
            performIntegrationStep(dt);
            
            // Update progress using existing utility
            //printProgressBar(i, steps);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(end - start).count();

    double performance = steps / elapsed_seconds; // steps per second

    //std::cout << "\nBenchmark completed in " << elapsed_seconds << " seconds." << std::endl;
    //std::cout << "Performance: " << performance << " steps/second" << std::endl;
    //std::cout << "             " << (performance * particles.n) / 1e6 << " million particle-steps/second" << std::endl;
}

// GPU benchmark implementation - with fallback to standard simulation
void System::runGPUBenchmark(double dt, int steps) {
    // Create a minimal OutputData instance for fallback
    OutputData dummyOutput(particles.n, 1, ExecutionMode::GPU);
    
    #ifndef BENCHMARK_MODE
    std::cout << "Benchmark mode is not enabled in this build. Using standard simulation." << std::endl;
    runGPUSimulation(dt, steps, 1, dummyOutput);
    #else                             
    runBenchmarkOnGPU(particles, method, forceMethod, dt, steps);
    #endif
}

// Visualization method implementation using real-time rendering
void System::runVisualization(double dt, int steps, double stepFreq) {
    std::cout << "Running visualization with " << steps << " steps..." << std::endl;
    
    if (executionMode == ExecutionMode::GPU) {
        runGPUVisualization(dt, steps, stepFreq);
    } else {
        // CPU visualization implementation directly here
        int n = particles.n;
        std::cout << "Starting CPU visualization with " << n << " particles..." << std::endl;
        
        // Print keyboard controls
        VisualizationUtils::printKeyboardControls();
        
        // Initialize visualization - explicitly specify CPU mode
        bool useCuda = false; // CPU mode
        if (!VisualizationUtils::initVisualization(n, "N-body Simulation (CPU)", useCuda)) {
            std::cerr << "Failed to initialize visualization. Aborting." << std::endl;
            return;
        }
        
        // Setup keyboard handlers
        VisualizationUtils::setupKeyboardCallback();
        
        // Analyze particle distribution to set a good scale factor
        float maxDistance = 0.0f;
        
        for (int i = 0; i < n; i++) {
            float dist = std::sqrt(
                particles.posMass[i].x * particles.posMass[i].x +
                particles.posMass[i].y * particles.posMass[i].y +
                particles.posMass[i].z * particles.posMass[i].z
            );
            maxDistance = std::max(maxDistance, dist);
        }
        
        // Adjust scale based on particle distribution
        if (maxDistance > 0) {
            float targetViewSize = 2.0f; // Target size in view space
            VisualizationUtils::scale = targetViewSize / maxDistance;
        }
        
        // Set up timing and performance tracking
        auto simStartTime = std::chrono::high_resolution_clock::now();
        int currentStep = 0;
        int framesRendered = 0;
        int lastProgressPercent = 0;
        
        // Target FPS and automatic physics step calculation
        const double targetFPS = 30.0;
        
        // Calculate steps per frame based on desired duration
        double simulationDurationSeconds = steps * dt;
        double totalFrames = targetFPS * simulationDurationSeconds;
        
        // Adaptive physics steps per frame calculation
        int calculatedStepsPerFrame = std::max(1, static_cast<int>(steps / totalFrames));
        
        // Use user provided value if given, otherwise use calculated value
        int physicsStepsPerFrame = 0; // Auto-calculate
        const int adaptiveStepsPerFrame = (physicsStepsPerFrame > 0) ? 
                                        physicsStepsPerFrame : calculatedStepsPerFrame;
        
        std::cout << "Simulation duration: " << simulationDurationSeconds << " seconds" << std::endl;
        std::cout << "Estimated total frames: " << totalFrames << std::endl; 
        std::cout << "Physics timestep: " << dt << " seconds" << std::endl;
        std::cout << "Target rendering frame rate: " << targetFPS << " FPS" << std::endl;
        std::cout << "Calculated physics steps per frame: " << calculatedStepsPerFrame << std::endl;
        
        // Set default color mode to velocity-based and initialize view
        if (VisualizationUtils::renderer) {
            VisualizationUtils::renderer->setColorMode(ParticleRenderer::ColorMode::VELOCITY_MAGNITUDE);
            VisualizationUtils::renderer->resetView();
            VisualizationUtils::renderer->setPointSize(30.0f);
        }
        
        // Initial update of the renderer with current particles
        if (VisualizationUtils::renderer) {
            VisualizationUtils::renderer->updatePositions(
                particles.posMass, particles.vel, n, 
                VisualizationUtils::scale, 
                VisualizationUtils::renderer->getTranslateX(), 
                VisualizationUtils::renderer->getTranslateY(), 
                VisualizationUtils::renderer->getZoom()
            );
        }
        
        // Main simulation loop
        while (currentStep < steps && !glfwWindowShouldClose(VisualizationUtils::window)) {
            // Handle window events
            glfwPollEvents();
            
            // Process physics steps - forcing at least 1 physics step per frame
            int stepsThisFrame = 0;
            
            do {
                // Perform integration step using the system
                performIntegrationStep(dt);
                
                currentStep++;
                stepsThisFrame++;
                
            } while (stepsThisFrame < adaptiveStepsPerFrame && currentStep < steps);
            
            // Update renderer with current particles
            if (VisualizationUtils::renderer) {
                VisualizationUtils::renderer->updatePositions(
                    particles.posMass, particles.vel, n, 
                    VisualizationUtils::scale, 
                    VisualizationUtils::renderer->getTranslateX(), 
                    VisualizationUtils::renderer->getTranslateY(), 
                    VisualizationUtils::renderer->getZoom()
                );
            }
            
            // Render the scene
            if (VisualizationUtils::renderer) {
                VisualizationUtils::renderer->render();
            }
            
            // Swap buffers
            glfwSwapBuffers(VisualizationUtils::window);
            framesRendered++;
            
            // Calculate progress and update console
            int progressPercent = (currentStep * 100) / steps;
            if (progressPercent > lastProgressPercent) {
                printProgressBar(currentStep, steps);
                lastProgressPercent = progressPercent;
            }
            
            // Update performance stats in window title
            if (framesRendered % 10 == 0) {
                auto currentTime = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> totalElapsed = currentTime - simStartTime;
                double elapsedSeconds = totalElapsed.count();
                double actualFPS = framesRendered / elapsedSeconds;
                double stepsPerSecond = currentStep / elapsedSeconds;
                
                VisualizationUtils::updateWindowTitle(currentStep, steps, actualFPS, stepsPerSecond);
            }
        }
        
        // Clean up
        VisualizationUtils::cleanupVisualization();
        
        // Print final performance statistics
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> totalTime = endTime - simStartTime;
        
        std::cout << "\nCPU Visualization completed: " << currentStep << " steps in "
                  << totalTime.count() << " seconds ("
                  << (currentStep / totalTime.count()) << " steps/sec, "
                  << (framesRendered / totalTime.count()) << " FPS)" << std::endl;
    }
}

// GPU visualization implementation with fallback to standard simulation
void System::runGPUVisualization(double dt, int steps, double stepFreq) {
    // Track simulation time
    auto start = std::chrono::high_resolution_clock::now();
    
    // Pass 0 for physicsStepsPerFrame to enable automatic calculation
    // The visualization will calculate steps per frame based on dt and steps
    int useAutoStepCalculation = 0;
    
    runVisualizationOnGPU(particles, method, forceMethod, dt, steps, stepFreq, useAutoStepCalculation);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    double simulationTime = diff.count();
    
    std::cout << "Simulation time: " << simulationTime << " seconds (using GPU)" << std::endl;
}

// New: Euler integrator using accelerations directly
void System::eulerIntegration(double dt)
{
    auto vcm = centerOfMassVelocity(particles);
    
    // Process particles in blocks for better cache utilization
    const size_t BLOCK_SIZE = 64; // Cache-friendly block size
    
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < particles.n; i += BLOCK_SIZE) {
        size_t end = std::min(i + BLOCK_SIZE, particles.n);
        for (size_t j = i; j < end; ++j) {
            particles.vel[j].x -= vcm[0];
            particles.vel[j].y -= vcm[1];
            particles.vel[j].z -= vcm[2];
        }
    }

    // Use the central acceleration calculation method
    Accelerations accel = calculateAccelerations();
    
    // Update positions and velocities with blocked processing
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < particles.n; i += BLOCK_SIZE) {
        size_t end = std::min(i + BLOCK_SIZE, particles.n);
        
        for (size_t j = i; j < end; ++j) {
            // Prefetch next particle data
            if (j + 1 < end) {
                __builtin_prefetch(&particles.posMass[j+1], 1);
                __builtin_prefetch(&particles.vel[j+1], 1);
                __builtin_prefetch(&accel.accel[j+1], 0);
            }
            
            // Update position
            particles.posMass[j].x += particles.vel[j].x * dt;
            particles.posMass[j].y += particles.vel[j].y * dt;
            particles.posMass[j].z += particles.vel[j].z * dt;
            
            // Apply accelerations directly without dividing by mass
            particles.vel[j].x += accel.accel[j].x * dt;
            particles.vel[j].y += accel.accel[j].y * dt;
            particles.vel[j].z += accel.accel[j].z * dt;
        }
    }
}

// Updated Velocity Verlet to use accelerations directly
void System::velocityVerlet(double dt)
{
    auto vcm = centerOfMassVelocity(particles);
    
    const size_t BLOCK_SIZE = 64;
    
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < particles.n; i += BLOCK_SIZE) {
        size_t end = std::min(i + BLOCK_SIZE, particles.n);
        for (size_t j = i; j < end; ++j) {
            particles.vel[j].x -= vcm[0];
            particles.vel[j].y -= vcm[1];
            particles.vel[j].z -= vcm[2];
        }
    }

    // Calculate accelerations
    Accelerations accel = calculateAccelerations();

    // Position update with blocked processing
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < particles.n; i += BLOCK_SIZE) {
        size_t end = std::min(i + BLOCK_SIZE, particles.n);
        
        for (size_t j = i; j < end; ++j) {
            if (j + 1 < end) {
                __builtin_prefetch(&particles.posMass[j+1], 1);
                __builtin_prefetch(&particles.vel[j+1], 1);
                __builtin_prefetch(&accel.accel[j+1], 0);
            }
            
            particles.posMass[j].x += particles.vel[j].x * dt + 0.5 * accel.accel[j].x * (dt*dt);
            particles.posMass[j].y += particles.vel[j].y * dt + 0.5 * accel.accel[j].y * (dt*dt);
            particles.posMass[j].z += particles.vel[j].z * dt + 0.5 * accel.accel[j].z * (dt*dt);
        }
    }

    // Recompute accelerations with new positions
    Accelerations newAccel = calculateAccelerations();

    // Velocity update with blocked processing
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < particles.n; i += BLOCK_SIZE) {
        size_t end = std::min(i + BLOCK_SIZE, particles.n);
        
        for (size_t j = i; j < end; ++j) {
            if (j + 1 < end) {
                __builtin_prefetch(&particles.vel[j+1], 1);
                __builtin_prefetch(&accel.accel[j+1], 0);
                __builtin_prefetch(&newAccel.accel[j+1], 0);
            }
            
            particles.vel[j].x += 0.5 * (accel.accel[j].x + newAccel.accel[j].x) * dt;
            particles.vel[j].y += 0.5 * (accel.accel[j].y + newAccel.accel[j].y) * dt;
            particles.vel[j].z += 0.5 * (accel.accel[j].z + newAccel.accel[j].z) * dt;
        }
    }
}

// Perform a single integration step and return the current energy
void System::performIntegrationStep(double dt) {
    // Use the appropriate integration method based on the selected method
    if (method == IntegrationMethod::EULER) {
        eulerIntegration(dt);
    } else {
        velocityVerlet(dt);
    }
}